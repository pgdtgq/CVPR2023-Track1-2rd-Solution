import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
from functools import partial


def _reduce(loss, reduction, **kwargs):
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        normalizer = loss.numel()
        if kwargs.get('normalizer', None):
            normalizer = kwargs['normalizer']
        ret = loss.sum() / normalizer
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        raise ValueError(reduction + ' is not valid')
    return ret

class EqualizedFocalLoss(nn.Layer):
    def __init__(self,
                 num_classes=1204,
                 ignore_index=-1,
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 scale_factor=8.0,
                 fpn_levels=4):
        super().__init__()
        # cfg for focal loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.ignore_index = ignore_index

        # ignore bg class and ignore idx
        self.num_classes = num_classes

        # cfg for efl loss
        self.scale_factor = scale_factor
        # initial variables
        self.register_buffer('pos_grad', paddle.zeros([self.num_classes]))
        self.register_buffer('neg_grad', paddle.zeros([self.num_classes]))
        self.register_buffer('pos_neg', paddle.ones([self.num_classes]))

        # grad collect
        self.grad_buffer = []
        self.fpn_levels = fpn_levels

    def forward(self, input, target, expand_target, reduction='mean', normalizer=None):
        # input: [B, num_queries, num_cls] cuda
        # target: [B, num_queries] 应该就是每个对应的类别
        # expand_target: [B, num_queries, num_cls] cls_label的one-hot编码形式
        self.n_c = input.shape[-1] 
        self.input = input.reshape([-1, self.n_c]) # 变成[B*num_queries, num_cls]
        self.target = target.reshape([-1]) # 变成[B]
        self.n_i, _ = self.input.shape # B*num_queries

        inputs = self.input  # [B*num_queries, num_cls] cuda
        expand_target = expand_target.reshape([-1, self.n_c])
        targets = expand_target  # [B, num_queries, num_cls] cuda
        self.cache_target = expand_target

        pred = F.sigmoid(inputs) # [B*num_queries, num_cls]
        pred_t = pred * targets + (1 - pred) * (1 - targets)  # pt  [B*num_queries, num_cls] cuda

        map_val = (1 - self.pos_neg.detach()) # [num_cls] 
        # self.focal_gamma: r_b
        dy_gamma = self.focal_gamma + self.scale_factor * map_val  # (r_b + (r_v ^ j))  [num_cls]
        # focusing factor
        dyre = dy_gamma.reshape([1, -1])  # [1, num_cls]
        ff = dy_gamma.reshape([1, -1]).expand([self.n_i, self.n_c])    # [B*num_queries, num_cls]
        # weighting factor
        wf = ff / self.focal_gamma  # [B*num_queries, num_cls]

        # ce_loss
        ce_loss = -paddle.log(pred_t)
        cls_loss = ce_loss * paddle.pow((1 - pred_t), ff.detach()) * wf.detach()


        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            cls_loss = alpha_t * cls_loss

        if normalizer is None:
            normalizer = 1.0

        return _reduce(cls_loss, reduction, normalizer=normalizer)

    def collect_grad(self, grad_in):
        bs = grad_in.shape[0]
        self.grad_buffer.append(grad_in.detach().permute(0, 2, 3, 1).reshape([bs, -1, self.num_classes]))
        if len(self.grad_buffer) == self.fpn_levels:
            target = self.cache_target
            grad = paddle.cat(self.grad_buffer[::-1], dim=1).reshape([-1, self.num_classes])

            grad = paddle.abs(grad)
            pos_grad = paddle.sum(grad * target, dim=0)
            neg_grad = paddle.sum(grad * (1 - target), dim=0)

            dist.allreduce(pos_grad)
            dist.allreduce(neg_grad)

            self.pos_grad += pos_grad
            self.neg_grad += neg_grad
            self.pos_neg = paddle.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)

            self.grad_buffer = []


