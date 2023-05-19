from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.nn as nn
from paddle.nn import AdaptiveAvgPool2D
from modeling.losses import cross_entropy_loss


__all__ = ['ClsHead']

class ClsHead(nn.Layer):
    def __init__(self, embedding_size, class_num, **kwargs):
        super(ClsHead, self).__init__()
        self.embedding_size = embedding_size
        self.class_num = class_num

        self.avg_pool = AdaptiveAvgPool2D(1, data_format="NCHW")

        self.fc1 = nn.Linear(self.embedding_size, self.embedding_size // 2)        
        self.fc2 = nn.Linear(self.embedding_size // 2, self.class_num) 
        self.flatten = nn.Flatten()
        self.loss = None
        if 'loss' in kwargs:
            self.loss = kwargs['loss']

        self.neck = None
        self.use_neck = False
        if 'neck' in kwargs:
            self.neck = kwargs['neck']
            self.use_neck = True
            print('======================================>use neck in cls')

    def forward(self, feats, inputs=None, current_iter = None):
        if self.use_neck:
            feats = self.neck(feats)  #取特征图形式
            feats = feats[-1]  #取最后一层
        else:
            if isinstance(feats, list):
                feats = feats[0][-1] 

        out = self.avg_pool(feats)
        out = self.flatten(out)
        out = self.fc2(self.fc1(out))

        if self.training:
            return self.get_loss(out, inputs['targets'])
        else:
            return out

    def get_loss(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = {}
        if self.loss is None:
            loss_dict['loss_ce_cls'] = cross_entropy_loss(
                    outputs,
                    gt_labels,
                    0.1
                )
        else:
            loss_dict['loss_ce_cls'] = self.loss(outputs, gt_labels)
 
        return loss_dict