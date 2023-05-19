# !/usr/bin/env python3
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import paddle
from paddle import nn

from modeling.losses import triplet_loss, cross_entropy_loss, log_accuracy


from paddle.nn.initializer import Assign

def add_parameter(layer, datas, name=None):
    parameter = layer.create_parameter(
        shape=(datas.shape), default_initializer=Assign(datas))
    if name:
        layer.add_parameter(name, parameter)
    return parameter


class AutomaticWeightedLoss(nn.Layer):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = paddle.ones([num])
        self.params = paddle.create_parameter([num], 
                            dtype = 'float32', 
                            name = 'dymic_weight',
                            is_bias=False, 
                            default_initializer=Assign(params))
        self.init_weight()

    def init_weight(self):
        for _param in self.parameters():
            _param.optimize_attr['learning_rate'] = 0.5

    def forward(self, loss, index):  #损失加权
        return 0.5 / (self.params[index] ** 2) * loss + paddle.log(1 + self.params[index] ** 2)


task_id = {    
            'segmentation': 0,
            'fgvc': 1,
            'trafficsign': 2,

}

class MultiTaskBatchFuse(nn.Layer):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    def __init__(
            self,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            freeze_backbone = False,
            task_loss_kwargs=None,
            task2head_mapping=None,
            dynamic_loss = False,
            dynamic_start_iter = 10000,
            Moe = False,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        # head
        # use nn.LayerDict to ensure head modules are properly registered
        self.heads = nn.LayerDict(heads)

        if task2head_mapping is None:
            task2head_mapping = {}
            for key in self.heads:
                task2head_mapping[key] = key
        self.task2head_mapping = task2head_mapping

        self.task_loss_kwargs = task_loss_kwargs

        self.register_buffer('pixel_mean', paddle.to_tensor(list(pixel_mean)).reshape((1, -1, 1, 1)), False)
        self.register_buffer('pixel_std', paddle.to_tensor(list(pixel_std)).reshape((1, -1, 1, 1)), False)


    @property
    def device(self):
        """
        Get device information
        """
        return self.pixel_mean.device

    def forward(self, task_batched_inputs, current_iter = None):
        """
        NOTE: this forward function only supports `self.training is False`
        """
        losses = {}
        outputs = {}
        # print('current_iter:', current_iter)
        for task_name, batched_inputs in task_batched_inputs.items():

            features = self.backbone(self.preprocess_image(batched_inputs))  #是否训练

            #至此得到了基本的特征图features[0] 特征金字塔, features[1]有12个分别代表每一个的self-attention的输出
            if self.training:
                # assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
                # targets = batched_inputs["targets"]

                # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
                # may be larger than that in the original dataset, so the circle/arcface will
                # throw an error. We just set all the targets to 0 to avoid this problem.
                # if targets.sum() < 0: targets.zero_()
                task_outputs = self.heads[self.task2head_mapping[task_name]](features, batched_inputs, current_iter)
                losses.update(**{task_name + "_" + key: val for key, val in task_outputs.items()})
            else:
                task_outputs = self.heads[self.task2head_mapping[task_name]](features, batched_inputs)    #测试的时候不需要iter
                outputs[task_name] = task_outputs

        if self.training:
            return losses
        else:
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            if 'image' in batched_inputs:
                images = batched_inputs['image']
            if 'images' in batched_inputs:
                images = batched_inputs['images']
            # print(images.shape)
        elif isinstance(batched_inputs, paddle.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or Tensor, but get {}".format(type(batched_inputs)))

        # images.sub_(self.pixel_mean).div_(self.pixel_std)
        return {'image': images}
