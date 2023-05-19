# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import numpy as np


class FocalLoss(nn.Layer):
    """A wrapper around paddle.nn.functional.sigmoid_focal_loss.
    Args:
        use_sigmoid (bool): currently only support use_sigmoid=True
        alpha (float): parameter alpha in Focal Loss
        gamma (float): parameter gamma in Focal Loss
        loss_weight (float): final loss will be multiplied by this
    """
    def __init__(self,
                 gamma=2.0,
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight
        print('================================>use focal loss!!!')
    def forward(self, pred, target, reduction='none'):
        """forward function.
        """
        num_classes = pred.shape[1]
        probs = F.softmax(pred, axis=1)
        print(probs)
        logs = paddle.log(probs,1)
        print(logs)
        one_hot  = F.one_hot(target, num_classes).cast(pred.dtype)
        print(paddle.pow((1.0 - probs), self.gamma))
        weight = -1.0 * one_hot * paddle.pow((1.0 - probs), self.gamma)
        loss = (weight * logs).sum(axis=1).mean()
        print(loss)
        return loss



if __name__ == '__main__':
    cre = FocalLoss()

    inputs = paddle.randn((2,5))
    labels = paddle.to_tensor(np.array([0,4]))
    loss = cre(inputs,labels)

