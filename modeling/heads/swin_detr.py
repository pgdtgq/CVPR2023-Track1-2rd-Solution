# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init
from modeling.initializer import xavier_uniform_


def sub_weight_init(m):
    if isinstance(m, nn.Conv2D):
        xavier_uniform_(m.weight)
        if m.bias is not None:
            param_init.constant_init(m.bias, value = 0)

class DETR(nn.Layer):
    """d
    """
    def __init__(self, transformer, detr_head, post_process, neck = None, exclude_post_process=False):
        super().__init__()
        self.neck = neck
        self.transformer = transformer
        self.detr_head = detr_head
        self.post_process = post_process
        self.exclude_post_process = exclude_post_process
        self.start = 0

        if self.neck is not None:
            print('======================================>det use neck')
            self.neck.apply(sub_weight_init)  #参数初始化

    def forward(self, body_feats, inputs, current_iter = None):
        """d"""
        if isinstance(body_feats, list):
            body_feats = body_feats[0]  # 取得是det_feature，有4层
        else:
            body_feats = body_feats
        # Transformer
        if 'gt_bbox' in inputs:
            #hard code
            gt_bbox = [paddle.cast(inputs['gt_bbox'][i], 'float32') for i in range(len(inputs['gt_bbox']))]
            inputs['gt_bbox'] = gt_bbox
        pad_mask = inputs['pad_mask'] if self.training else None  # 这个pad mask是干嘛得，[1, img_h, img_w]
        
        if self.neck is not None:
            body_feats = self.neck([body_feats])
        out_transformer = self.transformer(body_feats, pad_mask, inputs)   # dinotransformer

        # DETR Head
        if self.training:
            losses = self.detr_head(out_transformer, body_feats, inputs, current_iter)  
            new_losses = {}
            new_losses.update({
                'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
            })
            return new_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bboxes, logits, masks = preds
                bbox_pred, bbox_num = bboxes, logits
            else:
                bbox, bbox_num = self.post_process(
                    preds, inputs['im_shape'], inputs['scale_factor'])
                bbox_pred, bbox_num = bbox, bbox_num
            
            output = {
                "bbox": bbox_pred,
                "bbox_num": bbox_num,
            }
            return output