import os
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init
import numpy as np

from .transformer_utils import DropPath,add_parameter
from .transformer_utils import ones_, zeros_, trunc_normal_,DropPath,add_parameter
from paddleseg.models import layers


__all__ = ['LadderSideAttentionFPN']



class PatchMerging(nn.Layer):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.reshape([-1, H, W, C])

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # paddle F.pad default data_format is 'NCHW'
            x = F.pad(x, [0, 0, 0, H % 2, 0, W % 2, 0, 0], data_format='NHWC')
            H += H % 2
            W += W % 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape([-1, H * W // 4, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x



class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias  = False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]   # make paddlescript happy (cannot use tensor as tuple)

    
        q = q * self.scale
        attn = q.matmul(k.transpose((0, 1, 3, 2)))
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias  = False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class CrossLevelFeatureFuseModule(nn.Layer):  
    def __init__(self, layer_feat_nums=4, 
                       hidden_dim=[192,384,768],
                       transformer_block_nums=1, 
                       transformer_block_num_heads=2, 
                       gate_T=0.1, 
                       gate_alpha=0, 
                       indices = [0,1,2,3]):
        super(CrossLevelFeatureFuseModule, self).__init__()
        self.transformer_block_nums = transformer_block_nums
        self.transformer_block_num_heads = transformer_block_num_heads

        self.hidden_dim = hidden_dim  #list
        self.gate_T = gate_T
        self.gate_alpha = gate_alpha

        self.indices = indices
        assert len(self.indices) == len(self.hidden_dim)
        self.layer_feat_nums = len(self.hidden_dim)


        self.side_gate_params = self.generate_gate_params()
        self.transformer_blocks = self.generate_transformer_blocks()

        self.down_layers = nn.LayerList([
                        PatchMerging(dim=dim)
                        for i, dim in enumerate(self.hidden_dim[:-1])
                    ])  

        self.norm = nn.LayerList([
                    nn.LayerNorm(dim)
                    for i, dim in enumerate(self.hidden_dim)
                    ])  

        self.apply(self._init_weights)

    def generate_gate_params(self):
        side_gate_params = nn.ParameterList(
                    [add_parameter(
                    self,paddle.ones(shape=[1]) * self.gate_alpha) 
                    for i in range(self.layer_feat_nums)]
                )
        return side_gate_params
    
    def generate_transformer_blocks(self):
        transformer_blocks = nn.LayerList()
        for i in range(1,self.layer_feat_nums):
            sub_blocks = nn.LayerList()
            for _ in range(self.transformer_block_nums):
                sub_blocks.append(TransformerBlock(dim=self.hidden_dim[i], num_heads=self.transformer_block_num_heads))
                    
            transformer_blocks.append(sub_blocks)
        return transformer_blocks
    

    def forward(self, inputs):
        if isinstance(inputs, list):
            x = [inputs[1][i] for i in self.indices]  #取原始维度
            shapes = [inputs[-1][i] for i in self.indices]  #对应的特征维度
        B = x[0].shape[0]
                
        out = []
        for i, block_feat in enumerate(x):
            Hp, Wp = shapes[i]
            if i == 0:  #第一层
                out.append(self.norm[i](block_feat).reshape((-1, Hp, Wp, self.hidden_dim[i])).transpose((0, 3, 1, 2)))
                feats = self.down_layers[0](block_feat,Hp, Wp)  #降低维度
            else:
                gate = F.sigmoid(self.side_gate_params[i - 1] / self.gate_T)
                feats = gate * feats + (1 - gate) * block_feat  #融合  
                for transformer_block in self.transformer_blocks[i - 1]:
                    feats = transformer_block(feats, Hp, Wp)
                out.append(self.norm[i](feats).reshape((-1, Hp, Wp, self.hidden_dim[i])).transpose((0, 3, 1, 2)))
                if i == (self.layer_feat_nums-1):break
                feats = self.down_layers[i](feats, Hp, Wp)  #降低维度
        # for o in out:
        #     print(o.shape)
        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)


if __name__=='__main__':
    model = LadderSideAttentionFPN(layer_feat_nums = 4, hidden_dim = [192,384,768,1536], use_reduct = False)
    x = [[], [paddle.randn((1, 14400, 192)),  paddle.randn((1, 3600, 384)),  paddle.randn((1, 920, 768)), paddle.randn((1, 240,1536))],
                [[90,160],[45,80],[23,40],[12,20]]]  #FPN形式
    output =model(x)
