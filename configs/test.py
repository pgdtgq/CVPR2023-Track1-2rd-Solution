from .common import train
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from modeling.meta_arch.multitask_v2 import MultiTaskBatchFuse

# segmentation
from data.transforms.seg_transforms import Normalize, RandomPaddingCrop
from data.build_segmentation import build_segmentation_trainloader, \
    build_segementation_test_dataset
from evaluation.segmentation_evaluator import SegEvaluatorInfer

# classification
from data.build import build_reid_test_loader_lazy
from data.transforms.build import build_transforms_lazy
from data.build_cls import build_hierachical_test_set
from evaluation.common_cls_evaluator import CommonClasEvaluatorSingleTaskInfer

# detection
from data.build_trafficsign import build_cocodet_set, build_cocodet_loader_lazy
from evaluation.cocodet_evaluator import CocoDetEvaluatorSingleTaskInfer


dataloader=OmegaConf.create()
# _root = os.getenv("FASTREID_DATASETS", "datasets")
_root = os.getcwd()
print("==========>current path is " + _root)
seg_num_classes=19

# NOTE
#eval
#FGVCInferDataset是infer的dataset名称
#FGVCDataset是eval的dataset名称
dataloader.test = [
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            segmentation=L(build_segmentation_trainloader)(
                data_set=L(build_segementation_test_dataset)(
                        dataset_name="InferDataset",
                        dataset_root=_root + '/datasets/track1_test_data/seg/', 
                        transforms=[L(Normalize)()],
                        mode='test',
                        is_padding=True),
                total_batch_size=8, #
                worker_num=4, 
                drop_last=False, 
                shuffle=False,
                num_classes=seg_num_classes,
                is_train=False,
            ),
        ),
    ),

    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            fgvc=L(build_reid_test_loader_lazy)(
                test_set=L(build_hierachical_test_set)(
                    dataset_name = "FGVCInferDataset",
                    test_dataset_dir = _root + '/datasets/track1_test_data/cls/test/',
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[640, 640],
                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    ),
                    is_train=False  # eval mode 
                ),
                test_batch_size=16,
                num_workers=4,
            ),
        ),
    ),

    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(           
            trafficsign=L(build_cocodet_loader_lazy)(
                data_set=L(build_cocodet_set)(
                    is_padding=True,
                    dataset_name="COCOInferDataSet",
                    transforms=[
                        dict(Decode=dict(),),
                        dict(Resize=dict(
                            target_size=[1312, 1312],  
                            keep_ratio=False)
                            ),
                        dict(NormalizeImage=dict(
                            is_scale=True, 
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
                            ),
                        dict(Permute=dict()),
                    ],
                    image_dir='test.txt',  #数据库已经更改
                    anno_path='test.json',
                    dataset_dir= _root + '/datasets/track1_test_data/dec/',
                    data_fields=['image', 'im_id', 'im_file'],
                ),
                total_batch_size=16,
                num_workers=4,
                batch_transforms=[
                    dict(PadMaskBatch=dict(pad_to_stride=32, return_pad_mask=False),),
                ],
                is_train=False,
                shuffle=False, 
                drop_last=False, 
                collate_batch=False,
            ),
        ),    
    ),

]

dataloader.evaluator = [
    #seg
    L(SegEvaluatorInfer)(
    ),  # train/eval mode

    # # classification
    # # CommonClasEvaluatorSingleTask表示eval模式 
    # # CommonClasEvaluatorSingleTaskInfer表示infer模式
    L(CommonClasEvaluatorSingleTaskInfer)(
        cfg=dict(), num_classes=196
    ), 

    L(CocoDetEvaluatorSingleTaskInfer)(
        classwise=False, 
        output_eval=None,
        bias=0, 
        IouType='bbox', 
        save_prediction_only=False,
        parallel_evaluator=True,
        num_valid_samples=3067, 
    ),  # train/eval mode

]

from ppdet.modeling import ShapeSpec
from modeling.backbones.swin_transformer import SwinTransformer
from modeling.heads.simple_cls_head import ClsHead
from modeling.heads.mask2former.mask2former import Mask2Former
from modeling.losses.mask2former_loss import Mask2FormerLoss
from modeling.heads.swin_detr import DETR
from ppdet.modeling.transformers.matchers import HungarianMatcher
from ppdet.modeling.post_process import DETRBBoxPostProcess
from modeling.losses.dino_loss import DINOLoss
from modeling.heads.detr_head import DINOHead
from modeling.transformers.dino_transformer import DINOTransformer
from modeling.heads.neck.clsneck import CrossLevelFeatureFuseModule


backbone = L(SwinTransformer)(  #swin large
        arch ='swin_L_224',
        patch_size = 4,
        pretrain_img_size=224,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        pretrained = None)

trafficsign_num_classes=45
use_focal_loss=True

model=L(MultiTaskBatchFuse)(
    backbone=backbone,
    heads=L(OrderedDict)(
        segmentation=L(Mask2Former)(
            in_channels= [192,384,768,1536],
            num_classes=seg_num_classes,
            maskformer_num_feature_levels = 4,
            pretrained = None,
            loss=L(Mask2FormerLoss)(num_classes = seg_num_classes,
                                    loss_ce = 1.0,  
                                    mask_weight = 3.0,
                                    dice_weight = 2.0,
                                    cost_loss_ce = 2.0,  
                                    cost_mask_weight = 5.0,
                                    cost_dice_weight = 5.0,
                                    seg_loss_weight = 1.0, 
                                    ),
        ),

        fgvc=L(ClsHead)(
            neck = L(CrossLevelFeatureFuseModule)(  #swin large
                       hidden_dim=[192,384,768,1536],
                       transformer_block_nums=1, 
                       transformer_block_num_heads=2, 
                       gate_T=0.1, 
                       gate_alpha=0, 
                       indices = [0,1,2,3]),
            embedding_size=1536,  
            class_num=196,
        ),
        
        #采用DERT
        trafficsign=L(DETR)(
            transformer=L(DINOTransformer)(
                            num_classes=trafficsign_num_classes,
                            hidden_dim=256,
                            num_queries=900,
                            position_embed_type='sine',
                            return_intermediate_dec=True,
                            backbone_feat_channels=[192, 384,768,1536],  #must match the Vit embed_dim
                            num_levels=5,
                            num_encoder_points=4,
                            num_decoder_points=4,
                            nhead=8,
                            num_encoder_layers=6,
                            num_decoder_layers=6,
                            dim_feedforward=2048,
                            dropout=0.0,
                            activation="relu",
                            num_denoising=100,
                            label_noise_ratio=0.5,
                            box_noise_scale=1.0,
                            learnt_init_query=True,
                            eps=1e-2,
                            use_neck= False),
            detr_head=L(DINOHead)(loss=L(DINOLoss)(
                            num_classes=trafficsign_num_classes,
                            loss_coeff={"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1},
                            aux_loss=True,
                            use_focal_loss=use_focal_loss,
                            use_imporved_der_loss = True, #是否采用改进的EqualizedFocalLoss
                            change_loss_iter = 45000, #切换loss的iter,use_imporved_der_loss为True时候才生效
                            matcher=L(HungarianMatcher)(
                                matcher_coeff={"class": 2, "bbox": 5, "giou": 2},
                                use_focal_loss=use_focal_loss,)   
                            )
           ),
            post_process=L(DETRBBoxPostProcess)(
                            num_classes=trafficsign_num_classes,
                            num_top_queries=50,
                            use_focal_loss=use_focal_loss,
                            ),
        ),
    ),
    pixel_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    pixel_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
)

train.amp.enabled = False
train.init_checkpoint = _root + '/checkpoints/best.pdmodel'

train.output_dir = _root + '/outputs/test'
