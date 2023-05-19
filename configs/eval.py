from .common import train
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from modeling.meta_arch.multitask_v2 import MultiTaskBatchFuse

# segmentation
from data.transforms.seg_transforms import ResizeStepScaling, RandomPaddingCrop, \
    RandomHorizontalFlip, RandomDistort, Normalize,GenerateInstanceTargets, One_of_aug,RandomSelectAug,RandomWeather
from data.build_segmentation import build_segmentation_dataset, build_segmentation_trainloader, \
    build_segementation_test_dataset
from evaluation.segmentation_evaluator import SegEvaluator

# classification
from data.build import build_reid_test_loader_lazy
from data.transforms.build import build_transforms_lazy

from data.build_cls import build_hierachical_softmax_train_set, \
    build_hierachical_test_set, build_vehiclemulti_train_loader_lazy
from evaluation.common_cls_evaluator import CommonClasEvaluatorSingleTask

# detection
from data.build_trafficsign import build_cocodet_set, build_cocodet_loader_lazy
from evaluation.cocodet_evaluator import CocoDetEvaluatorSingleTask
from solver.build import build_lr_optimizer_lazy, build_lr_scheduler_lazy
import albumentations as A


dataloader=OmegaConf.create()
# _root = os.getenv("FASTREID_DATASETS", "datasets")
_root = os.getcwd()
print("==========>current path is " + _root)
seg_num_classes=19




dataloader.train=L(MultiTaskDataLoader)(
    cfg=dict(sample_mode='batch',),
    task_loaders=L(OrderedDict)(
        #分割
        segmentation=L(build_segmentation_trainloader)(
            data_set=L(build_segmentation_dataset)(
                    dataset_name="BDD100K",
                    dataset_root=_root + '/datasets/track1_train_data/seg/', 
                    transforms=[
                        L(ResizeStepScaling)(min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.2),  
                        L(RandomPaddingCrop)(crop_size=[1280, 720]), 
                        L(RandomHorizontalFlip)(), 
                        L(One_of_aug)(method = [  #仅仅变换img
                                    A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
                                    A.GaussianBlur(blur_limit=[3,5], sigma_limit=[0,1],p = 0.5),
                                    ],p = 0.3 , only_img = True
                         ),   
                        L(One_of_aug)(method = [  #img和mask都变换
                                    A.Rotate (limit=15, p=0.4), 
                                    A.ShiftScaleRotate(rotate_limit=15, p=0.4),
                                    A.GridDistortion(num_steps = 10,p = 0.2),
                                    ],p = 0.3,only_img = False
                        ), 
                        L(RandomSelectAug)(
                                transforms1 = L(RandomDistort)(brightness_range=0.4, contrast_range=0.4, saturation_range=0.4),
                                transforms2 = L(RandomWeather)(), 
                                p = 0.9  #RandomDistort的概率为0.9
                        ),
                        L(GenerateInstanceTargets)(num_classes = seg_num_classes),
                        L(Normalize)()],
                    mode='train'),
            total_batch_size=16, 
            worker_num=4, 
            drop_last=True, 
            shuffle=True,
            num_classes=seg_num_classes,
            is_train=True,
            use_shared_memory=False,
        ),
        #汽车分类
        fgvc=L(build_vehiclemulti_train_loader_lazy)(
            sampler_config={'sampler_name': 'ClassAwareSampler'},
            train_set=L(build_hierachical_softmax_train_set)(
                names = ("FGVCDataset",),
                train_dataset_dir = _root + '/datasets/track1_train_data/cls/trainval/',
                test_dataset_dir = _root + '/datasets/track1_test_data/cls/test/',
                train_label = _root + '/datasets/track1_train_data/cls/trainval.txt',
                test_label = _root + '/datasets/track1_test_data/cls/test.txt',
                transforms=L(build_transforms_lazy)(
                    is_train=True,
                    do_rpt = True,     #增加gridshuffle
                    size_train=[640, 640], 
                    do_rea=True,
                    rea_prob=0.5,
                    do_flip=True,
                    do_autoaug=True,
                    autoaug_prob=0.5,
                    do_myaug = True,  #增加数据扩增
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                ),

                num_classes=196,
            ),
            total_batch_size=16,
            num_workers=4,
        ),

        #交通标志检测
        trafficsign=L(build_cocodet_loader_lazy)(
            data_set=L(build_cocodet_set)(
                dataset_name="COCODataSet",
                transforms=[
                    dict(Decode=dict(),),
                    dict(RandomSelect =dict(
                        transforms1 = [ #马赛克分支+简单数据扩增
                                    dict(Mosaic=dict(
                                                    degrees = [-2,2], 
                                                    translate = [-0.02,0.02], 
                                                    scale = [0.4,1.2],
                                                    enable_mixup = True,
                                                    mixup_prob = 0.7)),
                                    dict(One_of_aug_onlyimg=dict()),  #简单的图像增强
                        ],
                        transforms2 = [  #resize分支+困难数据扩增
                                    dict(RandomSelect=dict(
                                            transforms1=[
                                                dict(RandomShortSideResize=dict(
                                                    short_side_sizes=[864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184],  
                                                    max_size=1184)  
                                                    ),
                                            ],
                                            transforms2=[
                                                dict(RandomShortSideResize=dict(short_side_sizes=[800, 928, 1056, 1184]),),
                                                dict(RandomSizeCrop=dict(min_size=720, max_size=1184),),
                                                dict(RandomShortSideResize=dict(
                                                    short_side_sizes=[864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184],   
                                                    max_size=1184)  
                                                    ),
                                            ],
                                    )),
                                    dict(One_of_aug_onlyimg_hard=dict()),  #困难的数据增强
                                    ],
                    p = 0.4)),  #马赛克开启的prob
                    dict(RandomFlip=dict(prob=0.5)),
                    dict(NormalizeImage=dict(
                        is_scale=True, 
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
                        ),
                    dict(NormalizeBox=dict()),
                    dict(BboxXYXY2XYWH=dict()),
                    dict(Permute=dict()),
                ],
                image_dir='trainval', 
                anno_path='trainval.json', 
                dataset_dir= _root + '/datasets/track1_train_data/dec/',
                data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'],
                mosaic_epoch = 0,
            ),
            total_batch_size=8,
            num_workers=4,
            batch_transforms=[
                dict(PadMaskBatch=dict(pad_to_stride=-1, return_pad_mask=True),),
            ],
            is_train=True,
            shuffle=True, 
            drop_last=True, 
            collate_batch=False,
        ),
    ),
)

# NOTE
# trian/eval模式用于构建对应的train/eval Dataset, 需提供样本及标签;
# infer模式用于构建InferDataset, 只需提供测试数据, 最终生成结果文件用于提交评测, 在训练时可将该部分代码注释减少不必要评测

dataloader.test = [
    
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            segmentation=L(build_segmentation_trainloader)(
                data_set=L(build_segementation_test_dataset)(
                        dataset_name="BDD100K",
                        dataset_root=_root + '/datasets/track1_train_data/seg/', 
                        transforms=[L(Normalize)()],
                        mode='val',
                        is_padding=True),
                total_batch_size=16, 
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
                    dataset_name = "FGVCDataset",
                    train_dataset_dir = _root + '/datasets/track1_train_data/cls/trainval/',  #数据库已经更改
                    test_dataset_dir = _root + '/datasets/track1_test_data/cls/test/',
                    train_label = _root + '/datasets/track1_train_data/cls/trainval.txt',
                    test_label = _root + '/datasets/track1_test_data/cls/test.txt',
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[640, 640],  
                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    ),
                    is_train=True  # eval mode 
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
                    dataset_name="COCODataSet",
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
                    image_dir='test', 
                    anno_path='test.json',
                    dataset_dir= _root + '/datasets/track1_test_data/dec/',
                    data_fields=['image', 'gt_bbox', 'gt_class', 'difficult'],
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

# NOTE
# trian/eval模式用于eval;
# infer模式则用于生成测试集预测结果(可直接提交评测), 在训练时可注释该部分代码减少不必要评测

dataloader.evaluator = [
    # segmentation
    L(SegEvaluator)(
    ),  # train/eval mode

    # classification
    L(CommonClasEvaluatorSingleTask)(
        cfg=dict(), num_classes=196
    ),   # train/eval mode

    # detection
    L(CocoDetEvaluatorSingleTask)(
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


optimizer = L(build_lr_optimizer_lazy)(
    optimizer_type='AdamW',
    base_lr=1e-4,
    weight_decay=1e-4,
    grad_clip_enabled=True,
    grad_clip_norm=0.1,
    apply_decay_param_fun=None,
    lr_multiplier=L(build_lr_scheduler_lazy)(
        max_iters=900000,
        warmup_iters=200,
        solver_steps=[720000],
        solver_gamma=0.1,
        base_lr=1e-4,
        sched='CosineAnnealingLR',
    ),
)

train.amp.enabled = False

# data settings
sample_num = 7000
epochs= 120
dataloader.train.task_loaders.segmentation.total_batch_size = 8
dataloader.train.task_loaders.fgvc.total_batch_size = 8*8
dataloader.train.task_loaders.trafficsign.total_batch_size = 8

iters_per_epoch = sample_num // dataloader.train.task_loaders.segmentation.total_batch_size

max_iters = iters_per_epoch * epochs

# optimizer
optimizer.lr_multiplier.max_iters = max_iters
optimizer.base_lr = optimizer.lr_multiplier.learning_rate = 1e-4
optimizer.lr_multiplier.solver_steps = [int(max_iters * 0.8)]


train.max_iter = max_iters
train.eval_period = int(iters_per_epoch * 1)
train.checkpointer.period = int(iters_per_epoch * 1)
train.checkpointer.max_to_keep= 30
train.init_checkpoint = _root + '/checkpoints/best.pdmodel'  

train.output_dir = _root + '/outputs/eval'  

# resume settings (remember last_checkpoint and --resume)
train.log_period = 20