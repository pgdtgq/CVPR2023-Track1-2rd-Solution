English | [简体中文](README_ch.md)

# CVPR-2023-Track1-2rd-solution

## Instructions

We provide a three task  AllInOne joint training method of classification, detection, and segmentation.

Demo is based on 8 A100 cards.

### Environment

Please use python3.7 and cuda11.2. 

```bash
pip install -r requirements.txt
```

setup paddleseg.
```bash
cd /PaddleSeg
pip install -r requirements.txt
pip install -v -e .
```

setup mask2former
```bash
cd /PaddleSeg/contrib/PanopticSeg/paddlepanseg/models/ops
cd ms_deform_attn
python setup.py install
```
### Data Configuration

You can download the data from:,and decompress the data into the 'datasets' folder ,like this:

track1_train_data
        |_____seg
        |      |______trainval  
        |      |______val      
        |
        |_____cls
        |      |______trainval 
        |
        |_____det
               |______trainval 

track2_test_data
        |
        |_____cls
        |      |______test 
        |
        |_____det
               |______test  


### Pretrained model

We use swin trainsformer Large as our backbone, can be download from:

the pretrained model of seg head:

the pretrained model of det head:

Put all above into ./pretraind

### Training


```bash
sh scripts/train.sh
```

### Inference


```bash
sh scripts/test.sh
```
