简体中文 | [English](README.md)

# Track1 Codebase

## 使用方案

提供了分类、检测、分割AllInOne三任务联合训练方法
Demo为单机（8卡）40G A100的训练方法

### 环境配置

运行环境为python3.7，cuda11.2测试机器为A100。使用pip的安装依赖包，如下：
```bash
pip install -r requirements.txt
```

安装PaddleSeg：
```bash
cd /PaddleSeg
pip install -r requirements.txt
pip install -v -e .
```

编译mask2former
```bash
cd /PaddleSeg/contrib/PanopticSeg/paddlepanseg/models/ops
cd ms_deform_attn
python setup.py install
```

### 数据配置

数据下载地址：https://pan.baidu.com/s/10HDywKyzu6R8q-Lxrpn3pw (J69v)

下载后将数据解压在./datasets文件夹下

我们将train和val共同作为训练数据，基于开源数据找到对应的test数据的标注文件，将test作为val，因此数据的组织架构为：

(1)分类数据：
 
 训练：/datasets/track1_train_data/cls/trainval
 
 验证：/datasets/track1_test_data/cls/test

(2)检测数据

 训练：/datasets/track1_train_data/dec/trainval

 验证：/datasets/track1_test_data/dec/test

(3)分割数据

我们将train和val共同放置在train文件夹下作为训练数据，将test放置在val文件夹下，两者都放置在track1_train_data目录下

训练：/datasets/track1_train_data/seg/train

验证：/datasets/track1_train_data/seg/val

### pretrianed model

我们采用swin trainsformer Large作为我们的backbone， 采用mask2former作为我们的分类头, 采用dino作为我们的检测头, 下载链接:

https://pan.baidu.com/s/10HDywKyzu6R8q-Lxrpn3pw (J69v )

将三者放置在./pretrained文件夹下

### 训练


```bash
sh scripts/train.sh
```

### 预测



```bash
sh scripts/test.sh
```

### best model

下载地址:https://pan.baidu.com/s/10HDywKyzu6R8q-Lxrpn3pw (J69v)