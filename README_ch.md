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

数据下载地址：

下载后将数据解压在./datasets文件夹下,组织结构如下：

track1_train_data
        |_____seg
        |      |______trainval  #我们将train与val共同用作训练数据
        |      |______val       #将test作为验证集
        |
        |_____cls
        |      |______trainval  #我们将train与val共同用作训练数据
        |
        |_____det
               |______trainval  #我们将train与val共同用作训练数据

track2_test_data
        |
        |_____cls
        |      |______test  #分类的验证数据（源于test）
        |
        |_____det
               |______test   #检测的测试数据（源于test）


### pretrianed model

我们采用swin trainsformer Large作为我们的backbone，下载连接：

分割头预训练权重下载地址：

检测头预训练权重下载地址：

将三者放置在./pretrained文件夹下

### 训练


```bash
sh scripts/train.sh
```

### 预测



```bash
sh scripts/test.sh
```
