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

下载后将数据解压在./datasets文件夹下


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
