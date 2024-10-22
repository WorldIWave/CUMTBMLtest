#IMU 分类项目

##1. 项目介绍

本项目是一个基于惯性测量单元（IMU）数据的人体活动分类系统，使用卷积神经网络（CNN）与 SVM 风格的间隔损失来提取特征并进行分类。项目模块包括：

* dataload.py: 数据加载与预处理。

* dataprocess.py: 数据分割、滤波和标准化处理。

* models.py: 定义神经网络架构和损失函数。

* trainer.py: 负责模型训练、评估和结果可视化。



##2. 快速开始

###2.1 环境配置

请确保已安装以下依赖：
```
Python >= 3.7

Anaconda/Miniconda
```

###2.2 安装步骤

```
pip install -r requirements.txt
```

文件包括以下库：
```
numpy
pandas
matplotlib
scikit-learn
torch
tqdm
joblib

```

###2.3 运行项目

####数据预处理

使用 dataprocess.py 进行数据预处理：
```
python dataprocess.py
```

该脚本会清洗原始数据，应用低通滤波器，并将数据分割成固定大小的窗口，处理后的数据会保存为 .npy 文件。

####数据加载

使用 dataload.py 加载数据：
```
python dataload.py
```
该脚本会准备数据，进行标准化处理并转换为 PyTorch 张量。

####训练模型

使用 trainer.py 训练模型：
```
python trainer.py
```
训练过程中会显示每一折的训练状态和验证指标，训练好的模型权重和评估结果会保存在 output 目录中。

###2.4 输出文件

* segments.npy: 预处理后的数据片段。

* labels.npy: 每个数据片段的标签。

* cnn_model.pth: 训练好的模型权重。

* confusion_matrix.png: 混淆矩阵图。

* roc_curve.png: 每个类别的 ROC 曲线。

* training_validation_loss_curve.png: 训练和验证损失曲线。

这些输出文件保存在 output 目录中，用于后续分析和评估。



##3. 结果

模型训练完成后，可以通过准确率、精确率、召回率、F1 分数和 ROC 曲线等指标来评估模型性能。评估结果会打印在终端并保存为图片。

* 评估指标(Benchmarks)

* 准确率: 衡量整体分类性能。

* 精确率、召回率、F1 分数: 评估模型对每个类别的识别能力。

* 混淆矩阵: 提供模型分类错误的详细信息。

* ROC 曲线和 AUC: 显示每个类别的分类性能。


