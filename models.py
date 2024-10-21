import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.flatten = nn.Flatten()
        # 降维全连接层
        self.fc_reduce = nn.Linear(128, 64)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (batch_size, 32, seq_len)
        x = self.pool(x)  # (batch_size, 32, seq_len/2)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch_size, 64, seq_len/2)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch_size, 128, seq_len/2)
        x = self.global_pool(x)  # (batch_size, 128, 1)
        x = self.flatten(x)  # (batch_size, 128)
        x = self.fc_reduce(x)  # (batch_size, 64)
        return x  # 返回特征向量


class MultiClassHingeLoss(nn.Module):
    def __init__(self, num_classes, margin=1.0):
        super(MultiClassHingeLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin

    def forward(self, outputs, targets):
        # outputs: (batch_size, num_classes)
        # targets: (batch_size)
        one_hot_targets = torch.zeros_like(outputs)
        one_hot_targets[range(len(targets)), targets] = 1
        margins = self.margin - (outputs * one_hot_targets) + (outputs * (1 - one_hot_targets))
        margins = torch.clamp(margins, min=0)
        loss = margins.sum(dim=1).mean()
        return loss


class CNNWithSVM(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(CNNWithSVM, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(64, num_classes, bias=False)  # 不使用偏置，模拟SVM的线性决策
        self.num_classes = num_classes

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x
