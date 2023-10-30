import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def __init__(self):
        self.num_classes = 1
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.model_name = 'TextCNN_custom'
        self.num_epochs = 50
        self.batch_size = 64
        self.pad_size = 100
        self.learning_rate = 1e-3
        self.embed = 1600
        # LSTM parameters
        self.dropout_LSTM = 0
        self.hidden_size = 256
        self.num_layers = 1
        # CNN parameters
        self.dropout_CNN = 0.2
        self.filter_sizes = (4, 80, 80)
        self.num_filters = 40


class Config(object):
    def __init__(self):
        self.num_classes = 1
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.model_name = 'TextCNN_custom'
        self.num_epochs = 50
        self.batch_size = 64
        self.pad_size = 100
        self.learning_rate = 0.0001
        self.embed = 1600
        # LSTM parameters
        self.dropout_LSTM = 0
        self.hidden_size = 256
        self.num_layers = 1
        # CNN parameters
        self.dropout_CNN = 0.2
        self.filter_sizes = (4, 80, 80)
        self.num_filters = 40


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(10, 10), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 20, kernel_size=(10, 10), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(10, 10), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 20, kernel_size=(10, 10), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 全连接层
        self.dense1 = nn.Sequential(
            nn.Linear(149720, 2000),
            nn.ReLU(),
            nn.Dropout(p=0.35),
            nn.Linear(2000, 1)
        )
        # 全连接层
        self.dense2 = nn.Sequential(
            nn.Linear(366460, 2000),
            nn.ReLU(),
            nn.Dropout(p=0.35),  # 缓解过拟合，一定程度上正则化
            nn.Linear(2000, 1)
        )
        self.dense=nn.Linear(2,1)

    def forward(self, x1, x2):
        ## 波形数据
        x1 = self.conv1(x1.reshape(-1, 1, 100, 1600))
        x1 = self.conv2(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.dense1(x1)

        ## 频谱数据
        x2 = self.conv3(x2.reshape(-1, 1, 1025, 313))
        x2 = self.conv4(x2)
        x2 = x2.view(x2.size(0), -1)  # flatten张量铺平，便于全连接层的接收
        x2 = self.dense2(x2)
        
        return self.dense(torch.cat([x1,x2],axis=1))
