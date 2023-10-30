# coding: UTF-8
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)


def train(config, model, train_iter, dev_iter, test_iter):
    model.eval()
    alist=[]
    for trains, labels in train_iter:
        trains, labels = trains.cuda().float(), labels.float().cuda()
        outputs = model(trains)  # 10,73->10,5
        alist=list(outputs.flatten().cpu().detach().numpy())
        olist=[np.expm1(x) for x in list(outputs.flatten().cpu().detach().numpy())]
    pd.DataFrame([alist,olist]).T.to_csv('result_TPM.csv',index=False)
