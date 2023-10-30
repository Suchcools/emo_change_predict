# coding: UTF-8
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
gpus=['0']

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


def train(config, model, train_iter, dev_iter, test_iter,device_ids):
    global gpus
    gpus=device_ids
    start_time = time.time()
    model.train()
    # 优化器，反向传播后，还需要优化方法来更新网络的权重和参数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    recard = pd.DataFrame()
    for epoch in range(config.num_epochs):
        print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
        # scheduler.step() # 学习率衰减
        for trains, labels in train_iter:
            trains, labels = trains.cuda(device=device_ids[0]).float(), labels.cuda(device=device_ids[0]).float()
            # print('trains:',len(trains)) # 128,3948
            # print('labels',len(labels))# 128.view -1
            outputs = model(trains)  # 10,73->10,5
            model.zero_grad()
            # loss = F.cross_entropy(outputs, labels)
            loss = F.mse_loss(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                train_acc = metrics.r2_score(true, outputs.data.cpu().flatten())
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model, './result/model.pkl')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.4},  Val Loss: {3:>5.2},  Val Acc: {4:>6.4},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                recard = recard.append([{'train_loss': loss.item(), 'dev_loss': dev_loss.item(), 'train_acc': train_acc,
                                         'dev_acc': dev_acc}], ignore_index=True)
                model.train()
            total_batch += 1
            # if total_batch - last_improve > config.require_improvement:
            #     # 验证集loss超过1000batch没下降，结束训练
            #     print("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
        # if flag:
        #     break

    recard.to_csv('result/recard.csv', index=False)
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    # model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    global gpus
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            texts, labels = texts.cuda(device=gpus[0]).float(), labels.cuda(device=gpus[0])
            outputs = model(texts)
            # loss = F.cross_entropy(outputs, labels)
            loss = F.mse_loss(outputs.data.flatten(), labels)
            loss_total += loss
            labels = labels.data.cpu().flatten().numpy()
            predic = outputs.data.cpu().flatten().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    report = pd.DataFrame([labels_all, predict_all]).T
    report.columns = ['GroundTruth', 'Predict']
    report.to_csv('result/label.csv', index=False)
    acc = metrics.r2_score(labels_all, predict_all)
    torch.save(model, './result/tmodel.pkl')
    if test:
        # print('Truth : ', labels_all, '\nYpred : ', predict_all)
        return acc, loss_total / len(data_iter)
    return acc, loss_total / len(data_iter)
