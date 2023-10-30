# coding: UTF-8
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from numpy.random import default_rng
def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    x,y=np.array(x),np.array(y)
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc
def r(x,y):
    ''' Pearson Correlation Coefficient'''
    x,y=np.array(x),np.array(y)
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rho = sxy / (np.std(x)*np.std(y))
    return rho

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


def train(config, model, train_iter, dev_iter, test_iter,index):
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
        for trains, trains1, labels in train_iter:
            trains, trains1 ,labels = trains.cuda().float(),trains1.cuda().float() ,labels.float().cuda()
            # print('trains:',len(trains)) # 128,3948
            # print('labels',len(labels))# 128.view -1
            outputs = model(trains,trains1)  # 10,73->10,5
            model.zero_grad()
            # loss = F.cross_entropy(outputs, labels)
            loss = F.mse_loss(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                train_acc = ccc(true.numpy(), outputs.data.cpu().flatten())
                dev_acc, dev_loss = evaluate(config, model, dev_iter,index=index)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model, './result/model.pkl')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train CCC: {2:>6.4},  Val Loss: {3:>5.2},  Val CCC: {4:>6.4},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                recard = recard.append([{'train_loss': loss.item(), 'dev_loss': dev_loss.item(), 'train_CCC': train_acc,
                                         'dev_CCC': dev_acc}], ignore_index=True)
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
    test(config, model, test_iter,index)


def test(config, model, test_iter,index):
    model.eval()
    start_time = time.time()
    test_acc, test_loss = evaluate(config, model, test_iter,index,test=True)
    msg = 'Test Loss: {0:>5.2},  Test CCC: {1:>6.2}'
    print(msg.format(test_loss, test_acc))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter,index,test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, texts1,labels in data_iter:
            texts,texts1,labels = texts.cuda().float(),texts1.cuda().float() ,labels.cuda()
            outputs = model(texts,texts1)
            # loss = F.cross_entropy(outputs, labels)
            loss = F.mse_loss(outputs.data.flatten(), labels)
            loss_total += loss
            labels = labels.data.cpu().flatten().numpy()
            predic = outputs.data.cpu().flatten().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    report = pd.DataFrame([labels_all, predict_all]).T
    report.columns = ['GroundTruth', 'Predict']
    report.to_csv(f'result/label_{index}.csv', index=False)
    acc = ccc(labels_all, predict_all)
    torch.save(model, f'./result/tmodel_{index}.pkl')
    if test:
        # print('Truth : ', labels_all, '\nYpred : ', predict_all)
        X,Y=np.array(labels_all),np.array(predict_all)
        with plt.style.context(('seaborn-whitegrid')):
            plt.figure(figsize=(8,6))
            
            # Scatter plot of X vs Y
            plt.scatter(X,Y,edgecolors='k',alpha=0.5)
        
            # Plot of the 45 degree line
            plt.plot([0,1],[0,1],'r')
            
            rng = default_rng()
            sigma = 0.01
            tilt = 0
            for i in range(X.shape[0]):
                Y[i] = tilt*(X[i]-0.5) + rng.normal(X[i],sigma)

            plt.text(0, 0.75*Y.max(), "CCC: %5.5f"%(ccc(X,Y))+"\nrho: %5.5f"%(r(X,Y))+"\nR2: %5.5f"%(r2_score(X, Y)),\
                    fontsize=16, bbox=dict(facecolor='white', alpha=0.5))
            plt.text(0.8, 0.1, "$\sigma=$ %5.3f"%(sigma)+"\nTilt = %5.3f"%(tilt),\
                    fontsize=16, bbox=dict(facecolor='white', alpha=0.5))
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('X',fontsize=16)
            plt.ylabel('Y',fontsize=16)
            plt.savefig('result/CCC.png')
            plt.show()



        return acc, loss_total / len(data_iter)
    return acc, loss_total / len(data_iter)
