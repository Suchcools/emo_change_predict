
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
from sklearn import tree
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import copy
import os
import math
import random
def bymain(X_train_std, Y_train, X_test_std, Y_test):
    logreg = ComplementNB()
    logreg.fit(X_train_std, Y_train)
    predict = logreg.predict(X_test_std)
    lrpredpro = logreg.predict_proba(X_test_std)
    groundtruth = Y_test
    predictprob = lrpredpro
    return groundtruth, predict, predictprob

def lgmain(X_train_std, Y_train, X_test_std, Y_test):
    svcmodel = LogisticRegression(max_iter=10000)
    svcmodel.fit(X_train_std, Y_train, sample_weight=None)
    predict = svcmodel.predict(X_test_std)
    predictprob =svcmodel.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

    
def xgmain(X_train_std, Y_train, X_test_std, Y_test):
    model = XGBClassifier()
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

def dtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

def rfmain(X_train_std, Y_train, X_test_std, Y_test):
    model = RandomForestClassifier()
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

def gbdtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train_std, Y_train)
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

def evaluate(baslineName,dataset):
    X_train_std, Y_train, X_test_std, Y_test=dataset
    start =time.time()
    if baslineName == 'NaiveBayes':
        groundtruth, predict, predictprob = bymain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName == 'Logistic':
        groundtruth, predict, predictprob = lgmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='XGBoost':
        groundtruth, predict, predictprob = xgmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='DecisionTree':
        groundtruth, predict, predictprob = dtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='RandomForest':
        groundtruth, predict, predictprob = rfmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='GradientBoosting':
        groundtruth, predict, predictprob = gbdtmain(X_train_std, Y_train, X_test_std, Y_test)
    else:
        return

    acc = metrics.accuracy_score(groundtruth, predict)
    precision = metrics.precision_score(groundtruth, predict, zero_division=1 )
    recall = metrics.recall_score(groundtruth, predict)
    f1 = metrics.f1_score(groundtruth, predict)
    tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()
    ppv = tp/(tp+fp+1.4E-45)
    npv = tn/(fn+tn+1.4E-45)
    mcc=metrics.matthews_corrcoef(groundtruth, predict)
    end = time.time()
    spend=round(end-start,2)
    print('Running time: %s Seconds'%(end-start))
    item={'BaslineName':baslineName,'Accuracy':acc,'Precision':precision,'MCC':mcc,'PPV':ppv,'NPV':npv,'Recall':recall,'F1':f1,'Time':spend,'TP':tp,'FP':fp,'TN':tn,'FN':fn}
    return groundtruth, predict, predictprob,item

def ROC_plot(Y_test,y_score,filename):
    y_label=[]
    for i in range(len(Y_test)):
        y_label+=[[0,0]]
    for i in range(len(Y_test)):
        y_label[i][int(Y_test.values[i])]=1
    y_label=np.array(y_label)
    n_classes = 2

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    lw=2
    plt.figure(figsize=(8,8))
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i+1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(filename[filename.rfind('/')+1:filename.rfind('.')])
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()
    
def plot_matrix(y_true, y_pred,filename):
    cm = confusion_matrix(y_true, y_pred)#混淆矩阵
    #annot = True 格上显示数字 ，fmt：显示数字的格式控制
    ax = sns.heatmap(cm,annot=True,fmt='g',xticklabels=['Negative', 'Positive'],yticklabels=['Negative', 'Positive'],annot_kws={"fontsize":20})
    #xticklabels、yticklabels指定横纵轴标签
    ax.set_xlabel('Predict',size=20) #x轴
    ax.set_ylabel('GroundTruth',size=20) #y轴
    plt.xticks(fontsize=15) #x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=15) #y轴刻度的字体大小（文本包含在pd_data中了）
    plt.gcf().set_size_inches(8, 6)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=15)
    plt.savefig(filename)
    plt.show()