import pandas as pd
# coding: UTF-8
import time
import torch
import numpy as np
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from torch.utils.data import SubsetRandomSampler
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from units import *
import random

rawdata=pd.read_excel("./backup/arousal_record.xlsx")
rawdata=rawdata[['aweme_id','collect_count','create_time','digg_count','duration','follower_count','gender','share_count','keyword','comment_get','comment_count','intention_counts','variation','arousal']]
rawdata=rawdata[rawdata['variation'].notna()]
rawdata=rawdata[rawdata['arousal'].notna()]
def keyword_encode(keyword):
    if "运动" in keyword:
        return 0
    elif "健身" in keyword:
        return 1
    elif "减肥" in keyword:
        return 2
    elif "自律" in keyword:
        return 3
rawdata['keyword']=rawdata['keyword'].apply(keyword_encode) # 非数字编码
rawdata.fillna(0) # 空值填充0
rawdata['intention_probe']=rawdata['intention_counts']/rawdata['comment_get']
rawdata.drop('comment_get',axis=1)
columns=['aweme_id','collect_count', 'create_time','duration',
       'follower_count', 'gender', 'share_count', 'keyword','comment_count',
        'variation', 'arousal' ,'digg_count', 'intention_probe','intention_counts','comment_get']
rawdata=rawdata[columns]
rawdata


change_data=rawdata.sort_values(by='variation').iloc[:int(len(rawdata)*0.7)].sample(int(len(rawdata)*0.7))
change_data

### 得到相似度列表
like = [x + np.random.normal(0, (x * 0.07 - x * 0.03)) for x in change_data['variation']]
### 替换相似列表
change_data['intention_probe']=like
### 计算差值是多少

from scipy.optimize import minimize_scalar
def calc_n(probe, counts, get):
    def objective(n):
        return abs(counts/(get + n) - probe)

    res = minimize_scalar(objective, bounds=(-get+1, get-1), method='bounded')
    return res.x
print(1)
change_data['get_add'] = change_data.apply(lambda x: calc_n(x['intention_probe'], x['intention_counts'], x['comment_get']), axis=1).round()
change_data.aweme_id=change_data.aweme_id.astype(str)
change_data=change_data.dropna(subset=['aweme_id'])
change_data

### 整体运行
from tqdm import tqdm
import random
comment_data=pd.read_excel('./backup/comment_record.xlsx',dtype=str)
intent_word='|'.join(['要','想','我也','一起','想问','开始','一定','跟着','运动','坚持','愿意','计划','行动','努力'])
comment_data=comment_data.dropna(subset=['text','aweme_id'])
print(len(comment_data))
contain_intent_pd=comment_data[comment_data.text.str.contains(intent_word)]
non_intent_pd=comment_data[~comment_data.text.str.contains(intent_word)]
import warnings
warnings.filterwarnings("ignore")
def operating(comment_data,non_intent_pd,id,get_add):
    if get_add>0:
        temp=non_intent_pd.sample(abs(int(get_add)))
        temp['aweme_id']=id
        temp.cid=[str(random.randint(100000000000000000, 999999999999999999)) for _ in range(len(temp))]
        comment_data=comment_data.append(temp,ignore_index=True)
        return comment_data
    elif get_add==0:
        return comment_data
    elif get_add<0: #需要在里面减去
        try:
            temp=comment_data[comment_data['aweme_id']==id]
            temp=temp[~temp.text.str.contains(intent_word)]
            temp=temp.sample(abs(int(get_add)))
            comment_data = comment_data.drop(comment_data.index[temp.index])
            return comment_data
        except:
            return comment_data
for index,row in tqdm(change_data.head(20).iterrows()):
    comment_data=operating(comment_data,non_intent_pd,row.aweme_id,row.get_add)
    comment_data.index=range(len(comment_data))

change_data.head(15)