# coding: UTF-8
import time
import torch
import numpy as np
import torch.utils.data as Data

from train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args(args=[])
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class MyDataSet(Data.Dataset):

    def __init__(self, path_dir):
        data = np.load(path_dir, allow_pickle=True)
        self.x = data['x']
        self.y = data['y']

    def __getitem__(self, index):
        return self.x[index],np.log1p(self.y[index])

    def __len__(self):
        return len(self.x)
data = MyDataSet("data/yl_vec2200_dataset.npz")

dataset = 'THUCNews'  # 数据集
# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = 'embedding_SougouNews.npz'
if args.embedding == 'random':
    embedding = 'random'
args.model = "Transformer_custom"
model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer, ResNet,Transformer_custom
if model_name == 'FastText':
    from utils_fasttext import build_dataset, build_iterator, get_time_dif

    embedding = 'random'
else:
    from utils import build_dataset, build_iterator, get_time_dif

x = import_module(f'models.{model_name}')
config = x.Config(dataset, embedding)

start_time = time.time()
print("Loading data...")

data_index=np.load('data_index.npz',allow_pickle=True)

batch_size=64
for index in range(len(data_index['data'])):
# 首先产生数据索引的乱序排列
    train_idx = data_index['data'][index][1]
    np.random.shuffle(train_idx)
    val_idx = data_index['data'][index][0]
    train_iter = Data.DataLoader(data, batch_size=batch_size,shuffle=False, drop_last=False, sampler=train_idx)
    dev_iter = Data.DataLoader(data, batch_size=batch_size, shuffle=False,drop_last=False, sampler=val_idx)
    test_iter = Data.DataLoader(data, batch_size=batch_size, shuffle=False,drop_last=False, sampler=val_idx)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # config.n_vocab = len(vocab)
    # 把模型放到GPU上
    model = x.Model(config).to(config.device)
    if model_name not in  ['Transformer','ResNet','Transformer_custom']:
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter,index)
