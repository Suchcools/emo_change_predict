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


class MyDataSet(Data.Dataset):

    def __init__(self, path_dir):
        dataset = np.load(path_dir, allow_pickle=True)
        self.x = dataset['x1'],dataset['x2']
        self.y = dataset['y']

    def __getitem__(self, index):
        return self.x[0][index],self.x[1][index],self.y[index]

    def __len__(self):
        return len(self.x[0])
data = MyDataSet("./Data/dataset.npz")

# 首先产生数据索引的乱序排列
shuffled_indices = np.random.permutation(len(data))
train_idx = shuffled_indices[:int(0.88 * len(data))]
val_idx = shuffled_indices[int(0.88 * len(data)):]

dataset = 'THUCNews'  # 数据集
np.savez('./result/index.npz', train=train_idx, test=val_idx)
# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = 'embedding_SougouNews.npz'
if args.embedding == 'random':
    embedding = 'random'
args.model = "TextCNN_custom"
model_name = args.model  # TextCNN,Transformer_custom
if model_name == 'FastText':
    from utils_fasttext import build_dataset, build_iterator, get_time_dif

    embedding = 'random'
else:
    from utils import build_dataset, build_iterator, get_time_dif

x = import_module(f'models.{model_name}')
config = x.Config()
# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
# 分词+处理句子以及标签
# vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
# print('train: ,dev: ,test: ',train_data.shape,dev_data,test_data)
# 把数据放到GPU上面 # 128,3948
batch_size=64
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
train(config, model, train_iter, dev_iter, test_iter,index=0)
