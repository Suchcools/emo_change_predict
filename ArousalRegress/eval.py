# coding: UTF-8
import time
import torch
import numpy as np
import torch.utils.data as Data

from train_eval import train
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
        self.y = np.array(list(range(0,59)))

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)
data = MyDataSet("/home/linjw/BioML/PredictNN/data/YL/compare/test_dataset1.npz")


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
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
# 分词+处理句子以及标签
# vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
# print('train: ,dev: ,test: ',train_data.shape,dev_data,test_data)
# 把数据放到GPU上面 # 128,3948
batch_size=59
train_iter = Data.DataLoader(data, batch_size=batch_size, drop_last=False,shuffle=False)
dev_iter = Data.DataLoader(data, batch_size=batch_size, drop_last=False,shuffle=False)
test_iter = Data.DataLoader(data, batch_size=batch_size, drop_last=False,shuffle=False)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# train
# config.n_vocab = len(vocab)
# 把模型放到GPU上
model = torch.load('./Test_Model/TPM_model.pkl')
train(config, model, train_iter, dev_iter, test_iter)
