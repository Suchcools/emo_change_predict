# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

def get_mydata():
    # rhea_reaction_1_1 = pd.read_feather('/home/dengrui/anaconda3/PyProjects/bio_protein_reaction_commit/bio_protein_reaction/tasks/rhea_dataset_expa_20221011.feather')
    # rhea_relation_copy = rhea_reaction_1_1.copy()
    # rhea_uniprot = pd.read_feather('/home/dengrui/anaconda3/PyProjects/bio_protein_reaction_commit/bio_protein_reaction/data/featureBank/rhea_uniprot_226101_20220924.feather')
    # rhea_products = pd.read_feather('/home/dengrui/anaconda3/PyProjects/bio_protein_reaction_commit/bio_protein_reaction/data/featureBank/rhea_products_smile_20220928.feather')
    # rhea_substrates = pd.read_feather('/home/dengrui/anaconda3/PyProjects/bio_protein_reaction_commit/bio_protein_reaction/data/featureBank/rhea_substrates_smile_20220928.feather')
    # dataset = rhea_relation_copy.merge(rhea_uniprot,on='uniprot_id',how='inner')
    # dataset = dataset.merge(rhea_substrates,on='reaction_id',how='inner')
    # dataset = dataset.merge(rhea_products,on='reaction_id',how='inner')
    # test_data = dataset[dataset.reaction_enzymes==1]
    # train_data = dataset[dataset.reaction_enzymes!=1]
    # train_data,vali_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_data = pd.read_feather('/home/dengrui/PyProjects/Multilabel/Chinese_Text_Classification_Pytorch/THUCNews/data/train_data.feather')
    test_data = pd.read_feather('/home/dengrui/PyProjects/Multilabel/Chinese_Text_Classification_Pytorch/THUCNews/data/test_data.feather')
    vali_data= pd.read_feather('/home/dengrui/PyProjects/Multilabel/Chinese_Text_Classification_Pytorch/THUCNews/data/vali_data.feather')

    return train_data.iloc[:,:],test_data.iloc[:,:],vali_data.iloc[:,:]


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    args.model = "TextRNN_Att"
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
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
    train_data,test_data,dev_data = get_mydata()
    # 把数据放到GPU上面
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # config.n_vocab = len(vocab)
    # 把模型放到GPU上
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
