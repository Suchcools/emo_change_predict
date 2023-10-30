# coding: UTF-8
import torch
import numpy as np
import torch.utils.data as Data
import librosa
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import librosa.display
from moviepy.editor import *
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=60)
from importlib import import_module
model_name = "TextCNN_custom"


x = import_module(f'models.{model_name}')
config = x.Config()
device = torch.device("cpu")
model = x.Model(config).to(device)
model  =torch.load('./ArousalRegress/model_backup/CNN_CCC_0.61/tmodel_0.pkl').to(device)

def get_waveform(x):
    try:
        # np.append(temp,np.array([[x.P]*1600]), axis = 0)
        temp = x.reshape(100,1600)
        mu = np.mean(temp, axis=0)
        sigma = np.std(temp, axis=0)
        return (temp - mu) / sigma
    except:
        return None
def get_spectrogram(x):
    try:
        temp=librosa.amplitude_to_db(librosa.stft(x.flatten())).reshape(1025, 313)
        mu = np.mean(temp, axis=0)
        sigma = np.std(temp, axis=0)
        return (temp - mu) / sigma
    except:
        return None

class MyDataSet(Data.Dataset):

    def __init__(self,wave_data,spect_data):
        self.x = wave_data,spect_data

    def __getitem__(self, index):
        return self.x[0][index],self.x[1][index],0

    def __len__(self):
        return len(self.x[0])
def get_arousal(id):
    try:
        video = VideoFileClip(f'./video/{id}.mp4')
        audio = video.audio
        audio.write_audiofile(f'./voice/test{id}.wav')
        samples, sr = librosa.load(f'./voice/test{id}.wav', sr=16000)
        step = 160000
        wave_data = [samples[i:i+step] for i in range(0,len(samples),step)]
        wave_data[-1] = samples[-step:]
        wave_data = np.array([get_waveform(x) for x in wave_data])
        spect_data = np.array([get_spectrogram(x) for x in wave_data])
        data=MyDataSet(wave_data,spect_data)
        dev_iter = Data.DataLoader(data, batch_size=1000, shuffle=False)
        ## 模型预测
        model.eval()
        for trains1,trains2, labels in dev_iter:
            trains1, trains12, labels = trains1.float(),trains2.float(), labels.float()
            outputs = model(trains1,trains12)  # 10,73->10,5
            olist=[np.expm1(x) for x in list(outputs.flatten().detach().numpy())]
        return sum(olist)/len(olist)
    except :
        return 'NA'
rawdata=pd.read_excel('/home/linjw/code2/gnn_study_crawler/backup/variation_record.xlsx',dtype=str)
rawdata=rawdata.drop_duplicates(subset='aweme_id')
rawdata['arousal']=rawdata.aweme_id.parallel_apply(get_arousal)
rawdata.to_excel('./backup/arousal_record.xlsx', engine="xlsxwriter", encoding="utf-8", index=False)4