# coding: UTF-8
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")
import librosa.display
from moviepy.editor import *
video = VideoFileClip('./Voice_arousal/movie.mp4')
audio = video.audio
audio.write_audiofile('./Voice_arousal/result/test.wav')

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

samples, sr = librosa.load('./Voice_arousal/result/test.wav', sr=16000)
step = 160000
wave_data = [samples[i:i+step] for i in range(0,len(samples),step)]
wave_data[-1] = samples[-step:]
wave_data = np.array([get_waveform(x) for x in wave_data])
spect_data = np.array([get_spectrogram(x) for x in wave_data])
print(wave_data.shape,spect_data.shape)
