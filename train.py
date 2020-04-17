from feature_definition import *

import numpy as np
import librosa
import scipy as sp


import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns



train_path = ('Input/train.wav')
audio_train, Fs = librosa.load(train_path, sr=None)
ipd.Audio(audio_train, rate=Fs) # load a local WAV file




win_length = int(np.floor(0.01 * Fs))
hop_size = int(np.floor(0.0075 * Fs))

window = sp.signal.get_window(window='hanning', Nx=win_length)

features_names = ['Zero Crossing Rate', 'Spectral Decrease', 'Spectral Centroid']

train_win_number = int(np.floor((audio_train.shape[0] - win_length) / hop_size))

n_features = 3

train_features = np.zeros((train_win_number, n_features))
for i in np.arange(train_win_number):
    frame = audio_train[i * hop_size: i * hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind)
    nyquist = int(np.floor(spec.shape[0] / 2))
    spec = spec[1:nyquist]

    train_features[i, 0] = compute_zcr(frame_wind, Fs)
    train_features[i, 1] = compute_specdec(spec)
    train_features[i, 2] = compute_speccentr(spec)




train_labels = np.genfromtxt('Input/train_GT.csv', delimiter=' ')


