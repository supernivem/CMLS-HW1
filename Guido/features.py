import numpy as np
import librosa
import scipy as sp

import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns

from Guido.parameters import *


def compute_features(audio, sr):
	audio_mean = np.mean(audio)
	audio_spec = np.fft.fft(audio)
	audio_pow = np.sum((audio - audio_mean) ** 2)

	frames_audio = []
	frames_spec = []
	frames_pow = []

	n_frames = int(np.floor((len(audio) - win_length) / hop_size))

	for i in range(n_frames):
		frame = audio[(i * hop_size): np.min([i * hop_size + win_length, len(audio)])]
		window = sp.signal.get_window(window='hanning', Nx=win_length)
		frame = frame * window

		frame_mean = np.mean(frame)
		frame_spec = np.fft.fft(frame)
		frame_pow = np.sum((frame - audio_mean) ** 2) / len(frame)

		frames_audio.append(frame)
		frames_spec.append(frame_spec)
		frames_pow.append(frame_pow)

	computed_features = []

	for feat in features:

		if feat == 'ZCR':
			temp = librosa.feature.zero_crossing_rate(audio)
			temp = np.mean(temp)

		elif feat == 'SpecRollOff':
			temp = librosa.feature.spectral_rolloff(audio, sr=sr)
			temp = np.mean(temp)

		elif feat == 'SpecCentr':
			temp = librosa.feature.spectral_centroid(audio, sr=sr)
			temp = np.mean(temp)

		elif feat == 'SpecBandWidth':
			temp = librosa.feature.spectral_bandwidth(audio, sr)
			temp = np.mean(temp)

		elif feat == 'SpecContrast':
			temp = librosa.feature.spectral_bandwidth(audio, sr)
			temp = np.mean(temp)

		elif feat == 'der':
			temp = 1
			for i in range(n_frames - 2):
				if (frames_pow[i] > frames_pow[i+1]) & (frames_pow[i+1] < frames_pow[i+2]):
					temp += 1
			# temp = np.log(temp)

		computed_features.append(temp)

	return computed_features
