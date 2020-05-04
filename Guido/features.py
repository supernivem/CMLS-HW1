import numpy as np
import librosa
import scipy as sp

import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns


def compute_features(features, audio, sr):

	computed_features = []
	for feat in features:

		if feat == 'ZCR':
			temp = librosa.feature.zero_crossing_rate(audio)
			temp = np.mean(temp)

		elif feat == 'SpecRollOff':
			temp = librosa.feature.spectral_rolloff(audio, sr)
			temp = np.mean(temp)

		elif feat == 'SpecDec':
			pass

		computed_features.append(temp)
	return computed_features
