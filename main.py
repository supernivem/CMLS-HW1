import numpy as np
import librosa
import scipy as sp
import os
from feature_computation import *
import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns

instruments = ['BM', 'GM', 'GP']
classes = ['NOFX', 'TREM', 'DIST']

features = ['ZCR', 'SpecDec']

for inst in instruments:
	for cls in classes:
		root = 'Data/{}/{}/'.format(inst, cls)
		files = [f for f in os.listdir(files) if f.endswith('.wav')]
		for file_index, file in enumerate(files):
			audio, fs = librosa.load(os.path.join(root, file), sr=None)
			temp_features = np.zeros(len(features))
			for feat_index, feat in enumerate(features):
				temp_features[feat_index] = compute_feature(feat, audio, fs)

