import numpy as np
import librosa
import scipy as sp
import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns

import os
import sklearn

from Guido.parameters import *
from Guido.features import *
from Guido.metrics import *

# Compute features and labels
x_train, y_train, x_test, y_test = [], [], [], []
for inst in instruments:
	for cls in classes:
		root = 'Data/{}/{}/'.format(inst, cls)
		files = [f for f in os.listdir(root) if f.endswith('.wav')]
		n_files = len(files)
		last_file_excluded = np.floor(n_files * data_proportion)
		last_train_file_excluded = np.floor(n_files * data_proportion * (1 - test_proportion))

		for file_index, file in enumerate(files):

			if file_index < last_file_excluded:
				audio, fs = librosa.load(os.path.join(root, file), sr=None)
				temp_features = compute_features(features, audio, fs)

				if file_index < last_train_file_excluded:
					x_train.append(temp_features)
					y_train.append(cls)
				else:
					x_test.append(temp_features)
					y_test.append(cls)

x_train = np.array(x_train)
x_train.reshape(-1, n_features)
x_test = np.array(x_test)
x_test.reshape(-1, n_features)
y_train = np.array(y_train)
y_train.reshape(-1, 1)
y_test = np.array(y_test)
y_test.reshape(-1, 1)

# Normalization
feat_max = np.max(x_train, axis=0)
feat_min = np.min(x_train, axis=0)
x_train_norm = (x_train - feat_min) / (feat_max - feat_min)
x_test_norm = (x_test - feat_min) / (feat_max - feat_min)

# Training
clf = sklearn.svm.SVC(class_weight='balanced')
clf.fit(x_train_norm, y_train)

# Predict
y_test_predicted = clf.predict(x_test_norm)

# Metrics
cm = compute_cm(y_test, y_test_predicted)
# compute_metrics(y_test, y_test_predicted)
print(cm)
