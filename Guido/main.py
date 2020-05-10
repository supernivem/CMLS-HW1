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
from tqdm import tqdm

from Guido.parameters import *
from Guido.features import *
from Guido.metrics import *

# Compute features and labels
x_train, y_train, x_test, y_test = [], [], [], []
for inst in instruments:
	for class_index, cls in enumerate(classes):
		root = 'Data/{}/{}/'.format(inst, cls)
		files = [f for f in os.listdir(root) if f.endswith('.wav')]
		n_files = len(files)
		last_file_excluded = np.floor(n_files * data_proportion)
		last_train_file_excluded = np.floor(n_files * data_proportion * (1 - test_proportion))

		for i in tqdm(range(n_files)):

			if i < last_file_excluded:
				audio, fs = librosa.load(os.path.join(root, files[i]), sr=None)
				temp_features = compute_features(audio, fs)

				if i < last_train_file_excluded:
					x_train.append(temp_features)
					y_train.append(class_index)
				else:
					x_test.append(temp_features)
					y_test.append(class_index)

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
clf.fit(x_train_norm, y_train),
y_test_predicted = clf.predict(x_test_norm)

"""
SVM_parameters = {'C': 1, 'kernel': 'rbf'}

clf_01 = sklearn.svm.SVC(**SVM_parameters, probability=True, class_weight='balanced')
clf_02 = sklearn.svm.SVC(**SVM_parameters, probability=True, class_weight='balanced')
clf_12 = sklearn.svm.SVC(**SVM_parameters, probability=True, class_weight='balanced')

clf_01.fit(x_train_norm[y_train != 2],
			  y_train[y_train != 2])

clf_02.fit(x_train_norm[y_train != 1],
			  y_train[y_train != 1])

clf_12.fit(x_train_norm[y_train != 0],
			  y_train[y_train != 0])

# EVALUATE EACH CLASSIFIER

y_test_predicted_01 = clf_01.predict(x_test_norm).reshape(-1, 1)
y_test_predicted_02 = clf_02.predict(x_test_norm).reshape(-1, 1)
y_test_predicted_12 = clf_12.predict(x_test_norm).reshape(-1, 1)

# MAJORITY VOTING

y_test_predicted_mc = np.concatenate((y_test_predicted_01, y_test_predicted_02, y_test_predicted_12), axis=1)
y_test_predicted_mc = np.array(y_test_predicted_mc, dtype=np.int)

y_test_predicted = np.zeros((y_test_predicted_mc.shape[0],))
for i, e in enumerate(y_test_predicted_mc):
	y_test_predicted[i] = np.bincount(e).argmax()
"""


# Metrics
cm = compute_cm(y_test, y_test_predicted)
# compute_metrics(y_test, y_test_predicted)
print(cm)
