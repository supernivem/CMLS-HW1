import numpy as np
import librosa
import scipy as sp

import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns

from Guido.parameters import *


def compute_metrics(gt_labels, predicted_labels):
	tp = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 1))
	fp = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 0))
	tn = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 0))
	fn = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 1))
	accuracy = (tp + tn) / (tp + fp + tn + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * precision * recall / (precision + recall)
	print("Results : \n Accuracy = {} \n Precision = {} \n Recall = {} \n F1 f1_score = {}".format(
		accuracy, precision, recall, f1_score))


def compute_cm(real, predicted):
	cm = np.zeros((len(classes), len(classes)))

	for c in classes:
		pred_class = predicted[real == c]
		for d in pred_class:
			cm[classes.index(c), classes.index(d)] += 1
	return cm
