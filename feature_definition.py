import numpy as np
import librosa
import scipy as sp


import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns


def compute_zcr(win, Fs):
    win_sign = np.sign(win)

    N = len(win)

    sign_diff = np.abs(win_sign[1:] - win_sign[:-1])

    zcr = len(sign_diff[sign_diff != 0]) * Fs / N

    # equivalent to

    zcr2 = np.sum(sign_diff) * Fs / (2 * N)
    return zcr



def compute_speccentr(spec):
    k_axis = np.arange(1, spec.shape[0] + 1)
    centr = np.sum(np.transpose(k_axis) * np.abs(spec)) / np.sum(np.abs(spec))
    return centr


def compute_specdec(spec):
    k_axis = np.arange(len(spec)) + 1;
    mul = 1 / np.sum(np.abs(spec[1:]));
    num = np.abs(spec[1:]) - np.tile(np.abs(spec[0]), len(spec) - 1);

    spectral_decrease = mul * np.sum(num / k_axis[:-1]);

    return spectral_decrease


