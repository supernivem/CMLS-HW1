from feature_definition import *
from train import *
import numpy as np
import librosa
import scipy as sp


import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats
import seaborn as sns



plt.figure(figsize=(16, 16))

plt.subplot(5,1,1)
time_axis = np.arange(audio_train.shape[0]) / Fs
plt.plot(time_axis, audio_train)
plt.grid(True)
plt.title('Train audio')

plt.subplot(5,1,2)
feat_time_axis = np.arange(train_features.shape[0]) * hop_size / Fs
plt.title(features_names[0])
plt.plot(feat_time_axis, train_features[:, 0])
plt.grid(True)

plt.subplot(5,1,3)
feat_time_axis = np.arange(train_features.shape[0]) * hop_size / Fs
plt.title(features_names[1])
plt.plot(feat_time_axis, train_features[:, 1])
plt.grid(True)


plt.subplot(5,1,4)
feat_time_axis = np.arange(train_features.shape[0]) * hop_size / Fs
plt.title(features_names[2])
plt.plot(feat_time_axis, train_features[:, 2])
plt.grid(True)


plt.subplot(5,1,5)
plt.plot(feat_time_axis, train_labels)




train_features_0 = train_features[train_labels==0]
train_features_1 = train_features[train_labels==1]

for feat_index, feat in enumerate(features_names):
    plt.figure(figsize=(16, 8))
    sns.distplot(train_features_0[:, feat_index], label='Histogram for class 0 of feature {}'.format(feat));
    sns.distplot(train_features_1[:, feat_index], label='Histogram for class 1 of feature {}'.format(feat));
    plt.legend()
    plt.grid(True)
    plt.show()



test_path = ('Input/test.wav')
audio_test, Fs = librosa.load(test_path, sr=None)


test_win_number = int(np.floor((audio_test.shape[0] - win_length) / hop_size))

test_features = np.zeros((test_win_number, n_features))

for i in np.arange(test_win_number):
    frame = audio_test[i * hop_size: i * hop_size + win_length]
    frame_wind = frame * window

    spec = np.fft.fft(frame_wind)
    nyquist = int(np.floor(spec.shape[0] / 2))
    spec = spec[1:nyquist]

    test_features[i, 0] = compute_zcr(frame_wind, Fs)
    test_features[i, 1] = compute_specdec(spec)
    test_features[i, 2] = compute_speccentr(spec)


test_labels = np.genfromtxt('Input/test_GT.csv', delimiter=' ')






selected_feature_index = 0

mu0 = np.mean(train_features_0[:, selected_feature_index])
std0 = np.std(train_features_0[:, selected_feature_index])

mu1 = np.mean(train_features_1[:, selected_feature_index])
std1 = np.std(train_features_1[:, selected_feature_index])





# Creat a normal continuous random variable

gauss_0 = scipy.stats.norm(mu0, std0)
gauss_1 = scipy.stats.norm(mu1, std1)

plt.figure(figsize=(16, 8))

# sample 1000 points from the distribution for class 0
r0 = gauss_0.rvs(size=1000)

# plot their distribution
sns.distplot(r0, label='Histogram for class 0');

# sample 1000 points from the distribution for class 1
r1 = gauss_1.rvs(size=1000)

# plot their ditribution
sns.distplot(r1, label='Histogram for class 1');




# For each point in the test feature
pdf_0 = gauss_0.pdf(test_features[:, selected_feature_index].reshape(-1,1))
pdf_1 = gauss_1.pdf(test_features[:, selected_feature_index].reshape(-1,1))
pdf = np.concatenate([pdf_0, pdf_1], axis=1)

naive_predicted_test_labels = np.argmax(pdf, axis=1)







n_components = 3
gmm_0 = BayesianGaussianMixture(n_components=n_components, random_state=2)
gmm_1 = BayesianGaussianMixture(n_components=n_components, random_state=2)

gmm_0.fit(train_features_0)
gmm_1.fit(train_features_1)

mixt_pdf_0 = []
mixt_pdf_1 = []

sample_0 = []
sample_1 = []

for n in np.arange(n_components):
    # Create a normal continuous random variable using the parameters estimated by EM algorithm for each class

    mixt_gauss_0 = scipy.stats.multivariate_normal(gmm_0.means_[n, :], gmm_0.covariances_[n, :], allow_singular=True)
    mixt_gauss_1 = scipy.stats.multivariate_normal(gmm_1.means_[n, :], gmm_1.covariances_[n, :], allow_singular=True)

    sample_0.append(mixt_gauss_0.rvs(np.int(500 * gmm_0.weights_[n])))
    sample_1.append(mixt_gauss_1.rvs(np.int(500 * gmm_1.weights_[n])))

    mixt_pdf_0.append(gmm_0.weights_[n] * mixt_gauss_0.pdf(test_features))
    mixt_pdf_1.append(gmm_1.weights_[n] * mixt_gauss_1.pdf(test_features))

pdf_0 = np.sum(mixt_pdf_0, axis=0).reshape(-1, 1)
pdf_1 = np.sum(mixt_pdf_1, axis=0).reshape(-1, 1)

pdf = np.concatenate((pdf_0, pdf_1), axis=1)





predicted_test_labels = np.argmax(pdf, axis=1)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection='3d')

markers = ['x', 'o', '*']
for n in np.arange(n_components):
    x = sample_0[n][:, 0]
    y = sample_0[n][:, 1]
    z = sample_0[n][:, 2]
    ax.scatter(x, y, z, c='r', marker=markers[n])

    x = sample_1[n][:, 0]
    y = sample_1[n][:, 1]
    z = sample_1[n][:, 2]
    ax.scatter(x, y, z, c='b', marker=markers[n])

plt.show()


def compute_metrics(gt_labels, predicted_labels):
    TP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 1))
    FP = np.sum(np.logical_and(predicted_labels == 1, gt_labels == 0))
    TN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 0))
    FN = np.sum(np.logical_and(predicted_labels == 0, gt_labels == 1))
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    print("Results : \n accuracy = {} \n precision = {} \n recall = {} \n F1 score = {}".format(
        accuracy, precision, recall, F1_score))



compute_metrics(test_labels, predicted_test_labels)

compute_metrics(test_labels, naive_predicted_test_labels)


