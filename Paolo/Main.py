import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import sklearn.svm
import scipy as sp
import random
import itertools


#%% FEATURE DEFINITION
def compute_specdec(spec):
    mul_fact = 1 / (np.sum(np.abs(spec[1:])) + 1e-16)  # avoid division by zero
    num = np.abs(spec[1:]) - np.tile(np.abs(spec[0]), len(spec) - 1)  # costruisce un array ripetendo il valore specificato
    den = np.arange(1, len(spec))
    spectral_decrease = mul_fact * np.sum(num / den)
    return spectral_decrease


# SPECTRAL CENTROID
def compute_speccentr(spec):
    k_axis = np.arange(1, spec.shape[0] + 1)  # numeri che vanno da 1 incluso a lunghezza spec + 1 escluso
    centr = np.sum(np.transpose(k_axis) * np.abs(spec)) / (np.sum(np.abs(spec)) + 1e-16)  # prodotto matriciale! Tra vettori, il primo va trasposto
    return centr


# SPECTRAL FLUX
def compute_specflux(spec):
    X = np.c_[spec[:], spec]
    afDeltaX = np.diff(X, 1, axis=1)
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
    return vsf


#%% COMPUTE FEATURES' VECTOR
param = 6


def compute_vector(features):
    vector = np.zeros(len(features)*param)
    for i, feat in enumerate(features):
        vector[i*param] = np.mean(feat)  # mean value
        vector[i*param+1] = np.std(feat)  # standard deviation
        vector[i*param+2] = np.mean(np.diff(feat))  # mean value of first derivative
        vector[i*param+3] = np.std(np.diff(feat))  # standard deviation of first derivative
        vector[i*param+4] = np.mean(np.diff(np.diff(feat)))  # mean value of second derivative
        vector[i*param+5] = np.std(np.diff(np.diff(feat)))  # standard deviation of second derivative
    return vector


#%% PREPROCESSING
def compute_audio_power(frame):
    length = len(frame)
    total_power = np.sum(np.power(frame, 2))
    return total_power / length


def cut_before_power_max(audio):
    frame_length = 8192  # samples
    frames = int(len(audio) / frame_length)
    powers = np.zeros(frames)
    for i in range(frames):
        powers[i] = compute_audio_power(audio[i*frame_length: i*frame_length+frame_length])
    index_max = np.argmax(powers)
    return audio[index_max*frame_length:]


#%% COMPUTE TRAINING AND TEST FEATURE
classes = ['Tremolo', 'Distortion', 'NoFX']
instruments = ['Bass monophon', 'Gitarre monophon', 'Gitarre polyphon']
path = 'C:/Users/pao_b/Documents/POLITECNICO/AA 2019-20/CMLS/Homework1/'
groups = ['train', 'test']
dict_features = {}

for g in groups:
    dict_features[g] = {}
    for inst in instruments:
        for c in classes:
            root = path + 'IDMT-SMT-AUDIO-EFFECTS/{}/Samples/{}/'.format(inst, c)  # path of the file parametrized
            class_files = [f for f in os.listdir(root) if f.endswith('.wav')]  # array of traks' path
            random.shuffle(class_files)
            n_samples = len(class_files)
            pivot = int(np.floor(n_samples*0.8))
            limit = int(np.floor(n_samples*1))
            if g == 'train':
                class_files = class_files[:pivot]
            else:
                class_files = class_files[pivot:limit]
            n_train_samples = len(class_files)
            n_features = 5
            features = np.zeros((n_train_samples, n_features*param))
            for index, f in enumerate(class_files):
                audio, fs = librosa.load(os.path.join(root, f), sr=None)  # loads the file
                audio = cut_before_power_max(audio)
                audio_length = len(audio)
                win_length = 8192  # definisce la lunghezza della finestra
                hop_size = 512  # definisce il tempo tra finestre successive (previsto overlapping)
                window = sp.signal.get_window(window='hanning', Nx=win_length)
                win_number = int(np.floor((audio_length - win_length) / hop_size))  # totale delle finestre nella traccia, l'ultima risulterebbbe tronca e va tolta
                speccentr = np.zeros(win_number)
                specdec = np.zeros(win_number)
                specflux = np.zeros(win_number)
                power = np.zeros(win_number)
                for i in np.arange(win_number):
                    frame = audio[i * hop_size: i * hop_size + win_length]  # estrae i campioni della finestra in analisi
                    frame2 = audio[i * hop_size: i * hop_size + hop_size]
                    frame_wind = frame * window
                    spec = np.fft.fft(frame_wind)  # esegue la fast fourier transform della finestra di segnale
                    nyquist = int(np.floor(spec.shape[0] / 2))  # individua la frequenza di Nyquist (massima/2)
                    spec = spec[1:nyquist]  # taglia tutte le frequenze oltre quella di nyquist
                    speccentr[i] = compute_speccentr(spec)
                    specdec[i] = compute_specdec(spec)
                    power[i] = compute_audio_power(frame2)
                rolloff = librosa.feature.spectral_rolloff(audio, fs)
                specband = librosa.feature.spectral_bandwidth(audio, fs)
                features[index, :] = compute_vector([rolloff, speccentr, specdec, specband, power])
            dict_features[g][c] = features


# FEATURE VISUALIZATION
for c in classes:
    specdec = dict_features['train'][c].transpose()
    # Visualization
    fig = plt.figure(figsize=(16, 6))
    plt.imshow(specdec, origin='lower', aspect='auto')
    plt.xlabel('Training samples')
    plt.ylabel('Features coefficients')
    plt.title('Features coefficients (mean, std and decrease) for class {}'.format(c))
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# %% EVALUATE THE MODEL
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


# CLASSIFICATION ON MULTICLASSES
X = {}
y = {}

for g in groups:
    X[g] = {}
    y[g] = {}
    for i, c in enumerate(classes):
        X[g][c] = dict_features[g][c]
        y[g][c] = np.ones((X[g][c].shape[0],))*i
y_test_mc = np.concatenate(list(y['test'].values()), axis=0)


# NORMALIZE FEATURE

feat_max = np.max(np.concatenate(list(X['train'].values()), axis=0), axis=0)
feat_min = np.min(np.concatenate(list(X['train'].values()), axis=0), axis=0)

X_normalized = {}
for g in groups:
    X_normalized[g] = {}
    for c in classes:
        X_normalized[g][c] = (X[g][c] - feat_min) / (feat_max - feat_min)

X_test_mc_normalized = np.concatenate(list(X_normalized['test'].values()), axis=0)


# DEFINE AND TRAIN A MODEL FOR EACH COUPLE OF CLASSES

SVM_parameters = {'C': 1, 'kernel': 'rbf'}

combinations = list(itertools.combinations(classes, 2))
y_test_predicted = {}
for i, comb in enumerate(combinations):
    clf = sklearn.svm.SVC(**SVM_parameters, probability=True)
    x_conc = np.concatenate((X_normalized['train'][comb[0]], X_normalized['train'][comb[1]]), axis=0)
    y_conc = np.concatenate((y['train'][comb[0]], y['train'][comb[1]]), axis=0)
    clf.fit(x_conc, y_conc)
    y_test_predicted[i] = clf.predict(X_test_mc_normalized).reshape(-1, 1)  # evaluate each classifier


# MAJORITY VOTING

y_test_predicted_mc = np.concatenate(list(y_test_predicted.values()), axis=1)
y_test_predicted_mc = np.array(y_test_predicted_mc, dtype=np.int)

y_test_predicted_mv = np.zeros((y_test_predicted_mc.shape[0],))
for i, e in enumerate(y_test_predicted_mc):
    y_test_predicted_mv[i] = np.bincount(e).argmax()


# COMPUTING CONFUSION MATRIX FOR MULTICLASS

def compute_cm_multiclass(gt, predicted):
    classes = np.unique(gt)

    CM = np.zeros((len(classes), len(classes)))

    for i in np.arange(len(classes)):
        pred_class = predicted[gt == i]

        for j in np.arange(len(pred_class)):
            CM[i, int(pred_class[j])] = CM[i, int(pred_class[j])] + 1
    print(CM)
    print((CM / CM.sum(axis=1).reshape((-1, 1)))*100)


compute_cm_multiclass(y_test_mc, y_test_predicted_mv)
