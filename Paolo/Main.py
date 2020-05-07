import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import sklearn.svm
#import IPython.display as ipd
import scipy as sp


#%% SPECTRAL DECREASE
def compute_specdec(spec):
    mul_fact = 1 / (np.sum(np.abs(spec[1:])) + 1e-16)  # avoid division by zero
    num = np.abs(spec[1:]) - np.tile(np.abs(spec[0]), len(spec) - 1)  # costruisce un array ripetendo il valore specificato
    den = np.arange(1, len(spec))
    spectral_decrease = mul_fact * np.sum(num / den)
    return spectral_decrease


#%% SPECTRAL CENTROID
def compute_speccentr(spec):
    k_axis = np.arange(1, spec.shape[0] + 1)  # numeri che vanno da 1 incluso a lunghezza spec + 1 escluso
    centr = np.sum(np.transpose(k_axis) * np.abs(spec)) / (np.sum(np.abs(spec)) + 1e-16)  # prodotto matriciale! Tra vettori, il primo va trasposto
    return centr


#%% AUDIO POWER
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


#%% COMPUTE TRAINING FEATURE
classes = ['NoFX', 'Tremolo', 'Distortion']
path = 'C:/Users/pao_b/Documents/POLITECNICO/AA 2019-20/CMLS/Homework1/'
dict_train_features = {'Tremolo': [], 'Distortion': [], 'NoFX': []}
dict_test_features = {'Tremolo': [], 'Distortion': [], 'NoFX': []}

for c in classes:
    root = path + 'IDMT-SMT-AUDIO-EFFECTS/Bass monophon/Samples/{}/'.format(c)  # path of the file parametrized
    class_files = [f for f in os.listdir(root) if f.endswith('.wav')]  # array of the traks
    n_train_samples = len(class_files)
    windows = 11
    pivot = int(np.floor(n_train_samples*0.8))
    class_train_files = class_files[:pivot]
    n_train_samples = len(class_train_files)
    train_features = np.zeros((n_train_samples, windows))
    for index, f in enumerate(class_train_files):
        audio, fs = librosa.load(os.path.join(root, f), sr=None)  # loads the file
        audio = cut_before_power_max(audio)
        audio_length = len(audio)
        win_length = int(np.floor(audio_length/11))  # definisce la lunghezza della finestra
        hop_size = int(np.floor(audio_length/13))  # definisce il tempo tra finestre successive (previsto overlapping)
        window = sp.signal.get_window(window='hanning', Nx=win_length)
        train_win_number = int(np.floor((audio_length - win_length) / hop_size))  # totale delle finestre nella traccia, l'ultima risulterebbbe tronca e va tolta
        for i in np.arange(train_win_number):
            frame = audio[i * hop_size: i * hop_size + win_length]  # estrae i campioni della finestra in analisi
            frame_wind = frame * window
            spec = np.fft.fft(frame_wind)  # esegue la fast fourier transform della finestra di segnale
            nyquist = int(np.floor(spec.shape[0] / 2))  # individua la frequenza di Nyquist (massima/2)
            spec = spec[1:nyquist]  # taglia tutte le frequenze oltre quella di nyquist
            train_features[index, i] = compute_speccentr(spec)
    dict_train_features[c] = train_features
    class_test_files = class_files[pivot:]
    n_test_samples = len(class_test_files)
    test_features = np.zeros((n_test_samples, windows))
    for index, f in enumerate(class_test_files):
        audio, fs = librosa.load(os.path.join(root, f), sr=None)
        audio_length = len(audio)
        win_length = int(np.floor(audio_length/11))  # definisce la lunghezza della finestra
        hop_size = int(np.floor(audio_length/13))  # definisce il tempo tra finestre successive (previsto overlapping)
        window = sp.signal.get_window(window='hanning', Nx=win_length)
        test_win_number = int(np.floor((audio_length - win_length) / hop_size))  # totale delle finestre nella traccia, l'ultima risulterebbbe tronca e va tolta
        for i in np.arange(test_win_number):
            frame = audio[i * hop_size: i * hop_size + win_length]  # estrae i campioni della finestra in analisi
            frame_wind = frame * window
            spec = np.fft.fft(frame_wind)  # esegue la fast fourier transform della finestra di segnale
            nyquist = int(np.floor(spec.shape[0] / 2))  # individua la frequenza di Nyquist (massima/2)
            spec = spec[1:nyquist]  # taglia tutte le frequenze oltre quella di nyquist
            test_features[index, i] = compute_speccentr(spec)
    dict_test_features[c] = test_features


#%% FEATURE VISUALIZATION
for c in classes:
    specdec = dict_train_features[c].transpose()
    # Visualization
    fig = plt.figure(figsize=(16, 6))
    plt.imshow(specdec, origin='lower', aspect='auto')
    plt.xlabel('Training samples')
    plt.ylabel('Spectral decrease coefficients')
    plt.title('Spectral decrease coefficients for class {}'.format(c))
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


#%% CLASSIFICATION ON MULTICLASSES
class_0 = 'Tremolo'
class_1 = 'Distortion'
class_2 = 'NoFX'

X_train_0 = dict_train_features[class_0]
X_train_1 = dict_train_features[class_1]
X_train_2 = dict_train_features[class_2]

y_train_0 = np.zeros((X_train_0.shape[0],))
y_train_1 = np.ones((X_train_1.shape[0],))
y_train_2 = np.ones((X_train_2.shape[0],))*2

X_test_0 = dict_test_features[class_0]
X_test_1 = dict_test_features[class_1]
X_test_2 = dict_test_features[class_2]

y_test_0 = np.zeros((X_test_0.shape[0],))
y_test_1 = np.ones((X_test_1.shape[0],))
y_test_2 = np.ones((X_test_2.shape[0],))*2

y_test_mc = np.concatenate((y_test_0, y_test_1, y_test_2), axis=0)


# NORMALIZE FEATURE

feat_max = np.max(np.concatenate((X_train_0, X_train_1, X_train_2), axis=0), axis=0)
feat_min = np.min(np.concatenate((X_train_0, X_train_1, X_train_2), axis=0), axis=0)

X_train_0_normalized = (X_train_0 - feat_min) / (feat_max - feat_min)
X_train_1_normalized = (X_train_1 - feat_min) / (feat_max - feat_min)
X_train_2_normalized = (X_train_2 - feat_min) / (feat_max - feat_min)

X_test_0_normalized = (X_test_0 - feat_min) / (feat_max - feat_min)
X_test_1_normalized = (X_test_1 - feat_min) / (feat_max - feat_min)
X_test_2_normalized = (X_test_2 - feat_min) / (feat_max - feat_min)

X_test_mc_normalized = np.concatenate((X_test_0_normalized, X_test_1_normalized, X_test_2_normalized), axis=0)


# DEFINE AND TRAIN A MODEL FOR EACH COUPLE OF CLASSES

SVM_parameters = {'C': 1, 'kernel': 'rbf'}

clf_01 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_02 = sklearn.svm.SVC(**SVM_parameters, probability=True)
clf_12 = sklearn.svm.SVC(**SVM_parameters, probability=True)

clf_01.fit(np.concatenate((X_train_0_normalized, X_train_1_normalized), axis=0),
           np.concatenate((y_train_0, y_train_1), axis=0))

clf_02.fit(np.concatenate((X_train_0_normalized, X_train_2_normalized), axis=0),
           np.concatenate((y_train_0, y_train_2), axis=0))

clf_12.fit(np.concatenate((X_train_1_normalized, X_train_2_normalized), axis=0),
           np.concatenate((y_train_1, y_train_2), axis=0))


# EVALUATE EACH CLASSIFIER

y_test_predicted_01 = clf_01.predict(X_test_mc_normalized).reshape(-1, 1)
y_test_predicted_02 = clf_02.predict(X_test_mc_normalized).reshape(-1, 1)
y_test_predicted_12 = clf_12.predict(X_test_mc_normalized).reshape(-1, 1)


# MAJORITY VOTING

y_test_predicted_mc = np.concatenate((y_test_predicted_01, y_test_predicted_02, y_test_predicted_12), axis=1)
y_test_predicted_mc = np.array(y_test_predicted_mc, dtype=np.int)

y_test_predicted_mv = np.zeros((y_test_predicted_mc.shape[0],))
for i, e in enumerate(y_test_predicted_mc):
    y_test_predicted_mv[i] = np.bincount(e).argmax()


#%% COMPUTING CONFUSION MATRIX FOR MULTICLASS

def compute_cm_multiclass(gt, predicted):
    classes = np.unique(gt)

    CM = np.zeros((len(classes), len(classes)))

    for i in np.arange(len(classes)):
        pred_class = predicted[gt == i]

        for j in np.arange(len(pred_class)):
            CM[i, int(pred_class[j])] = CM[i, int(pred_class[j])] + 1
    print(CM)


compute_cm_multiclass(y_test_mc, y_test_predicted_mv)
