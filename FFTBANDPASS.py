import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import butter, filtfilt, lfilter
import sys
import numpy as np
from sklearn.decomposition import FastICA
import scipy.signal
import scipy.stats

def multi_kurtosis_bss(denoised_signals):
    
    signals_matrix = denoised_signals.squeeze(1)
    
    
    ica = FastICA(n_components=signals_matrix.shape[0])
    sources = ica.fit_transform(signals_matrix.T).T
    
   
    kurtosis_values = [scipy.stats.kurtosis(source) for source in sources]
    
    
    rppg_signal = sources[np.argmax(kurtosis_values)]
    
    return rppg_signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=1, highcut=4, fs=30.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_heart_rate(pulse_signal, frame_rate):
    
    filtered_signal = bandpass_filter(pulse_signal, lowcut=1, highcut=2.3, fs=frame_rate)
    
    
    n = len(filtered_signal)
    freqs = np.fft.fftfreq(n, d=1/frame_rate)
    fft_spectrum = np.fft.fft(filtered_signal)
    positive_freqs = freqs[:n//2]
    positive_spectrum = np.abs(fft_spectrum[:n//2])
    
    
    max_freq = positive_freqs[np.argmax(positive_spectrum)]
    
    
    heart_rate = max_freq * 60

    plt.figure(figsize=(12, 6))
    plt.plot(positive_freqs, positive_spectrum)
    # plt.plot(filtered_signal)
    plt.axvline(x=max_freq, color='r', linestyle='--', label=f'HR = {heart_rate:.2f} bpm')
    plt.title('Pulse Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    return heart_rate


# ytrainpath=r"D:\torchsave\interpolated_waveform.npy"

# y_train=np.load(ytrainpath)# preapring y_train here its a numpy with trace values i.e. the ppg signal

# y_train=torch.from_numpy(y_train)




# y_train=y_train.view(1,1950) #1950 are no. of frames in my video here
# y_train=(y_train-(torch.mean(y_train)))/(torch.std(y_train)) 

frame_rate=30

# heart_rate=compute_heart_rate(y_train[0], frame_rate)

final_prediction=np.load('SahilSaver/SahilSaver.npy')
# final_prediction1=np.load('groundtruth.npy')
final_prediction = final_prediction.reshape(150)
print(final_prediction)
plt.plot(final_prediction)
plt.show()



frame_rate = 30
heart_rate = compute_heart_rate(final_prediction, frame_rate)

print(f"Estimated Heart Rate: {heart_rate:.2f} bpm")