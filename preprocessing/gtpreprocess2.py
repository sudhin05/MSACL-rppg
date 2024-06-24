import os
import numpy as np
import cv2
import scipy.io as sio

from heartpy.filtering import filter_signal

path = 'DATASET_2/'
save_path = 'Generatedmat/'

winLength = 150  # Window length to match the frame segments

if not os.path.exists(save_path):
    os.makedirs(save_path)

for dir in os.listdir(path):
    PPG = np.loadtxt(path + dir + '/ground_truth.txt')[0]
    HR = np.loadtxt(path + dir + '/ground_truth.txt')[1]
    vidobj = cv2.VideoCapture(path + dir + '/output.mkv')
    vid_len = int(vidobj.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidobj.get(cv2.CAP_PROP_FPS)
    vidobj.release()
    duration = vid_len / fps
    sr = len(PPG) / duration
    d = len(PPG) / sr
    PPG_scaled = filter_signal(PPG, cutoff=3, sample_rate=sr, order=2, filtertype='lowpass')
    PPG_scaled = 2 * (PPG_scaled - np.min(PPG_scaled)) / (np.max(PPG_scaled) - np.min(PPG_scaled)) - 1

    signalDict = {'PPG': PPG, 'PPG_scaled': PPG_scaled, 'HR': HR, 'SampleRate': sr, 'Duration': duration}
    ppg_scaled_list = np.array(signalDict['PPG_scaled'])
    save_path1 = os.path.join(save_path,dir)
    # Saving PPG labels in chunks of winLength (150)
    for i in range(0, len(ppg_scaled_list) - winLength + 1, winLength):
        segment = ppg_scaled_list[i:i + winLength]
        mat_filename = f"gtPPG{str(i // winLength).zfill(4)}.mat"
        mat_path = os.path.join(save_path1, mat_filename)
        sio.savemat(mat_path, {'PPG_scaled_value': segment})
        print(f"Saved {mat_path}")

print("Conversion complete.")
