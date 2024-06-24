import os
import cv2
import numpy as np
import scipy.io as sio

image_folder1 = 'generated_dataset_test/'
output_folder1 = 'Generated_Sahil'
if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)

for d in os.listdir(image_folder1):
    image_folder = os.path.join(image_folder1,'Frames')
    winLength = 150  
    output_folder = os.path.join(output_folder1,d)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))], key=lambda x: int(''.join(filter(str.isdigit, x))))

    for i in range(0, len(image_files), winLength):
        segment_files = image_files[i:i + winLength]
        if len(segment_files) < winLength:
            break  

        layer0_list = []
        layer1_list = []
        layer2_list = []

        for filename in segment_files:
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)

            layer0 = cv2.resize(img, (192, 128))
            layer1 = cv2.resize(img, (96, 64))
            layer2 = cv2.resize(img, (48, 32))

            layer0_list.append(layer0)
            layer1_list.append(layer1)
            layer2_list.append(layer2)

        layer0_stack = np.stack(layer0_list, axis=0)
        layer1_stack = np.stack(layer1_list, axis=0)
        layer2_stack = np.stack(layer2_list, axis=0)

        segment_number = i // winLength
        mat_filename = f'{segment_number:04d}.mat'
        mat_path = os.path.join(output_folder, mat_filename)
        sio.savemat(mat_path, {'layer0': layer0_stack, 'layer1': layer1_stack, 'layer2': layer2_stack})

        print(f"Saved {mat_path}")

    print("Conversion complete.")
