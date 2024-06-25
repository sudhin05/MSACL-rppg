import torch
import matplotlib.pyplot as plt
from model.JAMSNet import JAMSNet
from Dataset.LoadDataset import LoadDataset
from torch.utils.data import DataLoader
import numpy as np

# Function to load the model
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model'].state_dict())
    return model

# Function to normalize data using min-max normalization
def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Function to plot the output
def plot_output(model, dataset_loader):
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(dataset_loader):
            stRep_0 = batch_data[0]
            stRep_1 = batch_data[1]
            stRep_2 = batch_data[2]
            gtPPG = batch_data[3]
            gtPPG = gtPPG.squeeze(0)
            # 1 128 192 3 150
            stRep_0 = stRep_0.squeeze(0).permute(0, 3, 2, 1)   # 150 3 192 128
            stRep_1 = stRep_1.squeeze(0).permute(0, 3, 2, 1)   # 150 3 96 64
            stRep_2 = stRep_2.squeeze(0).permute(0, 3, 2, 1)   # 150 3 48 32
            stRep_0 = stRep_0.to(device)
            stRep_1 = stRep_1.to(device)
            stRep_2 = stRep_2.to(device)
            gtPPG = gtPPG.to(device)

            stRep_pred_L = model(stRep_0, stRep_1, stRep_2)
            
            # Move predictions and ground truth to CPU for plotting
            stRep_pred_L = stRep_pred_L.cpu().numpy()
            gtPPG = gtPPG.cpu().numpy()
            print(gtPPG.shape)
            
            # Normalize the data
            stRep_pred_L = min_max_normalize(stRep_pred_L)
            gtPPG = min_max_normalize(gtPPG)

            np.save("ouput.npy",stRep_pred_L)
            np.save("groundtruth.npy",gtPPG)
            
            # Plot the ground truth and predictions
            plt.figure(figsize=(10, 5))
            plt.plot(gtPPG, label='Ground Truth')
            plt.plot(stRep_pred_L, label='Predicted')
            plt.xlabel('Time')
            plt.ylabel('Normalized PPG')
            plt.title('Ground Truth vs Predicted PPG')
            plt.legend()
            plt.show()
            
            break  # Just plot one batch

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'saves_2/JAMSNet_UBFC_pyramid_data_150_epoch560_loss0.000499.pth'  # Change to your model path
    model = load_model(model_path)
    model = model.to(device)

    data_folder = 'Saves150/json_1'  # Address of the test set
    test_dataset = LoadDataset(data_folder, split='test')
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4)
    
    plot_output(model, test_loader)
