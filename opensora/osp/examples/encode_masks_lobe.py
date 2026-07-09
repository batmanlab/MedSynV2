import sys
sys.path.append(".")
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from opensora.models.ae.videobase import CausalVAEModel
import os
from tqdm import tqdm
import argparse
# Set device to CUDA if available
import SimpleITK as sitk
import pandas as pd
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
import pandas as pd


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def read(path):
    img = sitk.ReadImage(path)
    return img

class NiiDataset(Dataset):
    def __init__(self, folder_path, data_type: str='nii', frame: int =69, resolution: int=72):
        self.folder_path = folder_path
        self.file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz') ])
        print('len(self.file_paths)', len(self.file_paths))
        self.type = data_type
        self.frame = frame
        self.resolution = resolution
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            nii_image = read(file_path)
            nii_image = sitk.Cast(nii_image, sitk.sitkInt16)
            edges = sitk.LabelContour(nii_image)
            edges = sitk.Cast(edges > 0, sitk.sitkUInt8)
            edges = sitk.BinaryDilate(edges, [3,]*3)
            data = sitk.GetArrayFromImage(edges)
            data[data>0] = 1
            data[data<=0] = 0

            data = np.expand_dims(data, axis=0)
            data = np.repeat(data,3, axis=0)
            data = torch.tensor(data)
        except:
            data = torch.tensor([0])
        
        example = {
            "pixel_values": data,
            "type": self.type,
            'filename': file_path.split('/')[-1],
        }
        return example
    


if __name__ == '__main__':
    LOW_THRESHOLD = -1024
    HIGH_THRESHOLD = 600
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_index', type=int, default=0, required=True)
    parser.add_argument('--pretrained_model_path', type=str, default=None, required=True)
    parser.add_argument('--folder', type=str, default=None, required=True)
    parser.add_argument('--save_folder', type=str, default=None, required=True)
    args = parser.parse_args()

    batch_index = args.batch_index
    model_path = args.pretrained_model_path
    model = CausalVAEModel.load_from_checkpoint(model_path)
    model = model.to(device)
    model.half()
    model.eval()
    
    nii_datapath = args.folder
    nii_dataset = NiiDataset(nii_datapath)
    nii_loader = DataLoader(nii_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    latent_folder = os.path.join(args.save_folder, f'CT_Rate_valid_vessel_latents_112_{batch_index}')
    os.makedirs(latent_folder, exist_ok=True)
    for i, data in tqdm(enumerate(nii_loader), total=len(nii_dataset), desc="Processing NII Files"):
        filename = data['filename'][0].replace('.nii.gz', '')  # Use filename directly, assume it is already a string
        save_path = os.path.join(latent_folder, f'{filename}.npy')
        if os.path.exists(save_path):
            continue
        nii_image = data['pixel_values']
        if nii_image.sum() == 0:
            continue
        nii_image = nii_image - 1/2
        data = torch.nn.functional.interpolate(nii_image, size=[448, 448, 448], mode='nearest')
        
        input_1 = data[:, :, 80*batch_index:80*(batch_index+1), :, :].to(device, dtype=torch.float16) 
        with torch.no_grad():
            latent = model.encode(input_1).sample()
        
            latent = latent.cpu().numpy()
            np.save(save_path, latent)
        print(f'Saved latent for {filename}')
    
    print("All latents have been processed and saved.")