import torch.utils.data as data
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import nibabel as nib
import numpy
from typing import Optional
from torch.utils.data import DataLoader
# import decord
# decord.bridge.set_bridge('torch')
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
import os
from functools import partial
import SimpleITK as sitk
import pandas as pd
import torch.nn.functional as F

def cycle(dl):
    while True:
        for data in dl:
            yield data

class MixedDataset(data.Dataset):
    def __init__(self, video_dataset, image_dataset, nii_dataset):
        self.video_dataset = video_dataset
        self.image_dataset = image_dataset
        self.nii_dataset = nii_dataset
        self.datasets = [self.video_dataset, self.image_dataset, self.nii_dataset]
        
    def __len__(self):
        return min(len(self.video_dataset), len(self.image_dataset), len(self.nii_dataset))
    
    def __getitem__(self, idx):
        dataset_choice = random.choice(self.datasets)
        data = dataset_choice[idx]
        return data


CHANNEL_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}

def exists(val):
    return val is not None
def convert_image_to_fn(img_type, image):
    if not exists(img_type) or image.mode == img_type:
        return image

    return image.convert(img_type)

def read(path):
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    return array

class ImageDataset(Dataset):
    def __init__(self, folder_path, channels = 3, resize_size=(256, 256),convert_image_to=None, data_type: str='image',):
        self.folder_path = folder_path
        self.file_paths = []
        if exists(channels) and not exists(convert_image_to):
            convert_image_to = CHANNEL_TO_MODE.get(channels)
        # Walk through the folder path and append all image files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.file_paths.append(os.path.join(root, file))
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.Lambda(partial(convert_image_to_fn, convert_image_to)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.type = data_type

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx])
        image = self.transform(image)
        example = {
            "pixel_values": image,
            "type": self.type
        }
        return example

def filter_readable_nii(paths):
    readable_paths = []
    for path in paths:
        try:
            _ = sitk.ReadImage(path)
            readable_paths.append(path)
        except Exception as e:
            print(f"Unreadable image skipped: {path} - Error: {e}")
    return readable_paths

LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600
    
class NiiDataset(Dataset):
    def __init__(self, folder_path, data_type: str='nii', frame: int =69, resolution: int=72):
        self.folder_path = folder_path
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz') ]
        print('len(self.file_paths)', len(self.file_paths))
        self.type = data_type
        self.frame = frame
        self.resolution = resolution
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # try:
        nii_image = read(file_path)
        data = np.expand_dims(nii_image, axis=0)
        data = np.repeat(data,3, axis=0)
        data = torch.tensor(data)
        
        example = {
            "pixel_values": data,
            "type": self.type,
            'filename': file_path.split('/')[-1],
        }
        return example
    
class VideoDataset(Dataset):
    def __init__(
            self,
            folder_path: str,
            prompt: Optional[str] = None,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            data_type: str='video',
    ):
        self.folder_path = folder_path
        self.prompt = prompt
        self.prompt_ids = None  # You might need to initialize or compute prompt_ids based on the prompt

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.type = data_type
        # List all video files in the folder
        self.video_files = []
        # self.video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv'))]
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv')):
                    self.video_files.append(os.path.join(root, file))
    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        video_path = self.video_files[index]
        vr = decord.VideoReader(video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> c f h w")
        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "type": self.type
        }
        return example
        
        
def get_vim_data(args):
    video_dataset = VideoDataset(args.video_datapath, width=args.video_width, height=args.video_height, n_sample_frames=args.video_frames) if args.video_datapath else None
    image_dataset = ImageDataset(args.image_datapath, resize_size=args.image_resize) if args.image_datapath else None
    nii_dataset = NiiDataset(args.nii_datapath) if args.nii_datapath else None
    whole_loader = []
    lengths = []

    # For video
    if video_dataset:
        video_loader = DataLoader(video_dataset, batch_size=args.batch_video, shuffle=True, num_workers=args.num_worker)
        length_1 = len(video_loader)
        video_loader = cycle(video_loader)
        print('video load successfully!')
        whole_loader.append(video_loader)
        lengths.append(length_1)
    else:
        print('video(None) load successfully!')

    # For images
    if image_dataset:
        image_loader = DataLoader(image_dataset, batch_size=args.batch_image, shuffle=True, num_workers=args.num_worker)
        length_2 = len(image_loader)
        image_loader = cycle(image_loader)
        print('image load successfully!')
        whole_loader.append(image_loader)
        lengths.append(length_2)
    else:
        print('image(None) load successfully!')

    # For nii
    if nii_dataset:
        nii_loader = DataLoader(nii_dataset, batch_size=args.batch_nii, shuffle=True, num_workers=args.num_worker)
        length_3 = len(nii_loader)
        nii_loader = cycle(nii_loader)
        print('nii load successfully!')
        whole_loader.append(nii_loader)
        lengths.append(length_3)
    else:
        print('nii(None) load successfully!')

    # Sorting loaders by their corresponding lengths in descending order
    loaders_and_lengths = list(zip(whole_loader, lengths))
    loaders_and_lengths.sort(key=lambda x: x[1], reverse=True)
    
    # Unzipping
    sorted_loaders, sorted_lengths = zip(*loaders_and_lengths)
    
    return list(sorted_loaders), list(sorted_lengths)