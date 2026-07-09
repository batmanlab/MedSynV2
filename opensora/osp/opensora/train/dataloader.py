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
import decord
decord.bridge.set_bridge('torch')
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
import os
from functools import partial
from empatches import EMPatches
import SimpleITK as sitk
import pandas as pd
import torch.nn.functional as F

def read(path):
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    return array


def extract_patches_5d(source, patchsize=(144,120,120), overlap=0.25, stride=None, emp=None, vox=True, return_indices=False):
    a= []
    assert len(source.shape) == 5
    for i in range(source.shape[1]):
        patches, indices = emp.extract_patches(source[0,i,:,:,:],patchsize=patchsize, overlap=overlap, stride=stride, vox=vox)
        a.append(torch.stack(patches).unsqueeze(1).unsqueeze(1))
    a = torch.cat(a, dim=2)
    #(1, channel, frame, h, w)
    if return_indices:
        return a, indices
    else:
        return a

def cycle(dl):
    while True:
        for data in dl:
            yield data

def random_crop(data, crop_size):
    assert data.shape[-3] >= crop_size[-3], f"Crop size is larger than data size in dimension 0. data.shape[-3] and crop_size[-3]: {data.shape[-3]}, {crop_size[-3]}"
    assert data.shape[-2] >= crop_size[-2], f"Crop size is larger than data size in dimension 1. data.shape[-2] and crop_size[-2]: {data.shape[-2]}, {crop_size[-2]}"
    assert data.shape[-1] >= crop_size[-1], f"Crop size is larger than data size in dimension 2. data.shape[-1] and crop_size[-1]: {data.shape[-1]}, {crop_size[-1]}"

    max_x = data.shape[-3] - crop_size[-3]
    max_y = data.shape[-2] - crop_size[-2]
    max_z = data.shape[-1] - crop_size[-1]

    start_x = np.random.randint(0, max_x + 1)
    start_y = np.random.randint(0, max_y + 1)
    start_z = np.random.randint(0, max_z + 1)

    cropped_data = data[..., start_x:start_x + crop_size[-3],
                        start_y:start_y + crop_size[-2],
                        start_z:start_z + crop_size[-1]]
    return cropped_data, [start_x, start_y, start_z]


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
        image = image.unsqueeze(1)
        example = {
            "video": image,
            "type": self.type
        }
        return example
    

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
        # nii_image = read(self.file_paths[idx])
        # print(nii_image.shape) # (417, 451, 451)
        # exit()
        # if nii_image.ndim == 5:
            # data = np.squeeze(nii_image, axis=1)
            #(1, 256 256 256)
            
        # LOW_THRESHOLD = -1024
        # HIGH_THRESHOLD = 600
        # nii_image[nii_image>HIGH_THRESHOLD] = HIGH_THRESHOLD
        # nii_image[nii_image<LOW_THRESHOLD] = LOW_THRESHOLD
        # nii_image = (nii_image - LOW_THRESHOLD) / (HIGH_THRESHOLD-LOW_THRESHOLD) # [-1024, 600] -> [0,1]
        # nii_image = 2*nii_image-1
 
        # # print('nii_image',  nii_image.shape) # nii_image (417, 451, 451) 

        # # nii_image = np.swapaxes(nii_image, -3, -1)

        # data = random_crop(nii_image, (self.frame, self.resolution, self.resolution))[0]
        # data = np.expand_dims(data, axis=0)
        # data = np.repeat(data,3, axis=0)
        # data = torch.tensor(data)
        
        example = {
            "pixel_values": self.file_paths[idx],
            "type": self.type
        }
        return example


# class NiiDataset(Dataset):
#     def __init__(self, folder_path, data_type: str='nii', frame: int =21, resolution: int=48, ):
#         # self.folder_path = folder_path
#         # self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz')  ]
#         # self.file_paths = filter_readable_nii(self.file_paths)
#         self.type = data_type
#         volumes = pd.read_csv('/ocean/projects/asc170022p/shared/CT_Rate_train_npy/selected_kernels.csv')['VolumeName'].to_list()
#         volumes = [item.replace('.nii.gz', '') for item in volumes]
        
#         folder1 = "/ocean/projects/asc170022p/wdai1/medsyn/combined_img_112_56"  
#         folder2 = "/ocean/projects/asc170022p/wdai1/medsyn/moved_img_448"  
#         # folder3 = "/ocean/projects/asc170022p/wdai1/medsyn/moved_lobe_448"  

#         # folder_vessel = '/ocean/projects/asc170022p/wdai1/medsyn/moved_vessel_448'
#         # folder_vessel1 = '/ocean/projects/asc170022p/wdai1/medsyn/moved_vessel_448_append'
#         # folder_airway = '/ocean/projects/asc170022p/wdai1/medsyn/moved_airway_448'
        
#         folder_latent_0 = "/ocean/projects/asc170022p/wdai1/medsyn/CT_Rate_train_image_latents_112_0"
#         folder_latent_1 = "/ocean/projects/asc170022p/wdai1/medsyn/CT_Rate_train_image_latents_112_1"
#         folder_latent_2 = "/ocean/projects/asc170022p/wdai1/medsyn/CT_Rate_train_image_latents_112_2"
#         folder_latent_3 = "/ocean/projects/asc170022p/wdai1/medsyn/CT_Rate_train_image_latents_112_3"
#         folder_latent_4 = "/ocean/projects/asc170022p/wdai1/medsyn/CT_Rate_train_image_latents_112_4"
#         folder_latent_5 = "/ocean/projects/asc170022p/wdai1/medsyn/CT_Rate_train_image_latents_112_5"
#         train_files = []
#         for img_dir in volumes:
#             img_name = img_dir+'_processed_Reg.nii.gz'
#             # lobe_name = img_dir+'_processed_lobe_Reg.nii.gz'
#             # airway_name = img_dir+'_processed_airway_Reg.nii.gz'
#             latent_name = img_dir+'_processed_Reg.npy'
#             if os.path.exists(os.path.join(folder1, img_dir+'.npy')) and os.path.exists(os.path.join(folder2, img_name)) and os.path.exists(os.path.join(folder_latent_0, latent_name))and os.path.exists(os.path.join(folder_latent_1, latent_name))and os.path.exists(os.path.join(folder_latent_2, latent_name))and os.path.exists(os.path.join(folder_latent_3, latent_name))and os.path.exists(os.path.join(folder_latent_4, latent_name))and os.path.exists(os.path.join(folder_latent_5, latent_name)):
#                 train_files.append(
#                                 {"image_sr": os.path.join(folder2, img_name),
#                                     # "lobe_sr": os.path.join(folder3, lobe_name),
#                                     # "airway_sr": os.path.join(folder_airway, airway_name),
#                                     # "vessel_sr": vessel_path,
#                                     "lr_combine": os.path.join(folder1, img_dir+'.npy'),
#                                 "img_feature_0": os.path.join(
#                                     folder_latent_0,
#                                     latent_name),
#                                 "img_feature_1": os.path.join(
#                                     folder_latent_1,
#                                     latent_name),
#                                 "img_feature_2": os.path.join(
#                                     folder_latent_2,
#                                     latent_name),
#                                 "img_feature_3": os.path.join(
#                                     folder_latent_3,
#                                     latent_name),
#                                 "img_feature_4": os.path.join(
#                                     folder_latent_4,
#                                     latent_name),
#                                 "img_feature_5": os.path.join(
#                                     folder_latent_5,
#                                     latent_name),
#                                     })
#         self.file_paths = train_files
#         self.frame = frame
#         self.resolution = resolution

        
#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         # nii_image = np.load(self.file_paths[idx])
#         # if nii_image.ndim == 5:
#         #     data = np.squeeze(nii_image, axis=1)
#         #     data = np.repeat(data,3, axis=0)
#         # data = torch.tensor(data)
#         # example = {
#         #     "pixel_values": data,
#         #     "type": self.type
#         # }
#         # return example
#         data = self.file_paths[idx]
        
#         img  = read(data["image_sr"])
#         # data = np.repeat(data,3, axis=0)
#         # lobe, airway, vessel = read(data['lobe_sr']), read(data['airway_sr']), read(data['vessel_sr'])
#         img = torch.from_numpy(img) #, torch.from_numpy(lobe), torch.from_numpy(airway), torch.from_numpy(vessel)
        
#         LOW_THRESHOLD = -1024
#         HIGH_THRESHOLD = 600
#         img[img>HIGH_THRESHOLD] = HIGH_THRESHOLD
#         img[img<LOW_THRESHOLD] = LOW_THRESHOLD
#         img = (img - LOW_THRESHOLD) / (HIGH_THRESHOLD-LOW_THRESHOLD) # [-1024, 600] -> [0,1]
#         img = 2*img-1 # [0,1] -> [-1,1]
        
        
#         img_fea, lobe_fea, airway_fea, vessel_fea, final_fea, extra_fea = np.load(data['img_feature_0']), np.load(data['img_feature_1']), np.load(data['img_feature_2']), np.load(data['img_feature_3']), np.load(data['img_feature_4']), np.load(data['img_feature_5'])
#         img_fea, lobe_fea, airway_fea, vessel_fea, final_fea, extra_fea = torch.from_numpy(img_fea), torch.from_numpy(lobe_fea), torch.from_numpy(airway_fea), torch.from_numpy(vessel_fea), torch.from_numpy(final_fea), torch.from_numpy(extra_fea)
#         # lobe = lobe.unsqueeze(dim=1)#.to(self.accelerator.device)
#         # airway = airway.unsqueeze(dim=1) # .to(self.accelerator.device)
#         # vessel = vessel.unsqueeze(dim=1)#.to(self.accelerator.device)
#         # lobe = F.interpolate(lobe, size=[448, 448, 448], mode='nearest')
#         # airway = F.interpolate(airway, size=[448, 448, 448], mode='nearest')
#         # vessel = F.interpolate(vessel, size=[448, 448, 448], mode='nearest')
        
#         img = img.unsqueeze(dim=0).unsqueeze(dim=0)
#         # img = F.interpolate(img, size=[448, 448, 448], mode='trilinear', align_corners=True) [:, :, 20: 420]
#         img = F.interpolate(img, size=[448, 448, 448], mode='trilinear', align_corners=True).squeeze(0)
#         # img = torch.cat([img, lobe, airway, vessel], dim=1)
        
#         feature = torch.cat([img_fea, lobe_fea, airway_fea, vessel_fea, final_fea, extra_fea], dim=-3).squeeze(0)
#         lowres = np.load(data['lr_combine'])
#         lowres = torch.from_numpy(lowres)

#         lowres, shape =  random_crop(lowres, [self.frame, self.resolution, self.resolution])
#         feature = feature[..., shape[0]:shape[0]+self.frame, shape[1]:shape[1]+self.resolution, shape[2]:shape[2]+self.resolution]

#         img = img[..., 4*shape[0]:4*(shape[0]+self.frame-1)+1, 8*shape[1]:8*(shape[1]+self.resolution), 8*shape[2]:8*(shape[2]+self.resolution)]
        
#         # nii_image = np.load(file_path)
#         # nii_image = sitk.ReadImage(file_path)
#         # nii_image = sitk.GetArrayFromImage(nii_image)
#         # if nii_image.ndim == 3:
#         #     data = np.expand_dims(nii_image, axis=0)
#         #     data = np.repeat(data, 3, axis=0)  # Replicate across channel dimension if needed
#         # data = torch.tensor(data)
#         # data[data>HIGH_THRESHOLD] = HIGH_THRESHOLD
#         # data[data<LOW_THRESHOLD] = LOW_THRESHOLD
#         # data = (data - LOW_THRESHOLD) / (HIGH_THRESHOLD-LOW_THRESHOLD) # [-1024, 600] -> [0,1]
#         # data = 2*data-1 # [0,1] -> [-1,1]
#         # data = torch.nn.functional.interpolate(data.unsqueeze(0), size=[448, 448, 448], mode='trilinear').squeeze(0)
#         # assert data.shape[0] == 3
#         # assert data.shape[1] == 448
#         # filename = os.path.basename(file_path)  # Extract filename from the full file path
        
#         example = {
#             'lowres': lowres,
#             'feature': feature,
#             "video": img,
#             "type": self.type,
#             # "filename": filename  # Include filename in the output dictionary
#         }
#         return example
    
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