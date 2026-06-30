"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import torch
import monai
from torch.utils.data import DataLoader
import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    # AddChanneld,
    SpatialPadd,
    ToTensord,
    RandRotated,
    RandZoomd,
    RandSpatialCropd,
    ConcatItemsd,
    MapLabelValued,
    MapTransform
)
from monai.data.image_reader import ImageReader
import SimpleITK as sitk
import numpy as np
from typing import Sequence, Union, Optional, Any

class LabelMinMaxToMinusOneOne(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue

            img = d[key]

            # work in numpy
            if hasattr(img, "numpy"):  # torch tensor
                img_np = img.numpy()
            else:
                img_np = img

            vmin = img_np.min()
            vmax = img_np.max()

            # Case 1: already 0–1 → do nothing
            if vmin >= 0 and vmax <= 1:
                d[key] = img
                continue

            # Case 2: 0–5 style multi-label → min-max to [-1, 1]
            if vmin >= 0 and vmax > 1:
                if vmax > vmin:
                    img_np = 2.0 * (img_np - vmin) / (vmax - vmin) - 1.0
                else:
                    img_np = np.zeros_like(img_np)

                d[key] = img_np.astype(np.float32)

        return d

class SimpleITKReaderNoMeta(ImageReader):
    def __init__(self):
        super().__init__()
        self.img_obj: Optional[sitk.Image] = None

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        filenames = [filename] if isinstance(filename, str) else filename
        return all(str(f).endswith((".nii", ".nii.gz")) for f in filenames)

    def read(self, data: Union[Sequence[str], str], **kwargs) -> Any:
        filenames = [data] if isinstance(data, str) else data
        if len(filenames) > 1:
            raise ValueError("Only single-file loading is supported in this SimpleITKReaderNoMeta.")
        self.img_obj = sitk.ReadImage(filenames[0])
        return self.img_obj

    def get_data(self, img: Any) -> tuple[np.ndarray, dict]:
        array = sitk.GetArrayFromImage(img).astype(float)
        return array, {}  # empty metadata


# src_labels = [-1.,  -0.6, -0.2,  0.2,  0.6,  1. ]
# dst_labels = [-1.,  0.6, -0.6,  -0.2,  0.2,  1. ]

# src_labels = [0, 1, 2, 3, 4, 5]
# dst_labels = [0, 4, 1, 2, 3, 5]


# def get_transforms_normal(shape):
#     train_target_transforms = Compose(
#         [
#             LoadImaged(keys=["image", "text", 'img_sr'], image_only=False, allow_missing_keys=True),
#             ToTensord(keys=["image", "text", 'img_sr'], allow_missing_keys=True),
#         ]
#     )

#     return train_target_transforms

    
# def get_transforms_normal(shape):
#     train_target_transforms = Compose(
#         [
#             LoadImaged(keys=["text", ], image_only=False, allow_missing_keys=True),
#             LoadImaged(keys=["image" ], image_only=False, allow_missing_keys=True, reader=SimpleITKReaderNoMeta()),
#             ToTensord(keys=["image", "text", ], allow_missing_keys=True),
#         ]
#     )

#     return train_target_transforms
def get_transforms_normal(shape):
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["text"], image_only=False, allow_missing_keys=True),
            LoadImaged(
                keys=["image"],
                image_only=False,
                allow_missing_keys=True,
                reader=SimpleITKReaderNoMeta(),
            ),

            # 👇 add here
            LabelMinMaxToMinusOneOne(keys=["image"]),

            ToTensord(keys=["image", "text"], allow_missing_keys=True),
        ]
    )

    return train_target_transforms


def get_transforms_normal_seg(shape):
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["image", "lobe", "airway", "vessel", "text"]),
            ToTensord(keys=["image", "lobe", "airway", "vessel", "text"]),
        ]
    )

    return train_target_transforms


def get_transforms_text():
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["text"], reader='NumpyReader', image_only=False),
            ToTensord(keys=["text"]),
        ]
    )

    return train_target_transforms


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))


def get_transforms_dit_112_patho(shape, crop_shape):
    train_target_transforms = Compose(
        [
            LoadImaged(keys=["img_feat", 
                             "lobe_feat", "airway_feat", "vessel_feat", "heart_feat",
                             "pleffu_feat", "cons_feat", "ggo_feat", 'perieffu_feat', 'nodule_feat',
                             "text_real", 
                             ], dtype=np.float32, allow_missing_keys=True, image_only=False),
            ToTensord(keys=["img_feat", 
                            "lobe_feat", "airway_feat", "vessel_feat", "heart_feat",
                             "pleffu_feat", "cons_feat", "ggo_feat", 'perieffu_feat', 'nodule_feat',
                             "text_real", 
                             ], allow_missing_keys=True)
        ]
    )

    return train_target_transforms

def cache_transformed_train_data_dit_112_patho(train_files, shape, crop_shape):
    train_transforms = get_transforms_dit_112_patho(shape, crop_shape)
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=0.0
    )

    return train_ds
    