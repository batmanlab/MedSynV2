import sys
sys.path.append(".")
import torch
import numpy as np
import argparse
from opensora.models.ae.videobase import CausalVAEModel
import os
import argparse
# Set device to CUDA if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import math
from monai.inferers.inferer import SlidingWindowInferer
inferer = SlidingWindowInferer(
        roi_size=[16, 32, 32],
        sw_batch_size=1,
        mode="gaussian",
        overlap=0.1,
        sw_device=device,
        device=torch.device("cpu"),
    )

@torch.no_grad()
def dynamic_infer(inferer, model, images):
    """
    Perform dynamic inference using a model and an inferer, typically a monai SlidingWindowInferer.

    This function determines whether to use the model directly or to use the provided inferer
    (such as a sliding window inferer) based on the size of the input images.

    Args:
        inferer: An inference object, typically a monai SlidingWindowInferer, which handles patch-based inference.
        model (torch.nn.Module): The model used for inference.
        images (torch.Tensor): The input images for inference, shape [N,C,H,W,D] or [N,C,H,W].

    Returns:
        torch.Tensor: The output from the model or the inferer, depending on the input size.
    """
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model.decode(images)
    else:
        # Extract the spatial dimensions from the images tensor (H, W, D)
        spatial_dims = images.shape[2:]
        orig_roi = inferer.roi_size

        # Check that roi has the same number of dimensions as spatial_dims
        if len(orig_roi) != len(spatial_dims):
            raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")

        # Iterate and adjust each ROI dimension
        adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
        inferer.roi_size = adjusted_roi
        output = inferer(network=model, inputs=images)
        inferer.roi_size = orig_roi
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str, default=None, required=True)
    parser.add_argument('--folder', type=str, default=None, required=True)
    parser.add_argument('--save_folder', type=str, default=None, required=True)

    args = parser.parse_args()

    model_path = args.pretrained_model_path
    model = CausalVAEModel.load_from_checkpoint(model_path)
    model = model.to(device)
    model.half()
    model.eval()
    image_means = torch.tensor([0.8824114, -0.051992204, 1.1762542, -7.0781837], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1).to(device)
    image_stds  = torch.tensor([3.4852407, 5.1318316, 4.399875, 3.8425775], dtype=torch.float16, requires_grad=False).view(1, 4, 1, 1, 1).to(device)

    folder = args.folder 
    save_folder = args.save_folder
    for item in sorted(os.listdir(folder)):
        if not item.endswith('.npy'):
            continue
        if 'nii' in item or 'SR' in item :
            continue

        to_save = os.path.join(save_folder, item.split('/')[-1][:-4]+'_SR.npy')
        if os.path.exists(to_save):
            continue
        image = np.load(os.path.join(folder, item))
        image_ = torch.from_numpy(image).to(device) 
        image_ = (image_*image_stds) + image_means
        synthetic_images = dynamic_infer(inferer, model, image_.to(device).half())
        data = synthetic_images.squeeze().cpu().detach().numpy()
        np.save(to_save, data)