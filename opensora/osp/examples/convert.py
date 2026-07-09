import argparse
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ------------------------
# Worker initializer
# ------------------------
_reference = None

def init_worker(reference_path):
    global _reference
    _reference = sitk.ReadImage(reference_path)

# ------------------------
# Single-file processing
# ------------------------
def process_one(args):
    (
        fname,
        npy_dir,
        out_dir,
        processed_min,
        processed_max,
    ) = args

    if not fname.endswith("_SR.npy"):
        return None

    npy_path = os.path.join(npy_dir, fname)
    out_path = os.path.join(
        out_dir, fname.replace(".npy_", "_").replace(".npy", ".nii.gz")
    )

    if os.path.exists(out_path):
        return None
    try:

        data = np.load(npy_path).astype(np.float32)
    except:
        return None

    if "processed" not in fname:
        # ---------- CT-like data ----------
        data = np.nan_to_num(
            data,
            nan=-1024.0,
            posinf=processed_max,
            neginf=processed_min,
        )

        # Clip to [-1, 1]
        data = np.clip(data, -1.0, 1.0)

        # Map [-1, 1] → [-1024, 600]
        data = ((data + 1.0) / 2.0) * (
            processed_max - processed_min
        ) + processed_min

        data = data.astype(np.int16)

    else:
        # ---------- Label data ----------
        return None
        # data = (data > 0).astype(np.uint8)

    # Convert to SimpleITK image
    img = sitk.GetImageFromArray(data[0])
    img.SetOrigin(_reference.GetOrigin())
    img.SetDirection(_reference.GetDirection())

    sitk.WriteImage(img, out_path)
    return out_path

# ------------------------
# Main entry
# ------------------------
def npy_to_nifti_mp(
    npy_dir,
    out_dir,
    reference_path,
    processed_min=-1024,
    processed_max=600,
    num_workers=None,
):
    os.makedirs(out_dir, exist_ok=True)

    fnames = sorted(os.listdir(npy_dir))

    tasks = [
        (
            fname,
            npy_dir,
            out_dir,
            processed_min,
            processed_max,
        )
        for fname in fnames
    ]

    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # safe default on SCC

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(reference_path,),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one, tasks),
            total=len(tasks),
        ):
            if result is not None:
                pass  # already saved

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=None, required=True)
    parser.add_argument('--save_folder', type=str, default=None, required=True)
    args = parser.parse_args()

    npy_to_nifti_mp(
        npy_dir=args.folder,
        out_dir=args.save_folder,
        reference_path="./fixed/fixed_resampled.nii.gz",
        num_workers=4,  # adjust for SCC node
    )
