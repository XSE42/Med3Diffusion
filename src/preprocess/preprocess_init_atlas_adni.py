import os
import sys
import json
import argparse

# Add `src` dir to `sys.path`
base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

from utils.data import ADNI_3D_Set


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for ADNI init atlas")
    parser.add_argument("--data_dir", type=str, default=None, required=True, help="Folder containing ADNI dataset")
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Folder to store init atlas")
    parser.add_argument("--meta", type=str, default=None, required=True, help="JSON file containing metadata of ADNI")
    parser.add_argument("--prefix", type=str, default="init_atlas", help="Prefix of init atlas filename")
    args = parser.parse_args()

    return args


def main(args):
    # 1. load metadata of ADNI dataset
    with open(args.meta, "r") as handle:
        metadata = json.load(handle)
    prefix = args.prefix

    # 2. ADNI dataset
    train_dataset = ADNI_3D_Set(
        data_dir=args.data_dir,
        scan_list=metadata["scan_list"],
        scale=True,
        dtype=torch.float32
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
    )

    # 3. init atlas
    init_atlas = torch.zeros((1, 144, 176, 144), dtype=torch.float32)
    scan_num = 0

    # 4. iterate over dataset
    progress_bar = tqdm(range(len(train_dataloader)))
    progress_bar.set_description("Iterating over dataset")
    for i, batch in enumerate(train_dataloader):
        batch_size = batch.shape[0]
        init_atlas += torch.sum(batch, dim=0)
        scan_num += batch_size
        progress_bar.update(1)

    # 5. normalize
    init_atlas = init_atlas / scan_num
    print(f"init_atlas.min: {init_atlas.min()}")
    print(f"init_atlas.max: {init_atlas.max()}")

    # 6. reference image
    ref_scan_info = metadata["scan_list"][0]
    ref_subject_name = ref_scan_info["subject"]  # 011_S_0005
    ref_scan_time = ref_scan_info["scan_time"]  # 2005-09-02
    ref_scan_path = os.path.join(args.data_dir, ref_subject_name, ref_scan_time, "t1.nii.gz")
    ref_scan_img = sitk.ReadImage(ref_scan_path)

    # 7. interpolated atlas
    data_shape = (104, 128, 104)
    interpolate_mode = "trilinear"
    atlas_128 = F.interpolate(init_atlas.unsqueeze(dim=0), size=data_shape, mode=interpolate_mode)
    atlas_128 = atlas_128[0]  # (1, 104, 128, 104)

    # 8. save init atlas
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    torch.save(init_atlas, os.path.join(args.output_dir, f"{prefix}.pt"))
    torch.save(atlas_128, os.path.join(args.output_dir, f"{prefix}_128.pt"))

    init_atlas_np = init_atlas.squeeze().numpy()
    np.save(os.path.join(args.output_dir, f"{prefix}.npy"), init_atlas_np)

    init_atlas_img = sitk.GetImageFromArray(init_atlas_np)
    init_atlas_img.SetOrigin(ref_scan_img.GetOrigin())
    init_atlas_img.SetSpacing(ref_scan_img.GetSpacing())
    init_atlas_img.SetDirection(ref_scan_img.GetDirection())
    sitk.WriteImage(init_atlas_img, os.path.join(args.output_dir, f"{prefix}.nii.gz"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
