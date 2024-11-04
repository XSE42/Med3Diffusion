import os
import json
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for ADNI_128 dataset")
    parser.add_argument("--data_dir", type=str, default=None, required=True, help="Folder containing ADNI dataset")
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Folder to store ADNI_128 dataset")
    parser.add_argument("--meta", type=str, default=None, required=True, help="JSON file containing metadata of ADNI")
    parser.add_argument(
        "--interpolate_mode",
        type=str,
        default="trilinear",
        help='Interpolation mode for resizing. Choose between ["nearest", "trilinear", "area", "nearest-exact"]',
    )
    args = parser.parse_args()

    return args


def main():
    # parse arguments
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    data_shape = (104, 128, 104)
    interpolate_mode = args.interpolate_mode

    ref_spacing = ((144 - 1) / (104 - 1), (176 - 1) / (128 - 1), (144 - 1) / (104 - 1))

    # load metadata of ADNI dataset
    with open(args.meta, "r") as handle:
        metadata = json.load(handle)
    scan_list = metadata["scan_list"]
    scan_num = len(scan_list)
    print(f"Processing {scan_num} scans ...")

    # loop through scan_list
    for idx in tqdm(range(scan_num)):
        scan_info = scan_list[idx]
        subject_name = scan_info["subject"]  # 011_S_0005
        scan_time = scan_info["scan_time"]  # 2005-09-02
        scan_path = os.path.join(data_dir, subject_name, scan_time, "t1.nii.gz")
        out_dir = os.path.join(output_dir, subject_name, scan_time)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # load 3D scan
        scan_img = sitk.ReadImage(scan_path)
        scan = sitk.GetArrayFromImage(scan_img)  # 138, 176, 138
        scan = torch.from_numpy(scan).to(dtype=torch.float32)
        scan = (scan - scan.min()) / (scan.max() - scan.min())  # min-max norm
        scan = F.pad(scan, pad=(3, 3, 0, 0, 3, 3), mode="constant", value=0)  # 144, 176, 144
        scan = scan.unsqueeze(dim=0).unsqueeze(dim=0)  # 1, 1, 144, 176, 144
        scan = F.interpolate(scan, size=data_shape, mode=interpolate_mode)  # 1, 1, 104, 128, 104
        scan = scan.squeeze()  # 104, 128, 104
        scan = scan * 2.0 - 1.0  # from [0, 1] to [-1, 1]
        # save interpolated 3D scan
        torch.save(scan, os.path.join(out_dir, "t1.pt"))
        out_img = sitk.GetImageFromArray(scan.numpy())
        out_img.SetOrigin(scan_img.GetOrigin())
        out_img.SetSpacing(ref_spacing)
        out_img.SetDirection(scan_img.GetDirection())
        sitk.WriteImage(out_img, os.path.join(out_dir, "t1.nii.gz"))


if __name__ == "__main__":
    main()
