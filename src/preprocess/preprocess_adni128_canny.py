import os
import json
import argparse

import torch
from tqdm import tqdm
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for ADNI_128 dataset")
    parser.add_argument("--data_dir", type=str, default=None, required=True, help="Folder containing ADNI_128 dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to store edge detection results")
    parser.add_argument("--meta", type=str, default=None, required=True, help="JSON file containing metadata of ADNI")
    args = parser.parse_args()

    return args


def main():
    # parse arguments
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = data_dir

    # load metadata of ADNI dataset
    with open(args.meta, "r") as handle:
        metadata = json.load(handle)
    scan_list = metadata["scan_list"]
    scan_num = len(scan_list)
    print(f"Processing {scan_num} scans ...")

    # Canny Filter
    # gaussian_op = sitk.DiscreteGaussianImageFilter()
    # gaussian_op.SetVariance(0)
    # gaussian_op.SetMaximumError(0.01)
    edge_op = sitk.CannyEdgeDetectionImageFilter()
    edge_op.SetLowerThreshold(0.05)
    edge_op.SetUpperThreshold(0.1)
    edge_op.SetVariance(0)
    edge_op.SetMaximumError(0.01)

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
        scan_img = sitk.ReadImage(scan_path)  # 104, 128, 104

        scan_spacing = scan_img.GetSpacing()
        if scan_spacing[0] < 1.3 or scan_spacing[1] < 1.3 or scan_spacing[2] < 1.3:
            print(f"Wrong Spacing: {scan_spacing}")
            print(scan_path)
            break
        if scan_img.GetPixelID() != 8:  # should be "32-bit float"
            print(f"Wrong Data Type: {sitk.GetPixelIDValueAsString(scan_img.GetPixelID())}")
            print(scan_path)
            break

        # scan_gaussian = gaussian_op.Execute(scan_img)  # 104, 128, 104
        # scan_gaussian.SetOrigin(scan_img.GetOrigin())
        # scan_gaussian.SetSpacing(scan_img.GetSpacing())
        # scan_gaussian.SetDirection(scan_img.GetDirection())

        scan_edge = edge_op.Execute(scan_img)  # 104, 128, 104
        scan_edge.SetOrigin(scan_img.GetOrigin())
        scan_edge.SetSpacing(scan_img.GetSpacing())
        scan_edge.SetDirection(scan_img.GetDirection())

        scan_edge_pt = torch.from_numpy(sitk.GetArrayFromImage(scan_edge)).to(torch.uint8)  # 104, 128, 104

        # save edge detection results
        # sitk.WriteImage(scan_gaussian, os.path.join(out_dir, "t1_gaussian.nii.gz"))
        sitk.WriteImage(scan_edge, os.path.join(out_dir, "t1_canny.nii.gz"))
        torch.save(scan_edge_pt, os.path.join(out_dir, "t1_canny.pt"))


if __name__ == "__main__":
    main()

# python src/preprocess/preprocess_adni128_canny.py --data_dir data/ADNI_128 --meta data/ADNI_meta/ADNI_CN_meta.json
