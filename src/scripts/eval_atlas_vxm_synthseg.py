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
import monai.metrics as metrics
from tqdm.auto import tqdm

from voxelmorph.networks import VxmDense, SpatialTransformer
from utils.eval import get_onehot_mask


SYNTHSEG_LABELS = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60
]
NONE_SYNTHSEG_LABELS = [i for i in range(61) if i not in SYNTHSEG_LABELS]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to pretrained VxmDense model")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing ADNI dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save evaluation results")
    parser.add_argument("--meta", type=str, default=None, help="JSON file containing metadata of ADNI dataset")
    parser.add_argument("--postfix", type=str, default="vxm_adni", help="Postfix of the output file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    return args


@torch.no_grad()
def main(args):
    # 1. load metadata
    with open(args.meta, "r") as handle:
        meta = json.load(handle)
    scan_list = meta["scan_list"]

    # 2. variables
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    output_dir = args.output_dir
    postfix = args.postfix
    DATA_SHAPE = (112, 128, 112)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. load pretrained model
    vxm: VxmDense = VxmDense.from_pretrained(args.pretrained_model, subfolder="vxm").to(device)
    spatial_transformer = SpatialTransformer(vxm.inshape).to(device)

    # 4. results
    dice_list_128 = []
    dice_list_resample = []

    # 5. loop
    progress_bar = tqdm(range(len(scan_list)))
    for i in range(len(scan_list)):
        # directory
        scan_info = scan_list[i]
        subject_name = scan_info["subject"]  # 011_S_0005
        scan_time = scan_info["scan_time"]  # 2005-09-02
        scan_dir = os.path.join(data_dir, subject_name, scan_time)

        # ground truth mask
        scan_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(scan_dir, f"t1_pad_synthseg.nii.gz")))
        scan_seg = torch.from_numpy(scan_seg).unsqueeze(0).to(device)  # 1, 156, 177, 156
        scan_seg = get_onehot_mask(scan_seg, num_classes=61, valid_labels=SYNTHSEG_LABELS)  # 1, 33, 156, 177, 156
        scan_seg = scan_seg.to(dtype=torch.float32)
        scan_seg = F.interpolate(scan_seg, size=DATA_SHAPE, mode="trilinear")  # 1, 33, 112, 128, 112
        scan_seg[scan_seg >= 0.5] = 1.0
        scan_seg[scan_seg < 0.5] = 0.0
        scan_seg = scan_seg.to(dtype=torch.uint8)

        # atlas seg mask
        atlas_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(scan_dir, f"atlas_{postfix}_synthseg.nii.gz")))
        atlas_seg = torch.from_numpy(atlas_seg).unsqueeze(0).to(device)  # 1, 156, 177, 156
        atlas_seg = get_onehot_mask(atlas_seg, num_classes=61, valid_labels=SYNTHSEG_LABELS)  # 1, 33, 156, 177, 156
        atlas_seg = atlas_seg.to(dtype=torch.float32)
        atlas_seg = F.interpolate(atlas_seg, size=DATA_SHAPE, mode="trilinear")  # 1, 33, 112, 128, 112
        atlas_seg[atlas_seg >= 0.5] = 1.0
        atlas_seg[atlas_seg < 0.5] = 0.0

        # displacement flow
        displacement_flow = torch.load(os.path.join(scan_dir, f"disp_flow_{postfix}.pt"))  # 1, 3, 112, 128, 112
        displacement_flow = displacement_flow.to(device)

        # warp atlas seg mask with displacement flow
        warped_atlas_seg = spatial_transformer(atlas_seg, displacement_flow)  # 1, 33, 112, 128, 112
        warped_atlas_seg[warped_atlas_seg >= 0.5] = 1.0
        warped_atlas_seg[warped_atlas_seg < 0.5] = 0.0
        warped_atlas_seg = warped_atlas_seg.to(dtype=torch.uint8)

        # calculate dice score
        dice = metrics.compute_dice(
            warped_atlas_seg,
            scan_seg,
            include_background=False,
            num_classes=33,
        )  # (1, 32)
        dice_list_128.append(dice)

        # resampled atlas & scan
        atlas_resample = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(scan_dir, f"atlas_{postfix}_resample.nii.gz")))
        atlas_resample = torch.from_numpy(atlas_resample).unsqueeze(0).unsqueeze(0).to(device)  # 1, 1, 156, 177, 156
        atlas_resample = F.interpolate(atlas_resample, size=DATA_SHAPE, mode="trilinear")  # 1, 1, 112, 128, 112

        scan_resample = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(scan_dir, f"t1_pad_resample.nii.gz")))
        scan_resample = torch.from_numpy(scan_resample).unsqueeze(0).unsqueeze(0).to(device)  # 1, 1, 156, 177, 156
        scan_resample = F.interpolate(scan_resample, size=DATA_SHAPE, mode="trilinear")  # 1, 1, 112, 128, 112

        # displacement flow
        _, displacement_flow_resample = vxm(atlas_resample, scan_resample, registration=True)
        warped_atlas_seg_resample = spatial_transformer(atlas_seg, displacement_flow_resample)  # 1, 33, 112, 128, 112
        warped_atlas_seg_resample[warped_atlas_seg_resample >= 0.5] = 1.0
        warped_atlas_seg_resample[warped_atlas_seg_resample < 0.5] = 0.0
        warped_atlas_seg_resample = warped_atlas_seg_resample.to(dtype=torch.uint8)

        # calculate dice score
        dice_resample = metrics.compute_dice(
            warped_atlas_seg_resample,
            scan_seg,
            include_background=False,
            num_classes=33,
        )  # (1, 32)
        dice_list_resample.append(dice_resample)

        progress_bar.update(1)

    dice_128 = torch.cat(dice_list_128, dim=0)  # (N, 32)
    dice_128 = dice_128.cpu()
    dice_resample = torch.cat(dice_list_resample, dim=0)  # (N, 32)
    dice_resample = dice_resample.cpu()

    torch.save(dice_128, os.path.join(output_dir, f"dice_128_{postfix}.pt"))
    torch.save(dice_resample, os.path.join(output_dir, f"dice_resample_{postfix}.pt"))

    print("Dice Score: ")
    print(np.nanmean(dice_128.numpy()), np.nanstd(dice_128.numpy()))
    print(np.nanmean(dice_resample.numpy()), np.nanstd(dice_resample.numpy()))


if __name__ == "__main__":
    args = parse_args()
    main(args)
