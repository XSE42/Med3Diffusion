import os
import sys
import json
import argparse

# Add `src` dir to `sys.path`
base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Filename of predictions without postfix .nii.gz"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Filename of ground truth without postfix .nii.gz"
    )
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
    DATA_SHAPE = (112, 128, 112)
    ORIG_VAL_SLICE_IDX = [38 + 3, 58 + 3, 78 + 3, 98 + 3, 118 + 3]  # 138, 176, 138 -> 144, 176, 144
    VAL_SLICE_IDX = [round(idx * 104 / 144) + 4 for idx in ORIG_VAL_SLICE_IDX]  # 144, 176, 144 -> 104, 128, 104 -> 112, 128, 112
    gt_name = args.gt  # "t1_pad"
    pred_name = args.pred  # "numcond_synth_pad"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    writer = SummaryWriter(os.path.join(output_dir))

    dice = []  # dice score between `onehot_warped_gt` and `onehoe_pred`
    dice_128 = []  # dice score between `onehot_warped_gt_128` and `onehot_pred`
    diff_gt_slices = {k: [] for k in VAL_SLICE_IDX}  # difference between `gt_128` and `scan_gt`
    diff_pred_slices = {k: [] for k in VAL_SLICE_IDX}  # difference between `pred_128` and `scan_pred`
    diff_warped_gt_slices = {k: [] for k in VAL_SLICE_IDX}  # difference between `warped_gt_128` and `warped_gt`

    # 3. load model
    vxm: VxmDense = VxmDense.from_pretrained(args.pretrained_model, subfolder="vxm").to(device)
    spatial_transformer = SpatialTransformer(vxm.inshape).to(device)

    # 4. loop
    progress_bar = tqdm(range(len(scan_list)))
    for i in range(len(scan_list)):
        scan_info = scan_list[i]
        subject_name = scan_info["subject"]  # 011_S_0005
        scan_time = scan_info["scan_time"]  # 2005-09-02
        scan_dir = os.path.join(data_dir, subject_name, scan_time)

        gt_128 = torch.load(os.path.join(scan_dir, "t1_pad.pt"))  # 112, 128, 112, [-1, 1]
        gt_128 = gt_128.unsqueeze(0).unsqueeze(0).to(device)  # 1, 1, 112, 128, 112

        pred_128 = torch.load(os.path.join(scan_dir, f"{pred_name}.pt"))  # 112, 128, 112, [-1, 1]
        pred_128 = pred_128.unsqueeze(0).unsqueeze(0).to(device)  # 1, 1, 112, 128, 112

        warped_gt_128, disp_flow_128 = vxm(gt_128, pred_128, registration=True)

        scan_gt = sitk.ReadImage(os.path.join(scan_dir, f"{gt_name}_resample.nii.gz"))
        scan_gt = torch.from_numpy(sitk.GetArrayFromImage(scan_gt))  # 156, 177, 156
        scan_gt = scan_gt.unsqueeze(0).unsqueeze(0).to(device)  # 1, 1, 156, 177, 156
        scan_gt = F.interpolate(scan_gt, size=DATA_SHAPE, mode="trilinear")  # 1, 1, 112, 128, 112

        scan_pred = sitk.ReadImage(os.path.join(scan_dir, f"{pred_name}_resample.nii.gz"))
        scan_pred = torch.from_numpy(sitk.GetArrayFromImage(scan_pred))  # 156, 177, 156
        scan_pred = scan_pred.unsqueeze(0).unsqueeze(0).to(device)  # 1, 1, 156, 177, 156
        scan_pred = F.interpolate(scan_pred, size=DATA_SHAPE, mode="trilinear")  # 1, 1, 112, 128, 112

        warped_gt, disp_flow = vxm(scan_gt, scan_pred, registration=True)

        if i < 16:
            gt_128 = (gt_128 / 2 + 0.5).clamp(0, 1)
            scan_gt = (scan_gt / 2 + 0.5).clamp(0, 1)
            diff_gt = (gt_128[:, :, VAL_SLICE_IDX] - scan_gt[:, :, VAL_SLICE_IDX]).squeeze().cpu().numpy()  # 5, 128, 112
            color_diff_gt = np.ones((*diff_gt.shape, 3))  # 5, 128, 112, 3
            color_diff_gt[diff_gt > 0, 1:3] -= diff_gt[diff_gt > 0, np.newaxis]  # Red
            color_diff_gt[diff_gt < 0, 0:2] += diff_gt[diff_gt < 0, np.newaxis]  # Blue

            pred_128 = (pred_128 / 2 + 0.5).clamp(0, 1)
            scan_pred = (scan_pred / 2 + 0.5).clamp(0, 1)
            diff_pred = (pred_128[:, :, VAL_SLICE_IDX] - scan_pred[:, :, VAL_SLICE_IDX]).squeeze().cpu().numpy()  # 5, 128, 112
            color_diff_pred = np.ones((*diff_pred.shape, 3))  # 5, 128, 112, 3
            color_diff_pred[diff_pred > 0, 1:3] -= diff_pred[diff_pred > 0, np.newaxis]  # Red
            color_diff_pred[diff_pred < 0, 0:2] += diff_pred[diff_pred < 0, np.newaxis]  # Blue

            warped_gt_128 = (warped_gt_128 / 2 + 0.5).clamp(0, 1)
            warped_gt = (warped_gt / 2 + 0.5).clamp(0, 1)
            diff_warped_gt = (warped_gt_128[:, :, VAL_SLICE_IDX] - warped_gt[:, :, VAL_SLICE_IDX]).squeeze().cpu().numpy()  # 5, 128, 112
            color_diff_warped_gt = np.ones((*diff_warped_gt.shape, 3))  # 5, 128, 112, 3
            color_diff_warped_gt[diff_warped_gt > 0, 1:3] -= diff_warped_gt[diff_warped_gt > 0, np.newaxis]  # Red
            color_diff_warped_gt[diff_warped_gt < 0, 0:2] += diff_warped_gt[diff_warped_gt < 0, np.newaxis]  # Blue

            for j, idx in enumerate(VAL_SLICE_IDX):
                diff_gt_slices[idx].append(color_diff_gt[j])  # 128, 112, 3
                diff_pred_slices[idx].append(color_diff_pred[j])  # 128, 112, 3
                diff_warped_gt_slices[idx].append(color_diff_warped_gt[j])  # 128, 112, 3

        seg_gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(scan_dir, f"{gt_name}_synthseg.nii.gz")))  # 156, 177, 156
        seg_gt = torch.from_numpy(seg_gt).unsqueeze(0).to(device)  # 1, 156, 177, 156
        onehot_gt = get_onehot_mask(seg_gt, num_classes=61, valid_labels=SYNTHSEG_LABELS)  # 1, 33, 156, 177, 156
        onehot_gt = onehot_gt.to(dtype=torch.float32)
        onehot_gt = F.interpolate(onehot_gt, size=DATA_SHAPE, mode="trilinear")  # 1, 33, 112, 128, 112
        onehot_gt[onehot_gt >= 0.5] = 1.0
        onehot_gt[onehot_gt < 0.5] = 0.0

        onehot_warped_gt_128 = spatial_transformer(onehot_gt, disp_flow_128)  # 1, 33, 112, 128, 112
        onehot_warped_gt_128[onehot_warped_gt_128 >= 0.5] = 1.0
        onehot_warped_gt_128[onehot_warped_gt_128 < 0.5] = 0.0
        onehot_warped_gt_128 = onehot_warped_gt_128.to(dtype=torch.uint8)

        onehot_warped_gt = spatial_transformer(onehot_gt, disp_flow)  # 1, 33, 112, 128, 112
        onehot_warped_gt[onehot_warped_gt >= 0.5] = 1.0
        onehot_warped_gt[onehot_warped_gt < 0.5] = 0.0
        onehot_warped_gt = onehot_warped_gt.to(dtype=torch.uint8)

        seg_pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(scan_dir, f"{pred_name}_synthseg.nii.gz")))  # 156, 177, 156
        seg_pred = torch.from_numpy(seg_pred).unsqueeze(0).to(device)  # 1, 156, 177, 156
        onehot_pred = get_onehot_mask(seg_pred, num_classes=61, valid_labels=SYNTHSEG_LABELS)  # 1, 33, 156, 177, 156
        onehot_pred = onehot_pred.to(dtype=torch.float32)
        onehot_pred = F.interpolate(onehot_pred, size=DATA_SHAPE, mode="trilinear")  # 1, 33, 112, 128, 112
        onehot_pred[onehot_pred >= 0.5] = 1.0
        onehot_pred[onehot_pred < 0.5] = 0.0
        onehot_pred = onehot_pred.to(dtype=torch.uint8)

        result = metrics.compute_dice(
            onehot_pred,
            onehot_warped_gt,
            include_background=False,
            num_classes=33,
        )  # (1, 32)
        dice.append(result)

        result_128 = metrics.compute_dice(
            onehot_pred,
            onehot_warped_gt_128,
            include_background=False,
            num_classes=33,
        )  # (1, 32)
        dice_128.append(result_128)

        progress_bar.update(1)

    dice = torch.cat(dice, dim=0)  # (N, 32)
    dice_128 = torch.cat(dice_128, dim=0)  # (N, 32)
    dice = dice.cpu()
    dice_128 = dice_128.cpu()
    torch.save(dice, os.path.join(output_dir, "dice.pt"))
    torch.save(dice_128, os.path.join(output_dir, "dice_128.pt"))
    print("Dice Score: ")
    print(np.nanmean(dice.numpy()), np.nanstd(dice.numpy()))
    print(np.nanmean(dice_128.numpy()), np.nanstd(dice_128.numpy()))

    for idx in VAL_SLICE_IDX:
        color_diff_gt = np.stack(diff_gt_slices[idx], axis=0)  # N, 128, 112, 3
        color_diff_pred = np.stack(diff_pred_slices[idx], axis=0)  # N, 128, 112, 3
        color_diff_warped_gt = np.stack(diff_warped_gt_slices[idx], axis=0)  # N, 128, 112, 3
        color_diff_gt = (color_diff_gt * 255).round().astype(np.uint8)
        color_diff_pred = (color_diff_pred * 255).round().astype(np.uint8)
        color_diff_warped_gt = (color_diff_warped_gt * 255).round().astype(np.uint8)
        writer.add_images(f"diff_gt_{idx}", color_diff_gt, dataformats="NHWC")
        writer.add_images(f"diff_pred_{idx}", color_diff_pred, dataformats="NHWC")
        writer.add_images(f"diff_warped_gt_{idx}", color_diff_warped_gt, dataformats="NHWC")

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
