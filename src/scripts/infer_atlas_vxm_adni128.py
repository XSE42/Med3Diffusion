import os
import sys
import json
import argparse

# Add `src` dir to `sys.path`
base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import torch
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm.auto import tqdm

from utils.models import VxmCondAtlas
from voxelmorph.networks import VxmDense


def parse_args():
    parser = argparse.ArgumentParser(description="Infer atlas with VoxelMorph model")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to pretrained VoxelMorph model")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing ADNI dataset")
    parser.add_argument("--init_atlas_dir", type=str, default=None, help="Folder containing initial atlas")
    parser.add_argument("--meta", type=str, default=None, help="JSON file containing metadata of ADNI test set")
    parser.add_argument(
        "--val_meta",
        type=str,
        default=None,
        help="JSON file containing metadata of ADNI validation set"
    )
    parser.add_argument("--postfix", type=str, default="vxm_adni", help="Postfix of the output file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    return args


def main(args):
    # 1. load metadata
    with open(args.meta, "r") as handle:
        test_meta = json.load(handle)
    scan_list = test_meta["scan_list"]
    scan_num = len(scan_list)

    if args.val_meta is not None:
        with open(args.val_meta, "r") as handle:
            val_meta = json.load(handle)
        val_scan_list = val_meta["scan_list"]
        scan_list = scan_list + val_scan_list
    total_scan_num = len(scan_list)

    # 2. load pretrained model
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    atlas_model = VxmCondAtlas.from_pretrained(args.pretrained_model, subfolder="atlas").to(device)
    vxm = VxmDense.from_pretrained(args.pretrained_model, subfolder="vxm").to(device)

    # 3. nii meta
    ref_origin = (188.0, -22.0, 4.0)
    ref_spacing = ((144 - 1) / (104 - 1), (176 - 1) / (128 - 1), (144 - 1) / (104 - 1))
    ref_direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

    # 4. preparation
    PAD_SIZE = 4
    PAD_SEQ = (PAD_SIZE, PAD_SIZE, 0, 0, PAD_SIZE, PAD_SIZE)
    data_dir = args.data_dir
    postfix = args.postfix

    # 5. load init atlas
    init_atlas = torch.load(os.path.join(args.init_atlas_dir, "init_atlas_128.pt")).unsqueeze(0)  # (1, 1, 104, 128, 104)
    init_atlas = F.pad(init_atlas, pad=PAD_SEQ, mode="constant", value=-1)  # (1, 1, 112, 128, 112)
    init_atlas = init_atlas.to(device)

    # 6. inference
    progress_bar = tqdm(range(total_scan_num))

    with torch.no_grad():
        for i in range(total_scan_num):
            scan_info = scan_list[i]
            subject_name = scan_info["subject"]  # 011_S_0005
            scan_time = scan_info["scan_time"]  # 2005-09-02
            scan_age = scan_info["scan_age"]  # 79.6
            scan_dir = os.path.join(data_dir, subject_name, scan_time)
            scan_path = os.path.join(scan_dir, "t1.pt")
            scan = torch.load(scan_path)  # 104, 128, 104
            scan = scan.unsqueeze(0).unsqueeze(0).to(device)  # 1, 1, 104, 128, 104
            scan = F.pad(scan, pad=PAD_SEQ, mode="constant", value=-1)  # 1, 1, 112, 128, 112

            num_cond = torch.tensor([scan_age], dtype=torch.float32, device=device).unsqueeze(0)  # 1, 1
            atlas_tensor = atlas_model(num_cond) + init_atlas  # 1, 1, 112, 128, 112

            warped_atlas, displacement_flow = vxm(atlas_tensor, scan, registration=True)

            atlas_tensor = atlas_tensor.squeeze().cpu()
            warped_atlas = warped_atlas.squeeze().cpu()
            displacement_flow = displacement_flow.cpu()
            scan = scan.squeeze().cpu()
            torch.save(atlas_tensor, os.path.join(scan_dir, f"atlas_{postfix}.pt"))
            torch.save(warped_atlas, os.path.join(scan_dir, f"warped_atlas_{postfix}.pt"))
            torch.save(displacement_flow, os.path.join(scan_dir, f"disp_flow_{postfix}.pt"))
            torch.save(scan, os.path.join(scan_dir, f"t1_pad.pt"))

            atlas_tensor = atlas_tensor.clamp(-1, 1).numpy()
            warped_atlas = warped_atlas.clamp(-1, 1).numpy()
            scan = scan.clamp(-1, 1).numpy()
            atlas_img = sitk.GetImageFromArray(atlas_tensor)
            atlas_img.SetOrigin(ref_origin)
            atlas_img.SetSpacing(ref_spacing)
            atlas_img.SetDirection(ref_direction)
            warped_atlas_img = sitk.GetImageFromArray(warped_atlas)
            warped_atlas_img.SetOrigin(ref_origin)
            warped_atlas_img.SetSpacing(ref_spacing)
            warped_atlas_img.SetDirection(ref_direction)
            scan_img = sitk.GetImageFromArray(scan)
            scan_img.SetOrigin(ref_origin)
            scan_img.SetSpacing(ref_spacing)
            scan_img.SetDirection(ref_direction)
            sitk.WriteImage(atlas_img, os.path.join(scan_dir, f"atlas_{postfix}.nii.gz"))
            sitk.WriteImage(warped_atlas_img, os.path.join(scan_dir, f"warped_atlas_{postfix}.nii.gz"))
            sitk.WriteImage(scan_img, os.path.join(scan_dir, f"t1_pad.nii.gz"))

            progress_bar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
