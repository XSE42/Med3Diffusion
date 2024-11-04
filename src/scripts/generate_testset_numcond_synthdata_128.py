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

from utils.pipeline_numcond3d import NumCond3DDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic data under numeric conditions in test set")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained numeric conditional LDM",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="Folder to store the generated synthetic data",
    )
    parser.add_argument("--meta", type=str, default=None, help="JSON file containing metadata of test set")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    return args


def main(args):
    # 1. load metadata of synthetic dataset
    with open(args.meta, "r") as handle:
        metadata = json.load(handle)
    scan_list = metadata["scan_list"]

    # 2. load pretrained numeric conditional LDM
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    pipeline = NumCond3DDiffusionPipeline.from_pretrained(args.pretrained_model).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # 3. params
    ref_origin = (188.0, -22.0, 4.0)
    ref_spacing = ((144 - 1) / (104 - 1), (176 - 1) / (128 - 1), (144 - 1) / (104 - 1))
    ref_direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

    PAD_SIZE = 4
    PAD_SEQ = (PAD_SIZE, PAD_SIZE, 0, 0, PAD_SIZE, PAD_SIZE)
    resample_shape = (156, 177, 156)
    resample_spacing = (1.0, 1.0, 1.0)

    # 4. data directory
    data_dir =args.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 5. generate synthetic data
    vae = pipeline.vae
    scaling_factor = vae.config.scaling_factor

    progress_bar = tqdm(range(len(scan_list)))
    with torch.no_grad():
        for scan_info in scan_list:
            subject_name = scan_info["subject"]  # 011_S_0005
            scan_time = scan_info["scan_time"]  # 2005-09-02
            scan_age = scan_info["scan_age"]  # 79.6
            scan_gend = scan_info["gender"]  # M

            scan_age = scan_age / 100.0
            if scan_gend == "F":
                gend = 0.0
            else:
                gend = 1.0
            num_cond = torch.tensor([scan_age, gend], dtype=torch.float32, device=device).unsqueeze(0)

            # generate synthetic data
            latents = pipeline(
                num_cond,
                num_inference_steps=20,
                generator=None,
                base_latents=None,
                output_type="latent",
                return_dict=False,
            )[0]  # (1, 2, 26, 32, 26)
            latents = latents / scaling_factor
            synth_scan = vae.decode(latents, return_dict=False)[0]  # (1, 1, 104, 128, 104)
            synth_scan_pad = F.pad(synth_scan, pad=PAD_SEQ, mode="constant", value=-1)  # 1, 1, 112, 128, 112
            synth_scan_resample = F.interpolate(synth_scan_pad, size=resample_shape, mode="trilinear")

            synth_scan = synth_scan.clamp(-1, 1)
            synth_scan = synth_scan.squeeze().cpu()
            synth_scan_pad = synth_scan_pad.clamp(-1, 1)
            synth_scan_pad = synth_scan_pad.squeeze().cpu()
            synth_scan_resample = synth_scan_resample.clamp(-1, 1)
            synth_scan_resample = synth_scan_resample.squeeze().cpu()

            # save synthetic data
            out_dir = os.path.join(data_dir, subject_name, scan_time)
            torch.save(synth_scan, os.path.join(out_dir, "numcond_synth.pt"))
            torch.save(synth_scan_pad, os.path.join(out_dir, "numcond_synth_pad.pt"))
            torch.save(synth_scan_resample, os.path.join(out_dir, "numcond_synth_resample.pt"))

            synth_img = sitk.GetImageFromArray(synth_scan.numpy())
            synth_img.SetOrigin(ref_origin)
            synth_img.SetSpacing(ref_spacing)
            synth_img.SetDirection(ref_direction)
            sitk.WriteImage(synth_img, os.path.join(out_dir, "numcond_synth.nii.gz"))

            synth_img_pad = sitk.GetImageFromArray(synth_scan_pad.numpy())
            synth_img_pad.SetOrigin(ref_origin)
            synth_img_pad.SetSpacing(ref_spacing)
            synth_img_pad.SetDirection(ref_direction)
            sitk.WriteImage(synth_img_pad, os.path.join(out_dir, "numcond_synth_pad.nii.gz"))

            synth_img_resample = sitk.GetImageFromArray(synth_scan_resample.numpy())
            synth_img_resample.SetOrigin(ref_origin)
            synth_img_resample.SetSpacing(resample_spacing)
            synth_img_resample.SetDirection(ref_direction)
            sitk.WriteImage(synth_img_resample, os.path.join(out_dir, "numcond_synth_resample.nii.gz"))

            progress_bar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
