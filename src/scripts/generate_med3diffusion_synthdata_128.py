import os
import sys
import json
import argparse

# Add `src` dir to `sys.path`
base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import torch
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

from utils.data import EdgeSampler
from utils.pipeline_numedge3d import NumEdge3DDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic data with pretrained Med3Diffusion model")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained Med3Diffusion",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="Folder to store the generated synthetic dataset",
    )
    parser.add_argument(
        "--edge_dir",
        type=str,
        default=None,
        required=True,
        help="Folder containing ADNI dataset",
    )
    parser.add_argument("--meta", type=str, default=None, help="JSON file containing metadata of synthetic dataset")
    parser.add_argument("--meta_sampler", type=str, default=None, help="JSON file containing ADNI metadata sampler")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    return args


def main(args):
    # 1. load metadata of synthetic dataset & ADNI metadata sampler
    with open(args.meta, "r") as handle:
        metadata = json.load(handle)
    scan_list = metadata["scan_list"]

    with open(args.meta_sampler, "r") as handle:
        sampler = json.load(handle)["sampler"]
    edge_sampler = EdgeSampler(
        args.edge_dir,
        sampler=sampler,
        edge_name="t1_canny",
        edge_scale=False,
    )

    # 2. load pretrained Med3Diffusion
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    pipeline = NumEdge3DDiffusionPipeline.from_pretrained(args.pretrained_model).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # 3. load reference img
    ref_origin = (188.0, -22.0, 4.0)
    ref_spacing = ((144 - 1) / (104 - 1), (176 - 1) / (128 - 1), (144 - 1) / (104 - 1))
    ref_direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

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
            scan_str = scan_info["subject"]  # train0123
            scan_age = scan_info["scan_age"]  # 79.6
            scan_gend = scan_info["gender"]  # M

            edge = edge_sampler.get_edge(scan_gend, scan_age).to(device)  # 104, 128, 104
            edge = edge.unsqueeze(0).unsqueeze(0)  # 1, 1, 104, 128, 104

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
                edges=edge,
                edge_fuse="concat",
                output_type="latent",
                return_dict=False,
            )[0]  # (1, 2, 26, 32, 26)
            latents = latents / scaling_factor
            synth_scan = vae.decode(latents, return_dict=False)[0]  # 1, 1, 104, 128, 104
            synth_scan = synth_scan.clamp(-1, 1)
            synth_scan = synth_scan.squeeze().cpu()  # [-1, 1]
            edge = edge.squeeze().cpu()  # [0, 1]

            # save synthetic data
            out_dir = os.path.join(data_dir, scan_str)
            os.mkdir(out_dir)
            torch.save(synth_scan, os.path.join(out_dir, "t1.pt"))
            torch.save(edge, os.path.join(out_dir, "t1_canny.pt"))

            synth_img = sitk.GetImageFromArray(synth_scan.numpy())
            synth_img.SetOrigin(ref_origin)
            synth_img.SetSpacing(ref_spacing)
            synth_img.SetDirection(ref_direction)
            sitk.WriteImage(synth_img, os.path.join(out_dir, "t1.nii.gz"))
            edge_img = sitk.GetImageFromArray(edge.numpy().astype(np.int32))
            edge_img.SetOrigin(ref_origin)
            edge_img.SetSpacing(ref_spacing)
            edge_img.SetDirection(ref_direction)
            sitk.WriteImage(edge_img, os.path.join(out_dir, "t1_canny.nii.gz"))

            progress_bar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
