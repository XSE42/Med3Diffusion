import os
import json
import argparse
from collections import OrderedDict

import torch
from tqdm.auto import tqdm

from models import resnet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for testset feature extraction with Med3D")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="Path to ADNI testset and to store extracted features",
    )
    parser.add_argument("--meta", type=str, default=None, help="JSON file containing metadata of test set")
    parser.add_argument("--infile", type=str, default="t1.pt", help="Input file name")
    parser.add_argument("--outfile", type=str, default="t1_feat_med3d.pt", help="Output file name")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained Med3D model",
    )
    parser.add_argument("--depth", type=int, default=104, help="Depth of input volume")
    parser.add_argument("--height", type=int, default=128, help="Height of input volume")
    parser.add_argument("--width", type=int, default=104, help="Width of input volume")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    return args


def main(args):
    # params
    use_gpu = False
    if args.gpu >= 0:
        use_gpu = True
    device = torch.device("cuda:{}".format(args.gpu) if use_gpu else "cpu")

    data_dir = args.data_dir
    infile = args.infile
    outfile = args.outfile

    OUTPUT_SIZE = 1
    # OUTPUT_SIZE = (3, 4, 3)

    # model and checkpoint
    model = resnet.resnet50(
        sample_input_W=args.width,
        sample_input_H=args.height,
        sample_input_D=args.depth,
        shortcut_type="B",
        no_cuda=not use_gpu,
        num_seg_classes=2,
    ).to(device)

    ckpt = torch.load(args.pretrained_model)
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        name = k[7:]  # remove `module.`
        state_dict[name] = v
    model.load_state_dict(state_dict)

    # metadata
    with open(args.meta, "r") as handle:
        metadata = json.load(handle)
    scan_list = metadata["scan_list"]

    # feature extraction
    progress_bar = tqdm(range(len(scan_list)))
    with torch.no_grad():
        for scan_info in scan_list:
            subject_name = scan_info["subject"]  # 011_S_0005
            scan_time = scan_info["scan_time"]  # 2005-09-02
            scan_dir = os.path.join(data_dir, subject_name, scan_time)
            scan = torch.load(os.path.join(scan_dir, infile))
            scan = scan.unsqueeze(0).unsqueeze(0).to(device)  # 1, 1, 104, 128, 104
            feat = model.extract_feat(scan, OUTPUT_SIZE)  # 1, 32, *OUTPUT_SIZE
            feat = feat.squeeze().cpu()
            torch.save(feat, os.path.join(scan_dir, outfile))

            progress_bar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
