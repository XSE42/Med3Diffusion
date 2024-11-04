import os
import sys
import json
import argparse

# Add `src` dir to `sys.path`
base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)

import torch
import numpy as np
from scipy import linalg
# from monai.metrics.fid import get_fid_score

from utils.eval import get_fid_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FID score")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing ADNI dataset")
    parser.add_argument("--meta1", type=str, default=None, help="JSON file containing metadata of dataset 1")
    parser.add_argument("--meta2", type=str, default=None, help="JSON file containing metadata of dataset 2")
    parser.add_argument("--file1", type=str, default=None, help="Filename of feature file 1")
    parser.add_argument("--file2", type=str, default=None, help="Filename of feature file 2")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    return args


def main(args):
    # 1. load metadata
    with open(args.meta1, "r") as handle:
        meta1 = json.load(handle)
    if args.meta2 is None:
        scan_list1 = meta1["scan_list1"]
        scan_list2 = meta1["scan_list2"]
    else:
        with open(args.meta2, "r") as handle:
            meta2 = json.load(handle)
        scan_list1 = meta1["scan_list"]
        scan_list2 = meta2["scan_list"]
    assert len(scan_list1) == len(scan_list2)
    print("Scan num: {}".format(len(scan_list1)))

    # 2. params
    device = torch.device("cuda:{}".format(args.gpu) if args.gpu >= 0 else "cpu")
    data_dir = args.data_dir
    file1 = args.file1
    file2 = args.file2
    num_scan = len(scan_list1)

    # 3. loop
    feat_list1 = [None for _ in range(num_scan)]
    feat_list2 = [None for _ in range(num_scan)]
    with torch.no_grad():
        for idx in range(num_scan):
            scan_info1 = scan_list1[idx]
            subject_name1 = scan_info1["subject"]  # 011_S_0005
            scan_time1 = scan_info1["scan_time"]  # 2005-09-02
            scan_dir1 = os.path.join(data_dir, subject_name1, scan_time1)
            scan1 = torch.load(os.path.join(scan_dir1, file1)).flatten().to(device)
            # scan1 = torch.load(os.path.join(scan_dir1, file1)).flatten().unsqueeze(0).numpy()
            feat_list1[idx] = scan1

            scan_info2 = scan_list2[idx]
            subject_name2 = scan_info2["subject"]
            scan_time2 = scan_info2["scan_time"]
            scan_dir2 = os.path.join(data_dir, subject_name2, scan_time2)
            scan2 = torch.load(os.path.join(scan_dir2, file2)).flatten().to(device)
            # scan2 = torch.load(os.path.join(scan_dir2, file2)).flatten().unsqueeze(0).numpy()
            feat_list2[idx] = scan2

    # 4. calculate FID
    feat_list1 = torch.stack(feat_list1, dim=0)
    feat_list2 = torch.stack(feat_list2, dim=0)
    print(feat_list1.shape)
    print("Computing FID...")
    fid = get_fid_score(feat_list1, feat_list2)
    print("FID: {:.6f}".format(fid))

    # feat_list1 = np.concatenate(feat_list1, axis=0)
    # feat_list2 = np.concatenate(feat_list2, axis=0)
    # print("Calculating mean and covariance...")
    # mu1 = np.mean(feat_list1, axis=0)  # (43264,)
    # mu2 = np.mean(feat_list2, axis=0)  # (43264,)
    # sigma1 = np.cov(feat_list1, rowvar=False)  # (43264, 43264)
    # sigma2 = np.cov(feat_list2, rowvar=False)  # (43264, 43264)

    # print("Calculating FID...")
    # diff = mu1 - mu2
    # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # # Product might be almost singular
    # if not np.isfinite(covmean).all():
    #     eps = 1e-6
    #     offset = np.eye(sigma1.shape[0]) * eps
    #     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # # Numerical error might give slight imaginary component
    # if np.iscomplexobj(covmean):
    #     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
    #         m = np.max(np.abs(covmean.imag))
    #         raise ValueError("Imaginary component {}".format(m))
    #     covmean = covmean.real

    # tr_covmean = np.trace(covmean)

    # fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    # print("FID: {:.6f}".format(fid))


if __name__ == "__main__":
    args = parse_args()
    main(args)
