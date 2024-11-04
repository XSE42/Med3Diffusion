import os
import sys
import json
import argparse

# Add `src` dir to `sys.path`
base_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a list of MRI for SynthSeg")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing ADNI dataset")
    parser.add_argument("--meta", type=str, default=None, help="JSON file containing metadata of ADNI dataset")
    parser.add_argument("--name", type=str, default="t1", help="Input MRI name")
    parser.add_argument("--postfix", type=str, default="_synthseg", help="Postfix of the output file")
    parser.add_argument("--resample_postfix", type=str, default="_resample", help="Postfix of the resampled file")
    parser.add_argument("--infile", type=str, default="synthseg_in.txt", help="Path of the input file for SynthSeg")
    parser.add_argument("--outfile", type=str, default="synthseg_out.txt", help="Path of the output file for SynthSeg")
    parser.add_argument("--resamplefile", type=str, default=None, help="Path of the resample file for SynthSeg")
    args = parser.parse_args()

    return args


def get_adni_lists(data_dir, scan_list, name, postfix, resample_postfix):
    in_list = []
    out_list = []
    resample_list = []
    for scan_info in scan_list:
        subject_name = scan_info["subject"]  # 011_S_0005
        scan_time = scan_info["scan_time"]  # 2005-09-02
        scan_dir = os.path.join(data_dir, subject_name, scan_time)
        in_path = os.path.join(scan_dir, f"{name}.nii.gz") + "\n"
        out_path = os.path.join(scan_dir, f"{name}{postfix}.nii.gz") + "\n"
        resample_path = os.path.join(scan_dir, f"{name}{resample_postfix}.nii.gz") + "\n"
        in_list.append(in_path)
        out_list.append(out_path)
        resample_list.append(resample_path)

    return in_list, out_list, resample_list


def get_synth_lists(data_dir, scan_list, name, postfix, resample_postfix):
    in_list = []
    out_list = []
    resample_list = []
    for scan_info in scan_list:
        subject_name = scan_info["subject"]  # train0123
        scan_dir = os.path.join(data_dir, subject_name)
        in_path = os.path.join(scan_dir, f"{name}.nii.gz") + "\n"
        out_path = os.path.join(scan_dir, f"{name}{postfix}.nii.gz") + "\n"
        resample_path = os.path.join(scan_dir, f"{name}{resample_postfix}.nii.gz") + "\n"
        in_list.append(in_path)
        out_list.append(out_path)
        resample_list.append(resample_path)

    return in_list, out_list, resample_list


def main(args):
    # 1. load metadata
    with open(args.meta, "r") as handle:
        meta = json.load(handle)
    scan_list = meta["scan_list"]

    # 2. generate lists
    if scan_list[0].get("scan_time") is None:
        in_list, out_list, resample_list = get_synth_lists(
            args.data_dir, scan_list, args.name, args.postfix, args.resample_postfix
        )
    else:
        in_list, out_list, resample_list = get_adni_lists(
            args.data_dir, scan_list, args.name, args.postfix, args.resample_postfix
        )

    # 3. write to file
    with open(args.infile, "w") as handle:
        handle.writelines(in_list)
    with open(args.outfile, "w") as handle:
        handle.writelines(out_list)

    if args.resamplefile is not None:
        with open(args.resamplefile, "w") as handle:
            handle.writelines(resample_list)


if __name__ == "__main__":
    args = parse_args()
    main(args)

# data/ADNI_128/099_S_0534/2006-05-04
