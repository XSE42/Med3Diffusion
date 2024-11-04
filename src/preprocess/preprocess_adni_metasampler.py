import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for ADNI metadata sampler")
    parser.add_argument("--meta", type=str, default=None, required=True, help="JSON file containing metadata of ADNI")
    parser.add_argument("--outfile", type=str, default=None, required=True, help="Path to store the result")
    args = parser.parse_args()

    return args


def main(args):
    # 1. load metadata
    with open(args.meta, "r") as handle:
        meta = json.load(handle)
    scan_list = meta["scan_list"]

    # 2. sampler dict
    sampler = {
        k: {
            age: [] for age in [60, 70, 80, 90]
        } for k in ["F", "M"]
    }

    # 3. process data
    for scan_info in scan_list:
        scan_gend = scan_info["gender"]  # M
        if scan_gend != "F":
            scan_gend = "M"

        scan_age = scan_info["scan_age"]  # 79.6
        scan_age = round(scan_age)
        if scan_age < 65:
            scan_age = 60
        elif scan_age >= 65 and scan_age < 75:
            scan_age = 70
        elif scan_age >= 75 and scan_age < 85:
            scan_age = 80
        elif scan_age >= 85:
            scan_age = 90

        sampler[scan_gend][scan_age].append(scan_info)

    # 4. save sampler dict
    # num = 0
    # for gend in ["F", "M"]:
    #     for age in [60, 70, 80, 90]:
    #         n = len(sampler[gend][age])
    #         print(f"{gend} {age} num: {n}")
    #         num += n
    # print("Sum: ", num)
    # print("Total: ", len(scan_list))
    if args.outfile is None:
        return
    with open(args.outfile, "w") as handle:
        json.dump({"sampler": sampler}, handle, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)


# python src/preprocess/preprocess_adni_metasampler.py --meta data/ADNI_meta/ADNI_CN_Train_meta.json --outfile data/ADNI_meta/ADNI_CN_Train_sampler.json
