import os
import json
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for metadata of synthetic dataset")
    parser.add_argument("--meta", type=str, default=None, required=True, help="JSON file containing metadata of ADNI")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to store synthetic metadata")
    parser.add_argument("--meta_prefix", type=str, default="Synth_CN_Train", help="Prefix of synthetic meta filename")
    parser.add_argument("--scan_prefix", type=str, default="train", help="Prefix of synthetic scan name")
    args = parser.parse_args()

    return args


def main(args):
    # 1. load metadata of ADNI training dataset
    with open(args.meta, "r") as handle:
        metadata = json.load(handle)
    scan_list = metadata["scan_list"]

    # 2. variables
    start_age = 55
    end_age = 95
    age_list = list(range(start_age, end_age))  # list of ages
    age_num_list = [0] * len(age_list)  # num of scans for each age in ADNI
    age_gender_list = [{"M": 0, "F": 0} for _ in range(len(age_list))]  # num of scans for each age and gender in ADNI

    # 3. loop over scan_list
    for scan_info in scan_list:
        scan_age = scan_info["scan_age"]  # 79.6
        scan_gend = scan_info["gender"]  # M
        scan_age = round(scan_age)
        if scan_age < start_age or scan_age >= end_age:
            continue
        age_num_list[scan_age - start_age] += 1
        age_gender_list[scan_age - start_age][scan_gend] += 1

    print("Age list: ", age_list)
    print("Scan num for each age in ADNI: ", age_num_list)
    print("Scan num for each age and gender in ADNI: ", age_gender_list)

    # 4. synthetic dataset variables
    target_num_per_age = 100  # target num of scans for each age (including ADNI and synthetic dataset)
    target_num_per_age_gend = target_num_per_age // 2  # target num of scans for each age and gender
    synth_age_num_list = [0] * len(age_list)
    synth_age_gender_list = [{"M": 0, "F": 0} for _ in range(len(age_list))]

    synth_scan_list = []
    synth_scan_list_age60 = []
    synth_scan_list_age70 = []
    synth_scan_list_age80 = []
    synth_scan_list_age90 = []

    # 5. generate metadata for synthetic dataset
    scan_id = 0
    scan_prefix = args.scan_prefix
    for i in range(len(age_list)):
        age = age_list[i]
        synth_male_num = target_num_per_age_gend - age_gender_list[i]["M"]
        synth_male_num = max(0, synth_male_num)
        synth_female_num = target_num_per_age_gend - age_gender_list[i]["F"]
        synth_female_num = max(0, synth_female_num)

        synth_age_num_list[i] = synth_male_num + synth_female_num
        synth_age_gender_list[i]["M"] = synth_male_num
        synth_age_gender_list[i]["F"] = synth_female_num

        for _ in range(synth_male_num):
            synth_age = age + np.random.uniform(-0.5, 0.5)
            synth_scan_info = {
                "subject": scan_prefix + str(scan_id).zfill(4),
                "gender": "M",
                "scan_age": round(synth_age, 3),
                "dataset": "Synth"
            }
            synth_scan_list.append(synth_scan_info)
            scan_id += 1

            if age < 65:
                synth_scan_list_age60.append(synth_scan_info)
            elif age >= 65 and age < 75:
                synth_scan_list_age70.append(synth_scan_info)
            elif age >= 75 and age < 85:
                synth_scan_list_age80.append(synth_scan_info)
            elif age >= 85:
                synth_scan_list_age90.append(synth_scan_info)

        for _ in range(synth_female_num):
            synth_age = age + np.random.uniform(-0.5, 0.5)
            synth_scan_info = {
                "subject": scan_prefix + str(scan_id).zfill(4),
                "gender": "F",
                "scan_age": round(synth_age, 3),
                "dataset": "Synth"
            }
            synth_scan_list.append(synth_scan_info)
            scan_id += 1

            if age < 65:
                synth_scan_list_age60.append(synth_scan_info)
            elif age >= 65 and age < 75:
                synth_scan_list_age70.append(synth_scan_info)
            elif age >= 75 and age < 85:
                synth_scan_list_age80.append(synth_scan_info)
            elif age >= 85:
                synth_scan_list_age90.append(synth_scan_info)

    # 6. summary
    print("")
    print("Scan num for each age in synthetic dataset: ", synth_age_num_list)
    print("Scan num for each age and gender in synthetic dataset: ", synth_age_gender_list)
    print("Scan num in synthetic dataset: ", len(synth_scan_list))
    print("Scan num for age group 60 in synthetic dataset: ", len(synth_scan_list_age60))
    print("Scan num for age group 70 in synthetic dataset: ", len(synth_scan_list_age70))
    print("Scan num for age group 80 in synthetic dataset: ", len(synth_scan_list_age80))
    print("Scan num for age group 90 in synthetic dataset: ", len(synth_scan_list_age90))

    total_age_num_list = [age_num_list[i] + synth_age_num_list[i] for i in range(len(age_list))]
    total_age_gender_list = [
        {
            "M": age_gender_list[i]["M"] + synth_age_gender_list[i]["M"],
            "F": age_gender_list[i]["F"] + synth_age_gender_list[i]["F"]
        } for i in range(len(age_list))
    ]
    print("")
    print("Total scan num for each age: ", total_age_num_list)
    print("Total scan num for each age and gender: ", total_age_gender_list)

    # 7. save synthetic metadata

    if args.output_dir is None:
        return

    meta_prefix = args.meta_prefix
    with open(os.path.join(args.output_dir, f"{meta_prefix}.json"), "w") as handle:
        json.dump({"scan_list": synth_scan_list}, handle, indent=4)
    with open(os.path.join(args.output_dir, f"{meta_prefix}_age60.json"), "w") as handle:
        json.dump({"scan_list": synth_scan_list_age60}, handle, indent=4)
    with open(os.path.join(args.output_dir, f"{meta_prefix}_age70.json"), "w") as handle:
        json.dump({"scan_list": synth_scan_list_age70}, handle, indent=4)
    with open(os.path.join(args.output_dir, f"{meta_prefix}_age80.json"), "w") as handle:
        json.dump({"scan_list": synth_scan_list_age80}, handle, indent=4)
    with open(os.path.join(args.output_dir, f"{meta_prefix}_age90.json"), "w") as handle:
        json.dump({"scan_list": synth_scan_list_age90}, handle, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
