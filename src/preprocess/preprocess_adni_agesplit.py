import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for spliting ADNI into age groups")
    parser.add_argument("--meta", type=str, default=None, required=True, help="JSON file containing metadata of ADNI")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to store splitted metadata")
    args = parser.parse_args()

    return args


def main(args):
    # 1. load metadata of ADNI dataset
    with open(args.meta, "r") as handle:
        metadata = json.load(handle)
    scan_list = metadata["scan_list"]
    meta_prefix = os.path.basename(args.meta).split(".")[0]

    # 2. variables for age groups
    min_age = 200
    max_age = 0

    start_age = 50
    age_list = list(range(start_age, 100))
    age_num_list = [0] * len(age_list)

    scan_list_age60 = []
    gend_num_age60 = {"M": 0, "F": 0}
    scan_list_age70 = []
    gend_num_age70 = {"M": 0, "F": 0}
    scan_list_age80 = []
    gend_num_age80 = {"M": 0, "F": 0}
    scan_list_age90 = []
    gend_num_age90 = {"M": 0, "F": 0}

    # 3. loop over scan_list
    for scan_info in scan_list:
        scan_age = scan_info["scan_age"]  # 79.6
        scan_gend = scan_info["gender"]  # M
        min_age = min(min_age, scan_age)
        max_age = max(max_age, scan_age)
        age_num_list[round(scan_age) - start_age] += 1

        scan_age = round(scan_age)
        if scan_age < 65:
            scan_list_age60.append(scan_info)
            gend_num_age60[scan_gend] += 1
        elif scan_age >= 65 and scan_age < 75:
            scan_list_age70.append(scan_info)
            gend_num_age70[scan_gend] += 1
        elif scan_age >= 75 and scan_age < 85:
            scan_list_age80.append(scan_info)
            gend_num_age80[scan_gend] += 1
        elif scan_age >= 85:
            scan_list_age90.append(scan_info)
            gend_num_age90[scan_gend] += 1
        else:
            print("Error: ", scan_age)

    # 4. results
    age_num_dict = dict(zip(age_list, age_num_list))
    print(meta_prefix)
    print("Min age:", min_age)
    print("Max age:", max_age)
    print("Total scan num: ", len(scan_list))
    print("Scan num of age group 60: ", len(scan_list_age60))
    print("\t gender: ", gend_num_age60)
    print("Scan num of age group 70: ", len(scan_list_age70))
    print("\t gender: ", gend_num_age70)
    print("Scan num of age group 80: ", len(scan_list_age80))
    print("\t gender: ", gend_num_age80)
    print("Scan num of age group 90: ", len(scan_list_age90))
    print("\t gender: ", gend_num_age90)
    print("Scan num of each age: ", age_num_dict)

    # 5. save scan_list of each age group
    if args.output_dir is None:
        return
    with open(os.path.join(args.output_dir, f"{meta_prefix}_age60.json"), "w") as handle:
        json.dump({"scan_list": scan_list_age60}, handle, indent=4)
    with open(os.path.join(args.output_dir, f"{meta_prefix}_age70.json"), "w") as handle:
        json.dump({"scan_list": scan_list_age70}, handle, indent=4)
    with open(os.path.join(args.output_dir, f"{meta_prefix}_age80.json"), "w") as handle:
        json.dump({"scan_list": scan_list_age80}, handle, indent=4)
    with open(os.path.join(args.output_dir, f"{meta_prefix}_age90.json"), "w") as handle:
        json.dump({"scan_list": scan_list_age90}, handle, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
