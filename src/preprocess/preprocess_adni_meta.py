import os
import glob
import json
import pickle
import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for ADNI metadata")
    parser.add_argument("--data_dir", type=str, default=None, required=True, help="Folder containing ADNI dataset")
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Folder to store extracted meta")
    parser.add_argument("--orig_meta", type=str, default=None, required=True, help="CSV file containing original meta")
    parser.add_argument("--gend_file", type=str, default=None, required=True, help="CSV file containing subject gender")
    args = parser.parse_args()

    return args


def extract_subject_gender_map(gend_info):
    gend_info = gend_info.groupby("Subject")

    sub_gend_map = dict()
    # extract subject gender
    for subject, subject_data in gend_info:
        sub_gend_map[subject] = subject_data.iloc[0]["Sex"]

    return sub_gend_map


def extract_metadata(data_dir, orig_meta, gend_info):
    # extract subject_gender_map
    sub_gend_map = extract_subject_gender_map(gend_info)

    # drop unnecessary info and reformat date
    orig_meta = orig_meta.loc[:, ["RID", "DX.scan", "ScanDate", "AGE.scan"]]
    orig_meta["AcqDate"] = pd.to_datetime(orig_meta["ScanDate"], format="%d/%m/%Y")
    orig_meta = orig_meta.groupby("RID")

    scan_list = []
    sub_set = set()
    rid_sub_map = dict()
    # extract meta
    for subject_rid, subject_data in orig_meta:
        subject_path = glob.glob(os.path.join(data_dir, "[0-9][0-9][0-9]_S_" + str(subject_rid).zfill(4)))[0]
        subject_str = os.path.basename(subject_path)  # 011_S_0005
        subject_gend = sub_gend_map[subject_str]  # M

        subject_contains_data = False
        for scan in subject_data.itertuples():
            if scan._2 != "CN":
                continue

            scan_time = scan.AcqDate.strftime("%Y-%m-%d")  # 2005-09-02
            scan_dir = os.path.join(subject_path, scan_time)
            if not os.path.exists(scan_dir):
                continue

            scan_age = scan._4
            scan_item = {
                "rid": subject_rid,
                "subject": subject_str,
                "gender": subject_gend,
                "scan_time": scan_time,
                "scan_age": scan_age,
                "scan_dir": scan_dir,
                "dataset": "ADNI"
            }
            scan_list.append(scan_item)
            subject_contains_data = True

        if subject_contains_data:
            sub_set.add(subject_str)
            rid_sub_map[subject_rid] = subject_str

    # metadata
    metadata = {
        "scan_list": scan_list,
        "sub_list": list(sub_set),
        "rid_sub_map": rid_sub_map,
        "sub_gend_map": sub_gend_map,
    }

    return metadata


def main():
    # parse arguments
    args = parse_args()

    # load files
    orig_meta = pd.read_csv(args.orig_meta, low_memory=False)
    gend_info = pd.read_csv(args.gend_file, low_memory=False)

    # extract metadata
    extracted_meta = extract_metadata(data_dir=args.data_dir, orig_meta=orig_meta, gend_info=gend_info)
    print("Subject num: {}".format(len(extracted_meta["sub_list"])))  # 523
    print("Scan num: {}".format(len(extracted_meta["scan_list"])))  # 1417

    # save metadata
    out_file = os.path.join(args.output_dir, "ADNI_CN_meta.json")
    with open(out_file, "w") as handle:
        json.dump(extracted_meta, handle, indent=4)


if __name__ == "__main__":
    main()
