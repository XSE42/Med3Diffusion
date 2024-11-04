import os
import json
import pickle
import argparse

import numpy as np


TRAINSET_RATIO = 0.7
VALID_SET_RATIO = 0.1
TEST_SET_RATIO = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for spliting ADNI")
    parser.add_argument("--meta", type=str, default=None, required=True, help="JSON file containing metadata of ADNI")
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Folder to store splitted metadata")
    args = parser.parse_args()

    return args


def split_metadata(meta, train_ratio, val_ratio):
    # extract subject_list and shuffle it
    subject_list = meta["sub_list"]
    np.random.shuffle(subject_list)

    # get number of subjects and split point
    subject_num = len(subject_list)
    train_split = round(subject_num * train_ratio)
    val_split = round(subject_num * val_ratio + train_split)

    # split subject_list
    train_sub_list = subject_list[:train_split]
    val_sub_list = subject_list[train_split:val_split]
    test_sub_list = subject_list[val_split:]
    print("Training set subject num: {}".format(len(train_sub_list)))
    print("Validation set subject num: {}".format(len(val_sub_list)))
    print("Test set subject num: {}".format(len(test_sub_list)))

    # split metadata
    train_scan_list = []
    val_scan_list = []
    test_scan_list = []
    for scan in meta["scan_list"]:
        if scan["subject"] in train_sub_list:
            train_scan_list.append(scan)
        elif scan["subject"] in val_sub_list:
            val_scan_list.append(scan)
        else:
            test_scan_list.append(scan)

    # generate metadata for training set, validation set, test set
    train_meta = {
        "scan_list": train_scan_list,
        "sub_list": train_sub_list,
        "rid_sub_map": meta["rid_sub_map"],
        "sub_gend_map": meta["sub_gend_map"],
    }
    val_meta = {
        "scan_list": val_scan_list,
        "sub_list": val_sub_list,
        "rid_sub_map": meta["rid_sub_map"],
        "sub_gend_map": meta["sub_gend_map"],
    }
    test_meta = {
        "scan_list": test_scan_list,
        "sub_list": test_sub_list,
        "rid_sub_map": meta["rid_sub_map"],
        "sub_gend_map": meta["sub_gend_map"],
    }
    print("Training set scan num: {}".format(len(train_scan_list)))
    print("Validation set scan num: {}".format(len(val_scan_list)))
    print("Test set scan num: {}".format(len(test_scan_list)))

    return train_meta, val_meta, test_meta


def main():
    # parse arguments
    args = parse_args()

    # load metadata of the whole dataset
    with open(args.meta, "r") as handle:
        meta = json.load(handle)

    # split metadata into training set, validation set, test set
    train_meta, val_meta, test_meta = split_metadata(
        meta=meta,
        train_ratio=TRAINSET_RATIO,
        val_ratio=VALID_SET_RATIO
    )

    # save splitted metadata
    with open(os.path.join(args.output_dir, "ADNI_CN_Train_meta.json"), "w") as handle:
        json.dump(train_meta, handle, indent=4)
    with open(os.path.join(args.output_dir, "ADNI_CN_Val_meta.json"), "w") as handle:
        json.dump(val_meta, handle, indent=4)
    with open(os.path.join(args.output_dir, "ADNI_CN_Test_meta.json"), "w") as handle:
        json.dump(test_meta, handle, indent=4)


if __name__ == "__main__":
    main()
