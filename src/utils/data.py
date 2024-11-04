import os

import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import Dataset


class ADNI_3D_Set(Dataset):
    """
    Dataset for ADNI 3D data
    """

    def __init__(self, data_dir, scan_list, scale=True, dtype=torch.float32):
        self.data_dir = data_dir
        self.scan_list = scan_list
        self.scan_num = len(scan_list)
        self.scale = scale
        self.dtype = dtype

    def __len__(self):
        return self.scan_num

    def __getitem__(self, idx):
        scan_info = self.scan_list[idx]
        subject_name = scan_info["subject"]  # 011_S_0005
        scan_time = scan_info["scan_time"]  # 2005-09-02
        scan_path = os.path.join(self.data_dir, subject_name, scan_time, "t1.nii.gz")
        # load 3D scan
        scan = sitk.GetArrayFromImage(sitk.ReadImage(scan_path))  # 138, 176, 138
        scan = torch.from_numpy(scan).to(dtype=self.dtype)
        scan = (scan - scan.min()) / (scan.max() - scan.min())  # min-max norm
        scan = F.pad(scan, pad=(3, 3, 0, 0, 3, 3), mode="constant", value=0)  # 144, 176, 144
        if self.scale:
            scan = scan * 2.0 - 1.0  # from [0, 1] to [-1, 1]
        scan = torch.unsqueeze(scan, dim=0)  # 1, 144, 176, 144
        return scan


class ADNI_3D_NumCond_Set(Dataset):
    """
    Dataset for ADNI 3D data with numeric condition (age, gend)
    """

    def __init__(self, data_dir, scan_list, scale=True, dtype=torch.float32):
        self.data_dir = data_dir
        self.scan_list = scan_list
        self.scan_num = len(scan_list)
        self.scale = scale
        self.dtype = dtype

    def __len__(self):
        return self.scan_num

    def __getitem__(self, idx):
        scan_info = self.scan_list[idx]
        subject_name = scan_info["subject"]  # 011_S_0005
        scan_time = scan_info["scan_time"]  # 2005-09-02
        scan_path = os.path.join(self.data_dir, subject_name, scan_time, "t1.nii.gz")
        # load 3D scan
        scan = sitk.GetArrayFromImage(sitk.ReadImage(scan_path))  # 138, 176, 138
        scan = torch.from_numpy(scan).to(dtype=self.dtype)
        scan = (scan - scan.min()) / (scan.max() - scan.min())  # min-max norm
        scan = F.pad(scan, pad=(3, 3, 0, 0, 3, 3), mode="constant", value=0)  # 144, 176, 144
        if self.scale:
            scan = scan * 2.0 - 1.0  # from [0, 1] to [-1, 1]
        scan = torch.unsqueeze(scan, dim=0)  # 1, 144, 176, 144
        # numeric condition
        scan_age = scan_info["scan_age"]  # 79.6
        scan_age = scan_age / 100.0
        scan_gend = scan_info["gender"]  # M
        if scan_gend == "F":
            gend = 0.0
        else:
            gend = 1.0
        num_cond = torch.tensor([scan_age, gend], dtype=self.dtype)
        return {"scan": scan, "num_cond": num_cond}


class ADNI_3D_128_Set(Dataset):
    """
    Dataset for ADNI 128 data
    """

    def __init__(self, data_dir, scan_list, synth_data_dir=None, synth_scan_list=None, dtype=torch.float32):
        self.data_dir = data_dir
        self.scan_list = scan_list
        self.scan_num = len(scan_list)
        self.total_scan_num = len(scan_list)
        self.synth_data_dir = synth_data_dir
        self.synth_scan_list = synth_scan_list
        if synth_scan_list is not None:
            self.total_scan_num += len(synth_scan_list)
        self.dtype = dtype

    def __len__(self):
        return self.total_scan_num

    def __getitem__(self, idx):
        if idx < self.scan_num:
            scan_info = self.scan_list[idx]
            subject_name = scan_info["subject"]  # 011_S_0005
            scan_time = scan_info["scan_time"]  # 2005-09-02
            scan_path = os.path.join(self.data_dir, subject_name, scan_time, "t1.pt")
        else:
            scan_info = self.synth_scan_list[idx - self.scan_num]
            subject_name = scan_info["subject"]
            scan_path = os.path.join(self.synth_data_dir, subject_name, "t1.pt")
        # load 3D scan
        scan = torch.load(scan_path)  # 104, 128, 104
        scan = scan.to(dtype=self.dtype)
        scan = torch.unsqueeze(scan, dim=0)  # 1, 104, 128, 104
        return scan


class ADNI_3D_128_NumCond_Set(Dataset):
    """
    Dataset for ADNI 128 data with numeric condition (age)
    """

    def __init__(self, data_dir, scan_list, synth_data_dir=None, synth_scan_list=None, dtype=torch.float32):
        self.data_dir = data_dir
        self.scan_list = scan_list
        self.scan_num = len(scan_list)
        self.total_scan_num = len(scan_list)
        self.synth_data_dir = synth_data_dir
        self.synth_scan_list = synth_scan_list
        if synth_scan_list is not None:
            self.total_scan_num += len(synth_scan_list)
        self.dtype = dtype

    def __len__(self):
        return self.total_scan_num

    def __getitem__(self, idx):
        if idx < self.scan_num:
            scan_info = self.scan_list[idx]
            subject_name = scan_info["subject"]  # 011_S_0005
            scan_time = scan_info["scan_time"]  # 2005-09-02
            scan_path = os.path.join(self.data_dir, subject_name, scan_time, "t1.pt")
        else:
            scan_info = self.synth_scan_list[idx - self.scan_num]
            subject_name = scan_info["subject"]
            scan_path = os.path.join(self.synth_data_dir, subject_name, "t1.pt")
        # load 3D scan
        scan = torch.load(scan_path)  # 104, 128, 104
        scan = scan.to(dtype=self.dtype)
        scan = torch.unsqueeze(scan, dim=0)  # 1, 104, 128, 104
        # numeric condition
        scan_age = scan_info["scan_age"]  # 79.6
        num_cond = torch.tensor([scan_age], dtype=self.dtype)
        return {"scan": scan, "num_cond": num_cond}


class ADNI_3D_128_NumEdge_Set(Dataset):
    """
    Dataset for ADNI 128 data with numeric condition (age, gend) and edge detection
    """

    def __init__(
        self,
        data_dir,
        scan_list,
        synth_data_dir=None,
        synth_scan_list=None,
        edge_name="t1_canny",
        edge_scale=False,
        dtype=torch.float32
    ):
        self.data_dir = data_dir
        self.scan_list = scan_list
        self.scan_num = len(scan_list)
        self.total_scan_num = len(scan_list)
        self.synth_data_dir = synth_data_dir
        self.synth_scan_list = synth_scan_list
        if synth_scan_list is not None:
            self.total_scan_num += len(synth_scan_list)
        self.edge_name = edge_name
        self.edge_scale = edge_scale
        self.dtype = dtype

    def __len__(self):
        return self.total_scan_num

    def __getitem__(self, idx):
        if idx < self.scan_num:
            scan_info = self.scan_list[idx]
            subject_name = scan_info["subject"]  # 011_S_0005
            scan_time = scan_info["scan_time"]  # 2005-09-02
            scan_dir = os.path.join(self.data_dir, subject_name, scan_time)
            scan_path = os.path.join(scan_dir, "t1.pt")
            edge_path = os.path.join(scan_dir, f"{self.edge_name}.pt")
        else:
            scan_info = self.synth_scan_list[idx - self.scan_num]
            subject_name = scan_info["subject"]
            scan_dir = os.path.join(self.synth_data_dir, subject_name)
            scan_path = os.path.join(scan_dir, "t1.pt")
            edge_path = os.path.join(scan_dir, f"{self.edge_name}.pt")
        # load 3D scan
        scan = torch.load(scan_path)  # 104, 128, 104
        scan = scan.to(dtype=self.dtype)
        scan = torch.unsqueeze(scan, dim=0)  # 1, 104, 128, 104
        # load edge detection
        edge = torch.load(edge_path)  # 104, 128, 104
        edge = edge.to(dtype=self.dtype)
        if self.edge_scale:
            edge = edge * 2.0 - 1.0  # from [0, 1] to [-1, 1]
        edge = torch.unsqueeze(edge, dim=0)  # 1, 104, 128, 104
        # numeric condition
        scan_age = scan_info["scan_age"]  # 79.6
        scan_age = scan_age / 100.0
        scan_gend = scan_info["gender"]  # M
        if scan_gend == "F":
            gend = 0.0
        else:
            gend = 1.0
        num_cond = torch.tensor([scan_age, gend], dtype=self.dtype)
        return {"scan": scan, "num_cond": num_cond, "edge": edge}


class EdgeSampler:
    def __init__(self, data_dir, sampler, edge_name="t1_canny", edge_scale=False, dtype=torch.float32):
        self.data_dir = data_dir
        self.sampler = sampler
        self.edge_name = edge_name
        self.edge_scale = edge_scale
        self.dtype = dtype

    def get_edge(self, gend, age, health_state=None):
        if gend != "F":
            gend = "M"

        scan_age = round(age)
        if scan_age < 65:
            scan_age = "60"
        elif scan_age >= 65 and scan_age < 75:
            scan_age = "70"
        elif scan_age >= 75 and scan_age < 85:
            scan_age = "80"
        elif scan_age >= 85:
            scan_age = "90"

        if health_state is not None:
            num = len(self.sampler[gend][health_state][scan_age])
            idx = np.random.randint(num)
            scan_info = self.sampler[gend][health_state][scan_age][idx]
        else:
            num = len(self.sampler[gend][scan_age])
            idx = np.random.randint(num)
            scan_info = self.sampler[gend][scan_age][idx]

        subject_name = scan_info["subject"]  # 011_S_0005
        scan_time = scan_info["scan_time"]  # 2005-09-02
        scan_dir = os.path.join(self.data_dir, subject_name, scan_time)
        edge_path = os.path.join(scan_dir, f"{self.edge_name}.pt")
        edge = torch.load(edge_path)  # 104, 128, 104
        edge = edge.to(dtype=self.dtype)
        if self.edge_scale:
            edge = edge * 2.0 - 1.0  # from [0, 1] to [-1, 1]

        return edge
