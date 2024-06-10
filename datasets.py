import math
import os
import cv2
import h5py
import torch
import numpy as np
import torch.utils.data as data
from pathlib import Path
from lightglue.utils import load_image
from gluefactory.datasets.eth3d import read_cameras, qvec2rotmat
from gluefactory.datasets.utils import scale_intrinsics
from collections import defaultdict
from torch.utils.data import Dataset



class SampleDataset(Dataset):
    def __init__(self, data, image_list, dataset='megadepth'):

        self.overlap_features = data
        self.image_list = image_list
        self.dataset = dataset

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            sample (dict): A dictionary containing the input and target data for the sample.
        """
        data = defaultdict(list)

        with h5py.File(str(self.overlap_features), 'r') as hfile:
            for k in hfile[self.image_list[idx]].keys():
                data[k] = [torch.Tensor(hfile[self.image_list[idx]][k].__array__())]

        input_data0 = {k+'0': torch.concat(data[k]) for k in data.keys()}
        input_data1 = {k+'1': torch.concat(data[k]) for k in data.keys()}
        return {**input_data0, **input_data1}
