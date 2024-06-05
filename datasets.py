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

class Dataset(data.Dataset):
    """Load images and extrinsic matrices."""
    def __init__(self, dataset_dir, dataset, query_list=None, data_base=None, device='cuda:0'):

        scenes = os.listdir(dataset_dir)
        image_list = []
        for scene in scenes:
            image_list += [scene + '/' + image for image in os.listdir(dataset_dir / scene / "images/dslr_images_undistorted")]
        if len(image_list) % 2 != 0: image_list += [image_list[-1]]
        # load all features
        self.data = defaultdict(list)
        overlap_features = Path("dumped_data") / dataset / "overlap_feats.h5"
        assert os.path.exists(overlap_features)
        with h5py.File(str(overlap_features), 'r') as hfile:
            for _, name in enumerate(image_list):
                self.data[name] = hfile[name]
                import pdb; pdb.set_trace()
                for k in hfile[name].keys():
                    self.data[k] += [torch.Tensor(hfile[name][k].__array__()).to(device)[None]]
        self.files = image_list
        self.query_list = query_list if query_list is not None else image_list[:len(image_list)//2]
        self.data_base = data_base if data_base is not None else image_list[len(image_list)//2:]

    def __len__(self):
        return len(self.files)//2

    def __getitem__(self, index):
        #
        import pdb; pdb.set_trace()
        input_data0 = {self.query_list[index] + '0': torch.concat(self.data[self.query_list[index]])}
        input_data1 = {self.data_base[index] + '1': torch.concat(self.data[self.data_base[index]])}
        data = {**input_data0, **input_data1}


        return data
