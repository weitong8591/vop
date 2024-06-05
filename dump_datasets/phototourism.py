
import os
import h5py
import torch
import gluefactory
import numpy as np
from tqdm import tqdm
from pathlib import Path
from args import *
from utils import loadh5
from collections import defaultdict
from lightglue.utils import load_image
from gluefactory.datasets.utils import scale_intrinsics

def dump(opt):

    overlap_features = Path(opt.dump_dir)/ 'aachen' / 'overlap_feats.h5'
    Path(opt.dump_dir).mkdir(exist_ok=True)
    (Path(opt.dump_dir) / 'aachen').mkdir(exist_ok=True)
    modelconf = {}
    model = gluefactory.load_experiment(opt.model, conf=modelconf).cuda().eval()
    if not os.path.exists(overlap_features) or opt.overwrite:
        with h5py.File(str(overlap_features), 'w') as hfile:
            for scene in os.listdir(opt.dataset_dir):
                if not os.path.exists(os.path.join(opt.dataset_dir, scene, 'K1_K2.h5')):
                    continue
                K0K1s = loadh5(os.path.join(opt.dataset_dir, scene, 'K1_K2.h5'))
                Rs = loadh5(os.path.join(opt.dataset_dir, scene, 'R.h5'))
                Ts = loadh5(os.path.join(opt.dataset_dir, scene, 'T.h5'))

                Ks = defaultdict()
                for name0_1 in K0K1s.keys():
                    name0, name1 = name0_1.split('-')
                    if name0 not in Ks.keys():
                        Ks[name0] = K0K1s[name0_1][0][0]
                    if name1 not in Ks.keys():
                        Ks[name1] = K0K1s[name0_1][0][1]

                dataset_dir = os.path.join(opt.dataset_dir, scene, 'images/')
                image_list = [i for i in Rs.keys()]

                # Read intrinsics and extrinsics data
                for image_path in tqdm(image_list):
                    image_ = load_image(str(dataset_dir /image_path) + str('.jpg'), resize=opt.imsize)
                    c, h, w = image_.shape
                    image = torch.zeros((3, opt.imsize, opt.imsize), device=opt.device, dtype=opt.dtype)
                    image[:, :h, :w] = image_
                    feats = model.extractor({'image': image[None]})
                    group = hfile.create_group(image_path+str('.jpg'))
                    for key in ['keypoints', 'descriptors', 'global_descriptor']:
                        group.create_dataset(key, data=feats[key][0].detach().cpu().numpy())
                    ori_h, ori_w = load_image(str(dataset_dir /image_path) + str('.jpg')).shape[-2:]
                    original_image_size = np.array([ori_w, ori_h])
                    scales = opt.imsize/original_image_size
                    ori_K = Ks[image_path]
                    K = scale_intrinsics(ori_K, scales)
                    T_w2cam = np.zeros((4, 4))
                    T_w2cam[-1, -1] = 1
                    T_w2cam[:3, :3] = Rs[image_path]
                    T_w2cam[:-1, -1] = Ts[image_path].T[0]

                    group.create_dataset("K", data=K)
                    group.create_dataset("original_K", data=ori_K)
                    group.create_dataset("T_w2cam", data=T_w2cam)
                    group.create_dataset('image_size', data=[w, h])