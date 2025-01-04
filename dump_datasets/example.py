
import os
import h5py
import torch
import gluefactory
import numpy as np
from tqdm import tqdm
from pathlib import Path
from args import *
from lightglue.utils import load_image

def dump(opt):

    # path to save the preprocessed data
    overlap_features = Path(opt.dump_dir)/ 'example' / 'overlap_feats.h5'
    (Path(opt.dump_dir) / 'example').mkdir(exist_ok=True, parents=True)
    modelconf = {}

    # load a model with frozen DINOv2 backbone, for patchfying the imgaes
    model = gluefactory.load_experiment(opt.model, conf=modelconf).cuda().eval()
    if not os.path.exists(overlap_features) or opt.overwrite:
        with h5py.File(str(overlap_features), 'w') as hfile:
            for scene in os.listdir(opt.dataset_dir):
                # load images and run DINOv2 backbone
                dataset_dir = os.path.join(opt.dataset_dir, scene, 'images')
                if not os.path.isdir(dataset_dir):
                    continue
                image_list = os.listdir(dataset_dir)
                for image_path in tqdm(image_list):
                    image_ = load_image(dataset_dir +'/' + image_path, resize=opt.imsize)
                    c, h, w = image_.shape
                    image = torch.zeros((3, opt.imsize, opt.imsize), device=opt.device, dtype=opt.dtype)
                    image[:, :h, :w] = image_
                    feats = model.extractor({'image': image[None]})
                    group = hfile.create_group(scene+'/images/' +image_path)
                    for key in ['keypoints', 'descriptors', 'global_descriptor']:
                        group.create_dataset(key, data=feats[key][0].detach().cpu().numpy())
                    group.create_dataset('image_size', data=[w, h])

                # enough for retrieval process, after dumping the data, running register/retrieve.py will return image pair lists
                # If pose estimation evaluation, pls load GT pose information
