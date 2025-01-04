from lightglue.utils import load_image
import gluefactory
from gluefactory.datasets.utils import scale_intrinsics
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from args import *
from utils import quaternion_to_rotation_matrix, camera_center_to_translation, Camera

def dump(opt, model):

    overlap_features = Path(opt.dump_dir)/ 'inloc' / 'overlap_feats.h5'
    (Path(opt.dump_dir)/ opt.dataset).mkdir(exist_ok=True, parents=True)

    # save intrinsic and extrinsic parameters for each image, from world to camera
    with h5py.File(str(overlap_features), 'w') as hfile:
        query_list = [f"query/iphone7/" + q  for q in os.listdir(opt.dataset_dir + "/query/iphone7/") if q !='LICENSE.txt']
        db_list = []
        for scene in ['DUC1', 'DUC2']:
            if scene != 'LICENSE.txt':
                for folder in os.listdir(opt.dataset_dir + "/database/cutouts/" + scene):
                    db_list += [f"database/cutouts/{scene}/{folder}/{f}" for f in os.listdir(f"{opt.dataset_dir}/database/cutouts/{scene}/{folder}") if f.endswith(".jpg")]
        image_list = np.concatenate((query_list, db_list))

        # save the indices of query and db images
        group = hfile.create_group("indices")
        group.create_dataset("query", data=query_list, dtype=h5py.string_dtype(encoding='utf-8'))
        group.create_dataset("db", data=db_list, dtype=h5py.string_dtype(encoding='utf-8'))
        print(f"in total {len(query_list)} queries and {len(db_list)} db images")

        # save the images, embeddings
        for image_path in tqdm(image_list):
            image_ = load_image(opt.dataset_dir + image_path, resize=opt.imsize)
            c, h, w = image_.shape
            image = torch.zeros((3, opt.imsize, opt.imsize), device=opt.device, dtype=opt.dtype)
            image[:, :h, :w] = image_
            feats = model.extractor({'image': image[None]})
            group = hfile.create_group(image_path)
            for key in ['keypoints', 'descriptors', 'global_descriptor']:
                group.create_dataset(key, data=feats[key][0].detach().cpu().numpy())
            ori_h, ori_w = load_image(opt.dataset_dir + image_path).shape[-2:]
            original_image_size = np.array([ori_w, ori_h])
            scales = opt.imsize/original_image_size
            group.create_dataset('image_size', data=[w, h])
        print(f"finished." )
