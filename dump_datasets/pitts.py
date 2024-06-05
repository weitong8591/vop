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

def dump(opt):

    overlap_features = Path(opt.dump_dir)/ 'pitts30k' / 'overlap_feats.h5'
    Path(opt.dump_dir).mkdir(exist_ok=True)
    (Path(opt.dump_dir)/ 'pitts30k').mkdir(exist_ok=True)
    modelconf = {}
    model = gluefactory.load_experiment(opt.model, conf=modelconf).cuda().eval()

    # load the query and db list, refer to https://github.com/serizba/salad/tree/main/datasets/Pittsburgh
    query_list = np.load(Path(opt.dataset_dir) / "pitts30k_test_qImages.npy")
    db_list = np.load(Path(opt.dataset_dir) / "pitts30k_test_dbImages.npy")

    image_list = np.concatenate((query_list,db_list))
    if not os.path.exists(overlap_features) or opt.overwrite:
        with h5py.File(str(overlap_features), 'w') as hfile:

            # save the indices of query and db images
            group = hfile.create_group("indices")
            group.create_dataset("query", data=[str(q) for q in query_list])
            group.create_dataset("db", data=[str(q) for q in db_list])
            print(f"in total {len(query_list)} queries and {len(db_list)} db images")
            
            # save the images, embeddings
            for image_path in tqdm(image_list):
                image_ = load_image(Path(opt.dataset_dir) / image_path, resize=opt.imsize)
                c, h, w = image_.shape
                image = torch.zeros((3, opt.imsize, opt.imsize), device=opt.device, dtype=opt.dtype)
                image[:, :h, :w] = image_
                feats = model.extractor({'image': image[None]})
                group = hfile.create_group(image_path)
                for key in ['keypoints', 'descriptors', 'global_descriptor']:
                    group.create_dataset(key, data=feats[key][0].detach().cpu().numpy())
                ori_h, ori_w = load_image(Path(opt.dataset_dir) / image_path).shape[-2:]
                original_image_size = np.array([ori_w, ori_h])
                scales = opt.imsize/original_image_size
                group.create_dataset('image_size', data=[w, h])
