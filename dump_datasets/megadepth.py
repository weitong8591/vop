import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

import gluefactory
from lightglue.utils import load_image
from gluefactory.datasets.utils import scale_intrinsics
from gluefactory.utils.image import ImagePreprocessor

torch.set_grad_enabled(False)

def dump(opt):
    opt.dataset_dir = Path(opt.dataset_dir)
    modelconf = {}
    model = gluefactory.load_experiment(opt.model, conf=modelconf).cuda().eval()

    def get_imagedata(name, suffix):
        idx = list(scene_info["image_paths"]).index(name)
        depth_process = ImagePreprocessor({'resize': opt.imsize, "side": "long", "square_pad": True})
        dpath = opt.dataset_dir / scene_info["depth_paths"][idx]
        with h5py.File(str(dpath), "r") as f:
            depth = f["/depth"].__array__().astype(np.float32, copy=False)
            depth = torch.Tensor(depth)[None]
        ddata = depth_process(depth, interpolation="nearest")
        K = scene_info['intrinsics'][idx]
        K = scale_intrinsics(K, ddata["scales"])
        return {
            'scales'+suffix: ddata["scales"],
            'original_image_size': torch.from_numpy(ddata["original_image_size"]),
            'original_K': torch.from_numpy(scene_info['intrinsics'][idx]),
            'depth'+suffix: ddata["image"][0],
            'K'+suffix: torch.tensor(K).float(),
            'T_w2cam'+suffix: torch.tensor(scene_info['poses'][idx]).float()
        }
    
    overlap_features = Path(opt.dump_dir)/ 'megadepth' / 'overlap_feats.h5'
    Path(opt.dump_dir).mkdir(exist_ok=True)
    (Path(opt.dump_dir) / 'megadepth').mkdir(exist_ok=True)
    
    import pdb; pdb.set_trace()
    if not os.path.exists(overlap_features) or opt.overwrite:
        with h5py.File(str(overlap_features), 'w') as hfile:
            for scene in ['0015','0022']:
                scene_file = opt.dataset_dir / 'scene_info' / (scene+'.npz')
                scene_info = np.load(scene_file, allow_pickle=True)
                image_list = [image for image in scene_info['image_paths'] if image is not None]
                for image_path in tqdm(image_list):
                    image_ = load_image(opt.dataset_dir / image_path, resize=opt.imsize)
                    c, h, w = image_.shape
                    image = torch.zeros((3, opt.imsize, opt.imsize), device=opt.device, dtype=opt.dtype)
                    image[:, :h, :w] = image_
                    feats = model.extractor({'image': image[None]})
                    group = hfile.create_group(image_path)
                    for key in ['keypoints', 'descriptors', 'global_descriptor']:
                        group.create_dataset(key, data=feats[key][0].detach().cpu().numpy())
                    imdata = get_imagedata(image_path, "")
                    for key, v in imdata.items():
                        group.create_dataset(key, data=v.numpy())
                    group.create_dataset('image_size', data=[w, h])
            print(f"finished." )

