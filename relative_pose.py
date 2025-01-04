import os
import torch
import numpy as np
import gluefactory
from pathlib import Path
from omegaconf import OmegaConf
from hloc import extract_features, match_features
from evaluate_utils import *
from args import *

opt = create_parser()

# check if gpu device is available
opt.device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
torch.set_grad_enabled(False)

# load the data dirs for all
dataset_dirs = OmegaConf.load('dump_datasets/data_dirs.yaml')
opt.dataset_dir = Path(dataset_dirs.get('dataset_dirs')[opt.dataset])
# load the query and db lists in the dumped data
overlap_features = Path(opt.dump_dir) / opt.dataset / "overlap_feats.h5"
avg_results = Path(opt.output_dir) / opt.model / "avg_results.txt"
assert os.path.exists(overlap_features)

scenes = ['Undistorted_SfM/0015/images','Undistorted_SfM/0022/images'] if opt.dataset == 'megadepth' else os.listdir(opt.dataset_dir)

all = []
all_num_matches = []

for scene in scenes:
    print("Start testing on scene:", scene)
    if opt.dataset == 'eth3d': scene = f"{scene}/images/dslr_images_undistorted"
    if opt.dataset == 'phototourism': scene = f"{scene}/images/"
    if not (opt.dataset_dir/scene).is_dir():
        print(f"skipped scene:{scene}")
        continue

    with h5py.File(str(overlap_features), 'r') as hfile:
        image_list = [f"{scene}/{f}" for f in hfile[scene].keys()]

    output_dir = Path(opt.output_dir) / opt.model / opt.dataset / scene
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = extract_features.confs['superpoint_aachen']
    features = output_dir / 'features.h5'
    if not os.path.exists(features) or opt.overwrite:
        extract_features.main(conf, opt.dataset_dir, feature_path=features, image_list=image_list)

    if opt.cls:
        output_dir = output_dir / ('cls_' + str(opt.pre_filter))
        output_dir.mkdir(exist_ok=True)

    baseline_helper = BaselineHelper(image_list)
    radius = np.loadtxt(output_dir / "median_radius.txt").item() if opt.radius == -1 else opt.radius

    matches = output_dir / (f"top{opt.k}_{radius}_overlap_matches.h5")
    overlap_pairs = output_dir / Path(f"top{opt.k}_{radius}_overlap_pairs.txt")
    overlap_results = output_dir  / "overlap_results.txt"

    if os.path.exists(overlap_pairs):
        if not os.path.exists(matches) or opt.overwrite:
            # local feature matching
            match_conf = match_features.confs['superpoint+lightglue']
            match_features.main(match_conf, overlap_pairs, features, matches=matches, overwrite=opt.overwrite, unique_pairs=False)
        # check the number of matches in the retrieved image pairs
        pred_matches, _ = baseline_helper.match_after_retrival(matches, overlap_pairs)
        # relative pose estimation on the retrieved image pairs
        pred_pose_results, best_thre = baseline_helper.relative_pose_after_retrieval(matches, overlap_pairs, features, overlap_features)
        pose_results = [pred_pose_results[key] for key in ['rel_pose_error@10°', 'mrel_pose_error', 'mransac_inl']]#pred_pose_results.keys()]
        print("Finished test on scene:", scene)

        with open(overlap_results, "a") as doc:
            doc.write(f"{scene} recall@{opt.k} {radius} {[pred_matches[key] for key in pred_matches.keys()][0]} " + " ".join(map(str, pose_results[:16])) + " \n")
            all += [pose_results]
            all_num_matches += [pred_matches[key] for key in pred_matches.keys()]
    else:
        print(f"Skipped scene:{scene}, as no pairs provided in {overlap_pairs}!")
        continue

with open(avg_results, "a") as doc:
    doc.write(f"{opt.dataset} {len(scenes)} {len(all)} recall@{opt.k} {opt.radius} {np.mean(all_num_matches)} " + " ".join(map(str, np.vstack(all).mean(0))) + " \n")
