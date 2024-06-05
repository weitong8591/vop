import torch
import os
import numpy as np
import gluefactory
from tqdm import tqdm
from pathlib import Path
from hloc import extract_features
from evaluate_utils import *
from args import *
from omegaconf import OmegaConf
from datasets import SampleDataset
from torch.utils.data import DataLoader

opt = create_parser()

# check if gpu device is available
opt.device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
torch.set_grad_enabled(False)

# load the data dirs for all 
dataset_dirs = OmegaConf.load('dump_datasets/data_dirs.yaml')
opt.dataset_dir = Path(dataset_dirs.get('dataset_dirs')[opt.dataset])
# load the query and db lists in the dumped data
overlap_features = Path(opt.dump_dir) / opt.dataset / "overlap_feats.h5"
assert os.path.exists(overlap_features)

scenes = ['0015','0022'] if opt.dataset == 'megadepth' else os.listdir(opt.dataset_dir)

# use the trained model to encode the input embeddings
model = gluefactory.load_experiment(opt.model).to(opt.device).eval()

for scene in scenes:
    scene_file = opt.dataset_dir / 'scene_info' / (scene+'.npz')
    output_dir = Path('outputs') / opt.dataset / scene / opt.model 
    output_dir.mkdir(exist_ok=True, parents=True)

    print("start testing on scene:", scene)
    scene_info = np.load(scene_file, allow_pickle=True)
    list(scene_info.keys())
    image_list = [image for image in scene_info['image_paths'] if image is not None]
    N = len(image_list)

    conf = extract_features.confs['superpoint_aachen']
    features = output_dir / 'features.h5'
    if not os.path.exists(features):
        extract_features.main(conf, opt.dataset_dir, feature_path=features, image_list=image_list)
    if opt.cls: 
        output_dir = output_dir / Path('cls_' + str(opt.pre_filter))
        output_dir.mkdir(exist_ok=True)

    baseline_helper = BaselineHelper(image_list)
    matches = output_dir / Path("top"+ str(opt.k) + "_overlap_matches_w_auc.h5") if opt.weighted else output_dir / Path("top"+ str(opt.k) + "_overlap_matches_auc.h5")   
    overlap_pairs = output_dir / Path('top' + str(opt.k) +'_overlap_pairs_w_auc.txt') if opt.weighted else output_dir / Path('top' + str(opt.k) +'_overlap_pairs_auc.txt')
    overlap_results = output_dir  / Path("overlap_results_w_auc.txt") if opt.weighted else output_dir  / Path("overlap_results_auc.txt")

    if os.path.exists(overlap_pairs):
        if not os.path.exists(matches):
            # local feature matching
            match_conf = match_features.confs['superpoint+lightglue']
            match_features.main(match_conf, overlap_pairs, features, matches=matches, overwrite=opt.overwrite, unique_pairs=False)

        # check the number of matches in the retrieved image pairs    
        pred_matches, _ = baseline_helper.match_after_retrival(matches, overlap_pairs)
        pred_pose_results, best_thre = baseline_helper.relative_pose_after_retrieval(matches, overlap_pairs, features, overlap_features)
        # relative pose estimation on the retrieved image pairs 
        pose_results = [pred_pose_results[key] for key in pred_pose_results.keys()]
        print('Overlap voting', scene, [pred_matches[key] for key in pred_matches.keys()][0], pose_results)
        with open(overlap_results, "a") as doc: 
             doc.write(f"{scene} recall@{opt.k} {opt.radius} {opt.vote} {[pred_matches[key] for key in pred_matches.keys()][0]} " + " ".join(map(str, pose_results[:16])) + " \n")
    