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

scenes = ['Undistorted_SfM/0015/images','Undistorted_SfM/0022/images'] if opt.dataset == 'megadepth' else os.listdir(opt.dataset_dir)

# use the trained model to encode the input embeddings
model = gluefactory.load_experiment(opt.model).to(opt.device).eval()

for scene in scenes:
    print("start testing on scene:", scene)
    if opt.dataset == 'eth3d': scene = f"{scene}/images/dslr_images_undistorted"

    with h5py.File(str(overlap_features), 'r') as hfile:
        image_list = [f"{scene}/{f}" for f in hfile[scene].keys()]

    N = len(image_list)
    if N < opt.pre_filter: opt.pre_filter //=  2

    output_dir = Path('outputs') / opt.dataset / scene / opt.model
    output_dir.mkdir(exist_ok=True, parents=True)

    if opt.cls:
        output_dir = output_dir / Path('cls_' + str(opt.pre_filter))
        output_dir.mkdir(exist_ok=True)
        scores = np.load(output_dir / Path(str(opt.radius) + '_results_w.npz'))['votings'][opt.vote] if opt.weighted else np.load(output_dir / Path(str(radius) + '_results.npz'))['votings'][opt.vote]

    if opt.cls:
        mask = np.load(output_dir / Path("results_cls.npz"))['scores']
        # import pdb; pdb.set_trace()
        mask_ = torch.zeros_like(torch.from_numpy(scores))
        mask_.scatter_(dim=1, index=torch.topk(torch.from_numpy(mask), opt.pre_filter).indices, src=torch.ones_like(mask_))
        voting_topk = torch.topk(torch.from_numpy(scores) * mask_, opt.k)
    else:
        voting_topk = torch.topk(torch.from_numpy(scores), opt.k)

    recall_results = output_dir / Path('recall_results_w_auc.txt') if opt.weighted else output_dir  / Path("recall_results_auc.txt")
    # TODO: fix the indices
    gt_score = np.load(str(output_dir).replace(opt.model, '').replace(f"cls_{opt.pre_filter}", '') + 'overlap_gt.npz')["scores_gt"] if scene != '0022' and str(output_dir).split("/")[-1] != '02' else np.load(f'outputs/{scene}/overlap_gt.npz')['scores_gt']
    gt_topk = torch.topk(torch.from_numpy(gt_score), opt.k)
    with open(recall_results, "a") as doc:
            doc.write(f"{scene} {opt.model} {opt.radius} {opt.vote} recall@{opt.k} {recall(voting_topk, gt_topk)} \n")
