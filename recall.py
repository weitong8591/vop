import torch
import os
import numpy as np
import gluefactory
from pathlib import Path
from evaluate_utils import *
from args import *
from omegaconf import OmegaConf
import requests
import zipfile

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

url = "https://cmp.felk.cvut.cz/~weitong/vop/gt_score_megadepth.zip"
local_zip_path = "gt_score_megadepth.zip"
extract_path = opt.dump_dir /opt.dataset/'gt'

if not os.path.exists(extract_path):
    response = requests.get(url)
    with open(local_zip_path, 'wb') as file:
        file.write(response.content)

    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(local_zip_path)
    print(f"successfully download {extract_path}!")
else:
    print(f"{extract_path} is ready!")

avg_recall = Path('outputs') / opt.model / "avg_recall.txt"

all_recall = []
for scene in scenes:

    print("start testing on scene:", scene)
    with h5py.File(str(overlap_features), 'r') as hfile:
        image_list = [f"{scene}/{f}" for f in hfile[scene].keys()]

    N = len(image_list)
    if N < opt.pre_filter: opt.pre_filter //=  2

    output_dir = Path('outputs') / opt.model / opt.dataset / scene
    output_dir.mkdir(exist_ok=True, parents=True)

    if opt.cls:
        output_dir = output_dir / Path('cls_' + str(opt.pre_filter))
        output_dir.mkdir(exist_ok=True)
    scores = np.load(output_dir / Path(str(opt.radius) + '_results_w.npz'))['votings'][opt.vote] if opt.weighted else np.load(output_dir / Path(str(opt.radius) + '_results.npz'))['votings'][opt.vote]

    if opt.cls:
        mask = np.load(output_dir / Path("results_cls.npz"))['scores']
        mask_ = torch.zeros_like(torch.from_numpy(scores))
        mask_.scatter_(dim=1, index=torch.topk(torch.from_numpy(mask), opt.pre_filter).indices, src=torch.ones_like(mask_))
        voting_topk = torch.topk(torch.from_numpy(scores) * mask_, opt.k)
    else:
        voting_topk = torch.topk(torch.from_numpy(scores), opt.k)

    recall_results = output_dir / Path('recall_results_w_auc.txt') if opt.weighted else output_dir  / Path("recall_results_auc.txt")
    gt_score = np.load(opt.dump_dir /opt.dataset/ f"gt/gt_score_{scene.split('/')[1]}.npy")
    gt_topk = torch.topk(torch.from_numpy(gt_score), opt.k)
    recall_score = recall(voting_topk, gt_topk)
    all_recall += [recall_score]
    with open(recall_results, "a") as doc:
            doc.write(f"{scene} {opt.model} {opt.radius} {opt.vote} recall@{opt.k} {recall_score} \n")

with open(avg_recall, "a") as doc:
    doc.write(f"{opt.dataset} recall@{opt.k} {opt.radius} {opt.vote} {np.mean(all_recall)} \n")
