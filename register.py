import os
import torch
import numpy as np
from args import *
from utils import *
import gluefactory
from tqdm import tqdm
from pathlib import Path
from evaluate_utils import *
from omegaconf import OmegaConf
from datasets import SampleDataset
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity

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

if not os.path.exists(f"outputs/training/{opt.model}/checkpoint_best.tar"):
    download_best(opt.model)

# load the trained model to encode the input embeddings
model = gluefactory.load_experiment(opt.model).to(opt.device).eval()

all_scores = {}
for scene in scenes:
    print("start testing on scene:", scene)
    if opt.dataset == 'eth3d': scene = f"{scene}/images/dslr_images_undistorted"
    if opt.dataset == 'phototourism' or opt.dataset == 'imc2023': scene = f"{scene}/images/"
    if not (opt.dataset_dir/scene).is_dir():
        print(f"skipped scene:{scene}")
        continue

    with h5py.File(str(overlap_features), 'r') as hfile:
        image_list = [f"{scene}/{f}" for f in hfile[scene].keys()]

    N = len(image_list)
    pre_filter = min(N, opt.pre_filter)
    print(scene, N)
    output_dir = Path(opt.output_dir) / opt.model / opt.dataset / scene

    if opt.cls:
        output_dir = output_dir / Path('cls_' + str(opt.pre_filter))
    output_dir.mkdir(exist_ok=True, parents=True)

    # load the dumped data, prepare for tests
    dataset = SampleDataset(overlap_features, image_list, opt.dataset)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    all_des = []
    cls_tokens = []
    for d in tqdm(data_loader):
        batch_data_cuda = {k: v.to(opt.device) for k, v in d.items()}
        pred = model.matcher({**batch_data_cuda})
        if model.conf.matcher['add_cls_tokens']:
            des0 = pred['desc0'][:, 1:, :]
            cls_tokens0 = pred['desc0'][:, 0, :]
        else:
            des0 = pred['desc0']
            cls_tokens0 = batch_data_cuda['global_descriptor0']
        all_des += [des0]
        cls_tokens += [cls_tokens0]

    all_des = torch.concat(all_des)
    cls_tokens = torch.concat(cls_tokens)

    # Step 1 (optional): global retrieval
    if not os.path.exists(output_dir / Path("results_cls.npz")) or opt.overwrite:
        cls_scores = torch.einsum('id,jd->ij', cls_tokens, cls_tokens)
        diagonal = torch.eye(N, N).bool().to(cls_scores.device)
        cls_scores.masked_fill_(diagonal, -torch.inf)
        np.savez(output_dir / Path("results_cls.npz"), **{'scores':cls_scores.cpu().numpy()})
    else:
        cls_scores = torch.from_numpy(np.load(output_dir / Path("results_cls.npz"))['scores'])

    # Step 2: patch-level retrieval: radius search + weighted voting
    mask = np.load(output_dir / Path("results_cls.npz"))['scores']

    for radius in [opt.radius]:
        if radius == -1:
            # automatically choose a radius based on 100 random samples
            torch.manual_seed(42)
            random_indices = torch.randperm(all_des.size(0))[:100]
            sample_des = all_des[random_indices]
            similarities = cosine_similarity(sample_des.unsqueeze(2), sample_des.unsqueeze(1))
            radius = round((torch.round(similarities.median() * 10) / 10).item(), 2)#torch.ceil(
            with open(output_dir / "median_radius.txt",'w') as txtfile:
                txtfile.write(f"{radius}\n")

        query_fun = Voting(radius, num_patches=all_des.shape[1], weighted=opt.weighted)
        retrieved = query_fun.query_each(all_des)
        np.savez(output_dir / Path(str(radius) + "_results.npz"), **retrieved)
        print(f"successfully test {opt.model} on {scene} with radius of {radius}.")

        # Step 3: save the retrieved image list
        overlap_pairs = output_dir / Path(f"top{opt.k}_{radius}_overlap_pairs.txt")
        scores = np.load(output_dir / Path(str(radius) + '_results.npz'))['scores']

        all_scores[scene] = {
            'weighted': scores[1],
            'scores': scores[0],
            'pre_filter': pre_filter,
            'cls_scores': cls_scores,
            'overlap_pairs': overlap_pairs,
            'image_list': image_list}

# check if the tf-idf weights make sense for the dataset, not using it if half of the scenes with 0 weights (all db images containing a neighbor patch)
score_key = 'weighted' if np.sum([(all_scores[scene]['weighted']==0).all() or opt.weighted==False for scene in all_scores.keys()]) < len(scenes) // 2 else 'scores'

for scene in all_scores.keys():
    scores = all_scores[scene][score_key]
    k = min(opt.k, all_scores[scene]['pre_filter'])

    if opt.cls:
        mask_ = torch.zeros_like(torch.from_numpy(scores))
        mask_.scatter_(dim=1, index=torch.topk(all_scores[scene]['cls_scores'].to(mask_.device), all_scores[scene]['pre_filter']).indices, src=torch.ones_like(mask_))
        voting_topk = torch.topk(torch.from_numpy(scores) * mask_, k)
    else:
        voting_topk = torch.topk(torch.from_numpy(scores), k)

    assert scores.shape[0] > radius

    if not os.path.exists(all_scores[scene]['overlap_pairs']) or opt.overwrite:
        with open(all_scores[scene]['overlap_pairs'], "w") as doc:
            for i, name in enumerate(all_scores[scene]['image_list']):
                for j in voting_topk.indices[i]:
                    pairs_i = all_scores[scene]['image_list'][j]
                    doc.write(f"{name} {pairs_i}\n")
