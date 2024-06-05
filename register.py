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

    if opt.cls: 
        output_dir = output_dir / Path('cls_' + str(opt.pre_filter))
        output_dir.mkdir(exist_ok=True)
    
    # load the dumped data, prepare for tests
    dataset = SampleDataset(overlap_features, image_list)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    all_des = []
    cls_tokens = []
    for d in tqdm(data_loader):
        batch_data_cuda = {k: v.cuda() for k, v in d.items()}
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
    if not os.path.exists(output_dir / Path("results_cls.npz")):
        cls_scores = torch.einsum('id,jd->ij', cls_tokens, cls_tokens)
        diagonal = torch.eye(N, N).bool().to(cls_scores.device)
        cls_scores.masked_fill_(diagonal, -torch.inf)
        np.savez(output_dir / Path("results_cls.npz"), **{'scores':cls_scores.cpu().numpy()})

    # Step 2: patch-level retrieval: radius search + votings
    mask = np.load(output_dir / Path("results_cls.npz"))['scores']
    filtered_indices = torch.topk(torch.from_numpy(mask), opt.pre_filter).indices
    
    for radius in [opt.radius]:
        query_fun = Voting(opt.radius, num_patches=all_des.shape[1], weighted=opt.weighted)
        retrieved = query_fun.query_each(all_des, filtered_indices) if opt.cls else query_fun.query_each(all_des)
        np.savez(output_dir / Path(str(radius) + "_results_w.npz"), **retrieved)
        print(f"successfully test {opt.model} on ds {opt.dataset} with radius of {radius}.")  

    # Step 3: save the retrieved image list 
    overlap_pairs = output_dir / Path('top' + str(opt.k) +'_overlap_pairs_w_auc.txt') if opt.weighted else output_dir / Path('top' + str(opt.k) +'_overlap_pairs_auc.txt')
    recall_results = output_dir / Path('recall_results_w_auc.txt') if opt.weighted else output_dir  / Path("recall_results_auc.txt")

    scores = np.load(output_dir / Path(str(opt.radius) + '_results_w.npz'))['votings'] if opt.weighted else np.load(output_dir / Path(str(radius) + '_results.npz'))['votings']
    assert scores.shape[0] > opt.radius
    voting_topk = torch.topk(torch.from_numpy(scores[opt.vote]), opt.k)

    if not os.path.exists(overlap_pairs):
        with open(overlap_pairs, "w") as doc:
            for i, name in enumerate(image_list):
                for j in voting_topk.indices[i]:
                    pairs_i = image_list[j]
                    doc.write(f"{name} {pairs_i}\n")

    gt_score = np.load(str(output_dir).replace(opt.model, '').replace(f"cls_{opt.pre_filter}", '') + 'overlap_gt.npz')["scores_gt"] if scene != '0022' and str(output_dir).split("/")[-1] != '02' else np.load(f'outputs/{scene}/overlap_gt.npz')['scores_gt']
    gt_topk = torch.topk(torch.from_numpy(gt_score), opt.k)
    with open(recall_results, "a") as doc: 
            doc.write(f"{scene} {opt.model} {opt.radius} {opt.vote} recall@{opt.k} {recall(voting_topk, gt_topk)} \n")





