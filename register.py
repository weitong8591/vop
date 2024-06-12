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

    # load the dumped data, prepare for tests
    dataset = SampleDataset(overlap_features, image_list, opt.dataset)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    all_des = []
    cls_tokens = []
    for d in tqdm(data_loader):
        batch_data_cuda = {k: v.cuda() for k, v in d.items()}
        import pdb; pdb.set_trace()
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

    # Step 2: patch-level retrieval: radius search + votings
    mask = np.load(output_dir / Path("results_cls.npz"))['scores']
    # import pdb; pdb.set_trace()
    filtered_indices = torch.topk(torch.from_numpy(mask), opt.pre_filter).indices

    for radius in [opt.radius]:
        query_fun = Voting(opt.radius, num_patches=all_des.shape[1], weighted=opt.weighted)
        retrieved = query_fun.query_each(all_des)
        np.savez(output_dir / Path(str(radius) + "_results_w.npz"), **retrieved)
        print(f"successfully test {opt.model} on ds {opt.dataset} with radius of {radius}.")

    # Step 3: save the retrieved image list
    overlap_pairs = output_dir / Path('top' + str(opt.k) +'_overlap_pairs_w_auc.txt') if opt.weighted else output_dir / Path('top' + str(opt.k) +'_overlap_pairs_auc.txt')

    scores = np.load(output_dir / Path(str(opt.radius) + '_results_w.npz'))['votings'][opt.vote] if opt.weighted else np.load(output_dir / Path(str(radius) + '_results.npz'))['votings'][opt.vote]
    if opt.cls:
        mask = np.load(output_dir / Path("results_cls.npz"))['scores']
        # import pdb; pdb.set_trace()
        mask_ = torch.zeros_like(torch.from_numpy(scores))
        mask_.scatter_(dim=1, index=torch.topk(torch.from_numpy(mask), opt.pre_filter).indices, src=torch.ones_like(mask_))
        voting_topk = torch.topk(torch.from_numpy(scores) * mask_, opt.k)
    else:
        voting_topk = torch.topk(torch.from_numpy(scores), opt.k)

    assert scores.shape[0] > opt.radius

    if not os.path.exists(overlap_pairs) or opt.overwrite:
        with open(overlap_pairs, "w") as doc:
            for i, name in enumerate(image_list):
                for j in voting_topk.indices[i]:
                    pairs_i = image_list[j]
                    doc.write(f"{name} {pairs_i}\n")
