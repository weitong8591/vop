import torch
import h5py
import os
import numpy as np
import gluefactory
from tqdm import tqdm
from pathlib import Path
from hloc import extract_features
from collections import defaultdict
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

# load the query and db lists in the dumped data
overlap_features = Path(opt.dump_dir) / opt.dataset / "overlap_feats.h5"
assert os.path.exists(overlap_features)

if opt.dataset == 'aachen':
    opt.dataset_dir = Path(dataset_dirs.get('dataset_dirs')[opt.dataset]) /'images/images_upright/'
    with h5py.File(str(overlap_features), 'r') as hfile:
        day_query_list = [dq.decode() for dq in hfile['indices']['day_query'].__array__()]
        night_query_list = [nq.decode() for nq in hfile['indices']['night_query'].__array__()]
        db_list = [db.decode() for db in hfile['indices']['db'].__array__()]
        image_list = np.concatenate((day_query_list, night_query_list, db_list))
        num_query = len(day_query_list) + len(night_query_list)

elif opt.dataset == 'pitts':
    opt.dataset_dir = Path(dataset_dirs.get('dataset_dirs')[opt.dataset])
    query_list = np.load(opt.dataset_dir / "pitts30k_test_qImages.npy")
    db_list = np.load(opt.dataset_dir / "pitts30k_test_dbImages.npy")
    image_list = np.concatenate((query_list, db_list))
    num_query = len(query_list)

elif opt.dataset == 'inloc':
    opt.dataset_dir = Path(dataset_dirs.get('dataset_dirs')[opt.dataset])
    with h5py.File(str(overlap_features), 'r') as hfile:
        query_list = [q.decode() for q in hfile['indices']['query'].__array__()]
        db_list = [db.decode() for db in hfile['indices']['db'].__array__()]
    image_list = np.concatenate((query_list, db_list))
    num_query = len(query_list)
    
else:
    raise NameError("Not implemented")

# load the dumped data, prepare for tests
dataset = SampleDataset(overlap_features, image_list, opt.dataset)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

# use the trained model to encode the input embeddings
model = gluefactory.load_experiment(opt.model).to(opt.device).eval()

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

N = len(image_list)
output_dir = Path(opt.output_dir) / opt.dataset / opt.model

output_dir.mkdir(exist_ok=True, parents=True)

if opt.cls: 
    output_dir = output_dir / Path('cls_' + str(opt.pre_filter))
output_dir.mkdir(exist_ok=True)

# Step 1 (optional): global retrieval 
if not os.path.exists(output_dir / Path("results_cls.npz")):
    cls_scores = torch.einsum('id,jd->ij', cls_tokens, cls_tokens)
    diagonal = torch.eye(N, N).bool().to(cls_scores.device)
    cls_scores.masked_fill_(diagonal, -torch.inf)
    cls_scores[:num_query, :num_query] = torch.zeros_like(cls_scores[:num_query, :num_query] )
    np.savez(output_dir / Path("results_cls.npz"), **{'scores':cls_scores.cpu().numpy()})

# Step 2: patch-level retrieval: radius search + votings
mask = np.load(output_dir / Path("results_cls.npz"))['scores'][:num_query]  
filtered_indices = torch.topk(torch.from_numpy(mask), opt.pre_filter).indices - num_query

query_des = all_des[:num_query]
db_des = all_des[num_query:]

for radius in [opt.radius]:
    query_fun = Voting(opt.radius, num_patches=all_des.shape[1], weighted=opt.weighted)
    retrieved = query_fun.rerank(query_des, db_des, filtered_indices = filtered_indices)
    np.savez(output_dir / Path(str(radius) + "_results_w.npz"), **retrieved)
    print(f"successfully test model {opt.model} on {opt.dataset} with radius of {radius}.")  

# Step 3: save the retrieved image list 
overlap_pairs = output_dir / Path('top' + str(opt.k) +'_overlap_pairs_w_auc.txt') if opt.weighted else output_dir / Path('top' + str(opt.k) +'_overlap_pairs_auc.txt')
if not os.path.exists(overlap_pairs):
    scores = np.load(output_dir / Path(str(opt.radius) + '_results_w.npz'))['votings'] if opt.weighted else np.load(output_dir / Path(str(radius) + '_results.npz'))['votings']
    assert scores.shape[0] > opt.radius
    voting_topk = torch.topk(torch.from_numpy(scores[opt.vote][:num_query]), opt.k)

    with open(overlap_pairs, "w") as doc:
        if opt.dataset == 'aachen':     
            for i, name in enumerate(day_query_list):
                for j in voting_topk.indices[i]:
                    pairs_i = db_list[j]
                    try: 
                        name = str(name).split("'")[1]
                    except:
                        name=name
                    try: 
                       pairs_i = str(pairs_i).split("'")[1]
                    except:
                        pairs_i = pairs_i
                    doc.write(f"{name} {pairs_i}\n")
            for i, name in enumerate(night_query_list):
                for j in voting_topk.indices[i+len(day_query_list)]:
                    pairs_i = db_list[j]
                    try: 
                        name = str(name).split("'")[1]
                    except:
                        name=name
                    try:
                        pairs_i = str(pairs_i).split("'")[1]
                    except:
                        pairs_i = pairs_i
                    doc.write(f"{name} {pairs_i}\n")
        else:
            for i, name in enumerate(query_list):
                for j in voting_topk.indices[i]:
                    pairs_i = db_list[j]
                    try: 
                        name = str(name).split("'")[1]
                        pairs_i = str(pairs_i).split("'")[1]
                    except:
                        name=name
                        pairs_i = pairs_i
                    doc.write(f"{name} {pairs_i}\n")

    print(f"successfully save the retrieved image list to {overlap_pairs}.")  


