import os
import torch
import importlib
import gluefactory
from omegaconf import OmegaConf
from args import *
from utils import *

opt = create_parser()
# check if gpu device is available
opt.device = torch.device('cuda:0' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
# load the data dirs for all
dataset_dirs = OmegaConf.load('dump_datasets/data_dirs.yaml')

def dump_data(opt, model, dataset):
    print(f"----------Start dumping the dataset: {dataset} on {opt.device}------")
    opt.dataset = dataset

    try:
        module = importlib.import_module(f"dump_datasets.{dataset}")
        module.dump(opt, model)
    except ImportError:
        print(f"Unknown dataset: {dataset}")

    print(f"----------Finished------")

if opt.dataset == 'all':
    datasets = ['inloc', 'megadepth', 'eth3d', 'phototourism']
else:
    datasets = [opt.dataset]

for dataset in datasets:
    # fetch the data paths from yaml
    opt.dataset_dir = dataset_dirs.get('dataset_dirs')[dataset]
    if not Path(opt.dataset_dir).exists():
        raise FileNotFoundError('dataset_dir is hardcoded, please adjust it to yours')
    if not os.path.exists(f"outputs/training/{opt.model}"):
        download_best(opt.model)
    modelconf = {}
    model = gluefactory.load_experiment(opt.model, conf=modelconf).to(opt.device).eval()
    dump_data(opt, model, dataset)
