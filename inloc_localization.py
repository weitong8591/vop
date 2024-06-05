from pathlib import Path
from pprint import pformat
from omegaconf import OmegaConf
from args import *
from hloc import extract_features, match_features, localize_inloc

opt = create_parser()
dataset_dirs = OmegaConf.load('dump_datasets/data_dirs.yaml')
dataset = Path(dataset_dirs.get('dataset_dirs')[opt.dataset])

loc_pairs = opt.loc_pairs # top 40 retrieved 
outputs = Path(f"outputs_local/inloc/{opt.method}{opt.num_loc}") if opt.method is not None else Path(f"outputs_local/inloc/{opt.model}{opt.num_loc}")# where everything will be saved
outputs.mkdir(exist_ok=True)

results = outputs / f'InLoc_hloc_superpoint+superglue.txt'

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for extraction and matching
# you can also simply write your own here!
feature_conf = extract_features.confs['superpoint_inloc']
matcher_conf = match_features.confs['superglue']

# ## Extract local features for database and query images
feature_path = extract_features.main(feature_conf, dataset, outputs)

# Here we assume that the localization pairs are already computed using image retrieval. To generate new pairs from your own global descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs are also used for the localization - see below.
match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)
# ## Localize!
# Perform hierarchical localization using the precomputed retrieval and matches. Different from when localizing with Aachen, here we do not need a 3D SfM model here: the dataset already has 3D lidar scans. The file `InLoc_hloc_superpoint+superglue_netvlad40.txt` will contain the estimated query poses.
localize_inloc.main(
    dataset, loc_pairs, feature_path, match_path, results,
    skip_matches=10)  # skip database images with too few matches

print(f"Done, submit {results} to LONG-TERM VISUAL LOCALIZATION benchmark!")
 