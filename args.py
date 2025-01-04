import argparse
import torch
from pathlib import Path

def create_parser():

    parser = argparse.ArgumentParser()
    # dump data
    parser.add_argument('--model', '-m', default='best',
                            help='The name of the model to be train/test.')
    parser.add_argument('--dataset_dir', '-dir', type=Path, default="data/ETH3D_undistorted")
    parser.add_argument('--dump_dir', '-dump', type=Path, default="dumped_data",
                            help='dir to save the features.')
    parser.add_argument('--dataset', '-ds', default="eth3d", choices=['all', 'aachen', 'pitts', 'inloc', 'megadepth', 'eth3d', 'phototourism', 'imc2023'],
                            help='dataset name.')
    parser.add_argument('--imsize', '-im', type=int, default=224,
                            help='The resized image shape.')
    parser.add_argument('--overwrite', '-ow', action='store_true', default=False,
                            help='overwrite the dump data.')

    # type
    parser.add_argument('--device', '-device', default='cuda')
    parser.add_argument('--dtype', '-dtype', default=torch.float32)

    # retrieval
    parser.add_argument('--radius', '-r', type=float, default=-1,
                            help='radius for radius knn search, default set it as -1 to \
                             trigger the procedure of automatically compute the median similarity over 100 samples as radius.')
    parser.add_argument('--cls', '-cls',  action='store_true', default=True,
                            help='use CLS tokens as prefilter.')
    parser.add_argument('--pre_filter', '-pre', type=int, default=20,
                            help='how many db images prefiltered for reranking')
    parser.add_argument('--weighted', '-w',  action='store_true', default=True,
                            help='use TF-IDF weights for voting scores.')
    parser.add_argument('--k', '-k', type=int, default=1,
                            help='top-k retrievals')
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--num_workers', '-nw', type=int, default=8)
    parser.add_argument('--output_dir', '-out', type=Path, default='outputs')
    parser.add_argument('--radius_round', '-round', default='round', choices=['round, ceil'],
                            help='vote methods')

    # localization
    parser.add_argument('--method', default=None)
    parser.add_argument('--num_loc', type=int, default=40,
                    help='Number of image pairs for loc, default: %(default)s')
    parser.add_argument('--loc_pairs', type=Path, default="outputs/inloc/09/cls_100/top40_overlap_pairs_w_auc.txt",
                    help='Number of image pairs for loc, default: %(default)s')

    return parser.parse_args()
