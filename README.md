# Breaking the Frame: Image Retrieval by Visual Overlap Prediction

[arxiv]()

# Summary
The proposed method enables the identification of visible image sections without requiring expensive feature detection and matching.
    By focusing on obtaining patch-level embeddings by Vision Transformer backbone and establishing patch-to-patch correspondences, our approach uses a voting mechanism to assess overlap scores for potential database images, thereby providing a nuanced image retrieval metric in challenging scenarios.

<p align="center">
  <img src="https://cmp.felk.cvut.cz/~weitong/vop/demo_756.png"  alt="Image 1" width="200"/>
  <img src="https://cmp.felk.cvut.cz/~weitong/vop/demo_378.png"  alt="Image 2" width="200"/>
  <img src="https://cmp.felk.cvut.cz/~weitong/vop/demo_189.png"  alt="Image 3" width="200"/>
</p>

# Installation
```
torch == 2.2.2
Python == 3.10.13
OmegaConf == 2.3.0
h5py == 3.11.0
tqdm == 4.66.2
faiss == 1.8.0
```
# Evaluation

Step 1. dump the image pairs, save the GT information (e.g., R, K), pretrained Dino V2 [CLS] tokens and patch embeddings (e.g., large in 1024 dim).

Step 2. load the trained encoders to build our own embeddings, eg, in 256-dim, run the retrieval process (CLS tokens for prefiltering, and VOP for reranking.) and save the retrieved image pair list.

Step 3. verify the retrieved image pairs by sending them for [relative pose estimation](relative_pose.py) or [hloc](https://github.com/cvg/Hierarchical-Localization.git) for [localization](inloc_localization.py).

Here are the instructions for testing each data used in our paper and how to test your own data.
Note that before data dumping, create an original dirs for the specific dataset in [dump_datasets/data_dirs.yaml](dump_datasets/data_dirs.yaml).

```
dataset_dirs:
  inloc:<src_path>
```
<details>
<summary>[Inloc]</summary>

1. download the [cutouts](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/cutouts.tar.gz) (db images) and format the data to database/cutouts/; download the [query images](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/queries/iphone7.tar.gz) into query/iphone7/.
2. dump the data and perform image retrieval to get the most overlapping image list. (top-40 on InLoc)
```
python dump_data.py -ds inloc
python retrieve.py -ds inloc -k 40 -m 09 -v 3 -r 0.3 -pre 100 -cls 1
```
47.5 / 72.2 / 82.3	60.3 / 77.1 / 85.5	May 10, 2024, 9:44 a.m.

49.5 / 69.7 / 82.8	60.3 / 77.9 / 84.7	June 5, 2024, 4:58 p.m.

3. install and run [hloc](https://github.com/cvg/Hierarchical-Localization.git) to localize the query images.
```
python inloc_localization.py --loc_pairs outputs/inloc/09/cls_100/top40_overlap_pairs_w_auc.txt -m 09 -ds inloc
```
4. submit the result poses to the [long-term visual localization benchmark](https://www.visuallocalization.net/).


</details>

<details>
<summary>[Megadepth]</summary>

1. download the data from glue-factory: [images](https://cvg-data.inf.ethz.ch/megadepth/Undistorted_SfM.tar.gz), [scene_info](https://cvg-data.inf.ethz.ch/megadepth/scene_info.tar.gz).

2. dump the data and perform image retrieval to get the most overlapping image list.
```
python dump_data.py -ds megadepth
python register.py -k 5 -m 09 -v 4 -r 0.2 -pre 20 -cls 1 -ds megadepth
```

3. run RANSAC on those pairs to estimate relative poses.
```
python relative_pose.py -k 5 -m 09 -v 4 -r 0.2 -pre 20 -cls 1 -ds megadepth
```

</details>

<details>
<summary>[ETH3D]</summary>

1. download [ETH3D](https://cvg-data.inf.ethz.ch/SOLD2/SOLD2_ETH3D_undistorted/ETH3D_undistorted.zip) data (5.6G).
2. dump the data and perform image retrieval to get the most overlapping image list.

```
python dump_data.py -ds eth3d
python register.py -k 5 -m 09 -v 3 -r 0.3 -pre 20 -cls 1 -ds eth3d
```

3. run RANSAC on those pairs to estimate relative poses.

```
python relative_pose.py -k 5 -m 09 -v 4 -r 0.2 -pre 20 -cls 1 -ds eth3d
```
</details>


<details>
<summary>[Your own data]</summary>

1. specify the data dir of your data in [data_dirs.yaml](dump_datasets/data_dirs.yaml), and put the dump script into [here](dump_datasets) to load the images, scene information (K, pose, etc.), and query and data base image lists if needed.

2. run [retrieve.py](retrieve.py) to retrieve the queries if there are query and db images split; while [register.py](register.py) is the case we retrieve each image in the pool from the rest.

3. run [relative_pose.py](relative_pose.py) for relative pose estimation; or [inloc_localization.py](inloc_localization.py) to localize the queries by the retrieved db images.

</details>



# Training

1. download depths of Megadepth to build the training supervision from [here](https://cvg-data.inf.ethz.ch/megadepth/depth_undistorted.tar.gz).

2. customize the configs and start training.

```
python -m gluefactory.train 09 --conf train_configs/09.yaml
```

Here the training is based on [glue-factory](https://github.com/cvg/glue-factory.git), we provide details of the configurations we focus on.
```
data:
    # choose the data augmentation type: 'flip, dark, lighglue'
    photometric: {
            "name": "flip",
           "p": 0.95,
            # 'difficulty': 1.0,  # currently unused
       }

model:
    matcher:
        name: overlap_predictor # our model
        add_voting_head: true # whether to train by the constastive loss on the patch-level negative/positive matches
        add_cls_tokens: false # whether to train the global embeddings
        attentions: false # whether to use the attentionsfor supervison
        input_dim: 1024 # the dimension of the pretrained Dino features

train:
  dropout_prob: 0.5    # dropout probability

```
# Notes
<details>
<summary>[Useful configs]</summary>

```
--radius, radius for radius knn search
--cls, default=0, whether to use CLS tokens as prefilter
--pre_filter', default=20, the number of db images prefiltered for reranking.
--weighted, default=1, whether to use TF-IDF weights for voting scores.
--vote, vote methods.
--k', top-k retrievals.
--overwrite', for data redump.
--num_workers, default=8, change it to fit your machine.
```
</details>

<details>
<summary>[Acknowledgement]</summary>

[glue-factory](https://github.com/cvg/glue-factory.git)
[long-term visual localization benchmark](https://www.visuallocalization.net/)
[pre-commit](https://pre-commit.com/)

</details>

<details>
<summary>[Contact]</summary>
Contact me at weitongln@gmail.com or weitong@fel.cvut.cz
</details>
