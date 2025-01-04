# Breaking the Frame: Visual Place Recognition by Overlap Prediction

# Updates
2024.06. Available on [arxiv](https://arxiv.org/abs/2406.16204).

2024.10. Accepted at WACV 2025.


# Summary
The proposed method enables the identification of visible image sections without requiring expensive feature detection and matching.
    By focusing on obtaining patch-level embeddings by DINOV2 backbone and establishing patch-to-patch correspondences, our approach uses a voting mechanism to assess overlap scores for potential database images, thereby providing a nuanced image retrieval metric in challenging scenarios.

<p align="center">
  <img src="https://cmp.felk.cvut.cz/~weitong/vop/pipeline_test.jpg"  alt="Image 1" width=""/>
</p>

# Installation
```
torch == 2.3.1
Python == 3.10.13
OpenCV == 4.10.0.84
OmegaConf == 2.3.0
h5py == 3.11.0
tqdm == 4.66.4
faiss-gpu == 1.7.2
lightglue
hloc
```

# [Demo](demo.ipynb)
Try the proposed VOP on one example image pair and visualize their matched patches.

# Evaluation

Step 1. Preprocess the test data and GT information (e.g., camera parameters R, K) if available. Load images and run the frozen DINOv2 on them, then, save the [CLS] tokens and patch embeddings.


Step 2. Load the best checkpoint and use the trained encoder on the test set. Do retrieval and save a list of images with high overlaps.

Step 3. Evaluate the retrieval results by running [Relative pose estimation](relative_pose.py) or [localization](inloc_localization.py).


Here are the instructions for the test sets used in the paper. The [best checkpoint](https://cmp.felk.cvut.cz/~weitong/vop/checkpoint_best.tar) is downloaded automatically.

:boom: important: before data preprocessing, create/update an original dirs for the specific dataset in [dump_datasets/data_dirs.yaml](dump_datasets/data_dirs.yaml).

```
dataset_dirs:
  inloc:<src_path>
```


<details>
<summary>[Megadepth]</summary>

1. Download the data from glue-factory including [images](https://cvg-data.inf.ethz.ch/megadepth/Undistorted_SfM.tar.gz) and [scene_info](https://cvg-data.inf.ethz.ch/megadepth/scene_info.tar.gz).

2. Data preprocess and top-1/5/10 retrieval.
```
python dump_data.py -ds megadepth
python register.py -k 5 -m best -pre 20 -ds megadepth
```
3. Relative pose estimation using RANSAC.
```
python relative_pose.py -k 5 -m best -pre 20 -ds megadepth
```

</details>

<details>
<summary>[ETH3D]</summary>

1. Download [ETH3D](https://cvg-data.inf.ethz.ch/SOLD2/SOLD2_ETH3D_undistorted/ETH3D_undistorted.zip) (5.6G).
2. Data preprocess and top-1/5/10 retrieval.
```
python dump_data.py -ds eth3d
python register.py -k 5 -m best -pre 20 -ds eth3d
```
3. Relative pose estimation using RANSAC.

```
python relative_pose.py -k 5 -m best -pre 20 -ds eth3d
```
</details>

<details>
<summary>[Inloc]</summary>

1. Download the [DB images](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/cutouts.tar.gz) and format the data to database/cutouts/; download the [queries](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/queries/iphone7.tar.gz) into query/iphone7/.
2. Data preprocess and top-40 retrieval.
```
python dump_data.py -ds inloc
python retrieve.py -ds inloc -k 40 -m best -pre 100
```
<!-- 47.5 / 72.2 / 82.3	60.3 / 77.1 / 85.5	May 10, 2024, 9:44 a.m.
49.5 / 69.7 / 82.8	60.3 / 77.9 / 84.7	June 5, 2024, 4:58 p.m. -->

3. Install and run [hloc](https://github.com/cvg/Hierarchical-Localization.git) for localization.
```
python inloc_localization.py --loc_pairs outputs/inloc/best/cls_100/top40_overlap_pairs.txt -m best -ds inloc -out output_local
```
4. Submit the result poses to the [long-term visual localization benchmark](https://www.visuallocalization.net/).


</details>


<details>
<summary>[Customized data]</summary>

1. Add the path of the custom data in [data_dirs.yaml](dump_datasets/data_dirs.yaml), and creat a dump script into [here](dump_datasets) to load images and GT pose information if needed and available.

2. Run [retrieve.py](retrieve.py) to find overlapping DB images for the queries or [register.py](register.py) to search overlapping images for each image in the pool.

3. Run the evaluation of [relative pose estimation](relative_pose.py) or [localization](inloc_localization.py), or use the saved retrieved pairs somewhere else as you want.


</details>



# Training

Step 1. Download GT depths of Megadepth to for training supervision from [here](https://cvg-data.inf.ethz.ch/megadepth/depth_undistorted.tar.gz).

Step 2. Customize the configs and start training based on [glue-factory](https://github.com/cvg/glue-factory.git). Here we provided a [default config](train_configs/best_easy.yaml) with fixed positive/negative image pairs saved (fast) and random positive/negative pairs in [this config](train_configs/best.yaml) (slow).

```
python -m gluefactory.train best_easy_retrain --conf train_configs/best_easy.yaml
```
Note that the easy version requires prepared labels, pls download it from [train](https://cmp.felk.cvut.cz/~weitong/vop/supervision_train_check.h5) and [validation](https://cmp.felk.cvut.cz/~weitong/vop/supervision_val_check.h5).

Important configs:
```
data:
    data_dir: ""
    info_dir: ""
    # choose the data augmentation type: 'flip, dark, lighglue'
    photometric: {
            "name": "flip",
           "p": 0.95,
            # 'difficulty': 1.0,  # currently unused
       }
    gt_label_path: ""


model:
    matcher:
        name: overlap_predictor # our model
        input_dim: 1024 # the dimension of the pretrained DINOv2 features
        embedding_dim: 256 # projected embedding dim
        dropout_prob: 0.5    # dropout probability

```
# Notes
<details>
<summary>[Useful configs]</summary>

```
--model, name of the loaded model.
--k, top-k retrievals.
--radius, default=-1, compute the median similarity over 100 random samples as the radius threshold.
--cls, default=True, action True, whether CLS tokens (prefilter) is used.
--pre_filter, default=20, shortlist length.
--weighted, default=True, action True, whether to use TF-IDF weights for voting.
--overwrite, default=False, action True.
--conf, config path used for training.
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
Contact me at weitongln@gmail.com or weitong@fel.cvut.cz.
</details>
