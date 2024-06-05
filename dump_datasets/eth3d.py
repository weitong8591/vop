
import os
import h5py
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import gluefactory

from lightglue.utils import load_image
from gluefactory.datasets.utils import scale_intrinsics
from gluefactory.datasets.eth3d import qvec2rotmat
from gluefactory.datasets.utils import scale_intrinsics
from evaluate_utils import read_cameras


def dump(opt):
    scenes = os.listdir(opt.dataset_dir)
    overlap_features = Path(opt.dump_dir)/ 'eth3d/overlap_feats.h5'
    Path(opt.dump_dir).mkdir(exist_ok=True)
    (Path(opt.dump_dir) / 'eth3d').mkdir(exist_ok=True)
    modelconf = {}
    model = gluefactory.load_experiment(opt.model, conf=modelconf).cuda().eval()

    all_images = []
    if not os.path.exists(overlap_features) or opt.overwrite:
        with h5py.File(str(overlap_features), 'w') as hfile:
            for scene in scenes:
                image_list = [image for image in os.listdir(Path(opt.dataset_dir) / scene / "images/dslr_images_undistorted")]
                cameras = read_cameras(str(Path(opt.dataset_dir, scene, "dslr_calibration_undistorted", "cameras.txt")),1,)
                name_to_cam_idx = {name: {} for name in image_list}
                with open(str(Path(opt.dataset_dir, scene, "dslr_calibration_jpg", "images.txt")), "r") as f:
                    raw_data = f.read().rstrip().split("\n")[4::2]
                for raw_line in raw_data:
                    line = raw_line.split(" ")
                    img_name = os.path.basename(line[-1])
                    name_to_cam_idx[img_name]["dist_camera_idx"] = int(line[-2])
                T_world_to_camera = {}
                image_visible_points3D = {}
                with open(str(Path(opt.dataset_dir, scene, "dslr_calibration_undistorted", "images.txt")), "r") as f:
                    lines = f.readlines()[4:]  # Skip the header
                    raw_poses = [line.strip("\n").split(" ") for line in lines[::2]]
                    raw_points = [line.strip("\n").split(" ") for line in lines[1::2]]
                for raw_pose, raw_pts in zip(raw_poses, raw_points):
                    img_name = os.path.basename(raw_pose[-1])
                    # Extract the transform from world to camera
                    target_extrinsics = list(map(float, raw_pose[1:8]))
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = qvec2rotmat(target_extrinsics[:4])
                    pose[:3, 3] = target_extrinsics[4:]
                    T_world_to_camera[img_name] = pose
                    name_to_cam_idx[img_name]["undist_camera_idx"] = int(raw_pose[-2])
                    # Extract the visible 3D points
                    point3D_ids = [id for id in map(int, raw_pts[2::3]) if id != -1]
                    image_visible_points3D[img_name] = set(point3D_ids)
                

                all_images += image_list
                for image_path in tqdm(image_list):
                    image_ = load_image(Path(opt.dataset_dir) / scene / Path("images/dslr_images_undistorted") / image_path, resize=opt.imsize)
                    c, h, w = image_.shape
                    image = torch.zeros((3, opt.imsize, opt.imsize), device=opt.device, dtype=opt.dtype)
                    image[:, :h, :w] = image_
                    feats = model.extractor({'image': image[None]})
                    group = hfile.create_group(scene + '/' + image_path)
                    for key in ['keypoints', 'descriptors', 'global_descriptor']:
                        group.create_dataset(key, data=feats[key][0].detach().cpu().numpy())
                    
                    ori_h, ori_w = load_image(Path(opt.dataset_dir) / scene / Path("images/dslr_images_undistorted") / image_path).shape[-2:]
                    original_image_size = np.array([ori_w, ori_h])
                    scales = opt.imsize/original_image_size
                    ori_K = cameras[name_to_cam_idx[image_path]["dist_camera_idx"]],
                    K = scale_intrinsics(ori_K[0], scales)

                    group.create_dataset("K", data=K)
                    group.create_dataset("original_K", data=ori_K[0])
                    group.create_dataset("T_w2cam", data=T_world_to_camera[image_path])
                    group.create_dataset('image_size', data=[w, h])

            # save the indices of images
            group = hfile.create_group("indices")
            group.create_dataset("images", data=all_images, dtype=h5py.string_dtype(encoding='utf-8'))
            print(f"in total {len(image_list)} images")
            print(f"finished." )
