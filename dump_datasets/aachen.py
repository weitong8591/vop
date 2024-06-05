from lightglue.utils import load_image
import gluefactory
from gluefactory.datasets.utils import scale_intrinsics
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from args import *
from utils import quaternion_to_rotation_matrix, camera_center_to_translation, Camera

def dump(opt):

    overlap_features = Path(opt.dump_dir)/ 'aachen' / 'overlap_feats.h5'
    Path(opt.dump_dir).mkdir(exist_ok=True)
    (Path(opt.dump_dir) / 'aachen').mkdir(exist_ok=True)
    modelconf = {}
    model = gluefactory.load_experiment(opt.model, conf=modelconf).cuda().eval()

    # save intrinsic and extrinsic parameters for each image, from world to camera
    if not os.path.exists(overlap_features) or opt.overwrite:
        with h5py.File(str(overlap_features), 'w') as hfile:
            camera_parameters = {}

            # Recover intrinsics.
            with open(os.path.join(Path(opt.dataset_dir), '3D-models/aachen_v_1_1/database_intrinsics_v1_1.txt')) as f:
                raw_intrinsics = f.readlines()
            
            for intrinsics in raw_intrinsics:
                intrinsics = intrinsics.strip('\n').split(' ')
                image_name = intrinsics[0]
                camera_model = intrinsics[1]
                intrinsics = [float(param) for param in intrinsics[2 :]]
                camera = Camera()
                camera.set_intrinsics(camera_model=camera_model, intrinsics=intrinsics)
                # camera model: eg, SIMPLE_RADIAL
                # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
                camera_parameters[image_name] = camera

            # Recover poses.
            with open(os.path.join(Path(opt.dataset_dir), '3D-models/aachen_v_1_1/aachen_v_1_1.nvm')) as f:
                raw_extrinsics = f.readlines()
            # Skip the header.
            n_cameras = int(raw_extrinsics[2])
            raw_extrinsics = raw_extrinsics[3 : 3 + n_cameras]

            for extrinsics in raw_extrinsics:
                extrinsics = extrinsics.strip('\n').split(' ')
                image_name = extrinsics[0]
                # Skip the focal length. Skip the distortion and terminal 0.
                qw, qx, qy, qz, cx, cy, cz = [float(param) for param in extrinsics[2 : -2]]
                qvec = np.array([qw, qx, qy, qz])
                c = np.array([cx, cy, cz])
                # NVM -> COLMAP.
                t = camera_center_to_translation(c, qvec)
                camera_parameters[image_name].set_pose(qvec=qvec, t=t)
            db_images = [k for k in camera_parameters.keys()]

            query_image_list_path = os.path.join(Path(opt.dataset_dir), 'queries/night_time_queries_with_intrinsics.txt')
            with open(query_image_list_path) as f:
                raw_queries = f.readlines()
                query_names = set()
                for raw_query in raw_queries:
                    raw_query = raw_query.strip('\n').split(' ')
                    query_name = raw_query[0]
                    query_names.add(query_name)
                    image_name = query_name
                    camera_model = raw_query[1]
                    intrinsics = [float(param) for param in raw_query[2 :]]
                    camera = Camera()
                    camera.set_intrinsics(camera_model=camera_model, intrinsics=intrinsics)
                    camera_parameters[image_name] = camera

            query_image_list_path_day = os.path.join(Path(opt.dataset_dir), 'queries/day_time_queries_with_intrinsics.txt')
            with open(query_image_list_path_day) as f:
                raw_queries = f.readlines()
                query_names_day = set()
                for raw_query in raw_queries:
                    raw_query = raw_query.strip('\n').split(' ')
                    query_name = raw_query[0]
                    query_names_day.add(query_name)
                    image_name = query_name
                    camera_model = raw_query[1]
                    intrinsics = [float(param) for param in raw_query[2 :]]
                    camera = Camera()
                    camera.set_intrinsics(camera_model=camera_model, intrinsics=intrinsics)
                    camera_parameters[image_name] = camera

            group_indices = hfile.create_group("indices")
            group_indices.create_dataset("day_query", data=query_names_day)
            group_indices.create_dataset("night_query", data=query_names)
            group_indices.create_dataset("db", data=db_images)
            print(f"in total {len(query_names)} queries and {len(db_images)} db images")
            image_list =camera_parameters.keys()
            for image_path in tqdm(image_list):
                image_ = load_image(Path(opt.dataset_dir) / "images/images_upright/" / image_path, resize=opt.imsize)
                c, h, w = image_.shape
                image = torch.zeros((3, opt.imsize, opt.imsize), device=opt.device, dtype=opt.dtype)
                image[:, :h, :w] = image_
                feats = model.extractor({'image': image[None]})
                group = hfile.create_group(image_path.replace('/', '_'))
                for key in ['keypoints', 'descriptors', 'global_descriptor']:
                    group.create_dataset(key, data=feats[key][0].detach().cpu().numpy())
                ori_h, ori_w = load_image(Path(opt.dataset_dir) / "images/images_upright/" / image_path).shape[-2:]
                original_image_size = np.array([ori_w, ori_h])
                scales = opt.imsize/original_image_size
                fx = camera_parameters[image_path].intrinsics[2]
                cx, cy = camera_parameters[image_path].intrinsics[3:5]
                ori_K = np.array([
                    [fx, 0, cx],
                    [0, fx, cy],
                    [0, 0, 1]
                ])

                K = scale_intrinsics(ori_K[0], scales)
                try: 
                    T_world_to_camera = np.zeros((4, 4))
                    T_world_to_camera[:3, :3] = quaternion_to_rotation_matrix(camera_parameters[image_path].qvec)
                    T_world_to_camera[:3, 3] = camera_parameters[image_path].t
                    group.create_dataset("T_w2cam", data=T_world_to_camera)
                except:
                    pass
                group.create_dataset("K", data=K)
                group.create_dataset("original_K", data=ori_K)
                group.create_dataset('image_size', data=[w, h])
            print(f"finished." )
