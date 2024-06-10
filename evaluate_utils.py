import os
import time
import faiss
import torch
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from hloc.utils.parsers import parse_retrieval, names_to_pair
from collections import defaultdict
from gluefactory.geometry.epipolar import relative_pose_error
from gluefactory.robust_estimators import load_estimator
from gluefactory.eval.utils import eval_poses, eval_poses_best
from gluefactory.geometry.wrappers import Pose, Camera
from hloc import pairs_from_retrieval, pairs_from_exhaustive, extract_features, match_features
from sklearn.neighbors import NearestNeighbors
from gluefactory.datasets.utils import scale_intrinsics
from gluefactory.utils.image import ImagePreprocessor
from gluefactory.utils.patch_helper import PatchCollect
from torch.nn.functional import cosine_similarity

def recall(topk, gt_topk):
        topk_recall = 0
        for i in range(topk.indices.shape[0]):
            for j in range(topk.indices.shape[1]):
                if topk.indices[i, j] in gt_topk.indices[i]:
                    topk_recall += 1
        return topk_recall/(topk.indices.shape[0]*topk.indices.shape[1])

def read_cameras(camera_file, scale_factor=None):
    """Read the camera intrinsics from a file in COLMAP format."""
    with open(camera_file, "r") as f:
        raw_cameras = f.read().rstrip().split("\n")
    raw_cameras = raw_cameras[3:]
    cameras = []
    for c in raw_cameras:
        data = c.split(" ")
        fx, fy, cx, cy = np.array(list(map(float, data[4:])))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        if scale_factor is not None:
            K = scale_intrinsics(K, np.array([scale_factor, scale_factor]))
        cameras.append(K)
    return cameras

class BaselineHelper:
    def __init__(self, image_list) -> None:
        self.image_list = image_list

    def match_after_retrival(self, match_file, pairs):
        retrieval = parse_retrieval(pairs)
        stats = defaultdict(list)
        matches = []
        with h5py.File(str(match_file), "r") as hfile:
            for query in self.image_list:
                refs = retrieval[query]
                num_matches = 0
                for ref_name in refs:
                    pair_name = names_to_pair(query, ref_name)
                    matches0 = hfile[pair_name]['matches0'].__array__()
                    num_matches += (matches0 > -1).sum()
                    matches += [(matches0 > -1).sum()]
                # avg. matched correspondences over different matched images
                stats['num_matches'].append(num_matches / len(refs))
        reduced_stats = {k: np.mean(v) for k, v in stats.items()}

        return reduced_stats, stats

    def relative_pose_after_retrieval(self, match_file, pairs, features, overlap_features):
        retrieval = parse_retrieval(pairs)
        pose_results = defaultdict(lambda: defaultdict(list))
        with h5py.File(str(match_file), "r") as hfile, \
            h5py.File(str(features), "r") as hfile1, \
            h5py.File(str(overlap_features), "r") as hfile2:
            for query in tqdm(self.image_list):
                refs = retrieval[query]
                coordinates0 = hfile1[query]['keypoints'].__array__()
                for ref_name in refs:

                    pair_name = names_to_pair(query, ref_name)

                    matches0 = hfile[pair_name]['matches0'].__array__()

                    matches1 = matches0[matches0 > -1]
                    coordinates1 = hfile1[ref_name]['keypoints'].__array__()
                    try:
                       matched_coordinates0 = coordinates0[matches0 > -1]
                       matched_coordinates1 = coordinates1[matches1]
                    except:
                        length = min(len(coordinates0), len(coordinates1))
                        matched_coordinates0 = coordinates0[:length]
                        matched_coordinates1 = coordinates1[:length]
                    # do RANSAC, pose estimation and find inliers
                    data0 = {k+'0': torch.Tensor(hfile2[query][k].__array__()).cuda()[None] for k in hfile2[query].keys()}
                    data1 = {k+'1': torch.Tensor(hfile2[ref_name][k].__array__()).cuda()[None] for k in hfile2[ref_name].keys()}
                    data = {**data0, **data1}

                    data["camera0"] = Camera.from_calibration_matrix(data["original_K0"])#[None, :]).cuda()
                    data["T_w2cam0"] = Pose.from_4x4mat(data["T_w2cam0"])
                    data["camera1"] = Camera.from_calibration_matrix(data["original_K1"])#[None, :]).cuda()
                    data["T_w2cam1"] = Pose.from_4x4mat(data["T_w2cam1"])

                    if "T_w2cam0" in data.keys():
                        data["T_0to1"] = data["T_w2cam1"] @ data["T_w2cam0"].inv()
                        data["T_1to0"] = data["T_w2cam0"] @ data["T_w2cam1"].inv()
                    T_gt = data["T_0to1"]
                    for th in [0.5, 1.5, 2, 3]:#
                         estimator = load_estimator("relative_pose", 'opencv')(conf = {'ransac_th':th})
                         data_estimate = {
                             "m_kpts0": torch.from_numpy(matched_coordinates0.astype(np.float32)).to(data["camera0"].device),
                             "m_kpts1": torch.from_numpy(matched_coordinates1.astype(np.float32)).to(data["camera1"].device),
                             "camera0":  data["camera0"],
                             "camera1":  data["camera1"],
                         }
                         est = estimator(data_estimate)

                         pose_results_i = {}
                         if not est["success"]:
                             pose_results_i["rel_pose_error"] = float("inf")
                             pose_results_i["ransac_inl"] = 0
                             pose_results_i["ransac_inl%"] = 0
                         else:
                             M = est["M_0to1"]
                             inl = est["inliers"].cpu().numpy()
                             t_error, r_error = relative_pose_error(T_gt, M.R, M.t)
                             pose_results_i["rel_pose_error"] = max(r_error, t_error).item()
                             pose_results_i["ransac_inl"] = np.sum(inl)
                             pose_results_i["ransac_inl%"] = np.mean(inl)
                         [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            best_pose_results, best_th = eval_poses_best(pose_results, auc_ths=[5, 10, 20], key="rel_pose_error", n = len(self.image_list))

        return best_pose_results, best_th

class Voting:
    """Overlap retrieval on db images, radius search + votings."""

    def __init__(self, radius, num_patches=256, weighted=True) -> None:
        self.radius = radius
        self.num_patches = num_patches
        self.EPS = 1e-5
        # IF-TDF weights
        self.weighted = weighted

    def rerank(self, query_descriptors, db_descriptors, filtered_indices=None):
        """Rerank each patch in the query image from the prefiltered db images by radius search and voting.

        Args:
            query_descriptors (num_images, num_patches. dim)
            filtered_indices: indices for the prefiltered db images, for reranking
        """

        # the dimensions of the patch-level embeddings
        dim = query_descriptors.shape[-1]

        query_normalized = torch.nn.functional.normalize(query_descriptors, dim=-1)
        db_normalized = torch.nn.functional.normalize(db_descriptors, dim=-1)

        # return voting results, in s shape of (#voting schemes, #queries, #db images)
        votings= np.zeros((7, query_descriptors.shape[0], db_descriptors.shape[0]))
        # the last two options are matched on each patch at the same location in the images, use mean/max as the score
        for i, des_i in enumerate(query_normalized):
            idx = filtered_indices[i]
            votings[5, i, idx] = cosine_similarity(des_i[None, :], db_normalized[idx], dim=-1).mean(-1).cpu().numpy()
            votings[6, i, idx] = cosine_similarity(des_i[None, :], db_normalized[idx], dim=-1).max(-1).values.cpu().numpy()

        if self.weighted:
            # IF-TDF
            # n_d: how many patches in total in each image
            n_d = self.num_patches
            # N: num of images in the data base (filtered)
            N = filtered_indices.shape[-1]

        assign_time = 0
        for i, query in tqdm(enumerate(query_normalized)):
            # Faiss CPU radius NN search
            index_flat_cpu = faiss.IndexFlatIP(dim)
            # indexing all the patches in db images
            index_flat_cpu.add(db_normalized[filtered_indices[i]].view(-1, dim).cpu().numpy().astype(np.float32))
            # find radius neighbors
            start = time.time()
            lims, D_cpu, I_cpu = index_flat_cpu.range_search(query.cpu().numpy(), 1-self.radius)
            assign_time += time.time()-start
            # faiss doesn't sort
            for lims_i in range(len(lims)-1):
                sorted_idx = np.argsort(D_cpu[lims[lims_i]: lims[lims_i+1]])[::-1]
                if len(sorted_idx) > 0:
                    D_cpu[lims[lims_i]: lims[lims_i+1]] = D_cpu[lims[lims_i]: lims[lims_i+1]][sorted_idx]
                    I_cpu[lims[lims_i]: lims[lims_i+1]] = I_cpu[lims[lims_i]: lims[lims_i+1]][sorted_idx]
            assignment, distance = [I_cpu[lims[idx]:lims[idx+1]] for idx in range(len(query))], [1-D_cpu[lims[idx]:lims[idx+1]] for idx in range(len(query))]


            find_images_all = [a//self.num_patches for a in assignment]
            find_images = np.hstack(find_images_all)

            # if no neighbors found
            if len(find_images) == 0:
                continue
            # num_matched_patches, within a given radius
            nid = np.array([len(a) for a in assignment])

            # check the first occurrence
            unique_indices = np.unique(find_images, return_index=True)[1] # indices of the images
            # ni: num of images that has at least on patch in the neighbor of query patch
            ni = np.array([len(np.unique(p_image)) for p_image in find_images_all])
            # tf-idf weights for each query patch
            ti = nid/n_d *np.log(N/(ni + self.EPS) )
            ti2 = (1/(nid + self.EPS))/(1/(nid + self.EPS)).max()
            ti[ti<0] = 0
            ti2[ti2<0] = 0

            similarity = np.hstack([1.-d for d in distance])
            weighted_similarity  = np.hstack([(1.-d) * ti[i] for i, d in enumerate(distance)])
            weighted_similarity2  = np.hstack([(1.-d)* ti2[i] for i, d in enumerate(distance)])

            #votings:
                # 0. vote once on the db image by the cloest patch, by 1
                # 1. same as 1 but add similarity
                # 2. vote on the db image by all neighbor patches by 1
                # 3. same as 3 but add similarity
                # 4. same as 4 but apply TI-IDF weights
                # 5.

            # as we sorted the assignment, the duplicate retrieved neighbors will only be voted for once, by the closest
            votings[0, i, filtered_indices[i][find_images[unique_indices]]] = 1
            votings[1, i, filtered_indices[i][find_images[unique_indices]]] = similarity[unique_indices]
            votings[2, i] = np.bincount(filtered_indices[i][find_images], minlength=votings.shape[-1])
            votings[3, i] = np.bincount(filtered_indices[i][find_images], weights=similarity, minlength=votings.shape[-1])
            votings[4, i] = np.bincount(filtered_indices[i][find_images], weights=weighted_similarity, minlength=votings.shape[-1])
            votings[5, i] = np.bincount(filtered_indices[i][find_images], weights=weighted_similarity2, minlength=votings.shape[-1])

        return {
                "votings": votings,
                "assign_time": assign_time/len(query_normalized)
                }

    def query_each(self, descriptors, filtered_indices=None):
        """Retrieve each image from the rest (as db images) in patch level by radius search and voting.

        Args:
            descriptors (num_images, num_patches. dim)
        """
        if filtered_indices is not None:
            return self.rerank(descriptors, descriptors, filtered_indices)
        else:
            dim = descriptors.shape[-1]
            embeddings_normalized = torch.nn.functional.normalize(descriptors, dim=-1)
            votings= np.zeros((7, descriptors.shape[0], descriptors.shape[0]))
            # N: num of images
            N = descriptors.shape[0]
            # the last two options are matched on each patch at the same location in the images, use mean/max as the score
            for i, des_i in enumerate(embeddings_normalized):
                votings[5, i] = cosine_similarity(des_i[None, :], embeddings_normalized, dim=-1).mean(-1).cpu().numpy()
                votings[6, i] = cosine_similarity(des_i[None, :], embeddings_normalized, dim=-1).max(-1).values.cpu().numpy()
            # no matches within themselves
            diagonal = np.eye(N, N, dtype=bool)
            votings[5][diagonal] = -np.inf
            votings[6][diagonal] = -np.inf

            if self.weighted:
                # IF-TDF
                # n_d: how many patches in total in each image
                n_d = self.num_patches

            assign_time = 0
            for i, query in tqdm(enumerate(embeddings_normalized)):
                # Faiss CPU radius NN search
                index_flat_cpu = faiss.IndexFlatIP(dim)
                # indexing all the patches in db images
                index_flat_cpu.add(embeddings_normalized.view(-1, dim).cpu().numpy().astype(np.float32))
                # find radius neighbors
                start = time.time()
                lims, D_cpu, I_cpu = index_flat_cpu.range_search(query.cpu().numpy(), 1-self.radius)
                assign_time += time.time()-start
                # faiss doesn't sort
                for lims_i in range(len(lims)-1):
                    sorted_idx = np.argsort(D_cpu[lims[lims_i]: lims[lims_i+1]])[::-1]
                    if len(sorted_idx) > 0:
                        D_cpu[lims[lims_i]: lims[lims_i+1]] = D_cpu[lims[lims_i]: lims[lims_i+1]][sorted_idx]
                        I_cpu[lims[lims_i]: lims[lims_i+1]] = I_cpu[lims[lims_i]: lims[lims_i+1]][sorted_idx]
                assignment, distance = [I_cpu[lims[idx]:lims[idx+1]] for idx in range(len(query))], [1-D_cpu[lims[idx]:lims[idx+1]] for idx in range(len(query))]

                find_images_all = [a[1:]//self.num_patches for a in assignment]
                find_images = np.hstack(find_images_all)

                # if no neighbors found
                if len(find_images) == 0:
                    continue

                # num_matched_patches
                nid = np.array([len(a)-1 for a in assignment])
                # no voting on this image i itself
                mask = find_images != i
                find_images = find_images[mask]
                # check the first occurrence
                unique_indices = np.unique(find_images, return_index=True)[1]
                # ni: num of images that has at least one patch in the neighbor of query patch
                ni = len(find_images[unique_indices])
                # tf-idf weights for each query patch
                ti = nid / n_d *np.log((N-1) / (ni+ self.EPS))
                ti2 = (1 / (nid + self.EPS))/(1 / (nid + self.EPS)).max()
                ti[ti<0] = 0
                ti2[ti2<0] = 0

                similarity = np.hstack([1.-d[1:] for d in distance])[mask]
                weighted_similarity  = np.hstack([(1.-d[1:]) * ti[i] for i, d in enumerate(distance)])[mask]
                weighted_similarity2  = np.hstack([(1.-d[1:])* ti2[i] for i, d in enumerate(distance)] )[mask]

                votings[0, i, find_images[unique_indices]] = 1
                votings[1, i, find_images[unique_indices]] = similarity[unique_indices]
                votings[2, i] = np.bincount(find_images, minlength=votings.shape[-1])
                votings[3, i] = np.bincount(find_images, weights=similarity, minlength=votings.shape[-1])
                votings[4, i] = np.bincount(find_images, weights=weighted_similarity, minlength=votings.shape[-1])
                votings[5, i] = np.bincount(find_images, weights=weighted_similarity2, minlength=votings.shape[-1])

            return {
                "votings": votings,
                "assign_time": assign_time/len(descriptors)
                }
