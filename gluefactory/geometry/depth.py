import kornia
import torch

from .utils import get_image_coords
from .wrappers import Camera

import numpy as np

def sample_fmap(pts, fmap):
    h, w = fmap.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    pts = (pts / pts.new_tensor([[w, h]]) * 2 - 1)[:, None]
    # @TODO: This might still be a source of noise --> bilinear interpolation dangerous
    interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
    interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")
    return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(
        0, 2, 1
    )


def sample_depth(pts, depth_):
    depth = torch.where(depth_ > 0, depth_, depth_.new_tensor(float("nan")))
    depth = depth[:, None]
    interp = sample_fmap(pts, depth).squeeze(-1)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def sample_normals_from_depth(pts, depth, K):
    depth = depth[:, None]
    normals = kornia.geometry.depth.depth_to_normals(depth, K)
    normals = torch.where(depth > 0, normals, 0.0)
    interp = sample_fmap(pts, normals)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def project(
    kpi,
    di,
    depthj,
    camera_i,
    camera_j,
    T_itoj,
    validi,
    ccth=None,
    sample_depth_fun=sample_depth,
    sample_depth_kwargs=None,
):
    if sample_depth_kwargs is None:
        sample_depth_kwargs = {}

    # 3d points from 2d points in image i 
    kpi_3d_i = camera_i.image2cam(kpi) # B, 224*224, 3
    # multiplied by depth, filter the ones with zero depths
    kpi_3d_i = kpi_3d_i * di[..., None]
    # transform 3d i to j camera
    kpi_3d_j = T_itoj.transform(kpi_3d_i)
    # 2d points in image j. from 3d points
    kpi_j, validj = camera_j.cam2image(kpi_3d_j)
    """"""
    kpi_i, _ = camera_i.cam2image(kpi_3d_i)

    """"""
    # di_j = kpi_3d_j[..., -1]
    validi = validi & validj
    if depthj is None or ccth is None:
        return kpi_j, validi & validj, kpi_i
    else:
        # circle consistency
        dj, validj = sample_depth_fun(kpi_j, depthj, **sample_depth_kwargs)
        kpi_j_3d_j = camera_j.image2cam(kpi_j) * dj[..., None]
        kpi_j_i, validj_i = camera_i.cam2image(T_itoj.inv().transform(kpi_j_3d_j))
        consistent = ((kpi - kpi_j_i) ** 2).sum(-1) < ccth
        visible = validi & consistent & validj_i & validj
        # visible = validi
        return kpi_j, visible, kpi_i


def dense_warp_consistency(
    depthi: torch.Tensor, # depth map, B, 224, 224
    depthj: torch.Tensor,
    T_itoj: torch.Tensor,
    camerai: Camera,
    cameraj: Camera,
    **kwargs,
):
    kpi = get_image_coords(depthi).flatten(-3, -2) # key points, (N, 2)
    di = depthi.flatten(
        -2,
    )
    validi = di > 0
    kpir, validir, kpi  = project(kpi, di, depthj, camerai, cameraj, T_itoj, validi, **kwargs)

    return kpir.unflatten(-2, depthi.shape[-2:]), validir.unflatten(
        -1, (depthj.shape[-2:])), kpi.unflatten(-2, depthi.shape[-2:])

@torch.no_grad()    
def dense_patch_matching(matches0, matches1, inl0, patch_helper, gt_vis_image0, gt_vis_image1, N=0, one_to_one=True):
    
    """
        find the matching patches based on the visiable pixel matches, given visiable patches
        
        input: 
            matches0, matches1 in a shape (B, 224, 224, 2) from B image pairs
            N - threshold of # matching pixels that can be regarded as a matching patch pair
            patch_helper - class including functions to find patches using pixels, match patches, etc
        return:
            patch matching labels (1/0) for each pair with the patches in the other image 
            label_confs - we save #matching pixels in the matched patch, as confidence
            
    """
    B, num_patches = inl0.shape[0], patch_helper.num_patches**2
    gt_labels = torch.zeros((B, num_patches, num_patches), device=matches0.device, dtype=torch.int16)
    label_confs = torch.zeros_like(gt_labels)
    for b in range(B):
        patch_confs = label_confs[b]
        # patches indices for the visible pixels
        patch_indices_0, patch_indices_1 = patch_helper.find_patch(matches0[b][inl0[b]]), patch_helper.find_patch(matches1[b][inl0[b]])
        # if patch_indices_0.shape != patch_indices_1.shape:
    
        # the retrieved patches should be all from the visible list
        # assert (torch.from_numpy(patch_indices_0).to(gt_vis_image0.device).unique() == gt_vis_image0).all() 
        # assert (torch.from_numpy(patch_indices_1).to(gt_vis_image1.device).unique() == gt_vis_image1).all()
        # match the patches, N=0 return all the patches that have at least one corresponding pixel
        matched_patches = patch_helper.match_patch(patch_indices_0, patch_indices_1, N=N, all=False)
        matched_patches_groups = [m[0] for m in matched_patches]
        matched_patches_conf = np.vstack([m[1] for m in matched_patches])
        match_patch_list = np.vstack(matched_patches_groups)
        # import pdb; pdb.set_trace()
        # remove the patches that we are interested, ie, not visible
        if gt_vis_image0.dim() == 0:
            gt_vis_image0 = gt_vis_image0.unsqueeze(0)
        if gt_vis_image1.dim() == 0:
            gt_vis_image1 = gt_vis_image1.unsqueeze(0)
        try:
            for i, pair in enumerate(match_patch_list):
                if (not any(pair[0]==gt for gt in gt_vis_image0)) or (not any(pair[1]==gt for gt in gt_vis_image1)):
                    # import pdb; pdb.set_trace()
                    np.delete(match_patch_list, i)
                    np.delete(matched_patches_conf, i)
        except:
            continue
            # import pdb; pdb.set_trace()

        # to conf matrix
        for m in matched_patches:
            patch_confs[m[0]] = m[1]
        assert np.sum(matched_patches_conf) == patch_confs.sum().item()
        matched_patches = torch.from_numpy(np.vstack(matched_patches_groups)).to(gt_vis_image0.device)
        if one_to_one:
            # look for one-to-one patch matching pairs selcection from image 1s to 2s 
            best_match0_1 = []
            for idx_0, gt_vis0 in enumerate(gt_vis_image0):
                # multiple patches from image 1 correpondending to this patch, need to choose the best (most matching pixels)
                match_idxs = (matched_patches[:, 0] == gt_vis0).nonzero()
                if len(match_idxs) > 1:
                    all_possible_patch1 = []
                    for i in match_idxs:
                        all_possible_patch1 += [patch_confs[gt_vis0.item(), matched_patches[:, 1][i].item()]]
                    match_id = match_idxs[torch.argmax(torch.stack(all_possible_patch1)).item()].item()
                elif len(match_idxs) < 1:
                    continue
                else:
                    match_id = match_idxs.item()
                best_match0_1 += [(gt_vis0.item(), matched_patches[:, 1][match_id].item())]
            # get the new list of matching patches, further check from image 2 to 1
            try:
                best_match0_1 = torch.from_numpy(np.vstack(best_match0_1)).to(gt_vis_image0.device)
            except:
                continue
            # one-to-one matching pairs selcection from image 2s to 1s 
            best_match1_0 = []
            for idx_1, gt_vis1 in enumerate(gt_vis_image1):
                match_idxs = (best_match0_1[:, 1] == gt_vis1).nonzero()
                if len(match_idxs) > 1:
                    all_possible_patch0 = []
                    for i in match_idxs:
                        all_possible_patch0 += [patch_confs[best_match0_1[:, 0][i].item(), gt_vis1.item()]]
                    match_id = match_idxs[torch.argmax(torch.stack(all_possible_patch0)).item()].item()
                elif len(match_idxs) < 1:
                    continue
                else:
                    match_id = match_idxs.item()
                best_match1_0 += [(best_match0_1[:, 0][match_id].item(), gt_vis1.item())]
            try:
                best_match1_0 = torch.from_numpy(np.vstack(best_match1_0)).to(gt_vis_image0.device)
            except:
                continue
        else:
            best_match1_0 = matched_patches
        # print(len(matched_patches), len(best_match0_1), len(best_match1_0)) #195 67 56


        for i, m in enumerate(best_match1_0): 
            gt_labels[b, m[0], m[1]] = 1
            label_confs[b, m[0], m[1]] = patch_confs[m[0], m[1]].item()
        
    return gt_labels, label_confs