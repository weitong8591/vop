# import kornia as K
# import cv2
from time import time
# import pymagsac
# import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
import numpy as np
# import matplotlib.pyplot as plt
# from ..loftr.util import load_torch_image, readh5, loadh5
# import kornia as K
import torch

class PatchCollect():

    def __init__(self, patch_size=16, resize_shape=224, channel=3):
        self.num_patches = resize_shape //patch_size
        self.patch_size = patch_size
        self.resize_shape = resize_shape
        self.channel = channel

    def match_patch(self, patch_indices_1, patch_indices_2, N=5, all=True):
        """find how many correspondences on each patch between image 1 and 2."""

        patch_matches_count = defaultdict(int)

        for idx, patch_idx_2 in enumerate(patch_indices_2):
            # import pdb; pdb.set_trace()
            try:
                patch_idx_1 = patch_indices_1[idx]
                patch_matches_count[(patch_idx_1, patch_idx_2)] += 1
            except:
                continue         
        if all:
            # return both all the matched patches and the keys where includes more than N correspondences
            return patch_matches_count, [(key, value) for key, value in patch_matches_count.items() if value > N]
        else:
            return [(key, value) for key, value in patch_matches_count.items() if value > N]

    def find_patch(self, coordinates):
        patch_indices = []
        for point in coordinates:
            x, y = point
            patch_row = y // self.patch_size
            patch_col = x // self.patch_size
            idx = int(patch_row * self.num_patches + patch_col)
            if 0 <= idx < self.num_patches**2: 
                patch_indices.append(idx)
            else:
                patch_indices.append(-1)
        return np.array(patch_indices)
    
    def patch_mask(self, detected_points):
        """find those patches where no points are detected/matched"""
        return np.unique(self.find_patch(detected_points[:, :2]))
    
    def patch_range(self):
      patch_coordinates = []
      patch_centers = []
      for r in range(self.num_patches):
        for c in range(self.num_patches):
            x1 = c * self.patch_size
            y1 = r * self.patch_size
            x2 = x1 + self.patch_size
            y2 = y1 + self.patch_size
            patch_coordinates.append(((x1, y1), (x2, y2)))
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            patch_centers.append((center_x, center_y))
      return np.array(patch_coordinates), np.array(patch_centers)

    def show_patch(self, image_file, name='a'):
        img = Image.open(image_file).convert('RGB')
        x = np.array(img.resize((self.resize_shape, self.resize_shape)))
        P = self.patch_size
        C = self.channel

        # split image into patches using numpy
        patches = x.reshape(x.shape[0]//P, P, x.shape[1]//P, P, C).swapaxes(1, 2).reshape(-1, P, P, C)
        print(patches.shape)
        # flatten patches
        x_p = np.reshape(patches, (-1, P * P * C))
        # import pdb; pdb.set_trace()

        print('Image shape: ', x.shape)
        print('Number of patches: {} with resolution ({}, {})'.format(self.num_patches, P, P))
        print('Patches shape: ', patches.shape)
        print('Flattened patches shape: ', x_p.shape)

        fig = plt.figure(figsize=(10, 10))
        gridspec = fig.add_gridspec(self.num_patches, self.num_patches, hspace=-0.8)

        # display patches
        
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                  num = i * self.num_patches + j
                  ax = fig.add_subplot(gridspec[i, j])
                  ax.set(xticks=[], yticks=[])
                  ax.imshow(patches[num])
                  ax.text(0.25, 0.25, str(num), color='red', fontsize=12, ha='center')

        plt.savefig(name + '_patches.pdf')
        
    def show_patch_index(self, image_file, name='a', index=0):
        img = Image.open(image_file).convert('RGB')
        x = np.array(img.resize((self.resize_shape, self.resize_shape)))
        P = self.patch_size
        C = self.channel
        # split image into patches using numpy
        patches = x.reshape(int(x.shape[0]//P), P, int(x.shape[1]//P), P, C).swapaxes(1, 2).reshape(-1, P, P, C)
        # flatten patches
        x_p = np.reshape(patches, (-1, P * P * C))
        # # import pdb; pdb.set_trace()

        # print('Image shape: ', x.shape)
        # print('Number of patches: {} with resolution ({}, {})'.format(self.num_patches, P, P))
        # print('Patches shape: ', patches.shape)
        # print('Flattened patches shape: ', x_p.shape)

        patch_to_show = patches[index]
        return patch_to_show

def match_inliers(image0, image1, loftr):
    img1 = load_torch_image(image0)
    img2 = load_torch_image(image1)
    input = {"image0":K.color.rgb_to_grayscale(img1),
              "image1":K.color.rgb_to_grayscale(img2)}
    results  = loftr(input)

    # 1. filter the LoFTR correspondences by the confidence predicted from LoFTR
    mask = results['confidence']>0.9
    kpts0 = results['keypoints0']
    kpts1 = results['keypoints1']
    mkpts0 = kpts0[mask]
    mkpts1 = kpts1[mask]
    mconf = results['confidence'][mask]
    # correspondences = torch.concat((mkpts1, mkpts1), dim=-1)
    if mkpts0.shape[0]<7:
        return False, 0
    # try:
    #     normalized_mkpts1 = cv2.undistortPoints(np.expand_dims(mkpts0, axis=1), cameraMatrix=K1, distCoeffs=None)[:, 0, :]
    # except Exception as e:
    #     print(e)
    #     return False
    # normalized_mkpts2 = cv2.undistortPoints(np.expand_dims(mkpts1, axis=1), cameraMatrix=K2, distCoeffs=None)[:, 0, :]

    # avgDiagonal = (K1[0][0] + K1[1][1] + K2[0][0] + K2[1][1]) / 4 
    # normalizedThreshold = 0.75/ avgDiagonal
    # import pdb; pdb.set_trace()
    # find the inlier masks by RANSAC and keep only the inliers
    try:
        F, inliers = cv2.findFundamentalMat(mkpts0.numpy(), mkpts1.numpy(), ransacReprojThreshold=0.75, confidence=0.99, method=cv2.USAC_MAGSAC, maxIters=10000)
        # E, inliers = cv2.findEssentialMat(normalized_mkpts1, normalized_mkpts2, np.eye(3), threshold=normalizedThreshold, prob=0.99,method=cv2.USAC_MAGSAC)
    except:
        return False, 0

    mkpts0_inliers = mkpts0[(inliers==1).squeeze()]
    mkpts1_inliers = mkpts1[(inliers==1).squeeze()]

    return True, mkpts0_inliers.shape[0]

