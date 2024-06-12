
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import h5py

def precalculate_patch_colors(descriptors):
    """Use PCA to reduce the dimensionality of the patch descriptors to 3 and normalize these for RGB color space.

    Parameters:
    - descriptors: The high-dimensional descriptors for each patch.

    Returns:
    - normalized_colors: An array of RGB colors for each patch.
    """
    # Apply PCA to reduce descriptors to 3 dimensions
    pca = PCA()#n_components=3
    pca.n_components = 3
    reduced_descriptors = pca.fit_transform(descriptors)

    # Normalize the reduced descriptors to [0, 255] for RGB values
    min_val, max_val = np.min(reduced_descriptors, axis=0), np.max(reduced_descriptors, axis=0)
    normalized_colors = 255 * (reduced_descriptors - min_val) / (max_val - min_val)

    return normalized_colors

def overlay_colors_on_image(image, colors, patch_height, patch_width, gap_size, alpha=0.6):
    """Overlay precalculated colors on their corresponding patches in the image with alpha blending.

    Parameters:
    - image: The original image on which to overlay colors.
    - colors: The array of RGB colors for each patch.
    - patch_height, patch_width: Dimensions of each patch.
    - gap_size: The size of the gap between patches.
    - alpha: The alpha value for blending the color overlay and the original image.

    Returns:
    - processed_image: The image with colors overlaid on patches.
    """
    processed_image = np.copy(image)
    num_patches_x = image.shape[1] // (patch_width + gap_size)
    num_patches_y = image.shape[0] // (patch_height + gap_size)

    patch_index = 0
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            if patch_index >= len(colors):  # Safety check
                break

            # Define patch coordinates
            x_start = j * (patch_width + gap_size) + gap_size
            y_start = i * (patch_height + gap_size) + gap_size
            x_end = x_start + patch_width
            y_end = y_start + patch_height

            # Assign color with alpha blending
            color = colors[patch_index]
            for c in range(3):  # RGB channels
                processed_image[y_start:y_end, x_start:x_end, c] = (
                    alpha * color[c] + (1 - alpha) * processed_image[y_start:y_end, x_start:x_end, c]
                )

            patch_index += 1

    return processed_image

def process_image_into_patches(image, patch_height, patch_width, gap_size):
    """Divides an image into patches with specified height and width, including gaps around them, returning a new
    image."""
    rows, cols, _ = image.shape
    num_patches_x = cols // patch_width
    num_patches_y = rows // patch_height

    # Calculating new dimensions
    new_cols = num_patches_x * (patch_width + gap_size) + gap_size
    new_rows = num_patches_y * (patch_height + gap_size) + gap_size

    # Creating a new image with gaps
    new_image = np.full((new_rows, new_cols, 3), 255, dtype=np.uint8)  # Start with a white image

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patch = image[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width]
            y = i * (patch_height + gap_size) + gap_size
            x = j * (patch_width + gap_size) + gap_size
            new_image[y:y+patch_height, x:x+patch_width] = patch

    return new_image

def create_composite_image(image1, image2, gap_between_images, orientation='vertical'):
    """Combines two images with a specified gap between them into a single composite image, either horizontally or
    vertically based on the orientation parameter.

    Parameters:
    - image1, image2: The two images to combine.
    - gap_between_images: The number of pixels for the gap between the two images.
    - orientation: 'horizontal' or 'vertical' - the orientation for combining the images.

    Returns:
    - composite_image: The combined image with a gap between image1 and image2.
    """
    if orientation == 'horizontal':
        total_height = max(image1.shape[0], image2.shape[0])
        total_width = image1.shape[1] + gap_between_images + image2.shape[1]
        composite_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)  # Start with a white image
        composite_image[:image1.shape[0], :image1.shape[1]] = image1
        composite_image[:image2.shape[0], image1.shape[1] + gap_between_images:] = image2
    elif orientation == 'vertical':
        total_width = max(image1.shape[1], image2.shape[1])
        total_height = image1.shape[0] + gap_between_images + image2.shape[0]
        composite_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)  # Start with a white image
        composite_image[:image1.shape[0], :image1.shape[1]] = image1
        composite_image[image1.shape[0] + gap_between_images:image1.shape[0] + gap_between_images + image2.shape[0], :image2.shape[1]] = image2
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'")

    return composite_image

def draw_connections_with_numbers(composite_image, matches, patch_height, patch_width, gap_size, gap_between_images, fontsize=18, orientation='horizontal'):
    """Places a number in the middle of corresponding patches across two processed images within a composite image,
    based on a list of matches in row-major order. Supports both horizontal and vertical orientations.

    Parameters:
    - composite_image: The composite image containing both processed images and a gap.
    - matches: A list of tuples, where each tuple contains the indices [idx1, idx2] of matching patches.
    - patch_height, patch_width: The dimensions of each patch.
    - gap_size: The size of the gap around each patch.
    - gap_between_images: The gap between the two processed images within the composite image.
    - fontsize: The font size of the numbers to be placed.
    - orientation: 'horizontal' or 'vertical' - the orientation for combining the images.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(composite_image)

    coordinates = []

    if orientation == 'horizontal':
        num_patches_x_per_image = (composite_image.shape[1] - gap_between_images) // (patch_width + gap_size) // 2
        num_patches_y = composite_image.shape[0] // (patch_height + gap_size)

        for match_number, [idx1, idx2] in enumerate(matches, start=1):
            # Calculate positions in row-major order

            row1, col1 = divmod(idx1, num_patches_x_per_image)
            row2, col2 = divmod(idx2, num_patches_x_per_image)

            y1 = row1 * (patch_height + gap_size) + gap_size + patch_height // 2
            x1 = col1 * (patch_width + gap_size) + gap_size + patch_width // 2

            y2 = row2 * (patch_height + gap_size) + gap_size + patch_height // 2
            x2 = (col2 + num_patches_x_per_image) * (patch_width + gap_size) + gap_size + patch_width // 2 + gap_between_images

            coordinates.append((x1, y1, x2, y2, match_number))

    elif orientation == 'vertical':
        num_patches_x = composite_image.shape[1] // (patch_width + gap_size)
        num_patches_y_per_image = (composite_image.shape[0] - gap_between_images) // (patch_height + gap_size) // 2

        for match_number, [idx1, idx2] in enumerate(matches, start=1):
            # Calculate positions in column-major order for vertical orientation
            col1, row1 = divmod(idx1, num_patches_y_per_image)
            col2, row2 = divmod(idx2, num_patches_y_per_image)

            x1 = row1 * (patch_width + gap_size) + gap_size + patch_width // 2#col1 * (patch_width + gap_size) + gap_size + patch_width // 2
            y1 =  col1 * (patch_height + gap_size) + gap_size + patch_height // 2#row1 * (patch_height + gap_size) + gap_size + patch_height // 2

            x2 = row2 * (patch_width + gap_size) + gap_size + patch_width // 2#col2 * (patch_width + gap_size) + gap_size + patch_width // 2
            y2 = (col2 + num_patches_y_per_image) * (patch_height + gap_size) + gap_size + patch_height // 2 + gap_between_images + 2 * fontsize
            #(row2 + num_patches_y_per_image) * (patch_height + gap_size) + gap_size + patch_height // 2 + gap_between_images + 2 * fontsize

            coordinates.append((x1, y1, x2, y2, match_number))

    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'")

    # Place the connection number in the middle of each corresponding patch for both orientations
    for (x1, y1, x2, y2, match_number) in coordinates:
        plt.text(x1, y1, str(match_number), ha='center', va='center', color='white', fontsize=fontsize, weight='bold')
        plt.text(x2, y2, str(match_number), ha='center', va='center', color='white', fontsize=fontsize, weight='bold')

    plt.axis('off')
    plt.show()
    plt.savefig(f"demo/demo_{patch_height}.pdf")

def draw_connections(composite_image, patch_height, patch_width, gap_size, gap_between_images, num_connections):
    """Draws lines connecting randomly selected patches across two processed images within a composite image."""
    rows, cols, _ = composite_image.shape
    num_patches_x = (cols - gap_between_images) // (patch_width + gap_size)
    num_patches_y = rows // (patch_height + gap_size)

    plt.figure(figsize=(12, 8))
    plt.imshow(composite_image)
    for _ in range(num_connections):
        patch_y = random.randint(0, num_patches_y - 1)
        patch_x1 = random.randint(0, num_patches_x // 2 - 2)
        patch_x2 = random.randint(num_patches_x // 2, num_patches_x - 2)

        y = patch_y * (patch_height + gap_size) + gap_size + patch_height // 2
        x1 = patch_x1 * (patch_width + gap_size) + gap_size + patch_width // 2
        x2 = patch_x2 * (patch_width + gap_size) + gap_size + patch_width // 2 + gap_between_images

        plt.plot([x1, x2], [y, y], 'r-')

    plt.axis('off')
    plt.show()

def downsample_descriptors(descriptors, original_grid_size, new_grid_size, operation='mean'):
    """Downsample descriptors by averaging.

    Parameters:
    - descriptors: A (256 x 256) array where each row is a descriptor for a patch.
    - original_grid_size: The grid size of the original descriptors (e.g., (16, 16) for 16x16 patches).
    - new_grid_size: The desired grid size after downsampling (e.g., (8, 8)).

    Returns:
    - A new array of downsampled descriptors.
    """
    # Calculate the number of original patches that fit into a new patch
    patches_per_new_patch = (original_grid_size[0] // new_grid_size[0]) * (original_grid_size[1] // new_grid_size[1])

    # Initialize an array to hold the downsampled descriptors
    downsampled_descriptors = np.zeros((new_grid_size[0] * new_grid_size[1], descriptors.shape[1]))

    # Helper function to calculate the index in the flattened grid
    def calc_index(row, col, cols):
        return row * cols + col

    # Iterate over the new grid
    for new_row in range(new_grid_size[0]):
        for new_col in range(new_grid_size[1]):
            # Find the original patches that correspond to the current new patch
            start_row = new_row * (original_grid_size[0] // new_grid_size[0])
            start_col = new_col * (original_grid_size[1] // new_grid_size[1])
            patch_descriptors = []

            # Aggregate descriptors of the original patches
            for i in range(start_row, start_row + (original_grid_size[0] // new_grid_size[0])):
                for j in range(start_col, start_col + (original_grid_size[1] // new_grid_size[1])):
                    idx = calc_index(i, j, original_grid_size[1])
                    patch_descriptors.append(descriptors[idx])

            # Average the descriptors and assign to the downsampled descriptor
            if operation == 'mean':
                downsampled_descriptors[calc_index(new_row, new_col, new_grid_size[1])] = np.mean(patch_descriptors, axis=0)
            elif operation == 'max':
                downsampled_descriptors[calc_index(new_row, new_col, new_grid_size[1])] = np.max(patch_descriptors, axis=0)

    return downsampled_descriptors

def update_matches_for_downsampled_grid(matches, original_grid_size, new_grid_size):
    """Update the match indices for a downsampled grid ensuring each patch appears only once.

    Parameters:
    - matches: List of tuples/lists with the original matching indices.
    - original_grid_size: Tuple indicating the size of the original grid (rows, cols).
    - new_grid_size: Tuple indicating the size of the new (downsampled) grid (rows, cols).

    Returns:
    - Updated list of matches with indices corresponding to the downsampled grid, ensuring uniqueness.
    """
    # Calculate the factor by which the number of patches per row/column is reduced
    row_reduction_factor = original_grid_size[0] // new_grid_size[0]
    col_reduction_factor = original_grid_size[1] // new_grid_size[1]

    # Function to map original index to new index
    def map_index_to_downsampled(idx, original_grid_size, new_grid_size, reduction_factors):
        original_row, original_col = divmod(idx, original_grid_size[1])
        new_row = original_row // reduction_factors[0]
        new_col = original_col // reduction_factors[1]
        return new_row * new_grid_size[1] + new_col

    used_indices = set()  # Keep track of used indices to ensure uniqueness
    updated_matches = []

    for match in matches:
        # Update match indices for the downsampled grid
        new_idx1 = map_index_to_downsampled(match[0], original_grid_size, new_grid_size, (row_reduction_factor, col_reduction_factor))
        new_idx2 = map_index_to_downsampled(match[1], original_grid_size, new_grid_size, (row_reduction_factor, col_reduction_factor))

        # Ensure uniqueness by checking if either index has already been used
        if new_idx1 not in used_indices and new_idx2 not in used_indices:
            updated_matches.append([new_idx1, new_idx2])
            # Mark these indices as used
            used_indices.add(new_idx1)
            used_indices.add(new_idx2)

    return updated_matches

def quaternion_to_rotation_matrix(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                  [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                  [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
    return R

def camera_center_to_translation(c, qvec):
    R = quaternion_to_rotation_matrix(qvec)
    return (-1) * np.matmul(R, c)

# Simple COLMAP camera class.
class Camera:
    def __init__(self):
        self.camera_model = None
        self.intrinsics = None
        self.qvec = None
        self.t = None

    def set_intrinsics(self, camera_model, intrinsics):
        self.camera_model = camera_model
        self.intrinsics = intrinsics

    def set_pose(self, qvec, t):
        self.qvec = qvec
        self.t = t

def loadh5(dump_file_full_name):
    """Loads a h5 file as dictionary."""

    try:
        with h5py.File(dump_file_full_name, 'r') as h5file:
            dict_from_file = readh5(h5file)
    except Exception as e:
        print("Error while loading {}".format(dump_file_full_name))
        raise e

    return dict_from_file


def readh5(h5node):
    """Recursive function to read h5 nodes as dictionary."""
    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key][:]
    return dict_from_file
