import torch
class oldMaskGenerator:
    """
    A simplified, self-contained class for generating random block masks for
    batches of images or videos.

    This class generates a binary mask for each item in a batch. A value of '1'
    in the mask indicates a patch is visible, and '0' indicates it is masked.
    The masking is performed by creating a specified number of random blocks
    and setting their locations to zero.

    Attributes:
        input_size (tuple): The dimensions of the input data (T, H, W) for videos
                            or (H, W) for images.
        patch_size (tuple): The size of each patch (P_t, P_h, P_w) for videos or
                            (P_h, P_w) for images.
        num_blocks (int): The number of masking blocks to generate per sample.
        masking_ratio (float): The approximate fraction of total patches to mask.
        is_video (bool): Flag indicating if the input is video data.
        grid_size (tuple): The dimensions of the input in terms of patches.
        total_patches (int): The total number of patches in the grid.
        patches_per_block (int): The number of patches to include in each block.
    """

    def __init__(
        self,
        input_size,
        patch_size,
        num_blocks=1,
        masking_ratio=0.5,
    ):
        
        raise DeprecationWarning("This class is deprecated.")
        """
        Initializes the MaskGenerator.

        Args:
            input_size (tuple): The size of the input data. Should be a tuple of
                                (height, width) for images or
                                (frames, height, width) for videos.
            patch_size (tuple): The size of each patch. Should be a tuple of
                                (patch_h, patch_w) for images or
                                (patch_t, patch_h, patch_w) for videos.
            num_blocks (int, optional): The number of masking blocks to generate.
                                        Defaults to 1.
            masking_ratio (float, optional): The target fraction of patches to mask.
                                             Must be between 0 and 1. Defaults to 0.5.
        """
        if not (0 < masking_ratio < 1):
            raise ValueError("masking_ratio must be between 0 and 1.")

        self.is_video = (len(input_size) == 3)
        if self.is_video:
            if len(patch_size) != 3:
                raise ValueError("Patch size must have 3 dimensions for video data (t, h, w).")
            self.input_size = input_size
            self.patch_size = patch_size
            self.grid_size = (
                input_size[0] // patch_size[0],
                input_size[1] // patch_size[1],
                input_size[2] // patch_size[2],
            )
        else:
            if len(input_size) != 2 or len(patch_size) != 2:
                raise ValueError("Input and patch size must have 2 dimensions for image data (h, w).")
            # Prepend a temporal dimension of 1 for consistent logic
            self.input_size = (1,) + input_size
            self.patch_size = (1,) + patch_size
            self.grid_size = (
                1,
                input_size[0] // patch_size[0],
                input_size[1] // patch_size[1],
            )

        self.num_blocks = num_blocks
        self.masking_ratio = masking_ratio

        self.total_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        num_masked_patches = int(self.total_patches * self.masking_ratio)

        if self.total_patches == 0:
            raise ValueError("Input size is smaller than patch size in at least one dimension.")

        # Ensure at least one patch is masked per block
        self.patches_per_block = max(1, num_masked_patches // self.num_blocks)

    def _generate_block_mask(self):
        """Generates a single mask with random blocks."""
        mask = torch.ones(self.grid_size, dtype=torch.int32)

        for _ in range(self.num_blocks):
            # 1. Determine the shape of the masking block
            # We approximate the block volume and find integer factors.
            # This is a simplified version of the aspect ratio logic.
            target_volume = self.patches_per_block
            
            # Sample block dimensions
            block_t = torch.randint(1, self.grid_size[0] + 1, (1,)).item()
            block_h = torch.randint(1, self.grid_size[1] + 1, (1,)).item()
            block_w = torch.randint(1, self.grid_size[2] + 1, (1,)).item()

            # Scale dimensions to approximate target volume
            scale = (target_volume / (block_t * block_h * block_w)) ** (1/3)
            block_t = min(self.grid_size[0], max(1, int(block_t * scale)))
            block_h = min(self.grid_size[1], max(1, int(block_h * scale)))
            block_w = min(self.grid_size[2], max(1, int(block_w * scale)))

            # 2. Determine the top-left-front position of the block
            start_t = torch.randint(0, self.grid_size[0] - block_t + 1, (1,)).item()
            start_h = torch.randint(0, self.grid_size[1] - block_h + 1, (1,)).item()
            start_w = torch.randint(0, self.grid_size[2] - block_w + 1, (1,)).item()

            # 3. Apply the mask
            mask[
                start_t : start_t + block_t,
                start_h : start_h + block_h,
                start_w : start_w + block_w,
            ] = 0

        return mask

    def __call__(self, batch_size):
        """
        Generates a batch of masks.

        Args:
            batch_size (int): The number of masks to generate for the batch.

        Returns:
            torch.Tensor: A tensor of masks.
                          Shape for images: (batch_size, grid_h, grid_w).
                          Shape for videos: (batch_size, grid_t, grid_h, grid_w).
        """
        batch_masks = [self._generate_block_mask() for _ in range(batch_size)]
        
        collated_masks = torch.stack(batch_masks)

        # Remove the temporal dimension for image data
        if not self.is_video:
            collated_masks = collated_masks.squeeze(1)

        return collated_masks
    

import torch
import math

import math
import random
from typing import Tuple, Union

import torch


import math
import torch
from typing import Tuple, Union

class MaskGenerator:
    """
    A simplified random block mask generator for images or videos.

    This class generates a boolean mask to be applied to a batch of data.
    The mask indicates which patches to keep (`True`) and which to mask out (`False`).
    It is designed to be easily integrated into a PyTorch training loop.

    Args:
        input_size (tuple): The size of the input data. Should be a tuple of
                            (height, width) for images or
                            (frames, height, width) for videos.
        patch_size (tuple): The size of each patch. Should be a tuple of
                            (patch_h, patch_w) for images or
                            (patch_t, patch_h, patch_w) for videos.
        num_blocks (int, optional): The number of masking blocks to generate.
                                    Defaults to 1.
        masking_ratio (float, optional): The target fraction of patches to mask.
                                           Must be between 0 and 1. Defaults to 0.5.
    """
    def __init__(self, input_size, patch_size, num_blocks=1, masking_ratio=0.5):
        if not (0 < masking_ratio < 1):
            raise ValueError("masking_ratio must be between 0 and 1.")

        self.is_video = len(input_size) == 3
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.masking_ratio = masking_ratio

        if self.is_video:
            self.frames, self.height, self.width = self.input_size
            self.patch_t, self.patch_h, self.patch_w = self.patch_size
            self.num_patches_t = self.frames // self.patch_t
        else:
            self.height, self.width = self.input_size
            self.patch_h, self.patch_w = self.patch_size
            self.num_patches_t = 1
            self.patch_t = 1


        self.num_patches_h = self.height // self.patch_h
        self.num_patches_w = self.width // self.patch_w
        self.total_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w

    def __call__(self, batch_size):
        """
        Generates a batch of masks.

        Args:
            batch_size (int): The number of samples in the batch.

        Returns:
            torch.Tensor: A boolean tensor of shape `[B, N]`, where `B` is the
                          batch size and `N` is the total number of patches.
                          `True` indicates a patch to keep. All rows in the
                          batch will have the same number of `True` values.
        """
        all_masks = []
        for _ in range(batch_size):
            mask = torch.ones(self.num_patches_t, self.num_patches_h, self.num_patches_w, dtype=torch.bool)
            num_masked_patches = 0
            for _ in range(self.num_blocks):
                # Determine block size
                target_masked_for_block = int(self.total_patches * self.masking_ratio / self.num_blocks)
                if self.is_video:
                    block_t = torch.randint(1, self.num_patches_t + 1, (1,)).item()
                    block_h = torch.randint(1, self.num_patches_h + 1, (1,)).item()
                    block_w = int(target_masked_for_block / (block_t * block_h))
                    block_w = max(1, min(block_w, self.num_patches_w))

                else:
                    block_t = 1
                    aspect_ratio = torch.rand(1).item() * 2 + 0.5  # Random aspect ratio
                    block_h = int(math.sqrt(target_masked_for_block * aspect_ratio))
                    block_w = int(math.sqrt(target_masked_for_block / aspect_ratio))
                    block_h = max(1, min(block_h, self.num_patches_h))
                    block_w = max(1, min(block_w, self.num_patches_w))


                # Determine block position
                start_t = torch.randint(0, self.num_patches_t - block_t + 1, (1,)).item() if self.is_video else 0
                start_h = torch.randint(0, self.num_patches_h - block_h + 1, (1,)).item()
                start_w = torch.randint(0, self.num_patches_w - block_w + 1, (1,)).item()

                # Apply mask
                mask[start_t:start_t + block_t, start_h:start_h + block_h, start_w:start_w + block_w] = False
            all_masks.append(mask.flatten())

        # Find the minimum number of kept patches across all masks in the batch
        kept_patches_counts = [m.sum().item() for m in all_masks]
        min_kept_patches = min(kept_patches_counts)

        # Adjust each mask to have the same number of kept patches
        final_masks = []
        for mask in all_masks:
            kept_indices = torch.where(mask)[0]
            
            # Randomly select which patches to keep from the available set
            shuffled_indices = kept_indices[torch.randperm(len(kept_indices))]
            trimmed_indices = shuffled_indices[:min_kept_patches]

            # Create the final mask with the exact number of kept patches
            new_mask = torch.zeros_like(mask, dtype=torch.bool)
            new_mask[trimmed_indices] = True
            final_masks.append(new_mask)
            
        return torch.stack(final_masks)
