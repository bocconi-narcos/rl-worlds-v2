import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from .vit import ViT, Transformer, PreNorm, Attention, FeedForward
from src.utils.weight_init import initialize_weights


class MaskedViT(nn.Module):
    """
    Masked Vision Transformer for self-supervised pretraining.
    Extends ViT to support random masking of patches and learnable mask tokens.
    """
    def __init__(self, 
                 image_size, 
                 patch_size, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 channels=3, 
                 dim_head=64, 
                 dropout=0., 
                 emb_dropout=0.,
                 mask_token_dim=None,
                 num_mask_tokens=1):
        super().__init__()
        
        # Use mask_token_dim if provided, otherwise use dim
        self.mask_token_dim = mask_token_dim if mask_token_dim is not None else dim
        
        # Create the base ViT without classification head
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=0,  # No classification head
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool='cls',  # Use CLS token pooling
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        
        # Calculate number of patches
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        # Learnable mask tokens
        self.mask_tokens = nn.Parameter(torch.randn(1, num_mask_tokens, self.mask_token_dim) * 0.02)
        
        # Projection layer to match mask token dimension to ViT dimension
        if self.mask_token_dim != dim:
            self.mask_projection = nn.Linear(self.mask_token_dim, dim)
        else:
            self.mask_projection = nn.Identity()
        
        self.apply(initialize_weights)
    
    def random_masking(self, x, mask_ratio=0.75):
        """
        Randomly mask patches in the input.
        
        Args:
            x: Input tensor of shape [B, N, D] where N is number of patches
            mask_ratio: Ratio of patches to mask
        
        Returns:
            masked_x: Input with masked patches replaced by mask tokens
            mask: Boolean mask indicating which patches are masked
            ids_restore: Indices to restore original order
        """
        N, D = x.shape[1], x.shape[2]  # N: number of patches, D: patch dimension
        len_keep = int(N * (1 - mask_ratio))
        
        # Generate random noise for sorting
        noise = torch.rand(x.shape[0], N, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([x.shape[0], N], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x, mask_ratio=0.75, return_mask=False):
        """
        Forward pass with optional masking.
        
        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask (0 = no masking)
            return_mask: Whether to return the mask for loss computation
        
        Returns:
            If return_mask=False: encoded representation [B, D]
            If return_mask=True: (encoded_representation, mask, ids_restore)
        """
        if mask_ratio == 0:
            # No masking, use standard ViT forward pass
            return self.vit(x)
        
        # Get patch embeddings from ViT's patch embedding layer
        x = self.vit.to_patch_embedding(x)  # [B, N, D]
        B, N, D = x.shape
        
        # Add positional embeddings
        x = x + self.vit.pos_embedding[:, 1:, :]  # Skip CLS token position
        
        # Apply random masking
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add CLS token
        cls_tokens = repeat(self.vit.cls_token, '() n d -> b n d', b=B)
        x_masked = torch.cat([cls_tokens, x_masked], dim=1)
        
        # Add positional embeddings for visible patches + CLS
        pos_embed = self.vit.pos_embedding[:, :x_masked.shape[1], :]
        x_masked = x_masked + pos_embed
        
        # Apply transformer
        x_masked = self.vit.dropout(x_masked)
        x_masked = self.vit.transformer(x_masked)
        
        # Get CLS token representation
        encoded = x_masked[:, 0]  # [B, D]
        
        if return_mask:
            return encoded, mask, ids_restore
        else:
            return encoded
    
    def forward_with_mask_tokens(self, x, mask_ratio=0.75):
        """
        Forward pass that replaces masked patches with learnable mask tokens.
        This is used for the predictor branch.
        
        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask
        
        Returns:
            encoded_representation: [B, D]
        """
        # Get patch embeddings
        x = self.vit.to_patch_embedding(x)  # [B, N, D]
        B, N, D = x.shape
        
        # Add positional embeddings
        x = x + self.vit.pos_embedding[:, 1:, :]
        
        # Apply random masking
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Create full sequence with mask tokens
        mask_tokens = repeat(self.mask_tokens, '() n d -> b n d', b=B)
        mask_tokens = self.mask_projection(mask_tokens)  # Project to ViT dimension
        
        # Replace masked patches with mask tokens
        x_full = x.clone()
        mask_bool = mask.bool()
        
        # For each sample, replace masked patches with mask tokens
        for b in range(B):
            masked_indices = torch.where(mask_bool[b])[0]
            if len(masked_indices) > 0:
                # Use mask tokens cyclically if we have more masked patches than mask tokens
                num_mask_tokens = self.mask_tokens.shape[1]
                for i, idx in enumerate(masked_indices):
                    token_idx = i % num_mask_tokens
                    x_full[b, idx] = mask_tokens[b, token_idx]
        
        # Add CLS token
        cls_tokens = repeat(self.vit.cls_token, '() n d -> b n d', b=B)
        x_full = torch.cat([cls_tokens, x_full], dim=1)
        
        # Add positional embeddings
        pos_embed = self.vit.pos_embedding[:, :x_full.shape[1], :]
        x_full = x_full + pos_embed
        
        # Apply transformer
        x_full = self.vit.dropout(x_full)
        x_full = self.vit.transformer(x_full)
        
        # Get CLS token representation
        encoded = x_full[:, 0]  # [B, D]
        
        return encoded, mask, ids_restore 