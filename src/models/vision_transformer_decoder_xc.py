# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Vision Transformer Decoder with Cross-Attention (VisionTransformerDecoderXC)

This decoder is architecturally identical to the VisionTransformer encoder, except that
every block starts with a cross-attention sub-layer whose keys/values come from the 
predictor's latent tokens (Zpred). The decoder follows the same per-frame causal mask
as produced by build_action_block_causal_attention_mask.

Architecture flow:
raw_frames → VisionTransformer → Zenc → VisionTransformerPredictorAC → Zpred
                                                    │
                                                    ▼
                              VisionTransformerDecoderXC → reconstructed_frames
"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import ACBlock, CrossAttention, build_action_block_causal_attention_mask
from src.models.utils.patch_embed import PatchEmbed, PatchEmbed3D
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.fb_tensors import trunc_normal_


class CrossAttentionACBlock(nn.Module):
    """
    ACBlock with prepended cross-attention layer.
    Cross-attention uses keys/values from predictor tokens, queries from decoder tokens.
    """
    
    def __init__(
        self,
        dim: int,
        memory_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer = nn.GELU,
        wide_silu: bool = True,
        norm_layer = nn.LayerNorm,
        use_sdpa: bool = True,
        is_causal: bool = False,
        grid_size: int = 16,
        use_rope: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        # Cross-attention layer (prepended)
        self.norm_cross = norm_layer(dim)
        self.norm_memory = norm_layer(memory_dim)
        
        # Project memory to same dimension as decoder if needed
        if memory_dim != dim:
            self.memory_proj = nn.Linear(memory_dim, dim, bias=False)
        else:
            self.memory_proj = nn.Identity()
            
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_sdpa=use_sdpa
        )
        self.cross_drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Standard ACBlock components
        self.ac_block = ACBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            wide_silu=wide_silu,
            norm_layer=norm_layer,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
            grid_size=grid_size,
            use_rope=use_rope,
            **kwargs
        )
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
        H: Optional[int] = None,
        W: Optional[int] = None,
        action_tokens: int = 0
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention followed by self-attention.
        
        Args:
            x: Decoder tokens [B, N, dim]
            memory: Predictor tokens [B, M, memory_dim] 
            mask: Optional mask for self-attention
            attn_mask: Causal attention mask
            T, H, W: Temporal/spatial dimensions for RoPE
            action_tokens: Number of action tokens for ACBlock
        
        Returns:
            Updated decoder tokens [B, N, dim]
        """
        # Cross-attention: queries from decoder, keys/values from predictor
        q_norm = self.norm_cross(x)
        kv_norm = self.norm_memory(memory)
        kv_proj = self.memory_proj(kv_norm)
        
        # Apply cross-attention - CrossAttention returns the result directly
        cross_out = self.cross_attn(q_norm, kv_proj)
        x = x + self.cross_drop_path(cross_out)
        
        # Self-attention via ACBlock
        x = self.ac_block(
            x, 
            mask=mask, 
            attn_mask=attn_mask, 
            T=T, 
            H=H, 
            W=W, 
            action_tokens=action_tokens
        )
        
        return x


class VisionTransformerDecoderXC(nn.Module):
    """
    Vision Transformer Decoder with Cross-Attention
    
    Architecturally identical to VisionTransformer encoder, except every block
    starts with a cross-attention sub-layer. Uses the same causal masking strategy
    as the predictor for temporal consistency.
    """
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        memory_dim: int = 768,  # Dimension of predictor tokens
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer = nn.LayerNorm,
        init_std: float = 0.02,
        uniform_power: bool = False,
        use_silu: bool = False,
        wide_silu: bool = True,
        use_sdpa: bool = True,
        use_activation_checkpointing: bool = False,
        use_rope: bool = False,
        handle_nonsquare_inputs: bool = True,
        is_frame_causal: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.is_frame_causal = is_frame_causal
        
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        
        self.use_activation_checkpointing = use_activation_checkpointing
        
        # Grid dimensions for causal masking
        self.grid_height = self.img_height // self.patch_size
        self.grid_width = self.img_width // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        
        # Calculate number of patches
        if self.is_video:
            self.num_patches = self.grid_depth * self.grid_height * self.grid_width
        else:
            self.num_patches = self.grid_height * self.grid_width
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Positional embeddings (same as encoder)
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), 
                requires_grad=False
            )
        
        # Decoder blocks with cross-attention
        self.blocks = nn.ModuleList([
            CrossAttentionACBlock(
                dim=embed_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=nn.SiLU if use_silu else nn.GELU,
                wide_silu=wide_silu,
                norm_layer=norm_layer,
                use_sdpa=use_sdpa,
                is_causal=is_frame_causal,
                grid_size=self.grid_height,
                use_rope=use_rope,
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim)
        
        # Token-to-patch projection head
        self.decoder_proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans, bias=True)
        
        # Learnable decoder tokens - initialized properly as parameter
        self.decoder_tokens = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * init_std)
        
        # Build and cache causal attention mask
        self.attn_mask = None
        if self.is_frame_causal and self.is_video:
            self.attn_mask = build_action_block_causal_attention_mask(
                self.grid_depth, self.grid_height, self.grid_width, add_tokens=0
            )
        
        # Initialize weights
        self.init_std = init_std
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)
        self.apply(self._init_weights)
        self._rescale_blocks()
    
    def _init_pos_embed(self, pos_embed: torch.Tensor) -> None:
        """Initialize positional embeddings using sincos (same as encoder)."""
        embed_dim = pos_embed.size(-1)
        if self.is_video:
            sincos = get_3d_sincos_pos_embed(
                embed_dim, 
                self.grid_height, 
                self.grid_depth, 
                cls_token=False, 
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(
                embed_dim, 
                self.grid_height, 
                cls_token=False
            )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
    
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights (same pattern as encoder)."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _rescale_blocks(self) -> None:
        """Rescale block weights (same as encoder)."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.ac_block.attn.proj.weight.data, layer_id + 1)
            rescale(layer.ac_block.mlp.fc2.weight.data, layer_id + 1)
            # Rescale cross-attention if it has learnable projection
            if hasattr(layer, 'memory_proj') and hasattr(layer.memory_proj, 'weight'):
                rescale(layer.memory_proj.weight.data, layer_id + 1)
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens back to pixel space.
        
        Args:
            x: Patch tokens [B, N, patch_size^2 * in_chans]
        
        Returns:
            Reconstructed images/videos [B, C, H, W] or [B, C, T, H, W]
        """
        B, N, D = x.shape
        
        if self.is_video:
            # Video case: [B, T*H*W, tubelet_size * patch_size^2 * C]
            T = self.grid_depth
            H = self.grid_height  
            W = self.grid_width
            
            # Extract channel dimension
            patch_dim = self.patch_size * self.patch_size
            in_chans = D // patch_dim
            
            # Reshape to [B, T, H, W, patch_size, patch_size, C]
            x = x.view(B, T, H, W, self.patch_size, self.patch_size, in_chans)
            
            # Rearrange to [B, C, T, H*patch_size, W*patch_size]  
            x = x.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
            x = x.view(B, in_chans, T, H * self.patch_size, W * self.patch_size)
            
        else:
            # Image case: [B, H*W, patch_size^2 * C]
            H = self.grid_height
            W = self.grid_width
            
            # Extract channel dimension
            patch_dim = self.patch_size * self.patch_size
            in_chans = D // patch_dim
            
            # Reshape to [B, H, W, patch_size, patch_size, C]
            x = x.view(B, H, W, self.patch_size, self.patch_size, in_chans)
            
            # Rearrange to [B, C, H*patch_size, W*patch_size]
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            x = x.view(B, in_chans, H * self.patch_size, W * self.patch_size)
        
        return x
    
    def interpolate_pos_encoding(
        self, 
        target_shape: Tuple[int, ...], 
        pos_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings for different input sizes.
        Mirrors the encoder's interpolation logic.
        """
        _, N, dim = pos_embed.shape
        
        if self.is_video:
            T, H, W = target_shape
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size
            
            # If already correct size, return as is
            if (T * H * W) == N:
                return pos_embed
            
            # Compute original grid dimensions
            N_t = self.num_frames // self.tubelet_size
            N_h = self.img_height // self.patch_size
            N_w = self.img_width // self.patch_size
            
            assert N_h * N_w * N_t == N, f"Positional embedding size mismatch: {N} vs {N_h * N_w * N_t}"
            
            # 3D interpolation
            scale_factor = (T / N_t, H / N_h, W / N_w)
            pos_embed = F.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            
        else:
            H, W = target_shape
            H = H // self.patch_size
            W = W // self.patch_size
            
            # If already correct size, return as is
            if (H * W) == N:
                return pos_embed
            
            # 2D interpolation  
            scale_factor = math.sqrt((H * W) / N)
            grid_size = int(math.sqrt(N))
            
            pos_embed = F.interpolate(
                pos_embed.reshape(1, grid_size, grid_size, dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return pos_embed
    
    def forward(
        self,
        x_pred: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            x_pred: Predictor latent tokens [B, N, memory_dim]
            masks: Optional mask indices for attention
        
        Returns:
            Reconstructed pixels [B, C, H, W] or [B, C, T, H, W]
        """
        B, N, memory_dim = x_pred.shape
        assert memory_dim == self.memory_dim, f"Memory dim mismatch: {memory_dim} vs {self.memory_dim}"
        
        # Use learnable decoder tokens, adjust size if needed
        if N != self.num_patches:
            # If input size differs from initialized size, interpolate decoder tokens
            decoder_tokens = F.interpolate(
                self.decoder_tokens.permute(0, 2, 1),  # [1, embed_dim, num_patches]
                size=N,
                mode='linear'
            ).permute(0, 2, 1)  # [1, N, embed_dim]
        else:
            decoder_tokens = self.decoder_tokens
            
        x = decoder_tokens.expand(B, -1, -1)
        
        # Add positional embeddings if not using RoPE
        if not self.use_rope and self.pos_embed is not None:
            # Determine target shape for interpolation
            if self.is_video:
                target_shape = (self.num_frames, self.img_height, self.img_width)
            else:
                target_shape = (self.img_height, self.img_width)
                
            pos_embed = self.interpolate_pos_encoding(target_shape, self.pos_embed)
            
            # Ensure pos_embed matches token count
            if pos_embed.size(1) != N:
                pos_embed = pos_embed[:, :N, :]
            
            x = x + pos_embed
        
        # Prepare causal mask
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[:N, :N].to(x.device, non_blocking=True)
        
        # Determine spatial/temporal dimensions for RoPE
        T = H = W = None
        if self.use_rope or self.is_video:
            T = self.grid_depth if self.is_video else 1
            H = self.grid_height
            W = self.grid_width
        
        # Forward through decoder blocks with cross-attention
        for i, block in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    x_pred,  # memory
                    masks,
                    attn_mask,
                    T,
                    H, 
                    W,
                    0,  # action_tokens
                    use_reentrant=False,
                )
            else:
                x = block(
                    x, 
                    x_pred,  # memory
                    mask=masks, 
                    attn_mask=attn_mask,
                    T=T,
                    H=H,
                    W=W,
                    action_tokens=0
                )
        
        # Final normalization
        x = self.norm(x)
        
        # Project to pixel space
        x = self.decoder_proj(x)
        
        # Unpatchify to reconstruct images/videos
        x = self.unpatchify(x)
        
        return x


def tie_embed_linear_weights(encoder_patch_embed: nn.Module, decoder_proj: nn.Linear) -> None:
    """
    Utility to tie PatchEmbed weights with decoder projection weights.
    
    Args:
        encoder_patch_embed: Encoder's patch embedding layer
        decoder_proj: Decoder's projection layer
    """
    if hasattr(encoder_patch_embed, 'proj'):
        # Transpose convolution weights to match linear layer format
        conv_weight = encoder_patch_embed.proj.weight
        if isinstance(encoder_patch_embed.proj, nn.Conv2d):
            # [out_channels, in_channels, kH, kW] -> [in_channels * kH * kW, out_channels]
            linear_weight = conv_weight.view(conv_weight.size(0), -1).t()
        elif isinstance(encoder_patch_embed.proj, nn.Conv3d):
            # [out_channels, in_channels, kT, kH, kW] -> [in_channels * kT * kH * kW, out_channels]
            linear_weight = conv_weight.view(conv_weight.size(0), -1).t()
        else:
            raise ValueError(f"Unsupported patch embedding type: {type(encoder_patch_embed.proj)}")
        
        # Tie weights
        decoder_proj.weight.data = linear_weight.clone()
        print(f"Tied weights: encoder patch embed -> decoder proj")


# Export list
__all__ = [
    "VisionTransformerDecoderXC",
    "CrossAttentionACBlock", 
    "tie_embed_linear_weights"
] 