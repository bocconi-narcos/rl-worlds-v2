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

# Standard library imports
import math
from functools import partial
from typing import Optional, Tuple, Union

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports from utility modules
from src.models.utils.modules import ACBlock, CrossAttention, build_action_block_causal_attention_mask
from src.models.utils.patch_embed import PatchEmbed, PatchEmbed3D
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.fb_tensors import trunc_normal_


class CrossAttentionACBlock(nn.Module):
    """
    ACBlock with prepended cross-attention layer.
    Cross-attention uses keys/values from predictor tokens, queries from decoder tokens.
    
    This block combines:
    1. Cross-attention: decoder tokens attend to predictor tokens
    2. Self-attention: decoder tokens attend to themselves (via ACBlock)
    3. MLP: feed-forward network for feature transformation
    """
    
    def __init__(
        self,
        dim: int,                    # Dimension of decoder tokens
        memory_dim: int,             # Dimension of predictor tokens (memory)
        num_heads: int,              # Number of attention heads
        mlp_ratio: float = 4.0,      # Ratio of MLP hidden dim to input dim
        qkv_bias: bool = False,      # Whether to use bias in QKV projection
        qk_scale: Optional[float] = None,  # Scaling factor for QK attention
        drop: float = 0.0,           # Dropout rate for MLP
        attn_drop: float = 0.0,      # Dropout rate for attention
        drop_path: float = 0.0,      # Drop path rate for stochastic depth
        act_layer = nn.GELU,         # Activation function for MLP
        wide_silu: bool = True,      # Whether to use wide SiLU activation
        norm_layer = nn.LayerNorm,   # Normalization layer type
        use_sdpa: bool = True,       # Whether to use scaled dot-product attention
        is_causal: bool = False,     # Whether to use causal masking
        grid_size: int = 16,         # Grid size for positional encoding
        use_rope: bool = False,      # Whether to use RoPE (Rotary Position Embedding)
        **kwargs,                    # Additional keyword arguments
    ):
        super().__init__()
        
        # Cross-attention layer (prepended to self-attention)
        # Normalize decoder tokens before cross-attention
        self.norm_cross = norm_layer(dim)
        # Normalize memory (predictor) tokens before cross-attention
        self.norm_memory = norm_layer(memory_dim)
        
        # Project memory to same dimension as decoder if needed
        # This allows the cross-attention to work when memory_dim != dim
        if memory_dim != dim:
            self.memory_proj = nn.Linear(memory_dim, dim, bias=False)
        else:
            self.memory_proj = nn.Identity()
            
        # Cross-attention layer: queries from decoder, keys/values from predictor
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_sdpa=use_sdpa,
            is_causal=is_causal
        )
        # Drop path for cross-attention output (stochastic depth)
        self.cross_drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Standard ACBlock components (self-attention + MLP)
        # This handles the self-attention and feed-forward parts
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
        x: torch.Tensor,             # Decoder tokens [B, N, dim]
        memory: torch.Tensor,        # Predictor tokens [B, M, memory_dim] 
        mask: Optional[torch.Tensor] = None,      # Optional mask for self-attention
        attn_mask: Optional[torch.Tensor] = None, # Causal attention mask
        T: Optional[int] = None,     # Temporal dimension for RoPE
        H: Optional[int] = None,     # Height dimension for RoPE
        W: Optional[int] = None,     # Width dimension for RoPE
        action_tokens: int = 0       # Number of action tokens for ACBlock
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
        # Step 1: Cross-attention - decoder tokens attend to predictor tokens
        # Normalize decoder tokens for cross-attention
        q_norm = self.norm_cross(x)
        # Normalize memory (predictor) tokens for cross-attention
        kv_norm = self.norm_memory(memory)
        # Project memory to same dimension as decoder if needed
        kv_proj = self.memory_proj(kv_norm)
        
        # Apply cross-attention - CrossAttention returns the result directly
        # Queries come from decoder, keys/values from predictor
        cross_out = self.cross_attn(q_norm, kv_proj, attn_mask=attn_mask)
        # Residual connection with drop path
        x = x + self.cross_drop_path(cross_out)
        
        # Step 2: Self-attention via ACBlock
        # This includes self-attention + MLP with residual connections
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
    
    Key features:
    - Cross-attention to predictor tokens in every block
    - Learnable decoder tokens (no input patches)
    - Causal masking for temporal consistency
    - Support for both image and video inputs
    - Positional embeddings (sincos or RoPE)
    """
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = (224, 224),  # Input image size
        patch_size: int = 16,        # Size of patches
        num_frames: int = 1,         # Number of video frames (1 for images)
        tubelet_size: int = 2,       # Temporal patch size for videos
        in_chans: int = 3,           # Number of input channels
        embed_dim: int = 768,        # Embedding dimension
        memory_dim: int = 768,       # Dimension of predictor tokens
        depth: int = 12,             # Number of transformer blocks
        num_heads: int = 12,         # Number of attention heads
        mlp_ratio: float = 4.0,      # MLP expansion ratio
        qkv_bias: bool = True,       # Whether to use bias in QKV projection
        qk_scale: Optional[float] = None,  # QK scaling factor
        drop_rate: float = 0.0,      # Dropout rate
        attn_drop_rate: float = 0.0, # Attention dropout rate
        drop_path_rate: float = 0.0, # Drop path rate
        norm_layer = nn.LayerNorm,   # Normalization layer
        init_std: float = 0.02,      # Weight initialization standard deviation
        uniform_power: bool = False, # Whether to use uniform power for pos emb
        use_silu: bool = False,      # Whether to use SiLU activation
        wide_silu: bool = True,      # Whether to use wide SiLU
        use_sdpa: bool = True,       # Whether to use scaled dot-product attention
        use_activation_checkpointing: bool = False,  # Whether to use gradient checkpointing
        use_rope: bool = False,      # Whether to use RoPE
        handle_nonsquare_inputs: bool = True,  # Whether to handle non-square inputs
        is_frame_causal: bool = True, # Whether to use causal masking
        **kwargs                     # Additional keyword arguments
    ):
        super().__init__()
        
        # Store key parameters
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.is_frame_causal = is_frame_causal
        
        # Handle image size (convert int to tuple if needed)
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1  # Determine if input is video
        
        # Whether to use gradient checkpointing for memory efficiency
        self.use_activation_checkpointing = use_activation_checkpointing
        
        # Grid dimensions for causal masking
        # These represent the number of patches in each dimension
        self.grid_height = self.img_height // self.patch_size
        self.grid_width = self.img_width // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        
        # Calculate total number of patches
        if self.is_video:
            self.num_patches = self.grid_depth * self.grid_height * self.grid_width
        else:
            self.num_patches = self.grid_height * self.grid_width
        
        # Stochastic depth decay rule - increases dropout with depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Positional embeddings (same as encoder)
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        if self.use_rope:
            # If using RoPE, no need for learnable positional embeddings
            self.pos_embed = None
        else:
            # Learnable positional embeddings
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), 
                requires_grad=False  # Will be initialized with sincos embeddings
            )
        
        # Decoder blocks with cross-attention
        # Each block has cross-attention followed by self-attention
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
                drop_path=dpr[i],  # Different drop path for each layer
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
        
        # Final normalization layer
        self.norm = norm_layer(embed_dim)
        
        # Token-to-patch projection head
        # Projects decoder tokens back to pixel space
        self.decoder_proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans, bias=True)
        
        # Learnable decoder tokens - initialized properly as parameter
        # These are the starting tokens for the decoder (no input patches needed)
        self.decoder_tokens = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * init_std)
        
        # Build and cache causal attention mask for efficiency
        self.attn_mask = None
        if self.is_frame_causal and self.is_video:
            # Create causal mask that respects temporal ordering
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
        """
        Initialize positional embeddings using sincos (same as encoder).
        
        Args:
            pos_embed: Positional embedding tensor to initialize
        """
        embed_dim = pos_embed.size(-1)
        if self.is_video:
            # 3D sincos positional embeddings for video
            sincos = get_3d_sincos_pos_embed(
                embed_dim, 
                self.grid_height, 
                self.grid_depth, 
                cls_token=False, 
                uniform_power=self.uniform_power
            )
        else:
            # 2D sincos positional embeddings for images
            sincos = get_2d_sincos_pos_embed(
                embed_dim, 
                self.grid_height, 
                cls_token=False
            )
        # Copy the sincos embeddings to the parameter
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
    
    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize weights (same pattern as encoder).
        
        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            # Initialize linear layers with truncated normal distribution
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize layer norm with zeros and ones
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            # Initialize convolutions with truncated normal distribution
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _rescale_blocks(self) -> None:
        """
        Rescale block weights (same as encoder).
        This helps with training stability by scaling weights based on layer depth.
        """
        def rescale(param, layer_id):
            # Scale weights by sqrt(2 * layer_id) for better training dynamics
            param.div_(math.sqrt(2.0 * layer_id))
        
        for layer_id, layer in enumerate(self.blocks):
            # Rescale self-attention projection weights
            rescale(layer.ac_block.attn.proj.weight.data, layer_id + 1)
            # Rescale MLP output projection weights
            rescale(layer.ac_block.mlp.fc2.weight.data, layer_id + 1)
            # Rescale cross-attention if it has learnable projection
            if hasattr(layer, 'memory_proj') and hasattr(layer.memory_proj, 'weight'):
                rescale(layer.memory_proj.weight.data, layer_id + 1)
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens back to pixel space.
        
        This is the inverse operation of patch embedding - it reconstructs
        the original image/video from patch tokens.
        
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
            
            # Extract channel dimension from patch tokens
            patch_dim = self.patch_size * self.patch_size
            in_chans = D // patch_dim
            
            # Reshape to [B, T, H, W, patch_size, patch_size, C]
            # This organizes tokens by their spatial and temporal positions
            x = x.view(B, T, H, W, self.patch_size, self.patch_size, in_chans)
            
            # Rearrange to [B, C, T, H*patch_size, W*patch_size]  
            # This puts channels first and reconstructs the full spatial dimensions
            x = x.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
            x = x.view(B, in_chans, T, H * self.patch_size, W * self.patch_size)
            
        else:
            # Image case: [B, H*W, patch_size^2 * C]
            H = self.grid_height
            W = self.grid_width
            
            # Extract channel dimension from patch tokens
            patch_dim = self.patch_size * self.patch_size
            in_chans = D // patch_dim
            
            # Reshape to [B, H, W, patch_size, patch_size, C]
            # This organizes tokens by their spatial positions
            x = x.view(B, H, W, self.patch_size, self.patch_size, in_chans)
            
            # Rearrange to [B, C, H*patch_size, W*patch_size]
            # This puts channels first and reconstructs the full spatial dimensions
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            x = x.view(B, in_chans, H * self.patch_size, W * self.patch_size)
        
        return x
    

    
    def forward(
        self,
        x_pred: torch.Tensor,        # Predictor latent tokens [B, N, memory_dim]
        masks: Optional[torch.Tensor] = None,  # Optional mask indices for attention
    ) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        The decoder takes predictor tokens as input and reconstructs the original
        images/videos through a series of cross-attention and self-attention blocks.
        
        Args:
            x_pred: Predictor latent tokens [B, N, memory_dim]
            masks: Optional mask indices for attention
        
        Returns:
            Reconstructed pixels [B, C, H, W] or [B, C, T, H, W]
        """
        B, N, memory_dim = x_pred.shape
        assert memory_dim == self.memory_dim, f"Memory dim mismatch: {memory_dim} vs {self.memory_dim}"
        
        # Use learnable decoder tokens - enforce exact size match
        # The decoder starts with learnable tokens rather than input patches
        if N != self.num_patches:
            raise ValueError(
                f"Input token count mismatch: expected {self.num_patches} tokens but got {N} tokens. "
                f"Decoder was initialized for {self.num_frames} frames, "
                f"{self.img_height}x{self.img_width} image size, and {self.patch_size}x{self.patch_size} patches. "
                f"Please ensure predictor output matches decoder configuration."
            )
        
        decoder_tokens = self.decoder_tokens
            
        # Expand decoder tokens to batch size
        x = decoder_tokens.expand(B, -1, -1)
        
        # Add positional embeddings if not using RoPE
        if not self.use_rope and self.pos_embed is not None:
            # Check if positional embedding size matches expected token count
            if self.pos_embed.size(1) != N:
                raise ValueError(
                    f"Positional embedding size mismatch: expected {N} tokens but pos_embed has {self.pos_embed.size(1)} tokens. "
                    f"Decoder was initialized for {self.num_frames} frames, "
                    f"{self.img_height}x{self.img_width} image size, and {self.patch_size}x{self.patch_size} patches. "
                    f"Please ensure predictor output matches decoder configuration."
                )
            
            # Add positional embeddings to decoder tokens
            x = x + self.pos_embed
        
        # Prepare causal mask for temporal consistency
        attn_mask = None
        if self.attn_mask is not None:
            # Use cached causal mask, ensuring it matches token count
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
                # Use gradient checkpointing for memory efficiency
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    x_pred,  # memory (predictor tokens)
                    masks,
                    attn_mask,
                    T,
                    H, 
                    W,
                    0,  # action_tokens (not used in decoder)
                    use_reentrant=False,
                )
            else:
                # Standard forward pass
                x = block(
                    x, 
                    x_pred,  # memory (predictor tokens)
                    mask=masks, 
                    attn_mask=attn_mask,
                    T=T,
                    H=H,
                    W=W,
                    action_tokens=0  # No action tokens in decoder
                )
        
        # Final normalization
        x = self.norm(x)
        
        # Project to pixel space
        # This converts the final tokens back to patch-sized pixel values
        x = self.decoder_proj(x)
        
        # Unpatchify to reconstruct images/videos
        # This converts patch tokens back to the original image/video format
        x = self.unpatchify(x)
        
        return x


def tie_embed_linear_weights(encoder_patch_embed: nn.Module, decoder_proj: nn.Linear) -> None:
    """
    Utility to tie PatchEmbed weights with decoder projection weights.
    
    This function ties the weights between the encoder's patch embedding layer
    and the decoder's projection layer, which can help with training stability
    and potentially improve reconstruction quality.
    
    Args:
        encoder_patch_embed: Encoder's patch embedding layer
        decoder_proj: Decoder's projection layer
    """
    if hasattr(encoder_patch_embed, 'proj'):
        # Get the convolution weights from the encoder's patch embedding
        conv_weight = encoder_patch_embed.proj.weight
        if isinstance(encoder_patch_embed.proj, nn.Conv2d):
            # For 2D convolutions: [out_channels, in_channels, kH, kW] -> [in_channels * kH * kW, out_channels]
            # This transposes the weights to match the linear layer format
            linear_weight = conv_weight.view(conv_weight.size(0), -1).t()
        elif isinstance(encoder_patch_embed.proj, nn.Conv3d):
            # For 3D convolutions: [out_channels, in_channels, kT, kH, kW] -> [in_channels * kT * kH * kW, out_channels]
            # This transposes the weights to match the linear layer format
            linear_weight = conv_weight.view(conv_weight.size(0), -1).t()
        else:
            raise ValueError(f"Unsupported patch embedding type: {type(encoder_patch_embed.proj)}")
        
        # Tie weights by copying the transposed weights to the decoder projection
        decoder_proj.weight.data = linear_weight.clone()
        print(f"Tied weights: encoder patch embed -> decoder proj")


# Export list - defines what gets imported when using "from module import *"
__all__ = [
    "VisionTransformerDecoderXC",    # Main decoder class
    "CrossAttentionACBlock",         # Cross-attention block component
    "tie_embed_linear_weights"       # Utility function for weight tying
] 