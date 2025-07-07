import torch
import torch.nn as nn
import copy
from .masked_vit import MaskedViT
from .mlp import PredictorMLP
from src.utils.weight_init import initialize_weights, count_parameters, print_num_parameters


class MaskedPredictionModel(nn.Module):
    """
    Stage 1: Self-Supervised Encoder Pretraining Model
    Implements the V-JEPA2 inspired masked prediction architecture.
    
    Components:
    - Online Encoder: MaskedViT that processes masked images
    - Predictor: MLP that predicts masked token embeddings
    - Target Encoder: EMA-updated encoder for unmasked images
    """
    
    def __init__(self,
                 image_size,
                 patch_size,
                 input_channels,
                 latent_dim,
                 predictor_hidden_dims,
                 ema_decay=0.996,
                 # ViT parameters
                 vit_depth=4,
                 vit_heads=4,
                 vit_mlp_dim=512,
                 vit_dim_head=64,
                 vit_dropout=0.2,
                 vit_emb_dropout=0,
                 # Masking parameters
                 mask_token_dim=None,
                 num_mask_tokens=1,
                 # Predictor parameters
                 predictor_dropout_rate=0.0,
                 predictor_activation_fn_str='gelu'):
        super().__init__()
        
        self.ema_decay = ema_decay
        self._image_size_tuple = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        
        # Online Encoder (MaskedViT)
        self.online_encoder = MaskedViT(
            image_size=self._image_size_tuple,
            patch_size=patch_size,
            dim=latent_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            channels=input_channels,
            dim_head=vit_dim_head,
            dropout=vit_dropout,
            emb_dropout=vit_emb_dropout,
            mask_token_dim=mask_token_dim,
            num_mask_tokens=num_mask_tokens
        )
        
        # Target Encoder (EMA-updated)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._copy_weights_to_target_encoder()
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Predictor Network (MLP)
        # Input: concatenation of visible token embeddings and mask tokens
        # We'll use the latent_dim as input since we're predicting in latent space
        self.predictor = PredictorMLP(
            input_dim=latent_dim,  # Predictor takes encoded representation
            hidden_dims=predictor_hidden_dims,
            latent_dim=latent_dim,  # Output same dimension as encoder
            activation_fn_str=predictor_activation_fn_str,
            use_batch_norm=False,
            dropout_rate=predictor_dropout_rate
        )
        
        self.apply(initialize_weights)
        print_num_parameters(self)
    
    @torch.no_grad()
    def _copy_weights_to_target_encoder(self):
        """Copy weights from online encoder to target encoder."""
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)
    
    @torch.no_grad()
    def _update_target_encoder_ema(self):
        """Update target encoder weights using EMA."""
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = param_target.data * self.ema_decay + \
                param_online.data * (1. - self.ema_decay)
    
    def forward(self, x, mask_ratio=0.75):
        """
        Forward pass for masked prediction training.
        
        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask
        
        Returns:
            predicted_embeddings: Predicted embeddings for masked tokens [B, D]
            target_embeddings: Target embeddings from EMA encoder [B, D]
            mask: Binary mask indicating which patches were masked [B, N]
        """
        # Online encoder: process masked image with mask tokens
        online_encoded, mask, ids_restore = self.online_encoder.forward_with_mask_tokens(x, mask_ratio)
        
        # Predictor: predict embeddings for masked tokens
        predicted_embeddings = self.predictor(online_encoded)
        
        # Target encoder: process unmasked image (no gradients)
        with torch.no_grad():
            target_encoded = self.target_encoder(x, mask_ratio=0)  # No masking for target
            target_embeddings = target_encoded.detach()
        
        return predicted_embeddings, target_embeddings, mask
    
    def perform_ema_update(self):
        """Update target encoder weights using EMA."""
        self._update_target_encoder_ema()
    
    def get_encoder(self):
        """Return the trained encoder for use in Stage 2."""
        return self.online_encoder 