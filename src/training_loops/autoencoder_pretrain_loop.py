"""
Autoencoder Pre-training Loop for V-JEPA Architecture

This loop pre-trains the encoder and decoder using an autoencoder approach with JEPA-style masking:
Input [B, 1, T, H, W] -> VisionTransformer (with masking after tokenization) -> VisionTransformerPredictor -> VisionTransformerDecoderXC -> MSE Loss

The correct JEPA flow:
1. Full video input [B, 1, T, H, W]
2. Encoder tokenizes and applies masks to patches (masks_x = visible context)
3. Predictor predicts masked tokens (masks_y) from visible context
4. Decoder reconstructs full video from predictor output
5. MSE loss between original and reconstructed video
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.models.vision_transformer import VisionTransformer
from src.models.predictor_first_stage import VisionTransformerPredictor
from src.models.vision_transformer_decoder_xc import VisionTransformerDecoderXC
from src.masks.utils import apply_masks
from src.utils.fb_tensors import repeat_interleave_batch


class AutoencoderPretrainTrainer:
    """
    Trainer for autoencoder pre-training using V-JEPA architecture.
    """
    
    def __init__(self, config, device, train_dataloader, val_dataloader=None):
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Extract configuration
        self.pretrain_config = config.get('autoencoder_pretrain', {})
        
        # Training parameters
        self.masking_ratio = self.pretrain_config.get('masking_ratio', 0.75)
        self.num_epochs = self.pretrain_config.get('num_epochs', 100)
        self.learning_rate = self.pretrain_config.get('learning_rate', 0.0001)
        self.weight_decay = self.pretrain_config.get('weight_decay', 0.05)
        self.early_stopping_patience = self.pretrain_config.get('early_stopping_patience', 10)
        self.gradient_clip_norm = self.pretrain_config.get('gradient_clip_norm', 1.0)
        
        # Model parameters
        self.img_size = config.get('img_size', (224, 224))
        self.patch_size = config.get('patch_size', 16)
        self.num_frames = config.get('num_frames', 16)
        self.tubelet_size = config.get('tubelet_size', 2)
        self.in_chans = config.get('in_chans', 1)  # Greyscale
        self.embed_dim = config.get('embed_dim', 768)
        self.predictor_embed_dim = config.get('predictor_embed_dim', 384)
        self.decoder_embed_dim = config.get('decoder_embed_dim', 768)
        
        # Architecture parameters
        self.encoder_depth = config.get('encoder_depth', 12)
        self.predictor_depth = config.get('predictor_depth', 6)
        self.decoder_depth = config.get('decoder_depth', 12)
        self.num_heads = config.get('num_heads', 12)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        
        # Initialize models
        self.encoder = None
        self.predictor = None
        self.decoder = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.current_epoch = 0
        
        # Setup models
        self._setup_models()
        self._setup_optimizer()
        
    def _setup_models(self):
        """Setup encoder, predictor, and decoder models."""
        print("Setting up autoencoder models...")
        
        # Encoder: VisionTransformer
        self.encoder = VisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            depth=self.encoder_depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            use_rope=False,
            handle_nonsquare_inputs=True,
        ).to(self.device)
        
        # Predictor: VisionTransformerPredictor (with JEPA masking)
        self.predictor = VisionTransformerPredictor(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            embed_dim=self.embed_dim,
            predictor_embed_dim=self.predictor_embed_dim,
            depth=self.predictor_depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            uniform_power=False,
            use_mask_tokens=True,
            num_mask_tokens=2,
            zero_init_mask_tokens=True,
            use_silu=False,
            wide_silu=True,
            use_activation_checkpointing=False,
            return_all_tokens=False,
            chop_last_n_tokens=0,
            use_rope=False,
        ).to(self.device)
        
        # Decoder: VisionTransformerDecoderXC
        self.decoder = VisionTransformerDecoderXC(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            in_chans=self.in_chans,
            embed_dim=self.decoder_embed_dim,
            memory_dim=self.embed_dim,  # Predictor output dimension
            depth=self.decoder_depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            uniform_power=False,
            use_silu=False,
            wide_silu=True,
            use_sdpa=True,
            use_activation_checkpointing=False,
            use_rope=False,
            handle_nonsquare_inputs=True,
            is_frame_causal=True,
        ).to(self.device)
        
        # Tie encoder patch embedding weights with decoder projection
        from src.models.vision_transformer_decoder_xc import tie_embed_linear_weights
        tie_embed_linear_weights(self.encoder.patch_embed, self.decoder.decoder_proj)
        
        print(f"Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(f"Predictor parameters: {sum(p.numel() for p in self.predictor.parameters()):,}")
        print(f"Decoder parameters: {sum(p.numel() for p in self.decoder.parameters()):,}")
        print(f"Total parameters: {sum(p.numel() for p in self.encoder.parameters() + list(self.predictor.parameters()) + list(self.decoder.parameters())):,}")
        
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Combine all parameters
        all_params = list(self.encoder.parameters()) + list(self.predictor.parameters()) + list(self.decoder.parameters())
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            all_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        
    def _create_masks(self, batch_size, num_patches):
        """Create JEPA-style masks for the predictor."""
        # Create random masks for each sample in batch
        masks_x = []  # Context tokens (visible)
        masks_y = []  # Target tokens (masked)
        
        for _ in range(batch_size):
            # Randomly select tokens to mask
            num_masked = int(num_patches * self.masking_ratio)
            masked_indices = torch.randperm(num_patches)[:num_masked]
            context_indices = torch.randperm(num_patches)[num_masked:]
            
            masks_x.append(context_indices)
            masks_y.append(masked_indices)
        
        return masks_x, masks_y
    
    def _calculate_num_patches(self, video_frames):
        """Calculate number of patches from video dimensions."""
        B, C, T, H, W = video_frames.shape
        T_patches = T // self.tubelet_size
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        return T_patches * H_patches * W_patches

    def _train_epoch(self, epoch):
        """Train one epoch."""
        self.encoder.train()
        self.predictor.train()
        self.decoder.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Autoencoder Pre-train Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                video_frames = batch[0]
            else:
                video_frames = batch
            
            # Ensure video_frames is [B, 1, T, H, W] (greyscale)
            if video_frames.dim() == 4:
                video_frames = video_frames.unsqueeze(1)
            elif video_frames.dim() == 5 and video_frames.size(1) != 1:
                video_frames = video_frames.mean(dim=1, keepdim=True)
            
            video_frames = video_frames.to(self.device)
            batch_size = video_frames.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Step 1: Calculate number of patches for masking
            num_patches = self._calculate_num_patches(video_frames)
            
            # Step 2: Create JEPA-style masks (context vs target tokens)
            masks_x, masks_y = self._create_masks(batch_size, num_patches)
            
            # Step 3: Encode video frames with masking (JEPA style)
            # The encoder receives full video but applies masks to tokenized patches
            # masks_x = context tokens (visible), masks_y = target tokens (masked)
            tokens = self.encoder(video_frames, masks=masks_x)  # Only visible tokens
            
            # Step 4: Predict masked tokens using predictor
            # predictor_output: [B, N, embed_dim] (predicts all tokens including masked)
            predictor_output = self.predictor(tokens, masks_x, masks_y)
            
            # Step 5: Decode to reconstruct original video
            reconstructed = self.decoder(predictor_output)
            
            # Step 6: Compute MSE loss between original and reconstructed
            loss = nn.MSELoss()(video_frames, reconstructed)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.predictor.parameters()) + list(self.decoder.parameters()),
                max_norm=self.gradient_clip_norm
            )
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
            
            # Log to wandb every 10 batches
            if batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/avg_loss': total_loss / num_batches,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/batch': batch_idx
                })
        
        return total_loss / num_batches
    
    def _validate_epoch(self, epoch):
        """Validate one epoch."""
        self.encoder.eval()
        self.predictor.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    video_frames = batch[0]
                else:
                    video_frames = batch
                
                # Ensure video_frames is [B, 1, T, H, W] (greyscale)
                if video_frames.dim() == 4:
                    video_frames = video_frames.unsqueeze(1)
                elif video_frames.dim() == 5 and video_frames.size(1) != 1:
                    video_frames = video_frames.mean(dim=1, keepdim=True)
                
                video_frames = video_frames.to(self.device)
                batch_size = video_frames.size(0)
                
                # Calculate number of patches for masking
                num_patches = self._calculate_num_patches(video_frames)
                
                # Create JEPA-style masks
                masks_x, masks_y = self._create_masks(batch_size, num_patches)
                
                # Forward pass with correct JEPA logic
                tokens = self.encoder(video_frames, masks=masks_x)  # Only visible tokens
                predictor_output = self.predictor(tokens, masks_x, masks_y)
                reconstructed = self.decoder(predictor_output)
                
                # Compute loss
                loss = nn.MSELoss()(video_frames, reconstructed)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Main training loop."""
        print(f"Starting autoencoder pre-training for {self.num_epochs} epochs")
        print(f"Masking ratio: {self.masking_ratio}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        
        # Initialize wandb
        wandb.init(
            project="vjepa-autoencoder-pretrain",
            config={
                "masking_ratio": self.masking_ratio,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "early_stopping_patience": self.early_stopping_patience,
                "gradient_clip_norm": self.gradient_clip_norm,
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "num_frames": self.num_frames,
                "embed_dim": self.embed_dim,
                "predictor_embed_dim": self.predictor_embed_dim,
                "decoder_embed_dim": self.decoder_embed_dim,
                "encoder_depth": self.encoder_depth,
                "predictor_depth": self.predictor_depth,
                "decoder_depth": self.decoder_depth,
            }
        )
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(epoch)
            
            # Validation phase
            val_loss = 0.0
            if self.val_dataloader is not None:
                val_loss = self._validate_epoch(epoch)
            
            # Update scheduler
            if self.val_dataloader is not None:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)
            
            # Logging
            self._log_metrics(epoch, train_loss, val_loss)
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.2e}")
            
            # Early stopping
            if self.val_dataloader is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self._save_checkpoint()
                    print(f"New best validation loss: {val_loss:.4f}")
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (patience: {self.early_stopping_patience})")
                    break
        
        # Final logging
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/3600:.2f} hours")
        wandb.log({"training/total_time_hours": total_time/3600})
        
        # Load best model
        self._load_best_checkpoint()
        
        wandb.finish()
        
    def _log_metrics(self, epoch, train_loss, val_loss):
        """Log metrics to wandb."""
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
            'early_stopping/patience_counter': self.early_stopping_counter,
            'early_stopping/best_val_loss': self.best_val_loss,
        })
        
        # Log model parameters norm
        encoder_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=float('inf'))
        predictor_norm = torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=float('inf'))
        decoder_norm = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=float('inf'))
        
        wandb.log({
            'model/encoder_norm': encoder_norm,
            'model/predictor_norm': predictor_norm,
            'model/decoder_norm': decoder_norm,
        })
    
    def _save_checkpoint(self):
        """Save best model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, 'checkpoints/autoencoder_pretrain_best.pth')
        print("Saved best checkpoint")
    
    def _load_best_checkpoint(self):
        """Load best model checkpoint."""
        checkpoint_path = 'checkpoints/autoencoder_pretrain_best.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
    
    def get_trained_models(self):
        """Return trained models."""
        return {
            'encoder': self.encoder,
            'predictor': self.predictor,
            'decoder': self.decoder,
        }


def create_autoencoder_pretrain_trainer(config, device, train_dataloader, val_dataloader=None):
    """Factory function to create autoencoder pre-training trainer."""
    return AutoencoderPretrainTrainer(config, device, train_dataloader, val_dataloader) 