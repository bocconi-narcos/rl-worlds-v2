"""
Two-Stage V-JEPA2-Inspired World Model Training Loop

Stage 1: Self-Supervised Encoder Pretraining
Stage 2: Action-Conditioned World Model Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.masked_prediction_model import MaskedPredictionModel
from src.models.world_model_transformer import WorldModelTransformer
from src.losses.masked_prediction_loss import MaskedPredictionLoss, WorldModelLoss
from src.utils.weight_init import initialize_weights
from src.utils.env_utils import get_env_details
from src.utils.data_utils import TemporalSequenceDataset


class VJEPA2WorldModelTrainer:
    """
    Trainer for the two-stage V-JEPA2-inspired world model.
    """
    
    def __init__(self, config, device, train_dataloader, val_dataloader=None):
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Extract configuration
        self.stage1_config = config.get('vjepa2_world_model', {}).get('stage1', {})
        self.stage2_config = config.get('vjepa2_world_model', {}).get('stage2', {})
        
        # Stage 1 parameters
        self.masking_ratio = self.stage1_config.get('masking_ratio', 0.75)
        self.stage1_epochs = self.stage1_config.get('num_epochs', 50)
        self.stage1_lr = self.stage1_config.get('learning_rate', 0.0001)
        self.ema_decay = self.stage1_config.get('ema_decay', 0.996)
        self.stage1_weight_decay = self.stage1_config.get('weight_decay', 0.05)
        self.stage1_early_stopping_patience = self.stage1_config.get('early_stopping_patience', 3)
        
        # Stage 2 parameters
        self.sequence_length = self.stage2_config.get('sequence_length', 8)
        self.teacher_forcing_ratio = self.stage2_config.get('teacher_forcing_ratio', 0.8)
        self.rollout_steps = self.stage2_config.get('rollout_steps', 4)
        self.stage2_epochs = self.stage2_config.get('num_epochs', 30)
        self.stage2_lr = self.stage2_config.get('learning_rate', 0.0001)
        self.stage2_weight_decay = self.stage2_config.get('weight_decay', 0.05)
        self.stage2_early_stopping_patience = self.stage2_config.get('early_stopping_patience', 3)
        
        # Model parameters
        self.image_size = config['environment']['image_height']
        self.patch_size = config['models']['shared_patch_size']
        
        # Calculate correct input channels based on frame stacking and grayscale conversion
        base_channels = config['environment']['input_channels_per_frame']
        frame_stack_size = config['environment'].get('frame_stack_size', 1)
        grayscale_conversion = config['environment'].get('grayscale_conversion', False)
        
        if grayscale_conversion:
            # If grayscale conversion is enabled, each frame becomes 1 channel
            self.input_channels = frame_stack_size
        else:
            # If no grayscale conversion, multiply by frame stack size
            self.input_channels = base_channels * frame_stack_size
        
        self.latent_dim = config['models']['shared_latent_dim']
        self.action_dim = self._get_action_dim()
        self.action_type = self._get_action_type()
        
        # Initialize models and losses
        self.stage1_model = None
        self.stage2_model = None
        self.stage1_loss = None
        self.stage2_loss = None
        
        # Training state
        self.current_stage = 1
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
    def _get_action_dim(self):
        """Get action dimension from environment."""
        env_name = self.config.get('environment', {}).get('name', 'ALE/Pong-v5')
        try:
            action_dim, _, _ = get_env_details(env_name)
            return action_dim
        except Exception as e:
            print(f"Warning: Could not get action dimension from environment {env_name}: {e}")
            print("Using default action dimension of 6")
            return 6  # Default for most environments
    
    def _get_action_type(self):
        """Get action type from environment."""
        env_name = self.config.get('environment', {}).get('name', 'ALE/Pong-v5')
        try:
            _, action_type, _ = get_env_details(env_name)
            return action_type
        except Exception as e:
            print(f"Warning: Could not get action type from environment {env_name}: {e}")
            print("Using default action type of 'continuous'")
            return 'continuous'  # Default
    
    def setup_stage1(self):
        """Setup Stage 1: Masked Prediction Model."""
        print("Setting up Stage 1: Masked Prediction Model")
        
        # Create masked prediction model
        self.stage1_model = MaskedPredictionModel(
            image_size=self.image_size,
            patch_size=self.patch_size,
            input_channels=self.input_channels,
            latent_dim=self.latent_dim,
            predictor_hidden_dims=[256, 256],  # Use current MLP structure
            ema_decay=self.ema_decay,
            # ViT parameters from config
            vit_depth=self.config['models']['encoder']['params']['vit']['depth'],
            vit_heads=self.config['models']['encoder']['params']['vit']['heads'],
            vit_mlp_dim=self.config['models']['encoder']['params']['vit']['mlp_dim'],
            vit_dim_head=64,
            vit_dropout=self.config['models']['encoder']['params']['vit']['dropout'],
            vit_emb_dropout=self.config['models']['encoder']['params']['vit']['emb_dropout'],
            # Masking parameters
            mask_token_dim=None,  # Use same as latent_dim
            num_mask_tokens=1,
            # Predictor parameters
            predictor_dropout_rate=0.0,
            predictor_activation_fn_str='gelu'
        ).to(self.device)
        
        # Create loss function
        self.stage1_loss = MaskedPredictionLoss(
            loss_type='l1',
            reduction='mean'
        ).to(self.device)
        
        # Create optimizer
        self.stage1_optimizer = optim.AdamW(
            self.stage1_model.parameters(),
            lr=self.stage1_lr,
            weight_decay=self.stage1_weight_decay
        )
        
        # Create scheduler with more conservative learning rate decay
        self.stage1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.stage1_optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        print(f"Stage 1 model parameters: {sum(p.numel() for p in self.stage1_model.parameters()):,}")
    
    def setup_stage2(self):
        """Setup Stage 2: World Model Transformer."""
        print("Setting up Stage 2: World Model Transformer")
        
        # Get the trained encoder from Stage 1
        if self.stage1_model is None:
            raise ValueError("Stage 1 model must be trained before setting up Stage 2")
        
        # Freeze the encoder
        encoder = self.stage1_model.get_encoder()
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Create temporal sequence datasets
        print("Creating temporal sequence datasets...")
        self.temporal_train_dataset = TemporalSequenceDataset(
            self.train_dataloader.dataset,
            sequence_length=self.sequence_length,
            stride=1,  # Consecutive sequences
            config=self.config
        )
        
        if self.val_dataloader is not None:
            self.temporal_val_dataset = TemporalSequenceDataset(
                self.val_dataloader.dataset,
                sequence_length=self.sequence_length,
                stride=1,  # Consecutive sequences
                config=self.config
            )
        else:
            self.temporal_val_dataset = None
        
        # Create new dataloaders for temporal sequences
        from torch.utils.data import DataLoader
        self.temporal_train_dataloader = DataLoader(
            self.temporal_train_dataset,
            batch_size=self.train_dataloader.batch_size,
            shuffle=True,
            num_workers=self.train_dataloader.num_workers,
            pin_memory=self.train_dataloader.pin_memory
        )
        
        if self.temporal_val_dataset is not None:
            self.temporal_val_dataloader = DataLoader(
                self.temporal_val_dataset,
                batch_size=self.val_dataloader.batch_size,
                shuffle=False,
                num_workers=self.val_dataloader.num_workers,
                pin_memory=self.val_dataloader.pin_memory
            )
        else:
            self.temporal_val_dataloader = None
        
        # Create world model transformer
        self.stage2_model = WorldModelTransformer(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            action_emb_dim=32,  # From config
            action_type=self.action_type,
            sequence_length=self.sequence_length,
            # Transformer parameters
            transformer_depth=4,
            transformer_heads=8,
            transformer_mlp_dim=512,
            transformer_dim_head=64,
            transformer_dropout=0.1,
            # Positional encoding
            use_pos_encoding=True,
            pos_encoding_dim=None
        ).to(self.device)
        
        # Store encoder for use during training
        self.encoder = encoder
        
        # Create loss function with more balanced weights
        self.stage2_loss = WorldModelLoss(
            teacher_forcing_weight=0.7,
            rollout_weight=0.3,
            loss_type='l1',
            reduction='mean'
        ).to(self.device)
        
        # Create optimizer
        self.stage2_optimizer = optim.AdamW(
            self.stage2_model.parameters(),
            lr=self.stage2_lr,
            weight_decay=self.stage2_weight_decay
        )
        
        # Create scheduler with more conservative learning rate decay
        self.stage2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.stage2_optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        print(f"Stage 2 model parameters: {sum(p.numel() for p in self.stage2_model.parameters()):,}")
        print(f"Temporal training sequences: {len(self.temporal_train_dataset)}")
        if self.temporal_val_dataset is not None:
            print(f"Temporal validation sequences: {len(self.temporal_val_dataset)}")
    
    def train_stage1(self):
        """Train Stage 1: Masked Prediction Model."""
        print(f"Starting Stage 1 training for {self.stage1_epochs} epochs")
        
        for epoch in range(self.stage1_epochs):
            # Training phase
            train_loss = self._train_stage1_epoch(epoch)
            
            # Validation phase
            val_loss = 0.0
            if self.val_dataloader is not None:
                val_loss = self._validate_stage1_epoch(epoch)
            
            # Update scheduler based on validation loss
            if self.val_dataloader is not None:
                self.stage1_scheduler.step(val_loss)
            else:
                self.stage1_scheduler.step(train_loss)
            
            # Logging
            self._log_stage1_metrics(epoch, train_loss, val_loss)
            
            # Log current learning rate
            current_lr = self.stage1_optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
            
            # Early stopping
            if self.val_dataloader is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self._save_stage1_checkpoint()
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.stage1_early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (patience: {self.stage1_early_stopping_patience})")
                    break
    
    def _train_stage1_epoch(self, epoch):
        """Train one epoch of Stage 1."""
        self.stage1_model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Stage 1 Epoch {epoch+1}")
        
        for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(progress_bar):
            # Use current states for training
            s_t = s_t.to(self.device)
            
            # Handle frame stacking: reshape from (B, F, C, H, W) to (B, F*C, H, W)
            if s_t.dim() == 5:
                B, F, C, H, W = s_t.shape
                s_t = s_t.view(B, F * C, H, W)
            
            # Zero gradients
            self.stage1_optimizer.zero_grad()
            
            # Forward pass
            predicted_embeddings, target_embeddings, mask = self.stage1_model(
                s_t, mask_ratio=self.masking_ratio
            )
            
            # Compute loss
            loss = self.stage1_loss(predicted_embeddings, target_embeddings, mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.stage1_model.parameters(), max_norm=1.0)
            
            # Update weights
            self.stage1_optimizer.step()
            
            # Update EMA
            self.stage1_model.perform_ema_update()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def _validate_stage1_epoch(self, epoch):
        """Validate one epoch of Stage 1."""
        self.stage1_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for s_t, a_t, r_t, s_t_plus_1 in self.val_dataloader:
                s_t = s_t.to(self.device)
                
                # Handle frame stacking: reshape from (B, F, C, H, W) to (B, F*C, H, W)
                if s_t.dim() == 5:
                    B, F, C, H, W = s_t.shape
                    s_t = s_t.view(B, F * C, H, W)
                
                # Forward pass
                predicted_embeddings, target_embeddings, mask = self.stage1_model(
                    s_t, mask_ratio=self.masking_ratio
                )
                
                # Compute loss
                loss = self.stage1_loss(predicted_embeddings, target_embeddings, mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train_stage2(self):
        """Train Stage 2: World Model Transformer."""
        print(f"Starting Stage 2 training for {self.stage2_epochs} epochs")
        # Ensure best_val_loss and patience counter are reset at the start of every run
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        for epoch in range(self.stage2_epochs):
            # Training phase
            train_loss, tf_loss, rollout_loss = self._train_stage2_epoch(epoch)
            
            # Validation phase
            val_loss = 0.0
            if self.temporal_val_dataloader is not None:
                val_loss = self._validate_stage2_epoch(epoch)
            
            # Update scheduler based on validation loss
            if self.temporal_val_dataloader is not None:
                self.stage2_scheduler.step(val_loss)
            else:
                self.stage2_scheduler.step(train_loss)
            
            # Logging
            self._log_stage2_metrics(epoch, train_loss, tf_loss, rollout_loss, val_loss)
            
            # Log current learning rate
            current_lr = self.stage2_optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
            
            # Early stopping
            if self.temporal_val_dataloader is not None:
                # Debug print for early stopping state
                print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}, best_val_loss={self.best_val_loss:.4f}, patience={self.early_stopping_counter}/{self.stage2_early_stopping_patience}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self._save_stage2_checkpoint()
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.stage2_early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs (patience: {self.stage2_early_stopping_patience})")
                    break
    
    def _train_stage2_epoch(self, epoch):
        """Train one epoch of Stage 2."""
        self.stage2_model.train()
        self.encoder.eval()  # Encoder should be frozen
        
        total_loss_sum = 0.0
        total_tf_loss = 0.0
        total_rollout_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.temporal_train_dataloader, desc=f"Stage 2 Epoch {epoch+1}")
        
        for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(progress_bar):
            # Prepare sequences - now these are already temporal sequences
            s_t = s_t.to(self.device)  # [B, T, F*C, H, W]
            a_t = a_t.to(self.device)  # [B, T]
            s_t_plus_1 = s_t_plus_1.to(self.device)  # [B, T, F*C, H, W]
            
            # Zero gradients
            self.stage2_optimizer.zero_grad()
            
            # Encode observations using frozen encoder
            with torch.no_grad():
                B, T = s_t.shape[:2]
                
                # Reshape for batch encoding: [B*T, F*C, H, W]
                s_t_flat = s_t.view(B * T, -1, s_t.shape[-2], s_t.shape[-1])
                s_t_plus_1_flat = s_t_plus_1.view(B * T, -1, s_t_plus_1.shape[-2], s_t_plus_1.shape[-1])
                
                # Encode all frames at once
                latents_flat = self.encoder(s_t_flat)  # [B*T, D]
                target_latents_flat = self.encoder(s_t_plus_1_flat)  # [B*T, D]
                
                # Reshape back to sequence format: [B, T, D]
                latents = latents_flat.view(B, T, -1)
                target_latents = target_latents_flat.view(B, T, -1)
            
            # Prepare action sequences - already in correct format [B, T]
            if self.action_type == 'discrete':
                actions = a_t.long()
            else:
                actions = a_t.float()
            
            # Forward pass
            predicted_latents = self.stage2_model(latents, actions)
            
            # Create masks for teacher forcing vs rollout
            teacher_forcing_mask = torch.rand(B, T, device=self.device) < self.teacher_forcing_ratio
            rollout_mask = ~teacher_forcing_mask
            
            # Compute loss
            total_loss, tf_loss, rollout_loss = self.stage2_loss(
                predicted_latents, target_latents, teacher_forcing_mask, rollout_mask
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.stage2_model.parameters(), max_norm=1.0)
            
            # Update weights
            self.stage2_optimizer.step()
            
            # Update metrics
            total_loss_sum += total_loss.item()
            total_tf_loss += tf_loss.item()
            total_rollout_loss += rollout_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'tf_loss': f'{tf_loss.item():.4f}',
                'rollout_loss': f'{rollout_loss.item():.4f}'
            })
        
        return (total_loss_sum / num_batches, 
                total_tf_loss / num_batches, 
                total_rollout_loss / num_batches)
    
    def _validate_stage2_epoch(self, epoch):
        """Validate one epoch of Stage 2."""
        self.stage2_model.eval()
        self.encoder.eval()
        
        total_loss_sum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for s_t, a_t, r_t, s_t_plus_1 in self.temporal_val_dataloader:
                s_t = s_t.to(self.device)  # [B, T, F*C, H, W]
                a_t = a_t.to(self.device)  # [B, T]
                s_t_plus_1 = s_t_plus_1.to(self.device)  # [B, T, F*C, H, W]
                
                B, T = s_t.shape[:2]
                
                # Reshape for batch encoding: [B*T, F*C, H, W]
                s_t_flat = s_t.view(B * T, -1, s_t.shape[-2], s_t.shape[-1])
                s_t_plus_1_flat = s_t_plus_1.view(B * T, -1, s_t_plus_1.shape[-2], s_t_plus_1.shape[-1])
                
                # Encode all frames at once
                latents_flat = self.encoder(s_t_flat)  # [B*T, D]
                target_latents_flat = self.encoder(s_t_plus_1_flat)  # [B*T, D]
                
                # Reshape back to sequence format: [B, T, D]
                latents = latents_flat.view(B, T, -1)
                target_latents = target_latents_flat.view(B, T, -1)
                
                # Prepare actions - already in correct format [B, T]
                if self.action_type == 'discrete':
                    actions = a_t.long()
                else:
                    actions = a_t.float()
                
                # Forward pass
                predicted_latents = self.stage2_model(latents, actions)
                
                # Compute loss (for validation, use all steps for both teacher forcing and rollout)
                loss, _, _ = self.stage2_loss(predicted_latents, target_latents, None, None)
                
                total_loss_sum += loss.item()
                num_batches += 1
        
        return total_loss_sum / num_batches
    
    def _log_stage1_metrics(self, epoch, train_loss, val_loss):
        """Log Stage 1 metrics."""
        metrics = {
            'stage1/train_loss': train_loss,
            'stage1/learning_rate': self.stage1_optimizer.param_groups[0]['lr'],
            'stage1/epoch': epoch
        }
        
        if self.val_dataloader is not None:
            metrics['stage1/val_loss'] = val_loss
        
        if wandb.run is not None:
            wandb.log(metrics)
        
        print(f"Stage 1 Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def _log_stage2_metrics(self, epoch, train_loss, tf_loss, rollout_loss, val_loss):
        """Log Stage 2 metrics."""
        metrics = {
            'stage2/train_loss': train_loss,
            'stage2/teacher_forcing_loss': tf_loss,
            'stage2/rollout_loss': rollout_loss,
            'stage2/learning_rate': self.stage2_optimizer.param_groups[0]['lr'],
            'stage2/epoch': epoch
        }
        
        if self.val_dataloader is not None:
            metrics['stage2/val_loss'] = val_loss
        
        if wandb.run is not None:
            wandb.log(metrics)
        
        print(f"Stage 2 Epoch {epoch+1}: Train Loss: {train_loss:.4f}, TF Loss: {tf_loss:.4f}, "
              f"Rollout Loss: {rollout_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def _save_stage1_checkpoint(self):
        """Save Stage 1 checkpoint."""
        checkpoint = {
            'model_state_dict': self.stage1_model.state_dict(),
            'optimizer_state_dict': self.stage1_optimizer.state_dict(),
            'scheduler_state_dict': self.stage1_scheduler.state_dict(),
            'epoch': self.current_stage,
            'best_val_loss': self.best_val_loss
        }
        
        os.makedirs('trained_models', exist_ok=True)
        torch.save(checkpoint, 'trained_models/best_stage1_model.pth')
        print("Saved Stage 1 checkpoint")
    
    def _save_stage2_checkpoint(self):
        """Save Stage 2 checkpoint."""
        checkpoint = {
            'model_state_dict': self.stage2_model.state_dict(),
            'optimizer_state_dict': self.stage2_optimizer.state_dict(),
            'scheduler_state_dict': self.stage2_scheduler.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'epoch': self.current_stage,
            'best_val_loss': self.best_val_loss
        }
        
        os.makedirs('trained_models', exist_ok=True)
        torch.save(checkpoint, 'trained_models/best_stage2_model.pth')
        print("Saved Stage 2 checkpoint")
    
    def train(self):
        """Train both stages of the V-JEPA2 world model."""
        print("Starting V-JEPA2 World Model Training")
        
        # Stage 1: Self-Supervised Encoder Pretraining
        print("\n" + "="*50)
        print("STAGE 1: Self-Supervised Encoder Pretraining")
        print("="*50)
        
        self.setup_stage1()
        self.train_stage1()
        
        # Stage 2: Action-Conditioned World Model Training
        print("\n" + "="*50)
        print("STAGE 2: Action-Conditioned World Model Training")
        print("="*50)
        
        self.setup_stage2()
        self.train_stage2()
        
        print("\nTraining completed!")
    
    def get_trained_models(self):
        """Return the trained models."""
        return {
            'encoder': self.encoder,
            'world_model': self.stage2_model
        } 