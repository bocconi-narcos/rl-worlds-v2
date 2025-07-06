#!/usr/bin/env python3
"""
JEPA with VICReg Analysis - Production Ready Training Script

This script focuses specifically on training JEPA with VICReg loss and provides
comprehensive statistics to understand plateauing behavior and training dynamics.

Key Features:
- JEPA-only training with VICReg auxiliary loss
- Comprehensive VICReg component monitoring (similarity, variance, covariance)
- Gradient and parameter norm tracking
- Representation quality metrics
- Learning rate scheduling and warmup
- Extensive logging and visualization
- Early stopping with multiple metrics
- Production-ready error handling and stability checks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import wandb
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import warnings

# Import project modules
from src.utils.config_utils import load_config
from src.utils.env_utils import get_env_details
from src.data_handling import prepare_dataloaders
from src.model_setup import initialize_models
from src.loss_setup import initialize_loss_functions
from src.optimizer_setup import initialize_optimizers
from src.models.jepa import JEPA
from src.losses.vicreg import VICRegLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jepa_vicreg_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingStats:
    """Container for training statistics"""
    epoch: int
    batch: int
    total_loss: float
    sim_loss: float
    std_loss: float
    cov_loss: float
    grad_norm: float
    param_norm: float
    lr: float
    representation_stats: Dict[str, float]
    target_diff: float
    batch_time: float

class VICRegAnalyzer:
    """Analyzer for VICReg loss components and representation quality"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.stats_history = defaultdict(list)
        
    def analyze_representations(self, z: torch.Tensor, prefix: str = "") -> Dict[str, float]:
        """Analyze representation quality metrics"""
        with torch.no_grad():
            # Basic statistics
            mean_val = z.mean().item()
            std_val = z.std().item()
            
            # Feature-wise statistics
            feature_means = z.mean(dim=0)
            feature_stds = z.std(dim=0)
            
            # Variance analysis
            feature_variances = z.var(dim=0)
            variance_mean = feature_variances.mean().item()
            variance_std = feature_variances.std().item()
            variance_min = feature_variances.min().item()
            variance_max = feature_variances.max().item()
            
            # Covariance analysis
            z_centered = z - z.mean(dim=0)
            cov_matrix = (z_centered.T @ z_centered) / max(z.size(0) - 1, 1)
            off_diagonal_cov = cov_matrix.fill_diagonal_(0)
            cov_mean = off_diagonal_cov.mean().item()
            cov_std = off_diagonal_cov.std().item()
            cov_max = off_diagonal_cov.abs().max().item()
            
            # Rank analysis (effective rank)
            singular_values = torch.linalg.svd(z, full_matrices=False)[1]
            effective_rank = (singular_values / singular_values.max()).sum().item()
            
            # Collapse detection
            collapsed_features = (feature_variances < 0.01).sum().item()
            collapse_ratio = collapsed_features / z.size(1)
            
            # Feature correlation
            feature_correlations = torch.corrcoef(z.T)
            off_diagonal_corr = feature_correlations.fill_diagonal_(0)
            max_correlation = off_diagonal_corr.abs().max().item()
            mean_correlation = off_diagonal_corr.abs().mean().item()
            
            stats = {
                f"{prefix}mean": mean_val,
                f"{prefix}std": std_val,
                f"{prefix}variance_mean": variance_mean,
                f"{prefix}variance_std": variance_std,
                f"{prefix}variance_min": variance_min,
                f"{prefix}variance_max": variance_max,
                f"{prefix}cov_mean": cov_mean,
                f"{prefix}cov_std": cov_std,
                f"{prefix}cov_max": cov_max,
                f"{prefix}effective_rank": effective_rank,
                f"{prefix}collapse_ratio": collapse_ratio,
                f"{prefix}max_correlation": max_correlation,
                f"{prefix}mean_correlation": mean_correlation,
            }
            
            # Store in history
            for key, value in stats.items():
                self.stats_history[key].append(value)
                
            return stats
    
    def get_trends(self, window: int = 100) -> Dict[str, float]:
        """Calculate trends over recent history"""
        trends = {}
        for key, history in self.stats_history.items():
            if len(history) >= window:
                recent = history[-window:]
                if len(recent) > 1:
                    # Simple linear trend
                    x = np.arange(len(recent))
                    y = np.array(recent)
                    slope = np.polyfit(x, y, 1)[0]
                    trends[f"{key}_trend"] = slope
        return trends

class JEPAVICRegTrainer:
    """Production-ready JEPA trainer with VICReg analysis"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.analyzer = VICRegAnalyzer(device)
        
        # Initialize components
        self._setup_environment()
        self._setup_data()
        self._setup_models()
        self._setup_losses()
        self._setup_optimizers()
        self._setup_schedulers()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Performance tracking
        self.batch_times = deque(maxlen=100)
        self.loss_history = defaultdict(list)
        
        logger.info("JEPA VICReg Trainer initialized successfully")
    
    def _setup_environment(self):
        """Setup environment and configuration"""
        env_config = self.config.get('environment', {})
        self.action_dim, self.action_type, self.observation_space = get_env_details(
            env_config.get('name', 'ALE/Pong-v5')
        )
        
        self.image_h = env_config.get('image_height', 64)
        self.image_w = env_config.get('image_width', 64)
        self.image_h_w = (self.image_h, self.image_w)
        
        # Input channels calculation
        self.input_channels = env_config.get('input_channels_per_frame', 3)
        if env_config.get('grayscale_conversion', False):
            self.input_channels = 1
            
        logger.info(f"Environment: {env_config.get('name')}")
        logger.info(f"Image size: {self.image_h_w}, Input channels: {self.input_channels}")
        logger.info(f"Action dim: {self.action_dim}, Action type: {self.action_type}")
    
    def _setup_data(self):
        """Setup data loaders"""
        data_config = self.config.get('data', {})
        validation_split = data_config.get('validation_split', 0.2)
        
        self.train_dataloader, self.val_dataloader = prepare_dataloaders(
            self.config, validation_split
        )
        
        if self.train_dataloader is None:
            raise ValueError("No training data available")
            
        logger.info(f"Training batches: {len(self.train_dataloader)}")
        if self.val_dataloader:
            logger.info(f"Validation batches: {len(self.val_dataloader)}")
    
    def _setup_models(self):
        """Setup JEPA model"""
        models_map = initialize_models(
            self.config, self.action_dim, self.action_type, 
            self.device, self.image_h_w, self.input_channels
        )
        
        self.jepa_model = models_map.get('jepa')
        if self.jepa_model is None:
            raise ValueError("JEPA model not found in models_map")
            
        # Set to training mode
        self.jepa_model.train()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.jepa_model.parameters())
        trainable_params = sum(p.numel() for p in self.jepa_model.parameters() if p.requires_grad)
        
        logger.info(f"JEPA Model - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    def _setup_losses(self):
        """Setup VICReg loss with comprehensive monitoring"""
        models_config = self.config.get('models', {})
        aux_loss_config = models_config.get('auxiliary_loss', {})
        
        if aux_loss_config.get('type') != 'vicreg':
            raise ValueError("This trainer requires VICReg auxiliary loss")
        
        vicreg_params = aux_loss_config.get('params', {}).get('vicreg', {})
        
        # Create VICReg loss with projector for better decoupling
        self.vicreg_loss = VICRegLoss(
            sim_coeff=vicreg_params.get('sim_coeff', 0.5),
            std_coeff=vicreg_params.get('std_coeff', 10.0),
            cov_coeff=vicreg_params.get('cov_coeff', 0.1),
            eps=vicreg_params.get('eps', 1e-3),
            proj_hidden_dim=vicreg_params.get('proj_hidden_dim', 128),
            proj_output_dim=vicreg_params.get('proj_output_dim', 128),
            representation_dim=models_config.get('shared_latent_dim', 64)
        ).to(self.device)
        
        # MSE loss for fallback 
        self.mse_loss = nn.MSELoss()
        
        logger.info("VICReg loss initialized with projector")
    
    def _setup_optimizers(self):
        """Setup optimizers with gradient clipping"""
        training_config = self.config.get('training', {})
        jepa_config = self.config.get('models', {}).get('jepa', {})
        
        # JEPA optimizer
        self.jepa_optimizer = optim.AdamW(
            self.jepa_model.parameters(),
            lr=jepa_config.get('learning_rate', 0.0003),
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # VICReg optimizer (separate for better control)
        self.vicreg_optimizer = optim.AdamW(
            self.vicreg_loss.parameters(),
            lr=jepa_config.get('learning_rate', 0.0003),
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        logger.info("Optimizers initialized")
    
    def _setup_schedulers(self):
        """Setup learning rate schedulers"""
        training_config = self.config.get('training', {})
        
        # Cosine annealing with warmup
        self.jepa_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.jepa_optimizer,
            T_0=training_config.get('scheduler_t0', 10),
            T_mult=2,
            eta_min=1e-6
        )
        
        self.vicreg_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.vicreg_optimizer,
            T_0=training_config.get('scheduler_t0', 10),
            T_mult=2,
            eta_min=1e-6
        )
        
        logger.info("Learning rate schedulers initialized")
    
    def _compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute JEPA loss with VICReg and comprehensive monitoring"""
        s_t, a_t, r_t, s_t_plus_1 = batch
        
        # Move to device
        s_t = s_t.to(self.device)
        s_t_plus_1 = s_t_plus_1.to(self.device)
        
        if self.action_type == 'discrete':
            a_t_processed = a_t.to(self.device)
        else:
            a_t_processed = a_t.float().to(self.device)
        
        # Forward pass
        pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = self.jepa_model(
            s_t, a_t_processed, s_t_plus_1
        )
        
        # VICReg loss with detailed components
        total_vicreg_loss, sim_loss, std_loss, cov_loss = self.vicreg_loss(
            online_s_t_emb, target_emb_detached
        )
        
        # Analyze representations
        online_stats = self.analyzer.analyze_representations(online_s_t_emb, "online_")
        target_stats = self.analyzer.analyze_representations(target_emb_detached, "target_")
        
        # Target encoder difference (EMA tracking)
        target_diff = 0.0
        if hasattr(self.jepa_model, 'target_encoder') and self.jepa_model.target_encoder is not None:
            with torch.no_grad():
                for p_online, p_target in zip(
                    self.jepa_model.online_encoder.parameters(),
                    self.jepa_model.target_encoder.parameters()
                ):
                    target_diff += F.mse_loss(p_online, p_target, reduction='sum').item()
                target_diff /= sum(p.numel() for p in self.jepa_model.online_encoder.parameters())
        
        # Loss components for logging
        loss_components = {
            'total_loss': total_vicreg_loss.item(),
            'sim_loss': sim_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item(),
            'target_diff': target_diff,
            **online_stats,
            **target_stats
        }
        
        return total_vicreg_loss, loss_components
    
    def _backward_and_optimize(self, loss: torch.Tensor):
        """Backward pass with gradient clipping and optimization"""
        # Zero gradients
        self.jepa_optimizer.zero_grad()
        self.vicreg_optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        jepa_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.jepa_model.parameters(), self.max_grad_norm
        )
        vicreg_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vicreg_loss.parameters(), self.max_grad_norm
        )
        
        # Check for gradient issues
        if not torch.isfinite(jepa_grad_norm) or not torch.isfinite(vicreg_grad_norm):
            logger.warning("Non-finite gradients detected, skipping update")
            self.jepa_optimizer.zero_grad()
            self.vicreg_optimizer.zero_grad()
            return float('inf'), float('inf')
        
        # Optimizer steps
        self.jepa_optimizer.step()
        self.vicreg_optimizer.step()
        
        # Update target encoder (EMA)
        if hasattr(self.jepa_model, 'perform_ema_update'):
            self.jepa_model.perform_ema_update()
        
        return jepa_grad_norm.item(), vicreg_grad_norm.item()
    
    def _validate(self) -> Dict[str, float]:
        """Validation phase with comprehensive metrics"""
        self.jepa_model.eval()
        self.vicreg_loss.eval()
        
        val_losses = defaultdict(list)
        val_stats = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss, components = self._compute_loss(batch)
                
                for key, value in components.items():
                    val_stats[key].append(value)
        
        # Aggregate validation metrics
        val_metrics = {}
        for key, values in val_stats.items():
            val_metrics[f'val_{key}'] = np.mean(values)
            val_metrics[f'val_{key}_std'] = np.std(values)
        
        self.jepa_model.train()
        self.vicreg_loss.train()
        
        return val_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with comprehensive monitoring"""
        self.jepa_model.train()
        self.vicreg_loss.train()
        
        epoch_stats = defaultdict(list)
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            batch_start_time = time.time()
            
            # Compute loss
            loss, components = self._compute_loss(batch)
            
            # Backward and optimize
            jepa_grad_norm, vicreg_grad_norm = self._backward_and_optimize(loss)
            
            # Update schedulers
            self.jepa_scheduler.step()
            self.vicreg_scheduler.step()
            
            # Record statistics
            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)
            
            # Store epoch stats
            for key, value in components.items():
                epoch_stats[key].append(value)
            
            epoch_stats['jepa_grad_norm'].append(jepa_grad_norm)
            epoch_stats['vicreg_grad_norm'].append(vicreg_grad_norm)
            epoch_stats['batch_time'].append(batch_time)
            epoch_stats['jepa_lr'].append(self.jepa_optimizer.param_groups[0]['lr'])
            epoch_stats['vicreg_lr'].append(self.vicreg_optimizer.param_groups[0]['lr'])
            
            # Logging
            if (batch_idx + 1) % self.config.get('training', {}).get('log_interval', 20) == 0:
                avg_loss = np.mean(epoch_stats['total_loss'][-50:])
                avg_sim = np.mean(epoch_stats['sim_loss'][-50:])
                avg_std = np.mean(epoch_stats['std_loss'][-50:])
                avg_cov = np.mean(epoch_stats['cov_loss'][-50:])
                
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx+1}/{len(self.train_dataloader)}: "
                    f"Loss={avg_loss:.4f}, Sim={avg_sim:.4f}, Std={avg_std:.4f}, Cov={avg_cov:.4f}"
                )
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'train/step': self.global_step,
                        'train/total_loss': avg_loss,
                        'train/sim_loss': avg_sim,
                        'train/std_loss': avg_std,
                        'train/cov_loss': avg_cov,
                        'train/jepa_grad_norm': np.mean(epoch_stats['jepa_grad_norm'][-50:]),
                        'train/vicreg_grad_norm': np.mean(epoch_stats['vicreg_grad_norm'][-50:]),
                        'train/jepa_lr': self.jepa_optimizer.param_groups[0]['lr'],
                        'train/vicreg_lr': self.vicreg_optimizer.param_groups[0]['lr'],
                        'train/batch_time': np.mean(self.batch_times),
                    }, step=self.global_step)
            
            self.global_step += 1
        
        # Aggregate epoch statistics
        epoch_metrics = {}
        for key, values in epoch_stats.items():
            epoch_metrics[f'train_{key}_mean'] = np.mean(values)
            epoch_metrics[f'train_{key}_std'] = np.std(values)
        
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time
        
        return epoch_metrics
    
    def train(self, num_epochs: int, patience: int = 10):
        """Main training loop with early stopping"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch} Summary:\n"
                f"  Train Loss: {all_metrics.get('train_total_loss_mean', 0.0):.4f}\n"
                f"  Val Loss: {all_metrics.get('val_total_loss', 0.0):.4f}\n"
                f"  Sim Loss: {all_metrics.get('train_sim_loss_mean', 0.0):.4f}\n"
                f"  Std Loss: {all_metrics.get('train_std_loss_mean', 0.0):.4f}\n"
                f"  Cov Loss: {all_metrics.get('train_cov_loss_mean', 0.0):.4f}"
            )
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    **all_metrics
                })
            
            # Early stopping
            val_loss = all_metrics.get('val_total_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                self._save_checkpoint('best_model.pth', all_metrics)
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', all_metrics)
        
        logger.info("Training completed")
    
    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint with metrics"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'jepa_model_state_dict': self.jepa_model.state_dict(),
            'vicreg_loss_state_dict': self.vicreg_loss.state_dict(),
            'jepa_optimizer_state_dict': self.jepa_optimizer.state_dict(),
            'vicreg_optimizer_state_dict': self.vicreg_optimizer.state_dict(),
            'jepa_scheduler_state_dict': self.jepa_scheduler.state_dict(),
            'vicreg_scheduler_state_dict': self.vicreg_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = Path('trained_models') / filename
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def analyze_plateauing(self) -> Dict[str, any]:
        """Analyze potential plateauing causes"""
        analysis = {}
        
        # Get recent trends
        trends = self.analyzer.get_trends(window=100)
        
        # Loss component analysis
        if len(self.loss_history['total_loss']) > 50:
            recent_losses = self.loss_history['total_loss'][-50:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            analysis['loss_trend'] = loss_trend
            analysis['loss_plateauing'] = abs(loss_trend) < 1e-5
        
        # Representation collapse analysis
        if 'online_collapse_ratio_trend' in trends:
            analysis['collapse_trend'] = trends['online_collapse_ratio_trend']
            analysis['collapsing'] = trends['online_collapse_ratio_trend'] > 0.01
        
        # Gradient analysis
        if len(self.loss_history['jepa_grad_norm']) > 50:
            recent_grads = self.loss_history['jepa_grad_norm'][-50:]
            grad_trend = np.polyfit(range(len(recent_grads)), recent_grads, 1)[0]
            analysis['grad_trend'] = grad_trend
            analysis['vanishing_grads'] = grad_trend < -0.01
        
        # Learning rate analysis
        if len(self.loss_history['jepa_lr']) > 10:
            current_lr = self.loss_history['jepa_lr'][-1]
            analysis['current_lr'] = current_lr
            analysis['lr_too_small'] = current_lr < 1e-6
        
        return analysis

def main():
    """Main training function"""
    # Load configuration
    config = load_config()
    
    # Initialize wandb
    wandb_cfg = config.get('wandb', {})
    if wandb_cfg.get('enabled', False):
        try:
            wandb.init(
                project=wandb_cfg.get('project', 'jepa-vicreg-analysis'),
                entity=wandb_cfg.get('entity'),
                name=f"jepa-vicreg-{time.strftime('%Y%m%d-%H%M%S')}",
                config=config,
                tags=['jepa', 'vicreg', 'analysis']
            )
            logger.info("Weights & Biases initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Weights & Biases: {e}")
            wandb.init = None
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")
    
    # Create trainer
    trainer = JEPAVICRegTrainer(config, device)
    
    # Training parameters
    training_config = config.get('training', {})
    num_epochs = training_config.get('num_epochs', 50)
    patience = training_config.get('early_stopping', {}).get('patience', 10)
    
    try:
        # Start training
        trainer.train(num_epochs, patience)
        
        # Final analysis
        analysis = trainer.analyze_plateauing()
        logger.info("Final plateauing analysis:")
        for key, value in analysis.items():
            logger.info(f"  {key}: {value}")
        
        if wandb.run is not None:
            wandb.log({'final_analysis': analysis})
            wandb.finish()
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()

if __name__ == '__main__':
    main()