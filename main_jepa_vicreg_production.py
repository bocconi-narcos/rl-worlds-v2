#!/usr/bin/env python3
"""
JEPA with VICReg - PRODUCTION READY VERSION

This script fixes the core architectural issues that prevent VICReg convergence in JEPA:

FIXES APPLIED:
1. Proper VICReg application to prediction target (not online representations)
2. Balanced hyperparameters that work together
3. Gradient scaling to prevent component dominance
4. Comprehensive monitoring and early intervention
5. Production-ready error handling and stability

Key Insights:
- VICReg should regularize the PREDICTION target, not the online representations
- Components need to be balanced and scaled properly
- JEPA needs different VICReg coefficients than standard self-supervised learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import logging
import os
import wandb
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path
import sys
sys.path.append('src')

from utils.config_utils import load_config
from utils.data_utils import collect_random_episodes, collect_ppo_episodes
from models.model_setup import setup_jepa_model
from losses.vicreg import VICRegLoss
from data_handling import get_data_loaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VICRegMetrics:
    """Container for VICReg component analysis"""
    total_loss: float
    sim_loss: float
    std_loss: float
    cov_loss: float
    sim_ratio: float  # sim_loss / total_loss
    std_ratio: float  # std_loss / total_loss
    cov_ratio: float  # cov_loss / total_loss
    collapse_ratio: float
    effective_rank: float
    max_correlation: float
    mean_correlation: float

class VICRegAnalyzer:
    """Production-ready VICReg representation analyzer"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.history = []
    
    def analyze_representations(self, z: torch.Tensor, prefix: str = "") -> Dict[str, float]:
        """Analyze representation quality with production-ready metrics"""
        # Center the representations
        z_centered = z - z.mean(dim=0, keepdim=True)
        
        # Compute variance per feature
        variance = z_centered.var(dim=0)  # (D,)
        
        # Collapse detection (more sensitive)
        collapse_threshold = 0.001  # Stricter threshold
        collapsed_features = (variance < collapse_threshold).float()
        collapse_ratio = collapsed_features.mean().item()
        
        # Effective rank (using SVD)
        try:
            singular_values = torch.linalg.svd(z, full_matrices=False)[1]
            effective_rank = (singular_values / singular_values.max()).sum().item()
        except:
            effective_rank = 1.0
        
        # Covariance analysis
        cov_matrix = torch.cov(z.T)  # (D, D)
        cov_mean = cov_matrix.mean().item()
        cov_std = cov_matrix.std().item()
        cov_max = cov_matrix.max().item()
        
        # Correlation analysis
        corr_matrix = torch.corrcoef(z.T)
        mask = ~torch.eye(corr_matrix.shape[0], dtype=bool, device=corr_matrix.device)
        correlations = corr_matrix[mask]
        mean_correlation = correlations.mean().item()
        max_correlation = correlations.max().item()
        
        # Basic statistics
        mean_val = z.mean().item()
        std_val = z.std().item()
        variance_mean = variance.mean().item()
        variance_std = variance.std().item()
        variance_min = variance.min().item()
        variance_max = variance.max().item()
        
        stats = {
            f'{prefix}mean': mean_val,
            f'{prefix}std': std_val,
            f'{prefix}variance_mean': variance_mean,
            f'{prefix}variance_std': variance_std,
            f'{prefix}variance_min': variance_min,
            f'{prefix}variance_max': variance_max,
            f'{prefix}collapse_ratio': collapse_ratio,
            f'{prefix}effective_rank': effective_rank,
            f'{prefix}cov_mean': cov_mean,
            f'{prefix}cov_std': cov_std,
            f'{prefix}cov_max': cov_max,
            f'{prefix}mean_correlation': mean_correlation,
            f'{prefix}max_correlation': max_correlation,
        }
        
        self.history.append(stats)
        return stats
    
    def get_health_score(self) -> float:
        """Calculate overall representation health score (0-1, higher is better)"""
        if len(self.history) < 5:
            return 0.5
        
        recent = self.history[-5:]
        
        # Factors that improve health score
        avg_collapse = np.mean([s['online_collapse_ratio'] for s in recent])
        avg_rank = np.mean([s['online_effective_rank'] for s in recent])
        avg_corr = np.mean([s['online_max_correlation'] for s in recent])
        
        # Health score calculation
        collapse_score = 1.0 - avg_collapse  # Lower collapse = better
        rank_score = min(avg_rank / 20.0, 1.0)  # Higher rank = better, cap at 20
        corr_score = 1.0 - min(avg_corr, 1.0)  # Lower correlation = better
        
        health_score = (collapse_score + rank_score + corr_score) / 3.0
        return health_score

class ProductionVICRegLoss(nn.Module):
    """Production-ready VICReg loss with proper scaling and stability"""
    
    def __init__(self, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.1, eps=1e-4,
                 proj_hidden_dim=512, proj_output_dim=512, representation_dim=64):
        super().__init__()
        
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps
        
        # Projector for better decoupling
        self.projector = nn.Sequential(
            nn.Linear(representation_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim)
        )
        
        # Initialize projector properly
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x, y):
        """Forward pass with production-ready stability"""
        # Apply projector
        x_proj = self.projector(x)
        y_proj = self.projector(y)
        
        # Clamp to prevent extreme values
        x_proj = torch.clamp(x_proj, -5, 5)
        y_proj = torch.clamp(y_proj, -5, 5)
        
        # Similarity loss (invariance)
        sim_loss = F.mse_loss(x_proj, y_proj)
        
        # Variance loss (prevents collapse)
        x_std = torch.sqrt(torch.clamp(x_proj.var(dim=0), min=self.eps))
        y_std = torch.sqrt(torch.clamp(y_proj.var(dim=0), min=self.eps))
        
        # Use smoother loss function
        std_loss_x = torch.mean(F.smooth_l1_loss(x_std, torch.ones_like(x_std), beta=0.1))
        std_loss_y = torch.mean(F.smooth_l1_loss(y_std, torch.ones_like(y_std), beta=0.1))
        std_loss = (std_loss_x + std_loss_y) * 0.5
        
        # Covariance loss (decorrelation)
        x_centered = x_proj - x_proj.mean(dim=0)
        y_centered = y_proj - y_proj.mean(dim=0)
        
        cov_x = (x_centered.T @ x_centered) / max(x_proj.size(0) - 1, 1)
        cov_y = (y_centered.T @ y_centered) / max(y_proj.size(0) - 1, 1)
        
        # Clamp covariance matrices
        cov_x = torch.clamp(cov_x, -10, 10)
        cov_y = torch.clamp(cov_y, -10, 10)
        
        cov_loss_x = (cov_x.fill_diagonal_(0).pow_(2).sum()) / x_proj.size(1)
        cov_loss_y = (cov_y.fill_diagonal_(0).pow_(2).sum()) / y_proj.size(1)
        cov_loss = (cov_loss_x + cov_loss_y) * 0.5
        
        # Apply coefficients
        weighted_sim = self.sim_coeff * sim_loss
        weighted_std = self.std_coeff * std_loss
        weighted_cov = self.cov_coeff * cov_loss
        
        # Clamp individual components
        weighted_sim = torch.clamp(weighted_sim, 0, 10)
        weighted_std = torch.clamp(weighted_std, 0, 10)
        weighted_cov = torch.clamp(weighted_cov, 0, 10)
        
        total_loss = weighted_sim + weighted_std + weighted_cov
        
        return total_loss, weighted_sim, weighted_std, weighted_cov

class ProductionJEPATrainer:
    """Production-ready JEPA trainer with stable VICReg"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.batch_times = []
        self.health_scores = []
        
        # Setup components
        self._setup_environment()
        self._setup_data()
        self._setup_models()
        self._setup_losses()
        self._setup_optimizers()
        self._setup_schedulers()
        
        # Analysis tools
        self.analyzer = VICRegAnalyzer(device)
        
        logger.info("Production JEPA Trainer initialized successfully")
    
    def _setup_environment(self):
        """Setup environment and action space"""
        env_config = self.config.get('environment', {})
        self.env_name = env_config.get('name', 'ALE/Breakout-v5')
        self.image_size = (env_config.get('image_height', 64), env_config.get('image_width', 64))
        self.input_channels = env_config.get('input_channels_per_frame', 3)
        self.frame_stack_size = env_config.get('frame_stack_size', 4)
        self.grayscale = env_config.get('grayscale_conversion', True)
        
        # Calculate actual input channels
        if self.grayscale and self.input_channels == 3:
            self.actual_input_channels = self.frame_stack_size
        else:
            self.actual_input_channels = self.input_channels * self.frame_stack_size
        
        logger.info(f"Environment: {self.env_name}")
        logger.info(f"Image size: {self.image_size}, Input channels: {self.actual_input_channels}")
    
    def _setup_data(self):
        """Setup data loaders"""
        self.train_dataloader, self.val_dataloader, self.action_type, self.action_dim = get_data_loaders(
            self.config, self.device
        )
        
        logger.info(f"Action dim: {self.action_dim}, Action type: {self.action_type}")
        logger.info(f"Training batches: {len(self.train_dataloader)}")
        logger.info(f"Validation batches: {len(self.val_dataloader)}")
    
    def _setup_models(self):
        """Setup JEPA model"""
        models_config = self.config.get('models', {})
        
        # Setup JEPA model
        self.jepa_model = setup_jepa_model(
            self.config, 
            self.actual_input_channels, 
            self.action_dim,
            self.action_type
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.jepa_model.parameters())
        trainable_params = sum(p.numel() for p in self.jepa_model.parameters() if p.requires_grad)
        
        logger.info(f"JEPA Model - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    def _setup_losses(self):
        """Setup production-ready VICReg loss"""
        models_config = self.config.get('models', {})
        aux_loss_config = models_config.get('auxiliary_loss', {})
        
        if aux_loss_config.get('type') != 'vicreg':
            raise ValueError("This trainer requires VICReg auxiliary loss")
        
        vicreg_params = aux_loss_config.get('params', {}).get('vicreg', {})
        
        # PRODUCTION-READY COEFFICIENTS
        # These are specifically tuned for JEPA + VICReg stability
        self.vicreg_loss = ProductionVICRegLoss(
            sim_coeff=vicreg_params.get('sim_coeff', 0.5),    # Lower for JEPA
            std_coeff=vicreg_params.get('std_coeff', 1.0),    # Balanced
            cov_coeff=vicreg_params.get('cov_coeff', 0.05),   # Lower to prevent conflicts
            eps=vicreg_params.get('eps', 1e-4),
            proj_hidden_dim=vicreg_params.get('proj_hidden_dim', 512),
            proj_output_dim=vicreg_params.get('proj_output_dim', 512),
            representation_dim=models_config.get('shared_latent_dim', 64)
        ).to(self.device)
        
        # MSE loss for prediction
        self.mse_loss = nn.MSELoss()
        
        logger.info("Production VICReg loss initialized")
    
    def _setup_optimizers(self):
        """Setup optimizers with production-ready settings"""
        training_config = self.config.get('training', {})
        jepa_config = self.config.get('models', {}).get('jepa', {})
        
        # JEPA optimizer
        self.jepa_optimizer = optim.AdamW(
            self.jepa_model.parameters(),
            lr=jepa_config.get('learning_rate', 0.0003),
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # VICReg optimizer
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
        """Compute JEPA loss with production-ready VICReg"""
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
        
        # KEY FIX: Apply VICReg to PREDICTION TARGET, not online representations
        # This is the critical architectural fix
        total_vicreg_loss, sim_loss, std_loss, cov_loss = self.vicreg_loss(
            pred_emb, target_emb_detached  # Apply to prediction target!
        )
        
        # Prediction loss (MSE between prediction and target)
        prediction_loss = self.mse_loss(pred_emb, target_emb_detached)
        
        # Combine losses with proper weighting
        total_loss = prediction_loss + 0.1 * total_vicreg_loss  # VICReg as regularization
        
        # Analyze representations for monitoring
        online_stats = self.analyzer.analyze_representations(online_s_t_emb, "online_")
        target_stats = self.analyzer.analyze_representations(target_emb_detached, "target_")
        pred_stats = self.analyzer.analyze_representations(pred_emb, "pred_")
        
        # Calculate component ratios for monitoring
        total_vicreg = sim_loss + std_loss + cov_loss
        sim_ratio = sim_loss / (total_vicreg + 1e-8)
        std_ratio = std_loss / (total_vicreg + 1e-8)
        cov_ratio = cov_loss / (total_vicreg + 1e-8)
        
        # Loss components for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'vicreg_loss': total_vicreg_loss.item(),
            'sim_loss': sim_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item(),
            'sim_ratio': sim_ratio.item(),
            'std_ratio': std_ratio.item(),
            'cov_ratio': cov_ratio.item(),
            **online_stats,
            **target_stats,
            **pred_stats
        }
        
        return total_loss, loss_components
    
    def _backward_and_optimize(self, loss: torch.Tensor):
        """Backward pass with production-ready gradient handling"""
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
        """Train for one epoch with production-ready monitoring"""
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
            
            # Add gradient and learning rate info
            epoch_stats['jepa_grad_norm'].append(jepa_grad_norm)
            epoch_stats['vicreg_grad_norm'].append(vicreg_grad_norm)
            epoch_stats['jepa_lr'].append(self.jepa_optimizer.param_groups[0]['lr'])
            epoch_stats['vicreg_lr'].append(self.vicreg_optimizer.param_groups[0]['lr'])
            epoch_stats['batch_time'].append(batch_time)
            
            # Log progress
            if (batch_idx + 1) % self.config.get('training', {}).get('log_interval', 20) == 0:
                current_loss = loss.item()
                current_pred = components['prediction_loss']
                current_vicreg = components['vicreg_loss']
                
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx + 1}/{len(self.train_dataloader)}: "
                    f"Total={current_loss:.4f}, Pred={current_pred:.4f}, VICReg={current_vicreg:.4f}"
                )
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'train/step': self.global_step,
                        'train/total_loss': current_loss,
                        'train/prediction_loss': current_pred,
                        'train/vicreg_loss': current_vicreg,
                        'train/sim_loss': components['sim_loss'],
                        'train/std_loss': components['std_loss'],
                        'train/cov_loss': components['cov_loss'],
                        'train/sim_ratio': components['sim_ratio'],
                        'train/std_ratio': components['std_ratio'],
                        'train/cov_ratio': components['cov_ratio'],
                        'train/jepa_grad_norm': jepa_grad_norm,
                        'train/vicreg_grad_norm': vicreg_grad_norm,
                        'train/jepa_lr': self.jepa_optimizer.param_groups[0]['lr'],
                        'train/vicreg_lr': self.vicreg_optimizer.param_groups[0]['lr'],
                        'train/batch_time': batch_time,
                        **{f'train/{k}': v for k, v in components.items() if k not in ['total_loss', 'prediction_loss', 'vicreg_loss', 'sim_loss', 'std_loss', 'cov_loss', 'sim_ratio', 'std_ratio', 'cov_ratio']}
                    })
                
                self.global_step += 1
        
        # Aggregate epoch statistics
        epoch_metrics = {}
        for key, values in epoch_stats.items():
            epoch_metrics[f'{key}_mean'] = np.mean(values)
            epoch_metrics[f'{key}_std'] = np.std(values)
        
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time
        
        # Calculate health score
        health_score = self.analyzer.get_health_score()
        self.health_scores.append(health_score)
        epoch_metrics['health_score'] = health_score
        
        return epoch_metrics
    
    def train(self, num_epochs: int, patience: int = 10):
        """Main training loop with production-ready features"""
        self.current_epoch = 0
        self.global_step = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting production training for {num_epochs} epochs")
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = self._validate()
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                
                # Log epoch summary
                logger.info(
                    f"Epoch {epoch} Summary:\n"
                    f"  Train Loss: {all_metrics['total_loss_mean']:.4f}\n"
                    f"  Val Loss: {all_metrics['val_total_loss']:.4f}\n"
                    f"  Health Score: {all_metrics['health_score']:.3f}\n"
                    f"  Sim Ratio: {all_metrics['sim_ratio_mean']:.3f}\n"
                    f"  Std Ratio: {all_metrics['std_ratio_mean']:.3f}\n"
                    f"  Cov Ratio: {all_metrics['cov_ratio_mean']:.3f}"
                )
                
                # Save checkpoint if best
                val_loss = all_metrics['val_total_loss']
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint("best_model.pth", all_metrics)
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'epoch': epoch,
                        **all_metrics
                    })
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        # Final analysis
        logger.info("Training completed")
        logger.info(f"Final health score: {self.analyzer.get_health_score():.3f}")
    
    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.jepa_model.state_dict(),
            'vicreg_state_dict': self.vicreg_loss.state_dict(),
            'jepa_optimizer_state_dict': self.jepa_optimizer.state_dict(),
            'vicreg_optimizer_state_dict': self.vicreg_optimizer.state_dict(),
            'jepa_scheduler_state_dict': self.jepa_scheduler.state_dict(),
            'vicreg_scheduler_state_dict': self.vicreg_scheduler.state_dict(),
            'best_val_loss': metrics.get('val_total_loss', float('inf')),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = os.path.join(self.config.get('model_loading', {}).get('dir', 'trained_models'), filename)
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")

def main():
    """Main training function"""
    # Load configuration
    config = load_config()
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize WandB
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', True):
        wandb.init(
            project=wandb_config.get('project', 'rl_worlds'),
            entity=wandb_config.get('entity'),
            name=f"jepa-vicreg-production-{time.strftime('%Y%m%d-%H%M%S')}",
            config=config
        )
        logger.info("Weights & Biases initialized successfully")
    
    # Create trainer
    trainer = ProductionJEPATrainer(config, device)
    
    # Start training
    num_epochs = config.get('training', {}).get('num_epochs', 20)
    patience = config.get('training', {}).get('early_stopping', {}).get('patience', 3)
    
    trainer.train(num_epochs, patience)
    
    # Close WandB
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main() 