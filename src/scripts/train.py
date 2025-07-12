#!/usr/bin/env python3

import sys
import argparse
import logging
import os
import time
import copy
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import wandb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_utils import load_config
from utils.setups import init_opt, init_video_model, init_mask_generator
from scripts.collect_load_data import DataCollectionPipeline
from masks.utils import apply_masks


class VJEPATrainingPipeline:
    """
    State-of-the-art V-JEPA training pipeline with comprehensive monitoring and evaluation.
    
    This class encapsulates the complete V-JEPA training workflow, providing:
    - Data loading and preprocessing
    - Model initialization and setup
    - Training loop with logging and monitoring
    - Validation and evaluation
    - Checkpointing and model saving
    - Error handling and recovery
    """
    
    def __init__(self, config_path: str = "new_config.yaml"):
        """
        Initialize the V-JEPA training pipeline.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = None
        self.device = None
        self.wandb_run = None
        
        # Data components
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Model components
        self.encoder = None
        self.predictor = None
        self.target_encoder = None
        self.mask_generator = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.momentum_scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_start_time = None
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure comprehensive logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('vjepa_training.log', mode='a')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_device(self) -> torch.device:
        """Get the best available device: CUDA > MPS > CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_configuration(self) -> dict:
        """
        Load and validate the configuration file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        self.logger.info(f"Loading configuration from {self.config_path}")
        
        self.config = load_config(self.config_path)
        
        # Validate critical configuration parameters
        self._validate_config()
        
        # Set device
        self.device = self.get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        seed = self.config.get("training", {}).get("random_seed", 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        
        self.logger.info("Configuration loaded and validated successfully")
        return self.config
    
    def _validate_config(self):
        """Validate that all required configuration parameters are present."""
        required_sections = ['environment', 'data_collection', 'training', 'model']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate training config
        training_config = self.config['training']
        required_training_params = [
            'batch_size', 'num_epochs', 'learning_rate', 'weight_decay',
            'warmup_epochs', 'start_lr', 'end_lr', 'log_interval'
        ]
        for param in required_training_params:
            if param not in training_config:
                raise ValueError(f"Missing required training parameter: {param}")
    
    def setup_wandb(self) -> Optional[Any]:
        """
        Initialize Weights & Biases logging if enabled.
        
        Returns:
            wandb run object if successful, None otherwise
        """
        wandb_config = self.config.get('wandb', {})
        
        if not wandb_config.get('enabled', False):
            wandb.init(mode="disabled")
            self.logger.info("W&B logging disabled")
            return None
        
        try:
            run_name = f"{wandb_config.get('run_name_prefix', 'vjepa')}-{time.strftime('%Y%m%d-%H%M%S')}"
            
            self.wandb_run = wandb.init(
                project=wandb_config.get('project', 'vjepa-training'),
                entity=wandb_config.get('entity'),
                name=run_name,
                config=self.config,
                resume="allow"
            )
            
            self.logger.info(f"W&B logging initialized: {run_name}")
            return self.wandb_run
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize W&B: {e}. Continuing without logging.")
            wandb.init(mode="disabled")
            return None
    
    def load_data(self) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        """
        Load training and validation data using the data collection pipeline.
        
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        self.logger.info("Loading training data using DataCollectionPipeline")
        
        # Use the existing data collection pipeline
        data_pipeline = DataCollectionPipeline(self.config_path)
        self.train_dataloader, self.val_dataloader = data_pipeline.run_full_pipeline()
        
        self.logger.info(f"Training batches: {len(self.train_dataloader)}")
        if self.val_dataloader:
            self.logger.info(f"Validation batches: {len(self.val_dataloader)}")
        
        return self.train_dataloader, self.val_dataloader
    
    def initialize_models(self):
        """Initialize all models and components for training."""
        self.logger.info("Initializing models and training components")
        
        # Extract configuration parameters
        model_config = self.config.get("model", {})
        data_config = self.config.get("data_collection", {})
        training_config = self.config.get("training", {})
        env_config = self.config.get("environment", {})
        
        # Model parameters
        model_name = model_config.get("model_name", "vit_tiny")
        pred_depth = model_config.get("pred_depth", 4)
        pred_num_heads = model_config.get("pred_num_heads", 4)
        pred_embed_dim = model_config.get("pred_embed_dim", 16)
        tubelet_size = model_config.get("tubelet_size", 2)
        patch_size = model_config.get("patch_size", 8)
        frames_per_clip = model_config.get("frames_per_clip", 6)
        
        # Data parameters
        image_height = env_config.get("image_height", 64)
        image_width = env_config.get("image_width", 64)
        sequence_length = data_config.get("sequence_length", 6)
        
        # Validate image dimensions
        if image_height != image_width:
            self.logger.warning(f"Image height ({image_height}) != width ({image_width}). Using height for crop_size.")
        crop_size = image_height
        
        max_num_frames = frames_per_clip
        
        self.logger.info(f"Model: {model_name}, Patch size: {patch_size}, Frames: {max_num_frames}")
        self.logger.info(f"Predictor depth: {pred_depth}, heads: {pred_num_heads}, embed_dim: {pred_embed_dim}")
        
        # Initialize models
        self.encoder, self.predictor = init_video_model(
            num_mask_tokens=1,
            device=self.device,
            patch_size=patch_size,
            max_num_frames=max_num_frames,
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_num_heads=pred_num_heads,
            pred_embed_dim=pred_embed_dim,
        )
        
        # Create target encoder (EMA copy)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        # Initialize mask generator
        input_size = (sequence_length, image_height, image_width)
        patch_size_mask = (2, 16, 16)  # Default for masking
        masking_ratio = data_config.get("masking_ratio", 0.5)
        
        self.mask_generator = init_mask_generator(
            input_size=input_size,
            patch_size=patch_size_mask,
            num_blocks=1,
            masking_ratio=masking_ratio,
        )
        
        self.logger.info(f"Mask generator initialized with ratio: {masking_ratio}")
        
        # Initialize optimizer and scheduler
        self._initialize_optimizer()
        
        self.logger.info("Model initialization completed successfully")
    
    def _initialize_optimizer(self):
        """Initialize optimizer and learning rate schedulers."""
        training_config = self.config['training']
        
        # Optimizer parameters
        wd = float(training_config.get("weight_decay", 0.0001))
        num_epochs = training_config.get("num_epochs", 10)
        warmup_epochs = training_config.get("warmup_epochs", 2)
        start_lr = training_config.get("start_lr", 0.0001)
        lr = training_config.get("learning_rate", 0.0001)
        final_lr = training_config.get("end_lr", 0.00001)
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = init_opt(
            encoder=self.encoder,
            predictor=self.predictor,
            wd=wd,
            start_lr=start_lr,
            ref_lr=lr,
            final_lr=final_lr,
            warmup_epochs=warmup_epochs,
            num_epochs=num_epochs,
        )
        
        # EMA momentum schedule
        ema_start, ema_end = 0.996, 1.0
        total_steps = num_epochs * len(self.train_dataloader)
        self.momentum_scheduler = iter([
            ema_start + (ema_end - ema_start) * (step / total_steps) 
            for step in range(total_steps)
        ])
        
        self.logger.info(f"Optimizer initialized: lr={lr}, wd={wd}, warmup_epochs={warmup_epochs}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, additional_info: Dict = None):
        """
        Save model checkpoint with comprehensive state information.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            additional_info: Additional information to save
        """
        checkpoint_dir = self.config.get('training', {}).get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        save_dict = {
            "encoder": self.encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "training_time": time.time() - self.training_start_time if self.training_start_time else 0,
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        try:
            torch.save(save_dict, latest_path)
            self.logger.info(f"Checkpoint saved: {latest_path}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            try:
                torch.save(save_dict, best_path)
                self.logger.info(f"Best checkpoint saved: {best_path}")
            except Exception as e:
                self.logger.error(f"Error saving best checkpoint: {e}")
        
        # Save epoch checkpoint periodically
        save_every = self.config.get('training', {}).get('save_checkpoint_every', 5)
        if epoch % save_every == 0:
            epoch_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            try:
                torch.save(save_dict, epoch_path)
                self.logger.info(f"Epoch checkpoint saved: {epoch_path}")
            except Exception as e:
                self.logger.error(f"Error saving epoch checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.predictor.load_state_dict(checkpoint["predictor"])
            self.target_encoder.load_state_dict(checkpoint["target_encoder"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            
            self.current_epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Training batch data
            
        Returns:
            Dictionary containing loss values and metrics
        """
        state, next_state, action, reward = batch
        clip = torch.cat((state, next_state), dim=2).to(self.device)
        batch_size = clip.size(0)
        
        # Generate masks
        masks_enc, masks_pred = self.mask_generator(batch_size)
        masks_enc, masks_pred = masks_enc.to(self.device), masks_pred.to(self.device)
        
        # Forward pass through target encoder (no gradients)
        with torch.no_grad():
            h = self.target_encoder(clip)
            h = F.layer_norm(h, (h.size(-1),))
        
        # Forward pass through context encoder and predictor
        z = self.encoder(clip, masks_enc)
        z = self.predictor(z, masks_enc, masks_pred)
        
        # Apply masks to target representations
        h = apply_masks(h, masks_pred, concat=False)
        
        # Compute loss
        loss = 0
        n = 0
        for zij, hij in zip(z, h):
            loss += torch.mean(torch.abs(zij - hij))
            n += 1
        loss /= n
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        # EMA update of target encoder
        m = next(self.momentum_scheduler)
        with torch.no_grad():
            params_k = []
            params_q = []
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                params_k.append(param_k)
                params_q.append(param_q)
            torch._foreach_mul_(params_k, m)
            torch._foreach_add_(params_k, params_q, alpha=1 - m)
        
        # Return metrics
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item() if torch.isfinite(grad_norm) else 0.0,
            'momentum': m,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on validation set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.val_dataloader:
            return {}
        
        self.encoder.eval()
        self.predictor.eval()
        self.target_encoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                state, next_state, action, reward = batch
                clip = torch.cat((state, next_state), dim=2).to(self.device)
                batch_size = clip.size(0)
                
                # Generate masks
                masks_enc, masks_pred = self.mask_generator(batch_size)
                masks_enc, masks_pred = masks_enc.to(self.device), masks_pred.to(self.device)
                
                # Forward pass
                h = self.target_encoder(clip)
                h = F.layer_norm(h, (h.size(-1),))
                
                z = self.encoder(clip, masks_enc)
                z = self.predictor(z, masks_enc, masks_pred)
                
                # Apply masks and compute loss
                h = apply_masks(h, masks_pred, concat=False)
                
                loss = 0
                n = 0
                for zij, hij in zip(z, h):
                    loss += torch.mean(torch.abs(zij - hij))
                    n += 1
                loss /= n
                
                total_loss += loss.item()
                num_batches += 1
        
        # Reset to training mode
        self.encoder.train()
        self.predictor.train()
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'val_loss': avg_val_loss,
            'val_batches': num_batches
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing epoch training metrics
        """
        self.encoder.train()
        self.predictor.train()
        self.target_encoder.eval()
        
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0
        log_interval = self.config['training']['log_interval']
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            step_start_time = time.time()
            
            # Training step
            metrics = self.train_step(batch)
            
            # Update counters
            epoch_loss += metrics['loss']
            epoch_grad_norm += metrics['grad_norm']
            num_batches += 1
            self.global_step += 1
            
            step_time = time.time() - step_start_time
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                avg_grad_norm = epoch_grad_norm / num_batches
                
                self.logger.info(
                    f"Epoch {epoch}/{self.config['training']['num_epochs']} "
                    f"[{batch_idx + 1}/{len(self.train_dataloader)}] "
                    f"Loss: {metrics['loss']:.6f} (avg: {avg_loss:.6f}) "
                    f"Grad: {metrics['grad_norm']:.4f} (avg: {avg_grad_norm:.4f}) "
                    f"LR: {metrics['learning_rate']:.2e} "
                    f"Step: {step_time:.3f}s"
                )
                
                # W&B logging
                if self.wandb_run:
                    self.wandb_run.log({
                        'train/loss_step': metrics['loss'],
                        'train/grad_norm_step': metrics['grad_norm'],
                        'train/learning_rate': metrics['learning_rate'],
                        'train/momentum': metrics['momentum'],
                        'train/step_time': step_time,
                        'epoch': epoch,
                        'global_step': self.global_step
                    })
        
        # Epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_epoch_grad_norm = epoch_grad_norm / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_epoch_loss,
            'train_grad_norm': avg_epoch_grad_norm,
            'epoch_time': epoch_time,
            'num_batches': num_batches
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the complete training loop.
        
        Returns:
            Dictionary containing training results and final metrics
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING V-JEPA TRAINING PIPELINE")
        self.logger.info("=" * 60)
        
        training_config = self.config['training']
        num_epochs = training_config['num_epochs']
        
        self.training_start_time = time.time()
        
        # Training loop
        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Update current epoch
            self.current_epoch = epoch
            
            # Check if this is the best model
            current_val_loss = val_metrics.get('val_loss', float('inf'))
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch} Summary: "
                f"Train Loss: {train_metrics['train_loss']:.6f}, "
                f"Val Loss: {current_val_loss:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # W&B epoch logging
            if self.wandb_run:
                log_data = {
                    'train/loss_epoch': train_metrics['train_loss'],
                    'train/grad_norm_epoch': train_metrics['train_grad_norm'],
                    'train/epoch_time': epoch_time,
                    'epoch': epoch
                }
                
                if val_metrics:
                    log_data.update({
                        'val/loss_epoch': current_val_loss,
                        'val/num_batches': val_metrics['val_batches']
                    })
                
                self.wandb_run.log(log_data)
            
            # Save checkpoint
            self.save_checkpoint(
                epoch=epoch,
                is_best=is_best,
                additional_info={
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
            )
            
            # Early stopping check (optional)
            if self._should_early_stop(epoch, current_val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Training completed
        total_training_time = time.time() - self.training_start_time
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED SUCCESSFULLY")
        self.logger.info(f"Total training time: {total_training_time/3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info("=" * 60)
        
        # Save final checkpoint
        self.save_checkpoint(
            epoch=self.current_epoch,
            is_best=False,
            additional_info={
                'final_model': True,
                'total_training_time': total_training_time
            }
        )
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_training_time': total_training_time,
            'epochs_completed': self.current_epoch,
            'global_steps': self.global_step
        }
    
    def _should_early_stop(self, epoch: int, val_loss: float) -> bool:
        """
        Check if training should be stopped early.
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        # Implement early stopping logic if needed
        # For now, we don't implement early stopping
        return False
    
    def run_full_training(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        This is the main entry point that executes:
        1. Configuration loading
        2. W&B setup
        3. Data loading
        4. Model initialization
        5. Training
        6. Cleanup
        
        Returns:
            Dictionary containing training results
        """
        try:
            # Load configuration
            self.load_configuration()
            
            # Setup W&B
            self.setup_wandb()
            
            # Load data
            self.load_data()
            
            # Initialize models
            self.initialize_models()
            
            # Train
            results = self.train()
            
            # Cleanup
            if self.wandb_run:
                self.wandb_run.finish()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            if self.wandb_run:
                self.wandb_run.finish()
            raise


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="V-JEPA Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "config",
        nargs="?",
        default="new_config.yaml",
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = VJEPATrainingPipeline(args.config)
    
    # Disable W&B if requested
    if args.no_wandb:
        if pipeline.config is None:
            pipeline.load_configuration()
        pipeline.config['wandb'] = {'enabled': False}
    
    # Resume from checkpoint if provided
    if args.resume:
        if not pipeline.load_checkpoint(args.resume):
            print(f"Failed to load checkpoint: {args.resume}")
            return
    
    # Run training
    try:
        results = pipeline.run_full_training()
        
        print("\n" + "=" * 50)
        print("TRAINING RESULTS")
        print("=" * 50)
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Training time: {results['total_training_time']/3600:.2f} hours")
        print(f"Epochs completed: {results['epochs_completed']}")
        print(f"Global steps: {results['global_steps']}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()