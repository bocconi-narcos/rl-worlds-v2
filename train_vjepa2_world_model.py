#!/usr/bin/env python3
"""
V-JEPA2 World Model Training Script

This script implements the two-stage V-JEPA2-inspired world model training:
1. Stage 1: Self-Supervised Encoder Pretraining with masked prediction
2. Stage 2: Action-Conditioned World Model Training

Usage:
    python train_vjepa2_world_model.py
"""

import os
import sys
import yaml
import torch
import wandb
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_handling import prepare_dataloaders
from src.training_loops.vjepa2_world_model_loop import VJEPA2WorldModelTrainer
from src.utils.config_utils import load_config


def main():
    """Main training function for V-JEPA2 World Model."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train V-JEPA2 World Model")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--wandb", action="store_true", default=True,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                       help="Train only specific stage (1 or 2)")
    parser.add_argument("--stage1-checkpoint", type=str, default="trained_models/best_stage1_model.pth",
                       help="Path to Stage 1 checkpoint for Stage 2 training")
    parser.add_argument("--stage2-checkpoint", type=str, default=None,
                       help="Path to Stage 2 checkpoint for resuming Stage 2 training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # Setup Weights & Biases
    wandb_cfg = config.get('wandb', {})
    wandb_run = None
    if args.wandb and wandb_cfg.get('enabled', False):
        try:
            wandb_run = wandb.init(
                project=wandb_cfg.get('project'),
                entity=wandb_cfg.get('entity'),
                name=f"{wandb_cfg.get('run_name_prefix', 'exp')}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=config
            )
            print("Weights & Biases logging enabled")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}. Proceeding without W&B.")
            wandb_run = None
    else:
        wandb.init(mode="disabled")
        print("Weights & Biases logging disabled")
    
    # Create dataloaders
    print("Creating dataloaders...")
    data_config = config.get('data', {})
    validation_split = data_config.get('validation_split', 0.2)
    train_dataloader, val_dataloader = prepare_dataloaders(config, validation_split)
    
    if train_dataloader is None:
        print("Exiting due to no training data.")
        return
        
    print(f"Training samples: {len(train_dataloader.dataset)}")
    if val_dataloader:
        print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Initialize trainer
    trainer = VJEPA2WorldModelTrainer(
        config=config,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    # Train based on stage argument
    if args.stage == 1:
        print("Training Stage 1 only...")
        trainer.setup_stage1()
        trainer.train_stage1()
    elif args.stage == 2:
        print("Training Stage 2 only...")
        # Load Stage 1 model if it exists
        stage1_checkpoint = args.stage1_checkpoint
        if os.path.exists(stage1_checkpoint):
            print(f"Loading Stage 1 checkpoint: {stage1_checkpoint}")
            try:
                checkpoint = torch.load(stage1_checkpoint, map_location=device)
                
                # Setup Stage 1 model first
                trainer.setup_stage1()
                
                # Load the trained weights
                if 'model_state_dict' in checkpoint:
                    trainer.stage1_model.load_state_dict(checkpoint['model_state_dict'])
                    print("Successfully loaded Stage 1 model weights")
                    
                    # Optionally load optimizer and scheduler states (for resuming Stage 1 training)
                    if 'optimizer_state_dict' in checkpoint:
                        print("Note: Optimizer state found in checkpoint (not used for Stage 2)")
                    if 'scheduler_state_dict' in checkpoint:
                        print("Note: Scheduler state found in checkpoint (not used for Stage 2)")
                    if 'epoch' in checkpoint:
                        print(f"Note: Checkpoint was saved at epoch {checkpoint['epoch']}")
                    if 'best_val_loss' in checkpoint:
                        print(f"Note: Best validation loss in checkpoint: {checkpoint['best_val_loss']:.4f}")
                else:
                    print("Warning: No model_state_dict found in checkpoint. Using random weights.")
                
                # Setup Stage 2 with the loaded encoder
                trainer.setup_stage2()
                
                # Load Stage 2 checkpoint if provided
                if args.stage2_checkpoint and os.path.exists(args.stage2_checkpoint):
                    print(f"Loading Stage 2 checkpoint: {args.stage2_checkpoint}")
                    try:
                        stage2_checkpoint = torch.load(args.stage2_checkpoint, map_location=device)
                        if 'model_state_dict' in stage2_checkpoint:
                            trainer.stage2_model.load_state_dict(stage2_checkpoint['model_state_dict'])
                            print("Successfully loaded Stage 2 model weights")
                        if 'optimizer_state_dict' in stage2_checkpoint:
                            trainer.stage2_optimizer.load_state_dict(stage2_checkpoint['optimizer_state_dict'])
                            print("Successfully loaded Stage 2 optimizer state")
                        if 'scheduler_state_dict' in stage2_checkpoint:
                            trainer.stage2_scheduler.load_state_dict(stage2_checkpoint['scheduler_state_dict'])
                            print("Successfully loaded Stage 2 scheduler state")
                    except Exception as e:
                        print(f"Error loading Stage 2 checkpoint: {e}")
                        print("Proceeding with random initialization for Stage 2 model")
                
                # Train Stage 2
                trainer.train_stage2()
                
            except Exception as e:
                print(f"Error loading Stage 1 checkpoint: {e}")
                print("Please ensure Stage 1 was trained successfully first.")
                return
        else:
            print(f"Stage 1 checkpoint not found at: {stage1_checkpoint}")
            print("Please train Stage 1 first using: python train_vjepa2_world_model.py --stage 1")
            return
    else:
        # Train both stages
        print("Training both stages...")
        trainer.train()
    
    # Get trained models
    trained_models = trainer.get_trained_models()
    
    print("\nTraining completed successfully!")
    print("Trained models:")
    print(f"- Encoder: {type(trained_models['encoder']).__name__}")
    print(f"- World Model: {type(trained_models['world_model']).__name__}")
    
    # Save final models
    final_checkpoint = {
        'encoder_state_dict': trained_models['encoder'].state_dict(),
        'world_model_state_dict': trained_models['world_model'].state_dict(),
        'config': config,
        'stage': 'complete'
    }
    
    os.makedirs('trained_models', exist_ok=True)
    torch.save(final_checkpoint, 'trained_models/vjepa2_world_model_final.pth')
    print("Final models saved to: trained_models/vjepa2_world_model_final.pth")
    
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main() 