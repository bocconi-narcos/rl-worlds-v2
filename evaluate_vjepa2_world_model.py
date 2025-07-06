#!/usr/bin/env python3
"""
V-JEPA2 World Model Evaluation Script

This script evaluates the trained V-JEPA2 world model and demonstrates its capabilities.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.masked_prediction_model import MaskedPredictionModel
from src.models.world_model_transformer import WorldModelTransformer
from src.data_handling import prepare_dataloaders
from src.utils.config_utils import load_config
from src.utils.env_utils import get_env_details


def evaluate_stage1_model(model, dataloader, device, num_samples=5):
    """Evaluate Stage 1: Masked Prediction Model."""
    print("Evaluating Stage 1: Masked Prediction Model")
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            s_t = s_t.to(device)
            
            # Handle frame stacking: reshape from (B, F, C, H, W) to (B, F*C, H, W)
            if s_t.dim() == 5:
                B, F, C, H, W = s_t.shape
                s_t = s_t.view(B, F * C, H, W)
            
            # Test different masking ratios
            for mask_ratio in [0.25, 0.5, 0.75]:
                predicted_embeddings, target_embeddings, mask = model(s_t, mask_ratio=mask_ratio)
                
                # Compute L1 loss
                loss = torch.nn.L1Loss()(predicted_embeddings, target_embeddings)
                total_loss += loss.item()
                num_batches += 1
                
                print(f"Batch {batch_idx}, Mask Ratio {mask_ratio}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"Average Stage 1 Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_stage2_model(world_model, encoder, dataloader, device, num_samples=5):
    """Evaluate Stage 2: World Model Transformer."""
    print("Evaluating Stage 2: World Model Transformer")
    
    world_model.eval()
    encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            s_t = s_t.to(device)
            a_t = a_t.to(device)
            s_t_plus_1 = s_t_plus_1.to(device)
            
            # Handle frame stacking: reshape from (B, F, C, H, W) to (B, F*C, H, W)
            if s_t.dim() == 5:
                B, F, C, H, W = s_t.shape
                s_t = s_t.view(B, F * C, H, W)
            if s_t_plus_1.dim() == 5:
                B, F, C, H, W = s_t_plus_1.shape
                s_t_plus_1 = s_t_plus_1.view(B, F * C, H, W)
            
            # Encode observations
            latents = encoder(s_t)
            target_latents = encoder(s_t_plus_1)
            
            # Prepare actions
            if a_t.dtype == torch.long:
                actions = a_t
            else:
                actions = a_t.float()
            
            # Forward pass
            predicted_latents = world_model(latents, actions)
            
            # Compute loss
            loss = torch.nn.L1Loss()(predicted_latents, target_latents)
            total_loss += loss.item()
            num_batches += 1
            
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"Average Stage 2 Loss: {avg_loss:.4f}")
    return avg_loss


def demonstrate_rollout_prediction(world_model, encoder, dataloader, device, num_steps=8):
    """Demonstrate multi-step rollout prediction."""
    print(f"Demonstrating {num_steps}-step rollout prediction")
    
    world_model.eval()
    encoder.eval()
    
    with torch.no_grad():
        # Get a single batch
        s_t, a_t, r_t, s_t_plus_1 = next(iter(dataloader))
        s_t = s_t.to(device)
        a_t = a_t.to(device)
        
        # Handle frame stacking: reshape from (B, F, C, H, W) to (B, F*C, H, W)
        if s_t.dim() == 5:
            B, F, C, H, W = s_t.shape
            s_t = s_t.view(B, F * C, H, W)
        
        # Encode initial observation
        initial_latent = encoder(s_t[:, 0:1])  # Take first frame
        
        # Prepare action sequence
        if a_t.dtype == torch.long:
            actions = a_t[:, :num_steps]
        else:
            actions = a_t[:, :num_steps].float()
        
        # Perform rollout prediction
        predicted_latents = world_model.rollout_prediction(
            initial_latent.squeeze(1), actions, num_steps
        )
        
        print(f"Rollout prediction shape: {predicted_latents.shape}")
        print(f"Predicted latent norms: {torch.norm(predicted_latents, dim=-1)}")
        
        return predicted_latents


def visualize_predictions(world_model, encoder, dataloader, device, save_path="evaluation_plots"):
    """Visualize world model predictions."""
    print("Creating visualization plots")
    
    os.makedirs(save_path, exist_ok=True)
    
    world_model.eval()
    encoder.eval()
    
    with torch.no_grad():
        # Get a single batch
        s_t, a_t, r_t, s_t_plus_1 = next(iter(dataloader))
        s_t = s_t.to(device)
        a_t = a_t.to(device)
        s_t_plus_1 = s_t_plus_1.to(device)
        
        # Handle frame stacking: reshape from (B, F, C, H, W) to (B, F*C, H, W)
        if s_t.dim() == 5:
            B, F, C, H, W = s_t.shape
            s_t = s_t.view(B, F * C, H, W)
        if s_t_plus_1.dim() == 5:
            B, F, C, H, W = s_t_plus_1.shape
            s_t_plus_1 = s_t_plus_1.view(B, F * C, H, W)
        
        # Encode observations
        latents = encoder(s_t)
        target_latents = encoder(s_t_plus_1)
        
        # Prepare actions
        if a_t.dtype == torch.long:
            actions = a_t
        else:
            actions = a_t.float()
        
        # Get predictions
        predicted_latents = world_model(latents, actions)
        
        # Plot latent space trajectories
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot first two dimensions of latents
        batch_idx = 0
        axes[0, 0].plot(latents[batch_idx, :, 0].cpu(), latents[batch_idx, :, 1].cpu(), 'b-', label='Actual')
        axes[0, 0].plot(predicted_latents[batch_idx, :, 0].cpu(), predicted_latents[batch_idx, :, 1].cpu(), 'r--', label='Predicted')
        axes[0, 0].set_title('Latent Space Trajectory (Dim 0 vs Dim 1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot latent norms over time
        actual_norms = torch.norm(latents[batch_idx], dim=-1).cpu()
        predicted_norms = torch.norm(predicted_latents[batch_idx], dim=-1).cpu()
        time_steps = range(len(actual_norms))
        
        axes[0, 1].plot(time_steps, actual_norms, 'b-', label='Actual')
        axes[0, 1].plot(time_steps, predicted_norms, 'r--', label='Predicted')
        axes[0, 1].set_title('Latent Norms Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot prediction errors
        errors = torch.norm(predicted_latents - target_latents, dim=-1).cpu()
        axes[1, 0].plot(time_steps, errors[batch_idx], 'g-')
        axes[1, 0].set_title('Prediction Errors Over Time')
        axes[1, 0].set_ylabel('L2 Error')
        axes[1, 0].grid(True)
        
        # Plot action embeddings (if available)
        if hasattr(world_model, 'action_embedding'):
            if world_model.action_type == 'discrete':
                action_embeddings = world_model.action_embedding(actions[batch_idx])
            else:
                action_embeddings = world_model.action_embedding(actions[batch_idx])
            
            axes[1, 1].plot(time_steps, action_embeddings[batch_idx, :, 0].cpu(), 'purple')
            axes[1, 1].set_title('Action Embeddings (First Dimension)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'world_model_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {os.path.join(save_path, 'world_model_predictions.png')}")


def main():
    """Main evaluation function."""
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # Create dataloader
    data_config = config.get('data', {})
    validation_split = data_config.get('validation_split', 0.2)
    train_dataloader, val_dataloader = prepare_dataloaders(config, validation_split)
    eval_dataloader = val_dataloader if val_dataloader else train_dataloader
    
    # Load trained models
    checkpoint_path = "trained_models/vjepa2_world_model_final.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train_vjepa2_world_model.py")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get environment details for action dimensions
    env_name = config.get('environment', {}).get('name', 'ALE/Pong-v5')
    try:
        action_dim, action_type, _ = get_env_details(env_name)
    except Exception as e:
        print(f"Warning: Could not get environment details: {e}")
        action_dim, action_type = 6, 'continuous'  # Defaults
    
    # Initialize models
    stage1_model = MaskedPredictionModel(
        image_size=config['environment']['image_height'],
        patch_size=config['models']['shared_patch_size'],
        input_channels=config['environment']['input_channels_per_frame'],
        latent_dim=config['models']['shared_latent_dim'],
        predictor_hidden_dims=[256, 256],
        ema_decay=0.996
    ).to(device)
    
    world_model = WorldModelTransformer(
        latent_dim=config['models']['shared_latent_dim'],
        action_dim=action_dim,
        action_emb_dim=32,
        action_type=action_type,
        sequence_length=8
    ).to(device)
    
    # Load state dicts
    stage1_model.load_state_dict(checkpoint['encoder_state_dict'])
    world_model.load_state_dict(checkpoint['world_model_state_dict'])
    
    print("Models loaded successfully!")
    
    # Run evaluations
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Evaluate Stage 1
    stage1_loss = evaluate_stage1_model(stage1_model, eval_dataloader, device)
    
    # Evaluate Stage 2
    encoder = stage1_model.get_encoder()
    stage2_loss = evaluate_stage2_model(world_model, encoder, eval_dataloader, device)
    
    # Demonstrate rollout prediction
    predicted_latents = demonstrate_rollout_prediction(world_model, encoder, eval_dataloader, device)
    
    # Create visualizations
    visualize_predictions(world_model, encoder, eval_dataloader, device)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Stage 1 (Masked Prediction) Average Loss: {stage1_loss:.4f}")
    print(f"Stage 2 (World Model) Average Loss: {stage2_loss:.4f}")
    print(f"Rollout prediction completed successfully for {predicted_latents.shape[1]} steps")
    print("Visualization plots saved to evaluation_plots/")
    print("="*50)


if __name__ == "__main__":
    main() 