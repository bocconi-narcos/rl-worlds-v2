#!/usr/bin/env python3
"""
Diagnostic script to analyze JEPA training issues.
This script will help identify why training is failing.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.models.jepa import JEPA
from src.models.vit import ViT
from src.losses import VICRegLoss, BarlowTwinsLoss
from src.utils.config_utils import load_config
from src.utils.data_utils import ExperienceDataset
import os

def analyze_gradients(model, dataloader, device, num_batches=5):
    """Analyze gradient flow and magnitude."""
    print("\n=== GRADIENT ANALYSIS ===")
    
    model.train()
    gradients_info = {
        'grad_norms': [],
        'param_norms': [],
        'grad_ratios': []
    }
    
    for i, (s_t, a_t, r_t, s_t_plus_1) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
        a_t = a_t.to(device)
        
        # Forward pass
        pred_emb, target_emb, online_emb, target_emb_2 = model(s_t, a_t, s_t_plus_1)
        
        # Simple MSE loss for analysis
        loss = nn.MSELoss()(pred_emb, target_emb)
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        total_norm = 0
        param_norm = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm += param.norm().item() ** 2
                grad_norm = param.grad.norm().item() ** 2
                total_norm += grad_norm
                
                if i == 0:  # Print first batch details
                    print(f"  {name}: grad_norm={grad_norm:.6f}, param_norm={param.norm().item():.6f}")
        
        total_norm = total_norm ** 0.5
        param_norm = param_norm ** 0.5
        
        gradients_info['grad_norms'].append(total_norm)
        gradients_info['param_norms'].append(param_norm)
        gradients_info['grad_ratios'].append(total_norm / (param_norm + 1e-8))
        
        # Zero gradients
        model.zero_grad()
    
    print(f"  Average grad norm: {np.mean(gradients_info['grad_norms']):.6f}")
    print(f"  Average param norm: {np.mean(gradients_info['param_norms']):.6f}")
    print(f"  Average grad/param ratio: {np.mean(gradients_info['grad_ratios']):.6f}")
    
    return gradients_info

def analyze_loss_components(model, aux_loss_fn, dataloader, device, num_batches=5):
    """Analyze individual loss components."""
    print("\n=== LOSS COMPONENT ANALYSIS ===")
    
    model.train()
    loss_components = {
        'prediction_loss': [],
        'aux_loss': [],
        'total_loss': []
    }
    
    for i, (s_t, a_t, r_t, s_t_plus_1) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
        a_t = a_t.to(device)
        
        # Forward pass
        pred_emb, target_emb, online_emb, target_emb_2 = model(s_t, a_t, s_t_plus_1)
        
        # Prediction loss
        pred_loss = nn.MSELoss()(pred_emb, target_emb)
        
        # Auxiliary loss
        if aux_loss_fn is not None:
            if isinstance(aux_loss_fn, VICRegLoss):
                aux_loss = aux_loss_fn.calculate_reg_terms(online_emb, target_emb_2)
            elif isinstance(aux_loss_fn, BarlowTwinsLoss):
                aux_loss = aux_loss_fn.calculate_reg_terms(online_emb, target_emb_2)
            else:
                aux_loss = aux_loss_fn(online_emb, target_emb_2)
        else:
            aux_loss = torch.tensor(0.0, device=device)
        
        total_loss = pred_loss + aux_loss
        
        loss_components['prediction_loss'].append(pred_loss.item())
        loss_components['aux_loss'].append(aux_loss.item())
        loss_components['total_loss'].append(total_loss.item())
        
        if i < 3:  # Print first few batches
            print(f"  Batch {i+1}: pred={pred_loss.item():.6f}, aux={aux_loss.item():.6f}, total={total_loss.item():.6f}")
    
    print(f"  Average prediction loss: {np.mean(loss_components['prediction_loss']):.6f}")
    print(f"  Average auxiliary loss: {np.mean(loss_components['aux_loss']):.6f}")
    print(f"  Average total loss: {np.mean(loss_components['total_loss']):.6f}")
    print(f"  Auxiliary loss ratio: {np.mean(loss_components['aux_loss']) / (np.mean(loss_components['prediction_loss']) + 1e-8):.6f}")
    
    return loss_components

def analyze_representations(model, dataloader, device, num_batches=3):
    """Analyze learned representations."""
    print("\n=== REPRESENTATION ANALYSIS ===")
    
    model.eval()
    representations = {
        'online_emb': [],
        'target_emb': [],
        'pred_emb': []
    }
    
    with torch.no_grad():
        for i, (s_t, a_t, r_t, s_t_plus_1) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
            a_t = a_t.to(device)
            
            # Forward pass
            pred_emb, target_emb, online_emb, target_emb_2 = model(s_t, a_t, s_t_plus_1)
            
            representations['online_emb'].append(online_emb.cpu().numpy())
            representations['target_emb'].append(target_emb_2.cpu().numpy())
            representations['pred_emb'].append(pred_emb.cpu().numpy())
    
    # Analyze statistics
    for name, reps in representations.items():
        reps_array = np.concatenate(reps, axis=0)
        print(f"  {name}:")
        print(f"    Shape: {reps_array.shape}")
        print(f"    Mean: {np.mean(reps_array):.6f}")
        print(f"    Std: {np.std(reps_array):.6f}")
        print(f"    Min: {np.min(reps_array):.6f}")
        print(f"    Max: {np.max(reps_array):.6f}")
        print(f"    Norm: {np.linalg.norm(reps_array, axis=1).mean():.6f}")
    
    return representations

def analyze_numerical_stability(model, aux_loss_fn, dataloader, device, num_batches=3):
    """Check for numerical stability issues."""
    print("\n=== NUMERICAL STABILITY ANALYSIS ===")
    
    model.train()
    stability_issues = []
    
    for i, (s_t, a_t, r_t, s_t_plus_1) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
        a_t = a_t.to(device)
        
        # Check input statistics
        if torch.isnan(s_t).any() or torch.isinf(s_t).any():
            stability_issues.append(f"Batch {i+1}: NaN/Inf in input")
        
        # Forward pass
        pred_emb, target_emb, online_emb, target_emb_2 = model(s_t, a_t, s_t_plus_1)
        
        # Check embedding statistics
        for name, emb in [('pred_emb', pred_emb), ('target_emb', target_emb), 
                         ('online_emb', online_emb), ('target_emb_2', target_emb_2)]:
            if torch.isnan(emb).any():
                stability_issues.append(f"Batch {i+1}: NaN in {name}")
            if torch.isinf(emb).any():
                stability_issues.append(f"Batch {i+1}: Inf in {name}")
        
        # Check auxiliary loss
        if aux_loss_fn is not None:
            try:
                if isinstance(aux_loss_fn, VICRegLoss):
                    aux_loss = aux_loss_fn.calculate_reg_terms(online_emb, target_emb_2)
                elif isinstance(aux_loss_fn, BarlowTwinsLoss):
                    aux_loss = aux_loss_fn.calculate_reg_terms(online_emb, target_emb_2)
                else:
                    aux_loss = aux_loss_fn(online_emb, target_emb_2)
                
                if torch.isnan(aux_loss).any() or torch.isinf(aux_loss).any():
                    stability_issues.append(f"Batch {i+1}: NaN/Inf in auxiliary loss")
            except Exception as e:
                stability_issues.append(f"Batch {i+1}: Error in auxiliary loss: {e}")
    
    if stability_issues:
        print("  ⚠️  Numerical stability issues found:")
        for issue in stability_issues:
            print(f"    {issue}")
    else:
        print("  ✅ No numerical stability issues detected")
    
    return stability_issues

def analyze_training_dynamics(model, aux_loss_fn, dataloader, device, optimizer, num_steps=10):
    """Analyze training dynamics over multiple steps."""
    print("\n=== TRAINING DYNAMICS ANALYSIS ===")
    
    model.train()
    dynamics = {
        'losses': [],
        'grad_norms': [],
        'learning_rates': []
    }
    
    for step in range(num_steps):
        # Get batch
        batch = next(iter(dataloader))
        s_t, a_t, r_t, s_t_plus_1 = batch
        s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
        a_t = a_t.to(device)
        
        # Forward pass
        pred_emb, target_emb, online_emb, target_emb_2 = model(s_t, a_t, s_t_plus_1)
        
        # Loss computation
        pred_loss = nn.MSELoss()(pred_emb, target_emb)
        
        if aux_loss_fn is not None:
            if isinstance(aux_loss_fn, VICRegLoss):
                aux_loss = aux_loss_fn.calculate_reg_terms(online_emb, target_emb_2)
            elif isinstance(aux_loss_fn, BarlowTwinsLoss):
                aux_loss = aux_loss_fn.calculate_reg_terms(online_emb, target_emb_2)
            else:
                aux_loss = aux_loss_fn(online_emb, target_emb_2)
        else:
            aux_loss = torch.tensor(0.0, device=device)
        
        total_loss = pred_loss + aux_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Compute gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Optimizer step
        optimizer.step()
        
        # Record dynamics
        dynamics['losses'].append(total_loss.item())
        dynamics['grad_norms'].append(total_norm)
        dynamics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        if step < 5:  # Print first few steps
            print(f"  Step {step+1}: loss={total_loss.item():.6f}, grad_norm={total_norm:.6f}")
    
    # Analyze trends
    losses = dynamics['losses']
    if len(losses) > 1:
        loss_trend = "decreasing" if losses[-1] < losses[0] else "increasing" if losses[-1] > losses[0] else "stable"
        print(f"  Loss trend: {loss_trend}")
        print(f"  Loss change: {losses[-1] - losses[0]:.6f}")
    
    return dynamics

def main():
    """Run comprehensive training diagnostics."""
    print("JEPA Training Diagnostics")
    print("=" * 50)
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset from file
    print("\nLoading dataset from file...")
    dataset_path = "datasets/montezuma_rep_w.pth"
    data = torch.load(dataset_path, map_location=device, weights_only=False)
    train_dataset = data['train_dataset']
    # If it's not already a torch Dataset, wrap it
    if not isinstance(train_dataset, torch.utils.data.Dataset):
        train_dataset = ExperienceDataset(**train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    print("\nInitializing model...")
    action_dim = 4  # Breakout has 4 actions
    action_type = "discrete"
    image_h_w = (64, 64)
    input_channels = 1  # Grayscale
    
    # Create JEPA model
    encoder_type = 'vit'
    encoder_params = config['models']['encoder']['params']['vit']
    jepa_model = JEPA(
        image_size=64,
        patch_size=4,
        input_channels=input_channels * 4,  # 4 stacked frames
        action_dim=action_dim,
        action_emb_dim=32,  # Use config if available
        action_type=action_type,
        latent_dim=config['models']['shared_latent_dim'],
        predictor_hidden_dims=config['models']['jepa']['predictor_hidden_dims'],
        ema_decay=config['models']['jepa']['ema_decay'],
        encoder_type=encoder_type,
        encoder_params=encoder_params,
        target_encoder_mode=config['models']['jepa']['target_encoder_mode'],
        predictor_dropout_rate=config['models']['jepa']['predictor_dropout_rate']
    ).to(device)
    
    # Initialize auxiliary loss
    aux_loss_config = config['models']['auxiliary_loss']
    aux_loss_type = aux_loss_config['type']
    
    if aux_loss_type == 'vicreg':
        aux_loss_fn = VICRegLoss(
            sim_coeff=aux_loss_config['params']['vicreg']['sim_coeff'],
            std_coeff=aux_loss_config['params']['vicreg']['std_coeff'],
            cov_coeff=aux_loss_config['params']['vicreg']['cov_coeff'],
            eps=aux_loss_config['params']['vicreg']['eps']
        ).to(device)
    elif aux_loss_type == 'barlow_twins':
        aux_loss_fn = BarlowTwinsLoss(
            lambda_param=aux_loss_config['params']['barlow_twins']['lambda_param'],
            eps=aux_loss_config['params']['barlow_twins']['eps'],
            scale_loss=aux_loss_config['params']['barlow_twins']['scale_loss']
        ).to(device)
    else:
        aux_loss_fn = None
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        jepa_model.parameters(),
        lr=config['models']['jepa']['learning_rate']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in jepa_model.parameters()):,}")
    print(f"Auxiliary loss: {aux_loss_type}")
    print(f"Learning rate: {config['models']['jepa']['learning_rate']}")
    
    # Run diagnostics
    analyze_gradients(jepa_model, train_dataloader, device)
    analyze_loss_components(jepa_model, aux_loss_fn, train_dataloader, device)
    analyze_representations(jepa_model, train_dataloader, device)
    analyze_numerical_stability(jepa_model, aux_loss_fn, train_dataloader, device)
    analyze_training_dynamics(jepa_model, aux_loss_fn, train_dataloader, device, optimizer)
    
    print("\n" + "=" * 50)
    print("Diagnostics completed!")

if __name__ == "__main__":
    main() 