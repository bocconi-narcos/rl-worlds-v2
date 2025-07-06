#!/usr/bin/env python3
"""
Script to inspect the dataset structure and understand the data format.
"""

import torch
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config_utils import load_config
from src.data_handling import prepare_dataloaders

def inspect_dataset():
    """Inspect the dataset structure and sample data."""
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Create dataloaders
    print("Creating dataloaders...")
    validation_split = config.get('data', {}).get('validation_split', 0.2)
    train_dataloader, val_dataloader = prepare_dataloaders(config, validation_split)
    
    if train_dataloader is None:
        print("No training data available.")
        return
    
    print(f"\n=== DATASET INSPECTION ===")
    print(f"Training samples: {len(train_dataloader.dataset)}")
    if val_dataloader:
        print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Get a batch from training data
    print(f"\n=== SAMPLE BATCH STRUCTURE ===")
    batch = next(iter(train_dataloader))
    s_t, a_t, r_t, s_t_plus_1 = batch
    
    print(f"Batch shapes:")
    print(f"  s_t (current state): {s_t.shape}")
    print(f"  a_t (action): {a_t.shape}")
    print(f"  r_t (reward): {r_t.shape}")
    print(f"  s_t_plus_1 (next state): {s_t_plus_1.shape}")
    
    print(f"\nData types:")
    print(f"  s_t dtype: {s_t.dtype}")
    print(f"  a_t dtype: {a_t.dtype}")
    print(f"  r_t dtype: {r_t.dtype}")
    print(f"  s_t_plus_1 dtype: {s_t_plus_1.dtype}")
    
    print(f"\nValue ranges:")
    print(f"  s_t min/max: {s_t.min():.4f}/{s_t.max():.4f}")
    print(f"  a_t min/max: {a_t.min():.4f}/{a_t.max():.4f}")
    print(f"  r_t min/max: {r_t.min():.4f}/{r_t.max():.4f}")
    print(f"  s_t_plus_1 min/max: {s_t_plus_1.min():.4f}/{s_t_plus_1.max():.4f}")
    
    # Check if data is 5D (batch, frames, channels, height, width)
    if s_t.dim() == 5:
        B, F, C, H, W = s_t.shape
        print(f"\n=== FRAME STACKING ANALYSIS ===")
        print(f"Batch size: {B}")
        print(f"Number of frames: {F}")
        print(f"Channels per frame: {C}")
        print(f"Image height: {H}")
        print(f"Image width: {W}")
        
        # Check if frames are identical (which would be problematic)
        frame_diff = torch.abs(s_t[:, 1:] - s_t[:, :-1]).mean()
        print(f"Average difference between consecutive frames: {frame_diff:.6f}")
        
        if frame_diff < 1e-6:
            print("⚠️  WARNING: Frames appear to be identical! This is problematic for training.")
        else:
            print("✅ Frames have variation - good for training.")
    
    # Check validation data if available
    if val_dataloader:
        print(f"\n=== VALIDATION DATA COMPARISON ===")
        val_batch = next(iter(val_dataloader))
        val_s_t, val_a_t, val_r_t, val_s_t_plus_1 = val_batch
        
        print(f"Validation batch shapes:")
        print(f"  val_s_t: {val_s_t.shape}")
        print(f"  val_a_t: {val_a_t.shape}")
        print(f"  val_r_t: {val_r_t.shape}")
        print(f"  val_s_t_plus_1: {val_s_t_plus_1.shape}")
        
        # Compare statistics between train and val
        train_mean = s_t.mean()
        val_mean = val_s_t.mean()
        train_std = s_t.std()
        val_std = val_s_t.std()
        
        print(f"\nTrain vs Validation statistics:")
        print(f"  Train mean: {train_mean:.4f}, Val mean: {val_mean:.4f}")
        print(f"  Train std: {train_std:.4f}, Val std: {val_std:.4f}")
        
        # Check if validation data is significantly different
        mean_diff = abs(train_mean - val_mean) / train_mean
        std_diff = abs(train_std - val_std) / train_std
        
        if mean_diff > 0.5 or std_diff > 0.5:
            print("⚠️  WARNING: Train and validation data have significantly different statistics!")
        else:
            print("✅ Train and validation data have similar statistics.")
    
    print(f"\n=== CONFIGURATION SUMMARY ===")
    env_config = config.get('environment', {})
    print(f"Environment: {env_config.get('name', 'N/A')}")
    print(f"Frame stack size: {env_config.get('frame_stack_size', 'N/A')}")
    print(f"Input channels per frame: {env_config.get('input_channels_per_frame', 'N/A')}")
    print(f"Grayscale conversion: {env_config.get('grayscale_conversion', 'N/A')}")
    print(f"Image size: {env_config.get('image_height', 'N/A')}x{env_config.get('image_width', 'N/A')}")

if __name__ == "__main__":
    inspect_dataset() 