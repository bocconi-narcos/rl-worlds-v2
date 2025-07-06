#!/usr/bin/env python3
"""
Test script to verify dataset loading
"""

import torch
import os
from src.utils.config_utils import load_config

def test_dataset_loading():
    # Load config
    config = load_config()
    
    # Check dataset path
    dataset_dir = config['data']['dataset']['dir']
    load_path = config['data']['dataset']['load_path']
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Load path: {load_path}")
    
    if load_path:
        full_path = os.path.join(dataset_dir, load_path)
        print(f"Full dataset path: {full_path}")
        
        if os.path.exists(full_path):
            print(f"✅ Dataset file exists!")
            
            try:
                # Load the dataset
                print("Loading dataset...")
                data = torch.load(full_path, weights_only=False)
                
                # Check structure
                print(f"Dataset keys: {list(data.keys())}")
                
                if 'train_dataset' in data:
                    train_dataset = data['train_dataset']
                    print(f"✅ Train dataset loaded successfully!")
                    print(f"   Length: {len(train_dataset)}")
                    if len(train_dataset) > 0:
                        sample = train_dataset[0]
                        print(f"   Sample shape: {[t.shape if hasattr(t, 'shape') else type(t) for t in sample]}")
                
                if 'val_dataset' in data:
                    val_dataset = data['val_dataset']
                    print(f"✅ Validation dataset loaded successfully!")
                    print(f"   Length: {len(val_dataset)}")
                
                if 'metadata' in data:
                    metadata = data['metadata']
                    print(f"✅ Metadata loaded successfully!")
                    print(f"   Metadata: {metadata}")
                
                print("🎉 Dataset loading test passed!")
                return True
                
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
                return False
        else:
            print(f"❌ Dataset file does not exist at {full_path}")
            return False
    else:
        print("❌ No load_path specified in config")
        return False

if __name__ == "__main__":
    test_dataset_loading() 