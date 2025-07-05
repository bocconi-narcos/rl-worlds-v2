#!/usr/bin/env python3
"""
Test script to verify comparison configuration loading
"""

import yaml
from src.utils.config_utils import load_config

def test_config_loading():
    """Test loading of comparison configuration files."""
    
    config_files = [
        'config_comparison.yaml',
        'config_comparison_simple.yaml'
    ]
    
    for config_file in config_files:
        print(f"\nTesting {config_file}...")
        try:
            config = load_config(config_file)
            
            # Check required fields
            required_fields = ['base_config_path', 'experiments']
            for field in required_fields:
                if field not in config:
                    print(f"  ❌ Missing required field: {field}")
                    continue
                print(f"  ✅ Found field: {field}")
            
            # Check experiments
            experiments = config.get('experiments', [])
            print(f"  📊 Found {len(experiments)} experiments:")
            
            for i, exp in enumerate(experiments):
                name = exp.get('name', f'experiment_{i}')
                aux_type = exp.get('aux_loss_type', 'unknown')
                print(f"    {i+1}. {name} ({aux_type})")
            
            # Check base overrides
            if 'base_overrides' in config:
                print(f"  ⚙️  Base overrides configured")
            
            # Check comparison settings
            if 'comparison' in config:
                print(f"  📈 Comparison settings configured")
            
            print(f"  ✅ {config_file} loaded successfully")
            
        except Exception as e:
            print(f"  ❌ Error loading {config_file}: {e}")

def test_base_config_loading():
    """Test loading of base configuration."""
    print(f"\nTesting base config loading...")
    try:
        base_config = load_config('config.yaml')
        print(f"  ✅ Base config loaded successfully")
        
        # Check if auxiliary loss section exists
        aux_loss_config = base_config.get('models', {}).get('auxiliary_loss', {})
        if aux_loss_config:
            print(f"  ✅ Auxiliary loss configuration found")
            print(f"     Type: {aux_loss_config.get('type', 'not specified')}")
        else:
            print(f"  ⚠️  No auxiliary loss configuration found in base config")
            
    except Exception as e:
        print(f"  ❌ Error loading base config: {e}")

if __name__ == '__main__':
    print("Testing Comparison Configuration Loading")
    print("=" * 50)
    
    test_base_config_loading()
    test_config_loading()
    
    print(f"\n{'='*50}")
    print("Configuration test completed!")
    print("\nTo run the comparison:")
    print("  python main_comparison.py config_comparison_simple.yaml")
    print("  python main_comparison.py config_comparison.yaml") 