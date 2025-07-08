"""
Test suite for V-JEPA2 models.

This module tests the core models in the V-JEPA2 architecture:
1. VisionTransformer - The encoder backbone
2. VisionTransformerPredictor - First stage predictor (masked prediction)
3. VisionTransformerPredictorAC - Second stage predictor (action-conditioned)

All test parameters are configurable via tests_config.yaml

Usage Examples:
    # Run with default configuration (auto-detects best device)
    python tests/test_models.py
    
    # Force MPS device (Apple Silicon)
    python tests/test_models.py --device mps
    
    # Use MPS-optimized configuration
    python tests/test_models.py --config tests/tests_config_mps_example.yaml
    
    # Force CUDA device with timing
    python tests/test_models.py --device cuda --timing
    
    # Run on CPU only
    python tests/test_models.py --device cpu
    
    # Override random seed
    python tests/test_models.py --seed 123

Configuration Files:
    - tests_config.yaml: Default configuration
    - tests_config_mps_example.yaml: MPS-optimized configuration for Apple Silicon
"""

import torch
import torch.nn as nn
import yaml
import time
import sys
import os
from typing import Dict, Any, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.vision_transformer import VisionTransformer, vit_base, vit_small, vit_tiny
from src.models.predictor_first_stage import VisionTransformerPredictor, vit_predictor
from src.models.predictor_second_stage import VisionTransformerPredictorAC, vit_ac_predictor


class TestConfig:
    """Configuration loader and manager for tests."""
    
    def __init__(self, config_path: str = "tests/tests_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = self.setup_device()
        self.setup_random_seed()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            print("Using default configuration...")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML file: {e}")
            print("Using default configuration...")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {
            'general': {
                'random_seed': 42,
                'device': 'auto',  # auto, cpu, cuda, mps
                'verbose': True,
                'include_timing': False
            },
            'test_suites': {
                'vision_transformer': True,
                'predictor_first_stage': True,
                'predictor_second_stage': True,
                'model_integration': True
            }
        }
    
    def setup_device(self) -> torch.device:
        """Setup computation device based on configuration."""
        device_config = self.config['general']['device']
        
        if device_config == 'auto':
            # Auto-detect best available device: mps > cuda > cpu
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        else:
            # Validate device availability
            if device_config == 'mps':
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    print("⚠️  MPS requested but not available, falling back to CPU")
                    device = torch.device('cpu')
                else:
                    device = torch.device('mps')
            elif device_config == 'cuda':
                if not torch.cuda.is_available():
                    print("⚠️  CUDA requested but not available, falling back to CPU")
                    device = torch.device('cpu')
                else:
                    device = torch.device('cuda')
            else:
                device = torch.device(device_config)
        
        if self.config['general']['verbose']:
            print(f"✓ Using device: {device}")
            # Print additional device info
            if device.type == 'mps':
                print("  🍎 Apple Metal Performance Shaders (MPS) acceleration enabled")
            elif device.type == 'cuda':
                print(f"  🚀 CUDA acceleration enabled (GPU: {torch.cuda.get_device_name()})")
            else:
                print("  💻 Running on CPU")
        
        return device
    
    def setup_random_seed(self):
        """Set random seed for reproducibility."""
        seed = self.config['general']['random_seed']
        torch.manual_seed(seed)
        
        # Set seed for CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # Note: MPS uses the general torch.manual_seed for reproducibility
        
        if self.config['general']['verbose']:
            print(f"✓ Set random seed to {seed}")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("  📱 MPS will use the general torch random seed")
            if torch.cuda.is_available():
                print("  🎲 CUDA random seed set for all devices")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'vision_transformer.image_tests.batch_size')."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


class TimingContext:
    """Context manager for timing test execution."""
    
    def __init__(self, test_name: str, verbose: bool = True):
        self.test_name = test_name
        self.verbose = verbose
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose and self.start_time is not None:
            elapsed = time.time() - self.start_time
            print(f"   ⏱️  {self.test_name} completed in {elapsed:.3f}s")


class TestVisionTransformer:
    """Test suite for VisionTransformer encoder."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = config.device
    
    def test_image_input_output_shapes(self):
        """Test VisionTransformer with image inputs."""
        cfg = self.config.get('vision_transformer.image_tests')
        if not cfg:
            print("⚠️  Image test configuration not found, skipping...")
            return
        
        with TimingContext("Image input test", self.config.get('general.include_timing', False)):
            # Create model
            model = VisionTransformer(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                in_chans=cfg['in_chans'],
                embed_dim=cfg['embed_dim'],
                depth=cfg['depth'],
                num_heads=cfg['num_heads'],
                use_rope=cfg['use_rope']
            ).to(self.device)
            
            # Create input tensor
            x = torch.randn(
                cfg['batch_size'], cfg['in_chans'], 
                cfg['img_size'][0], cfg['img_size'][1]
            ).to(self.device)
            
            # Forward pass
            output = model(x)
            
            # Expected output shape
            num_patches = (cfg['img_size'][0] // cfg['patch_size']) * (cfg['img_size'][1] // cfg['patch_size'])
            expected_shape = (cfg['batch_size'], num_patches, cfg['embed_dim'])
            
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
            print(f"✓ Image input test passed: {x.shape} -> {output.shape}")
    
    def test_video_input_output_shapes(self):
        """Test VisionTransformer with video inputs."""
        cfg = self.config.get('vision_transformer.video_tests')
        if not cfg:
            print("⚠️  Video test configuration not found, skipping...")
            return
        
        with TimingContext("Video input test", self.config.get('general.include_timing', False)):
            # Create model
            model = VisionTransformer(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                num_frames=cfg['num_frames'],
                tubelet_size=cfg['tubelet_size'],
                in_chans=cfg['in_chans'],
                embed_dim=cfg['embed_dim'],
                depth=cfg['depth'],
                num_heads=cfg['num_heads'],
                use_rope=cfg['use_rope']
            ).to(self.device)
            
            # Create input tensor
            x = torch.randn(
                cfg['batch_size'], cfg['in_chans'], cfg['num_frames'],
                cfg['img_size'][0], cfg['img_size'][1]
            ).to(self.device)
            
            # Forward pass
            output = model(x)
            
            # Expected output shape
            temporal_patches = cfg['num_frames'] // cfg['tubelet_size']
            spatial_patches = (cfg['img_size'][0] // cfg['patch_size']) * (cfg['img_size'][1] // cfg['patch_size'])
            num_patches = temporal_patches * spatial_patches
            expected_shape = (cfg['batch_size'], num_patches, cfg['embed_dim'])
            
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
            print(f"✓ Video input test passed: {x.shape} -> {output.shape}")
    
    def test_rope_vs_sincos_consistency(self):
        """Test that models with RoPE and sincos positional encoding have consistent shapes."""
        cfg = self.config.get('vision_transformer.rope_comparison')
        if not cfg:
            print("⚠️  RoPE comparison configuration not found, skipping...")
            return
        
        with TimingContext("RoPE vs sincos test", self.config.get('general.include_timing', False)):
            # Create models with different positional encodings
            model_sincos = VisionTransformer(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                in_chans=cfg['in_chans'],
                embed_dim=cfg['embed_dim'],
                depth=cfg['depth'],
                num_heads=cfg['num_heads'],
                use_rope=False
            ).to(self.device)
            
            model_rope = VisionTransformer(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                in_chans=cfg['in_chans'],
                embed_dim=cfg['embed_dim'],
                depth=cfg['depth'],
                num_heads=cfg['num_heads'],
                use_rope=True
            ).to(self.device)
            
            # Create input
            x = torch.randn(
                cfg['batch_size'], cfg['in_chans'],
                cfg['img_size'][0], cfg['img_size'][1]
            ).to(self.device)
            
            # Forward pass
            output_sincos = model_sincos(x)
            output_rope = model_rope(x)
            
            # Should have same output shape
            assert output_sincos.shape == output_rope.shape, f"Shape mismatch: sincos {output_sincos.shape} vs rope {output_rope.shape}"
            print(f"✓ RoPE vs sincos consistency test passed: both output {output_sincos.shape}")
    
    def test_predefined_model_variants(self):
        """Test predefined model variants like vit_base, vit_small, etc."""
        cfg = self.config.get('vision_transformer.model_variants')
        if not cfg:
            print("⚠️  Model variants configuration not found, skipping...")
            return
        
        with TimingContext("Model variants test", self.config.get('general.include_timing', False)):
            models = {
                'vit_tiny': vit_tiny,
                'vit_small': vit_small, 
                'vit_base': vit_base
            }
            
            x = torch.randn(
                cfg['batch_size'], cfg['in_chans'],
                cfg['img_size'][0], cfg['img_size'][1]
            ).to(self.device)
            
            for name in cfg['test_variants']:
                if name in models:
                    model = models[name](img_size=tuple(cfg['img_size'])).to(self.device)
                    output = model(x)
                    
                    # All should have same spatial dimensions, different embed_dim
                    num_patches = (cfg['img_size'][0] // 16) * (cfg['img_size'][1] // 16)  # patch_size=16 default
                    expected_batch_patches = (cfg['batch_size'], num_patches)
                    
                    assert output.shape[:2] == expected_batch_patches, f"{name} shape mismatch: {output.shape}"
                    print(f"✓ {name} test passed: {output.shape}")


class TestVisionTransformerPredictor:
    """Test suite for VisionTransformerPredictor (first stage)."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = config.device
    
    def test_predictor_forward_with_masks(self):
        """Test predictor with proper mask inputs."""
        cfg = self.config.get('predictor_first_stage.basic_tests')
        if not cfg:
            print("⚠️  Basic predictor test configuration not found, skipping...")
            return
        
        with TimingContext("Predictor forward test", self.config.get('general.include_timing', False)):
            # Create model
            model = VisionTransformerPredictor(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                embed_dim=cfg['embed_dim'],
                predictor_embed_dim=cfg['predictor_embed_dim'],
                depth=cfg['depth'],
                num_heads=cfg['num_heads'],
                use_mask_tokens=cfg['use_mask_tokens'],
                num_mask_tokens=cfg['num_mask_tokens'],
                use_rope=cfg['use_rope']
            ).to(self.device)
            
            # Calculate patch numbers
            num_patches = (cfg['img_size'][0] // cfg['patch_size']) * (cfg['img_size'][1] // cfg['patch_size'])
            num_context = int(num_patches * cfg['context_ratio'])
            num_target = int(num_patches * cfg['target_ratio'])
            
            x = torch.randn(cfg['batch_size'], num_context, cfg['embed_dim']).to(self.device)
            
            # Create masks
            masks_x = [torch.randperm(num_patches)[:num_context] for _ in range(cfg['batch_size'])]
            masks_y = [torch.randperm(num_patches)[:num_target] for _ in range(cfg['batch_size'])]
            
            # Forward pass
            output = model(x, masks_x, masks_y)
            
            expected_shape = (cfg['batch_size'], num_target, cfg['embed_dim'])
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
            print(f"✓ Predictor forward test passed: context {x.shape} -> predictions {output.shape}")
    
    def test_predictor_video_mode(self):
        """Test predictor in video mode."""
        cfg = self.config.get('predictor_first_stage.video_tests')
        if not cfg:
            print("⚠️  Video predictor test configuration not found, skipping...")
            return
        
        with TimingContext("Video predictor test", self.config.get('general.include_timing', False)):
            # Create model in video mode
            model = VisionTransformerPredictor(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                num_frames=cfg['num_frames'],
                tubelet_size=cfg['tubelet_size'],
                embed_dim=cfg['embed_dim'],
                predictor_embed_dim=cfg['predictor_embed_dim'],
                depth=cfg['depth'],
                num_heads=cfg['num_heads'],
                use_mask_tokens=cfg['use_mask_tokens'],
                use_rope=cfg['use_rope']
            ).to(self.device)
            
            # Calculate patch dimensions
            temporal_patches = cfg['num_frames'] // cfg['tubelet_size']
            spatial_patches = (cfg['img_size'][0] // cfg['patch_size']) * (cfg['img_size'][1] // cfg['patch_size'])
            num_patches = temporal_patches * spatial_patches
            
            num_context = int(num_patches * cfg['context_ratio'])
            num_target = int(num_patches * cfg['target_ratio'])
            
            x = torch.randn(cfg['batch_size'], num_context, cfg['embed_dim']).to(self.device)
            masks_x = [torch.randperm(num_patches)[:num_context]]
            masks_y = [torch.randperm(num_patches)[:num_target]]
            
            # Forward pass
            output = model(x, masks_x, masks_y)
            
            expected_shape = (cfg['batch_size'], num_target, cfg['embed_dim'])
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
            print(f"✓ Video predictor test passed: {x.shape} -> {output.shape}")


class TestVisionTransformerPredictorAC:
    """Test suite for VisionTransformerPredictorAC (second stage, action-conditioned)."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = config.device
    
    def test_ac_predictor_forward(self):
        """Test action-conditioned predictor forward pass."""
        cfg = self.config.get('predictor_second_stage.basic_tests')
        if not cfg:
            print("⚠️  AC predictor basic test configuration not found, skipping...")
            return
        
        with TimingContext("AC predictor test", self.config.get('general.include_timing', False)):
            # Create model
            model = VisionTransformerPredictorAC(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                num_frames=cfg['num_frames'],
                tubelet_size=cfg['tubelet_size'],
                embed_dim=cfg['embed_dim'],
                predictor_embed_dim=cfg['predictor_embed_dim'],
                depth=cfg['depth'],
                num_heads=cfg['num_heads'],
                action_embed_dim=cfg['action_embed_dim'],
                use_rope=cfg['use_rope'],
                use_extrinsics=cfg['use_extrinsics']
            ).to(self.device)
            
            # Calculate dimensions
            temporal_patches = cfg['num_frames'] // cfg['tubelet_size']
            spatial_patches = (cfg['img_size'][0] // cfg['patch_size']) * (cfg['img_size'][1] // cfg['patch_size'])
            total_patches = temporal_patches * spatial_patches
            
            # Create inputs
            x = torch.randn(cfg['batch_size'], total_patches, cfg['embed_dim']).to(self.device)
            actions = torch.randn(cfg['batch_size'], temporal_patches, cfg['action_embed_dim']).to(self.device)
            states = torch.randn(cfg['batch_size'], temporal_patches, cfg['action_embed_dim']).to(self.device)
            
            # Forward pass
            output = model(x, actions, states)
            
            expected_shape = (cfg['batch_size'], total_patches, cfg['embed_dim'])
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
            print(f"✓ AC predictor test passed: context {x.shape}, actions {actions.shape} -> {output.shape}")
    
    def test_ac_predictor_with_extrinsics(self):
        """Test action-conditioned predictor with extrinsics."""
        cfg = self.config.get('predictor_second_stage.extrinsics_tests')
        if not cfg:
            print("⚠️  AC predictor extrinsics test configuration not found, skipping...")
            return
        
        with TimingContext("AC predictor extrinsics test", self.config.get('general.include_timing', False)):
            # Create model with extrinsics
            model = VisionTransformerPredictorAC(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                num_frames=cfg['num_frames'],
                tubelet_size=cfg['tubelet_size'],
                embed_dim=cfg['embed_dim'],
                predictor_embed_dim=cfg['predictor_embed_dim'],
                depth=cfg['depth'],
                num_heads=cfg['num_heads'],
                action_embed_dim=cfg['action_embed_dim'],
                use_rope=cfg['use_rope'],
                use_extrinsics=cfg['use_extrinsics']
            ).to(self.device)
            
            # Calculate dimensions
            temporal_patches = cfg['num_frames'] // cfg['tubelet_size']
            spatial_patches = (cfg['img_size'][0] // cfg['patch_size']) * (cfg['img_size'][1] // cfg['patch_size'])
            total_patches = temporal_patches * spatial_patches
            
            # Create inputs
            x = torch.randn(cfg['batch_size'], total_patches, cfg['embed_dim']).to(self.device)
            actions = torch.randn(cfg['batch_size'], temporal_patches, cfg['action_embed_dim']).to(self.device)
            states = torch.randn(cfg['batch_size'], temporal_patches, cfg['action_embed_dim']).to(self.device)
            extrinsics = torch.randn(cfg['batch_size'], temporal_patches, cfg['action_embed_dim'] - 1).to(self.device)
            
            # Forward pass
            output = model(x, actions, states, extrinsics)
            
            expected_shape = (cfg['batch_size'], total_patches, cfg['embed_dim'])
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
            print(f"✓ AC predictor with extrinsics test passed: {x.shape} -> {output.shape}")


class TestModelIntegration:
    """Integration tests showing how models work together."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = config.device
    
    def test_encoder_to_predictor_pipeline(self):
        """Test the pipeline from encoder to first-stage predictor."""
        cfg = self.config.get('integration_tests.encoder_predictor_pipeline')
        if not cfg:
            print("⚠️  Integration test configuration not found, skipping...")
            return
        
        with TimingContext("Encoder->Predictor pipeline test", self.config.get('general.include_timing', False)):
            # Create encoder
            encoder = VisionTransformer(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                embed_dim=cfg['embed_dim'],
                depth=cfg['encoder_depth'],
                num_heads=cfg['num_heads'],
                use_rope=cfg['use_rope']
            ).to(self.device)
            
            # Create predictor
            predictor = VisionTransformerPredictor(
                img_size=tuple(cfg['img_size']),
                patch_size=cfg['patch_size'],
                embed_dim=cfg['embed_dim'],
                predictor_embed_dim=cfg['predictor_embed_dim'],
                depth=cfg['predictor_depth'],
                num_heads=cfg['num_heads'],
                use_mask_tokens=True,
                use_rope=cfg['use_rope']
            ).to(self.device)
            
            # Create input and masks
            x = torch.randn(cfg['batch_size'], 3, cfg['img_size'][0], cfg['img_size'][1]).to(self.device)
            num_patches = (cfg['img_size'][0] // cfg['patch_size']) * (cfg['img_size'][1] // cfg['patch_size'])
            
            # Calculate mask sizes
            context_size = int(num_patches * cfg['context_ratio'])
            target_size = int(num_patches * cfg['target_ratio'])
            
            context_indices = torch.randperm(num_patches)[:context_size]
            target_indices = torch.randperm(num_patches)[:target_size]
            
            # Step 1: Encode with masking
            encoder_output = encoder(x, masks=[context_indices])
            
            # Step 2: Predict masked regions
            prediction = predictor(
                encoder_output, 
                masks_x=[context_indices], 
                masks_y=[target_indices]
            )
            
            expected_pred_shape = (cfg['batch_size'], target_size, cfg['embed_dim'])
            assert prediction.shape == expected_pred_shape, f"Prediction shape mismatch: {prediction.shape}"
            print(f"✓ Encoder->Predictor pipeline test passed: {x.shape} -> {encoder_output.shape} -> {prediction.shape}")


def run_tests(config=None):
    """Run all configured tests."""
    print("Running V-JEPA2 model tests...\n")
    
    # Load configuration if not provided
    if config is None:
        config = TestConfig()
    
    # Determine which test suites to run
    suites = config.get('test_suites', {})
    
    # Test VisionTransformer
    if suites.get('vision_transformer', True):
        print("=" * 50)
        print("Testing VisionTransformer")
        print("=" * 50)
        vit_tests = TestVisionTransformer(config)
        vit_tests.test_image_input_output_shapes()
        vit_tests.test_video_input_output_shapes()
        vit_tests.test_rope_vs_sincos_consistency()
        vit_tests.test_predefined_model_variants()
    
    # Test VisionTransformerPredictor
    if suites.get('predictor_first_stage', True):
        print("\n" + "=" * 50)
        print("Testing VisionTransformerPredictor (First Stage)")
        print("=" * 50)
        pred_tests = TestVisionTransformerPredictor(config)
        pred_tests.test_predictor_forward_with_masks()
        pred_tests.test_predictor_video_mode()
    
    # Test VisionTransformerPredictorAC
    if suites.get('predictor_second_stage', True):
        print("\n" + "=" * 50)
        print("Testing VisionTransformerPredictorAC (Second Stage)")
        print("=" * 50)
        ac_tests = TestVisionTransformerPredictorAC(config)
        ac_tests.test_ac_predictor_forward()
        ac_tests.test_ac_predictor_with_extrinsics()
    
    # Test integration
    if suites.get('model_integration', True):
        print("\n" + "=" * 50)
        print("Testing Model Integration")
        print("=" * 50)
        integration_tests = TestModelIntegration(config)
        integration_tests.test_encoder_to_predictor_pipeline()
    
    print("\n" + "=" * 60)
    print("All configured tests passed! ✓")
    print("=" * 60)
    
    # Print device and timing info if verbose
    if config.get('general.verbose', True):
        print(f"\nTests completed on device: {config.device}")
        if config.get('general.include_timing', False):
            print("Timing information included above.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run V-JEPA2 model tests')
    parser.add_argument('--config', type=str, default='tests/tests_config.yaml',
                        help='Path to configuration file (default: tests/tests_config.yaml)')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Override device setting from config')
    parser.add_argument('--timing', action='store_true',
                        help='Enable timing measurements')
    parser.add_argument('--seed', type=int,
                        help='Override random seed from config')
    
    args = parser.parse_args()
    
    # Load configuration with potential overrides
    config = TestConfig(args.config)
    
    # Apply command line overrides
    if args.device:
        config.config['general']['device'] = args.device
        config.device = config.setup_device()
    
    if args.timing:
        config.config['general']['include_timing'] = True
    
    if args.seed:
        config.config['general']['random_seed'] = args.seed
        config.setup_random_seed()
    
    print(f"Using configuration: {args.config}")
    if args.device:
        print(f"Device override: {args.device}")
    
    run_tests(config) 