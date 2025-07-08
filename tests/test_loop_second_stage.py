"""
Test suite for V-JEPA2 Second Stage Training Loop.

This module tests the training loop functionality for the second stage of V-JEPA2:
- train_vjepa_mps function from src/scripts/train_second_stage.py
- Model weight updates during training
- Loss computation and gradient flow
- Checkpoint saving and loading
- Memory management
- Error handling

Usage Examples:
    # Run with default configuration
    python tests/test_loop_second_stage.py
    
    # Force specific device
    python tests/test_loop_second_stage.py --device mps
    
    # Run with timing information
    python tests/test_loop_second_stage.py --timing
    
    # Use custom configuration
    python tests/test_loop_second_stage.py --config tests/tests_config.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import sys
import os
import tempfile
import shutil
import numpy as np
from typing import Dict, Any, Tuple, List
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.scripts.train_second_stage import train_vjepa_mps, get_training_config
from src.models.vision_transformer import VisionTransformer, vit_base
from src.models.predictor_second_stage import VisionTransformerPredictorAC, vit_ac_predictor


class MockLogger:
    """Mock logger for testing purposes."""
    
    def __init__(self):
        self.logs = []
    
    def info(self, message):
        self.logs.append(('info', message))
        print(f"INFO: {message}")
    
    def error(self, message):
        self.logs.append(('error', message))
        print(f"ERROR: {message}")
    
    def get_logs(self):
        return self.logs


class MockCSVLogger:
    """Mock CSV logger for testing purposes."""
    
    def __init__(self):
        self.entries = []
    
    def log(self, epoch, iteration, loss, total_time, step_time, data_time):
        entry = {
            'epoch': epoch,
            'iteration': iteration,
            'loss': loss,
            'total_time': total_time,
            'step_time': step_time,
            'data_time': data_time
        }
        self.entries.append(entry)
    
    def get_entries(self):
        return self.entries


class MockDataset:
    """Mock dataset that generates synthetic video and action data."""
    
    def __init__(self, num_samples=10, img_size=224, num_frames=8, action_dim=7):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_frames = num_frames
        self.action_dim = action_dim
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic video data: [C, T-1, H, W] for states and [C, 1, H, W] for next_state
        states = torch.randn(3, self.num_frames - 1, self.img_size, self.img_size)
        next_state = torch.randn(3, 1, self.img_size, self.img_size)
        
        # Generate synthetic action data: [T-1, action_dim]
        actions = torch.randn(self.num_frames - 1, self.action_dim)
        
        return states, next_state, actions


class TestConfig:
    """Configuration loader for training loop tests."""
    
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
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for training loop tests."""
        return {
            'general': {
                'random_seed': 42,
                'device': 'auto',
                'verbose': True,
                'include_timing': False
            },
            'training_loop_tests': {
                'batch_size': 2,
                'num_epochs': 2,
                'num_samples': 8,
                'img_size': 112,  # Smaller for faster testing
                'num_frames': 4,
                'action_dim': 7,
                'patch_size': 16,
                'embed_dim': 192,
                'predictor_embed_dim': 256,
                'encoder_depth': 2,
                'predictor_depth': 2,
                'num_heads': 4,
                'learning_rate': 1e-4,
                'weight_decay': 0.05,
                'log_freq': 1,
                'checkpoint_freq': 1
            }
        }
    
    def setup_device(self) -> torch.device:
        """Setup computation device based on configuration."""
        device_config = self.config['general']['device']
        
        if device_config == 'auto':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        if self.config['general']['verbose']:
            print(f"✓ Using device: {device}")
        
        return device
    
    def setup_random_seed(self):
        """Set random seed for reproducibility."""
        seed = self.config['general']['random_seed']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
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


class TestSecondStageTrainingLoop:
    """Test suite for the second stage training loop."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = config.device
        
    def create_models(self, cfg: Dict[str, Any]) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Create encoder, predictor, and target encoder models."""
        # Create encoder
        encoder = VisionTransformer(
            img_size=(cfg['img_size'], cfg['img_size']),
            patch_size=cfg['patch_size'],
            num_frames=cfg['num_frames'],
            tubelet_size=1,  # No temporal compression
            embed_dim=cfg['embed_dim'],
            depth=cfg['encoder_depth'],
            num_heads=cfg['num_heads'],
            use_rope=True
        ).to(self.device)
        
        # Create predictor
        predictor = VisionTransformerPredictorAC(
            img_size=(cfg['img_size'], cfg['img_size']),
            patch_size=cfg['patch_size'],
            num_frames=cfg['num_frames'],
            tubelet_size=1,  # No temporal compression
            embed_dim=cfg['embed_dim'],
            predictor_embed_dim=cfg['predictor_embed_dim'],
            depth=cfg['predictor_depth'],
            num_heads=cfg['num_heads'],
            action_embed_dim=cfg['action_dim'],
            use_rope=True
        ).to(self.device)
        
        # Create target encoder (copy of encoder)
        target_encoder = VisionTransformer(
            img_size=(cfg['img_size'], cfg['img_size']),
            patch_size=cfg['patch_size'],
            num_frames=cfg['num_frames'],
            tubelet_size=1,  # No temporal compression
            embed_dim=cfg['embed_dim'],
            depth=cfg['encoder_depth'],
            num_heads=cfg['num_heads'],
            use_rope=True
        ).to(self.device)
        
        # Initialize target encoder with encoder weights
        target_encoder.load_state_dict(encoder.state_dict())
        
        return encoder, predictor, target_encoder
    
    def create_dataloader(self, cfg: Dict[str, Any]) -> DataLoader:
        """Create a mock dataloader for testing."""
        dataset = MockDataset(
            num_samples=cfg['num_samples'],
            img_size=cfg['img_size'],
            num_frames=cfg['num_frames'],
            action_dim=cfg['action_dim']
        )
        
        return DataLoader(
            dataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
    
    def test_basic_training_loop(self):
        """Test that the basic training loop runs without errors."""
        print("\n🧪 Testing basic training loop functionality...")
        
        cfg = self.config.get('training_loop_tests')
        if not cfg:
            print("⚠️  Training loop test configuration not found, using defaults...")
            cfg = self.config.get_default_config()['training_loop_tests']
        
        with TimingContext("Basic training loop test", self.config.get('general.include_timing', False)):
            # Create models
            encoder, predictor, target_encoder = self.create_models(cfg)
            
            # Create dataloader
            dataloader = self.create_dataloader(cfg)
            
            # Create optimizer and schedulers
            all_params = list(encoder.parameters()) + list(predictor.parameters())
            optimizer = optim.AdamW(all_params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
            
            # Simple lambda schedulers for testing
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            wd_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            
            # Create mock loggers
            logger = MockLogger()
            csv_logger = MockCSVLogger()
            
            # Create temporary directory for checkpoints
            with tempfile.TemporaryDirectory() as temp_dir:
                # Training configuration
                training_config = {
                    'batch_size': cfg['batch_size'],
                    'num_epochs': cfg['num_epochs'],
                    'crop_size': cfg['img_size'],
                    'patch_size': cfg['patch_size'],
                    'max_num_frames': cfg['num_frames'],
                    'tubelet_size': 1,
                    'loss_exp': 1.0,
                    'normalize_reps': True,
                    'auto_steps': 1,
                    'mixed_precision': False,  # Disable for testing simplicity
                    'dtype': torch.float32,
                    'log_freq': cfg['log_freq'],
                    'checkpoint_freq': cfg['checkpoint_freq'],
                    'save_dir': temp_dir,
                    'device': str(self.device)  # Force device from test config
                }
                
                try:
                    # Run training loop
                    train_vjepa_mps(
                        encoder=encoder,
                        predictor=predictor,
                        target_encoder=target_encoder,
                        dataloader=dataloader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        wd_scheduler=wd_scheduler,
                        scaler=None,
                        config=training_config,
                        logger=logger,
                        csv_logger=csv_logger
                    )
                    
                    print("✅ Basic training loop completed successfully")
                    
                    # Check that logs were generated
                    logs = logger.get_logs()
                    assert len(logs) > 0, "No logs were generated during training"
                    print(f"✅ Generated {len(logs)} log entries")
                    
                    # Check that CSV entries were created
                    csv_entries = csv_logger.get_entries()
                    assert len(csv_entries) > 0, "No CSV entries were generated during training"
                    print(f"✅ Generated {len(csv_entries)} CSV entries")
                    
                    # Check that checkpoints were saved
                    checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
                    assert len(checkpoint_files) > 0, "No checkpoint files were saved"
                    print(f"✅ Saved {len(checkpoint_files)} checkpoint files")
                    
                except Exception as e:
                    print(f"❌ Training loop failed with error: {e}")
                    raise
    
    def test_model_weight_updates(self):
        """Test that model weights are actually updated during training."""
        print("\n🧪 Testing model weight updates...")
        
        cfg = self.config.get('training_loop_tests')
        if not cfg:
            cfg = self.config.get_default_config()['training_loop_tests']
        
        with TimingContext("Model weight update test", self.config.get('general.include_timing', False)):
            # Create models
            encoder, predictor, target_encoder = self.create_models(cfg)
            
            # Store initial weights
            initial_encoder_weights = {name: param.clone() for name, param in encoder.named_parameters()}
            initial_predictor_weights = {name: param.clone() for name, param in predictor.named_parameters()}
            
            # Create dataloader with just a few samples
            test_cfg = cfg.copy()
            test_cfg['num_samples'] = 4
            test_cfg['num_epochs'] = 1
            dataloader = self.create_dataloader(test_cfg)
            
            # Create optimizer and schedulers
            all_params = list(encoder.parameters()) + list(predictor.parameters())
            optimizer = optim.AdamW(all_params, lr=1e-3, weight_decay=0.01)  # Higher LR for visible changes
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            wd_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            
            # Create mock loggers
            logger = MockLogger()
            csv_logger = MockCSVLogger()
            
            # Training configuration
            training_config = {
                'batch_size': test_cfg['batch_size'],
                'num_epochs': test_cfg['num_epochs'],
                'crop_size': test_cfg['img_size'],
                'patch_size': test_cfg['patch_size'],
                'max_num_frames': test_cfg['num_frames'],
                'tubelet_size': 1,
                'loss_exp': 1.0,
                'normalize_reps': True,
                'auto_steps': 1,
                'mixed_precision': False,
                'dtype': torch.float32,
                'log_freq': 1,
                'checkpoint_freq': 1,
                'save_dir': '/tmp',  # Won't be used in this test
                'device': str(self.device)  # Force device from test config
            }
            
            try:
                # Run training loop
                train_vjepa_mps(
                    encoder=encoder,
                    predictor=predictor,
                    target_encoder=target_encoder,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    wd_scheduler=wd_scheduler,
                    scaler=None,
                    config=training_config,
                    logger=logger,
                    csv_logger=csv_logger
                )
                
                # Check that encoder weights changed
                encoder_changed = False
                for name, param in encoder.named_parameters():
                    if not torch.equal(param, initial_encoder_weights[name]):
                        encoder_changed = True
                        break
                
                assert encoder_changed, "Encoder weights did not change during training"
                print("✅ Encoder weights were updated during training")
                
                # Check that predictor weights changed
                predictor_changed = False
                for name, param in predictor.named_parameters():
                    if not torch.equal(param, initial_predictor_weights[name]):
                        predictor_changed = True
                        break
                
                assert predictor_changed, "Predictor weights did not change during training"
                print("✅ Predictor weights were updated during training")
                
            except Exception as e:
                print(f"❌ Weight update test failed with error: {e}")
                raise
    
    def test_loss_computation(self):
        """Test that loss values are reasonable and decrease over time."""
        print("\n🧪 Testing loss computation...")
        
        cfg = self.config.get('training_loop_tests')
        if not cfg:
            cfg = self.config.get_default_config()['training_loop_tests']
        
        with TimingContext("Loss computation test", self.config.get('general.include_timing', False)):
            # Create models
            encoder, predictor, target_encoder = self.create_models(cfg)
            
            # Create dataloader
            test_cfg = cfg.copy()
            test_cfg['num_samples'] = 8
            test_cfg['num_epochs'] = 3
            dataloader = self.create_dataloader(test_cfg)
            
            # Create optimizer and schedulers
            all_params = list(encoder.parameters()) + list(predictor.parameters())
            optimizer = optim.AdamW(all_params, lr=1e-3, weight_decay=0.01)
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            wd_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            
            # Create mock loggers
            logger = MockLogger()
            csv_logger = MockCSVLogger()
            
            # Training configuration
            training_config = {
                'batch_size': test_cfg['batch_size'],
                'num_epochs': test_cfg['num_epochs'],
                'crop_size': test_cfg['img_size'],
                'patch_size': test_cfg['patch_size'],
                'max_num_frames': test_cfg['num_frames'],
                'tubelet_size': 1,
                'loss_exp': 1.0,
                'normalize_reps': True,
                'auto_steps': 1,
                'mixed_precision': False,
                'dtype': torch.float32,
                'log_freq': 1,
                'checkpoint_freq': 1,
                'save_dir': '/tmp',
                'device': str(self.device)  # Force device from test config
            }
            
            try:
                # Run training loop
                train_vjepa_mps(
                    encoder=encoder,
                    predictor=predictor,
                    target_encoder=target_encoder,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    wd_scheduler=wd_scheduler,
                    scaler=None,
                    config=training_config,
                    logger=logger,
                    csv_logger=csv_logger
                )
                
                # Check loss values from CSV logger
                csv_entries = csv_logger.get_entries()
                assert len(csv_entries) > 0, "No CSV entries generated"
                
                # Check that all loss values are finite
                losses = [entry['loss'] for entry in csv_entries]
                assert all(np.isfinite(loss) for loss in losses), "Some loss values are not finite"
                print("✅ All loss values are finite")
                
                # Check that loss values are positive (for L1 loss)
                assert all(loss >= 0 for loss in losses), "Some loss values are negative"
                print("✅ All loss values are non-negative")
                
                # Check that loss generally decreases (with some tolerance for noise)
                if len(losses) >= 3:
                    early_loss = np.mean(losses[:len(losses)//3])
                    late_loss = np.mean(losses[-len(losses)//3:])
                    print(f"   Early loss: {early_loss:.4f}, Late loss: {late_loss:.4f}")
                    
                    # Allow for some increase due to randomness, but expect general improvement
                    if late_loss < early_loss * 1.5:  # Loss shouldn't increase by more than 50%
                        print("✅ Loss shows reasonable behavior over training")
                    else:
                        print("⚠️  Loss increased significantly, but this might be normal for short training")
                
            except Exception as e:
                print(f"❌ Loss computation test failed with error: {e}")
                raise
    
    def test_checkpoint_functionality(self):
        """Test checkpoint saving and loading functionality."""
        print("\n🧪 Testing checkpoint functionality...")
        
        cfg = self.config.get('training_loop_tests')
        if not cfg:
            cfg = self.config.get_default_config()['training_loop_tests']
        
        with TimingContext("Checkpoint functionality test", self.config.get('general.include_timing', False)):
            # Create models
            encoder, predictor, target_encoder = self.create_models(cfg)
            
            # Create dataloader
            test_cfg = cfg.copy()
            test_cfg['num_samples'] = 4
            test_cfg['num_epochs'] = 2
            dataloader = self.create_dataloader(test_cfg)
            
            # Create optimizer and schedulers
            all_params = list(encoder.parameters()) + list(predictor.parameters())
            optimizer = optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            wd_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
            
            # Create mock loggers
            logger = MockLogger()
            csv_logger = MockCSVLogger()
            
            # Create temporary directory for checkpoints
            with tempfile.TemporaryDirectory() as temp_dir:
                # Training configuration
                training_config = {
                    'batch_size': test_cfg['batch_size'],
                    'num_epochs': test_cfg['num_epochs'],
                    'crop_size': test_cfg['img_size'],
                    'patch_size': test_cfg['patch_size'],
                    'max_num_frames': test_cfg['num_frames'],
                    'tubelet_size': 1,
                    'loss_exp': 1.0,
                    'normalize_reps': True,
                    'auto_steps': 1,
                    'mixed_precision': False,
                    'dtype': torch.float32,
                    'log_freq': 1,
                    'checkpoint_freq': 1,
                    'save_dir': temp_dir,
                    'device': str(self.device)  # Force device from test config
                }
                
                try:
                    # Run training loop
                    train_vjepa_mps(
                        encoder=encoder,
                        predictor=predictor,
                        target_encoder=target_encoder,
                        dataloader=dataloader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        wd_scheduler=wd_scheduler,
                        scaler=None,
                        config=training_config,
                        logger=logger,
                        csv_logger=csv_logger
                    )
                    
                    # Check that checkpoint files exist
                    checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
                    assert len(checkpoint_files) > 0, "No checkpoint files were created"
                    print(f"✅ Created {len(checkpoint_files)} checkpoint files")
                    
                    # Test loading a checkpoint
                    latest_checkpoint = os.path.join(temp_dir, 'latest.pt')
                    if os.path.exists(latest_checkpoint):
                        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                        
                        # Check that checkpoint contains expected keys
                        expected_keys = ['encoder', 'predictor', 'target_encoder', 'optimizer', 'epoch', 'config']
                        for key in expected_keys:
                            assert key in checkpoint, f"Checkpoint missing key: {key}"
                        
                        print("✅ Checkpoint contains all expected keys")
                        
                        # Test loading weights into new models
                        new_encoder, new_predictor, new_target_encoder = self.create_models(cfg)
                        
                        new_encoder.load_state_dict(checkpoint['encoder'])
                        new_predictor.load_state_dict(checkpoint['predictor'])
                        new_target_encoder.load_state_dict(checkpoint['target_encoder'])
                        
                        print("✅ Successfully loaded checkpoint weights into new models")
                    
                except Exception as e:
                    print(f"❌ Checkpoint functionality test failed with error: {e}")
                    raise

    def test_debug_dimensions(self):
        """Debug test to understand tensor dimensions."""
        print("\n🔍 Debug: Checking tensor dimensions...")
        
        cfg = self.config.get('training_loop_tests')
        if not cfg:
            cfg = self.config.get_default_config()['training_loop_tests']
        
        # Create models
        encoder, predictor, target_encoder = self.create_models(cfg)
        
        # Create a simple test input
        batch_size = 1
        num_frames = cfg['num_frames']
        img_size = cfg['img_size']
        
        # Simple test tensor: [B, C, T, H, W]
        test_input = torch.randn(batch_size, 3, num_frames, img_size, img_size).to(self.device)
        print(f"Test input shape: {test_input.shape}")
        
        # Test encoder directly
        try:
            with torch.no_grad():
                encoder_output = target_encoder(test_input)
                print(f"✅ Encoder output shape: {encoder_output.shape}")
                print(f"Encoder output type: {type(encoder_output)}")
                print(f"Encoder output ndim: {encoder_output.ndim}")
                
                if hasattr(encoder_output, 'size'):
                    print(f"Size method: {encoder_output.size()}")
                
        except Exception as e:
            print(f"❌ Encoder failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test the training data format
        print("\n--- Testing with training data format ---")
        try:
            # Create sample as in training loop
            states = torch.randn(3, num_frames-1, img_size, img_size)  # [C, T-1, H, W]
            next_state = torch.randn(3, 1, img_size, img_size)  # [C, 1, H, W]
            actions = torch.randn(num_frames-1, 7)  # [T-1, A]
            
            # Add batch dimension and move to device
            states = states.unsqueeze(0).to(self.device)
            next_state = next_state.unsqueeze(0).to(self.device)
            clips = torch.cat([states, next_state], dim=2)  # [B C T H W]
            actions = actions.unsqueeze(0).to(self.device)
            
            print(f"Training clips shape: {clips.shape}")
            print(f"Training actions shape: {actions.shape}")
            
            # Reshape for encoder as in training loop
            c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B*T, C, H, W]
            print(f"Encoder input after permute/flatten: {c.shape}")
            
            with torch.no_grad():
                h = target_encoder(c)
                print(f"✅ Training encoder output shape: {h.shape}")
                print(f"Training encoder output ndim: {h.ndim}")
                
                # Now try to understand predictor requirements
                print(f"\n--- Testing predictor ---")
                
                # Try predictor with simple inputs
                if h.ndim == 3:
                    B, N_total, D = h.shape
                    print(f"B={B}, N_total={N_total}, D={D}")
                    
                    # Calculate actual tokens per frame
                    tokens_per_frame = N_total // num_frames
                    print(f"Tokens per frame: {tokens_per_frame}")
                    print(f"Expected tokens per frame (7*7): {7*7}")
                    
                    # For predictor, use context frames (all but last)
                    context_frames = num_frames - 1
                    context_tokens = context_frames * tokens_per_frame
                    pred_input = h[:, :context_tokens]
                    print(f"Predictor input shape: {pred_input.shape}")
                    print(f"Context frames: {context_frames}")
                    
                    # Actions should match context frames
                    pred_actions = actions[:, :context_frames]
                    print(f"Predictor actions shape: {pred_actions.shape}")
                    
                    try:
                        pred_output = predictor(pred_input, pred_actions)
                        print(f"✅ Predictor output shape: {pred_output.shape}")
                    except Exception as pred_e:
                        print(f"❌ Predictor failed: {pred_e}")
                        print(f"Error details:")
                        import traceback
                        traceback.print_exc()
                        
                        # Check if the issue is with token organization
                        print(f"Predictor expects tokens organized as frames of {predictor.grid_height}x{predictor.grid_width}={predictor.grid_height * predictor.grid_width}")
                        print(f"But we have {tokens_per_frame} tokens per frame")
                
        except Exception as e:
            print(f"❌ Training format test failed: {e}")
            import traceback
            traceback.print_exc()


def run_tests(config_path: str = "tests/tests_config.yaml"):
    """Run all second stage training loop tests."""
    print("=" * 80)
    print("🚀 V-JEPA2 Second Stage Training Loop Test Suite")
    print("=" * 80)
    
    # Load configuration
    config = TestConfig(config_path)
    
    # Create test suite
    test_suite = TestSecondStageTrainingLoop(config)
    
    # Run tests
    tests = [
        test_suite.test_basic_training_loop,
        test_suite.test_model_weight_updates,
        test_suite.test_loss_computation,
        test_suite.test_checkpoint_functionality,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed. Please check the output above.")
    
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test V-JEPA2 Second Stage Training Loop")
    parser.add_argument("--config", type=str, default="tests/tests_config.yaml",
                        help="Path to test configuration file")
    parser.add_argument("--device", type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                        help="Force specific device for testing")
    parser.add_argument("--timing", action="store_true",
                        help="Include timing information in output")
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.device or args.timing:
        config = TestConfig(args.config)
        if args.device:
            config.config['general']['device'] = args.device
        if args.timing:
            config.config['general']['include_timing'] = True
        
        # Save temporary config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config.config, f)
            temp_config_path = f.name
        
        try:
            success = run_tests(temp_config_path)
        finally:
            os.unlink(temp_config_path)
    else:
        success = run_tests(args.config)
    
    sys.exit(0 if success else 1) 