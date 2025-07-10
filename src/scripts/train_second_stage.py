import copy
import gc
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


def train_vjepa_mps(
    encoder,
    predictor, 
    target_encoder,
    dataloader,
    optimizer,
    scheduler,
    wd_scheduler,
    scaler=None,
    config=None,
    logger=None,
    csv_logger=None
):
    """
    Simplified V-JEPA training loop optimized for MPS and without states/extrinsics
    
    Args:
        encoder: Context encoder model
        predictor: Predictor model 
        target_encoder: Target encoder (EMA of encoder)
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        wd_scheduler: Weight decay scheduler  
        scaler: GradScaler for mixed precision (optional)
        config: Training configuration dict
        logger: Logger instance
        csv_logger: CSV logger instance
    """
    
    # Extract config parameters
    batch_size = config.get('batch_size', 8)
    num_epochs = config.get('num_epochs', 100)
    crop_size = config.get('crop_size', 256)
    patch_size = config.get('patch_size', 16)
    max_num_frames = config.get('max_num_frames', 16)
    tubelet_size = config.get('tubelet_size', 2)
    loss_exp = config.get('loss_exp', 1.0)
    normalize_reps = config.get('normalize_reps', True)
    auto_steps = config.get('auto_steps', 1)
    mixed_precision = config.get('mixed_precision', False)
    dtype = config.get('dtype', torch.float32)
    log_freq = config.get('log_freq', 10)
    checkpoint_freq = config.get('checkpoint_freq', 1)
    save_dir = config.get('save_dir', './checkpoints')
    
    # Device setup for MPS compatibility
    # Check if device is specified in config first
    config_device = config.get('device', None)
    if config_device:
        device = torch.device(config_device)
        device_type = config_device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Calculate tokens per frame (will be updated dynamically)
    tokens_per_frame = int((crop_size // patch_size) ** 2)
    ipe = len(dataloader)  # iterations per epoch
    
    # Training utilities
    class AverageMeter:
        def __init__(self):
            self.reset()
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def save_checkpoint(epoch, path):
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(), 
            "target_encoder": target_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "config": config,
        }
        try:
            torch.save(save_dict, path)
            logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def get_memory_usage():
        """Get memory usage based on device type"""
        if device_type == "mps":
            try:
                return torch.mps.current_allocated_memory() / 1024.0**2
            except:
                return 0.0
        elif device_type == "cuda":
            return torch.cuda.max_memory_allocated() / 1024.0**2
        else:
            return 0.0

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Initialize meters
        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter() 
        iter_time_meter = AverageMeter()
        data_time_meter = AverageMeter()
        
        encoder.train()
        predictor.train()
        target_encoder.eval()
        
        for itr, sample in enumerate(dataloader):
            itr_start_time = time.time()
            
            # Load data (only clips and actions, no states/extrinsics)
            def load_data():
                states = sample[0].to(device, non_blocking=True)  # [B C T-1 H W]
                next_state = sample[1].to(device, non_blocking=True)  # [B C 1 H W]
                clips = torch.cat([states, next_state], dim=2)  # [B C T H W]
                actions = sample[2].to(device, dtype=torch.float, non_blocking=True)  # [B T-1 A]
                return clips, actions
            
            clips, actions = load_data()
            data_time = (time.time() - itr_start_time) * 1000.0
            
            def train_step():
                # Update learning rate and weight decay
                scheduler.step()
                wd_scheduler.step()
                _new_lr = optimizer.param_groups[0]['lr']
                _new_wd = optimizer.param_groups[0]['weight_decay']
                
                def forward_target(c):
                    """Forward pass through target encoder"""
                    with torch.no_grad():
                        # VisionTransformer expects [B, C, T, H, W] format for video
                        h = target_encoder(c)  # Input is already [B, C, T, H, W]
                        
                        # h is now [B, N_total, D] where N_total = T * tokens_per_frame
                        B, N_total, embed_dim = h.shape
                        tokens_per_frame_actual = N_total // max_num_frames
                        
                        if normalize_reps:
                            h = F.layer_norm(h, (h.size(-1),))
                        return h
                
                def forward_predictions(z):
                    """Forward pass through predictor with actions only"""
                    # z is [B, N_total, D] where N_total = T * tokens_per_frame
                    B, N_total, D = z.shape
                    actual_tokens_per_frame = N_total // max_num_frames
                    
                    def _step_predictor(_z, _a):
                        """Single prediction step using only actions"""
                        pred_all = predictor(_z, _a)  # Predictor outputs tokens for all frames
                        # Extract only the last frame's predictions
                        _z = pred_all[:, -actual_tokens_per_frame:]
                        if normalize_reps:
                            _z = F.layer_norm(_z, (_z.size(-1),))
                        return _z
                    
                    # Teacher forcing step - use context frames to predict last frame
                    context_tokens = (max_num_frames - 1) * actual_tokens_per_frame
                    _z, _a = z[:, :context_tokens], actions
                    z_tf = _step_predictor(_z, _a)
                    
                    # Auto-regressive rollout 
                    _z = torch.cat([z[:, :actual_tokens_per_frame], z_tf], dim=1)
                    for n in range(1, auto_steps):
                        _a = actions[:, :n+1] if actions.size(1) > n else actions
                        _z_nxt = _step_predictor(_z, _a)  # Already returns only last frame tokens
                        _z = torch.cat([_z, _z_nxt], dim=1)
                    z_ar = _z[:, actual_tokens_per_frame:]
                    
                    return z_tf, z_ar
                
                def loss_fn(z, h):
                    """Compute loss between predictions and targets"""
                    # h is [B, N_total, D], z is predicted tokens for last frame(s)
                    B, N_total, D = h.shape
                    actual_tokens_per_frame = N_total // max_num_frames
                    
                    # Get target tokens starting from frame 1 (second frame)
                    # This properly aligns with auto-regressive predictions
                    _h = h[:, actual_tokens_per_frame : z.size(1) + actual_tokens_per_frame]
                    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp
                
                # Forward pass with autocast for mixed precision
                autocast_context = (
                    torch.autocast(device_type=device_type, dtype=dtype) 
                    if mixed_precision and device_type in ["cuda", "cpu"]
                    else torch.autocast(device_type="cpu", dtype=dtype)
                    if mixed_precision and device_type == "mps"
                    else torch.no_grad() if False else torch.enable_grad()
                )
                
                with autocast_context:
                    h = forward_target(clips)
                    z_tf, z_ar = forward_predictions(h)
                    jloss = loss_fn(z_tf, h)  # Teacher forcing loss
                    sloss = loss_fn(z_ar, h)  # Auto-regressive loss
                    loss = jloss + sloss
                
                # Backward pass
                if mixed_precision and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad()
                
                return float(loss), float(jloss), float(sloss), _new_lr, _new_wd
            
            # Execute training step
            step_start = time.time()
            loss, jloss, sloss, new_lr, new_wd = train_step()
            step_time = (time.time() - step_start) * 1000.0
            
            total_time = (time.time() - itr_start_time) * 1000.0
            
            # Update meters
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            iter_time_meter.update(total_time)
            data_time_meter.update(data_time)
            
            # Logging
            if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss):
                memory_usage = get_memory_usage()
                log_msg = (
                    f"[{epoch+1:3d}, {itr:5d}] "
                    f"loss: {loss_meter.avg:.3f} [{jloss_meter.avg:.2f}, {sloss_meter.avg:.2f}] "
                    f"[wd: {new_wd:.2e}] [lr: {new_lr:.2e}] "
                    f"[mem: {memory_usage:.1f}MB] "
                    f"[iter: {iter_time_meter.avg:.1f}ms] "
                    f"[data: {data_time_meter.avg:.1f}ms]"
                )
                logger.info(log_msg)
                
                if csv_logger is not None:
                    csv_logger.log(epoch + 1, itr, loss, total_time, step_time, data_time)
            
            # Check for NaN
            if np.isnan(loss):
                logger.error("Loss is NaN! Stopping training.")
                raise ValueError("Training diverged - loss is NaN")
        
        # End of epoch
        logger.info(f"Epoch {epoch+1} completed. Average loss: {loss_meter.avg:.3f}")
        
        # Save checkpoint
        if (epoch % checkpoint_freq == 0) or (epoch == num_epochs - 1):
            checkpoint_path = f"{save_dir}/checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(epoch + 1, checkpoint_path)
            
            # Also save as latest
            latest_path = f"{save_dir}/latest.pt"
            save_checkpoint(epoch + 1, latest_path)


# Example usage configuration
def get_training_config():
    return {
        'batch_size': 8,
        'num_epochs': 100,
        'crop_size': 256,
        'patch_size': 16,
        'max_num_frames': 16,
        'tubelet_size': 1,
        'loss_exp': 1.0,
        'normalize_reps': True,
        'auto_steps': 1,
        'mixed_precision': True,
        'dtype': torch.float16,
        'log_freq': 10,
        'checkpoint_freq': 5,
        'save_dir': './checkpoints',
    }