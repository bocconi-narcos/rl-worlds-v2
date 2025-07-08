import copy
import random
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.setups import init_opt, init_video_model
from src.scripts.collect_load_data import DataCollectionPipeline
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)


def get_device():
    """Get the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get("meta")
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    which_dtype = cfgs_meta.get("dtype")
    
    if which_dtype.lower() == "bfloat16":
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        mixed_precision = True
    else:
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)

    # -- DATA
    cfgs_data = args.get("data")
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    
    # Placeholder values for removed distributed params
    dataset_fpcs = cfgs_data.get("dataset_fpcs", [16])  # Default fallback
    max_num_frames = max(dataset_fpcs)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup_epochs = cfgs_opt.get("warmup_epochs")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # -- set device using device logic: CUDA > MPS > CPU
    device = get_device()
    print(f"Using device: {device}")

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(cfgs_mask) * len(dataset_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=cfgs_meta.get("use_sdpa", False),
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        print("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    # -- Load data using collect_load_data.py
    data_pipeline = DataCollectionPipeline()
    train_dataloader, val_dataloader = data_pipeline.run_full_pipeline()
    
    # Get data length for optimizer initialization
    _dlen = len(train_dataloader)
    if ipe is None:
        ipe = _dlen
    print(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        warmup_epochs=warmup_epochs,
        num_epochs=num_epochs,
    )

    
    # Remove distributed data parallel wrapping
    for p in target_encoder.parameters():
        p.requires_grad = False

     # -- EMA momentum schedule (simple linear schedule)
    ema_start, ema_end = 0.996, 1.0
    momentum_schedule = [
        ema_start + (ema_end - ema_start) * (epoch / num_epochs) 
        for epoch in range(num_epochs)
    ]

    # -- Initialize mask collator
    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=(crop_size, crop_size),
        patch_size=(patch_size, patch_size),
        tubelet_size=tubelet_size,
    )

    def save_checkpoint(epoch, path):
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "batch_size": batch_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
            print(f"Checkpoint saved to {path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    print("Training setup completed. Starting training loop...")


    print("Training setup completed. Ready for training loop implementation.")
    print("Configuration summary:")
    print(f"  - Device: {device}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Model: {model_name}")
    print(f"  - Mixed precision: {mixed_precision}")

    # -- SIMPLIFIED TRAINING LOOP
    for epoch in range(num_epochs):
        encoder.train()
        predictor.train()
        target_encoder.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_dataloader):

            print('batch: ', batch_idx)
            print('batc')
            
            # Get batch data - assuming it comes from mask collator
            if isinstance(batch, list):
                # Handle multiple frame rates
                all_loss = 0.0
                batch_count = 0
                
                for fpc_batch in batch:
                    if len(fpc_batch) == 3:
                        udata, masks_enc, masks_pred = fpc_batch
                        clips = udata[0].to(device)
                        masks_enc = [m.to(device) for m in masks_enc]  
                        masks_pred = [m.to(device) for m in masks_pred]
                    else:
                        clips = batch[0].to(device)
                        # Simple random masking if no mask collator
                        B, C, T, H, W = clips.shape
                        num_patches = (T // tubelet_size) * (H // patch_size) * (W // patch_size)
                        mask_ratio = 0.75
                        num_masked = int(mask_ratio * num_patches)
                        
                        masks_enc = [torch.randperm(num_patches, device=device)[:num_patches-num_masked].unsqueeze(0).repeat(B, 1)]
                        masks_pred = [torch.randperm(num_patches, device=device)[:num_masked].unsqueeze(0).repeat(B, 1)]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    # Target encoder forward (no gradients)
                    with torch.no_grad():
                        target_features = target_encoder(clips)
                        if not isinstance(target_features, list):
                            target_features = [target_features]
                        # Normalize target features
                        target_features = [F.layer_norm(h, (h.size(-1),)) for h in target_features]
                    
                    # Context encoder + predictor forward
                    context_features = encoder(clips, masks_enc)
                    if not isinstance(context_features, list):
                        context_features = [context_features]
                    
                    predicted_features = predictor(context_features, masks_enc, masks_pred)
                    if not isinstance(predicted_features, list):
                        predicted_features = [predicted_features]
                    
                    # Compute loss
                    loss = 0.0
                    loss_count = 0
                    
                    for pred_feats, target_feats, mask_pred in zip(predicted_features, target_features, masks_pred):
                        # Apply prediction masks to target features
                        if len(mask_pred.shape) == 2:  # [B, num_masked]
                            target_masked = apply_masks(target_feats, [mask_pred], concat=False)[0]
                        else:
                            target_masked = target_feats
                        
                        # Compute L1 loss
                        batch_loss = F.l1_loss(pred_feats, target_masked)
                        loss += batch_loss
                        loss_count += 1
                    
                    if loss_count > 0:
                        loss = loss / loss_count
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    all_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    batch_loss = all_loss / batch_count
                else:
                    batch_loss = 0.0
            else:
                # Simple batch handling
                clips = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                
                # Simple random masking
                B, C, T, H, W = clips.shape
                num_patches = (T // tubelet_size) * (H // patch_size) * (W // patch_size)
                mask_ratio = 0.75
                num_masked = int(mask_ratio * num_patches)
                
                masks_enc = [torch.randperm(num_patches, device=device)[:num_patches-num_masked].unsqueeze(0).repeat(B, 1)]
                masks_pred = [torch.randperm(num_patches, device=device)[:num_masked].unsqueeze(0).repeat(B, 1)]
                
                # Forward pass
                optimizer.zero_grad()
                
                # Target encoder forward (no gradients)
                with torch.no_grad():
                    target_features = target_encoder(clips)
                    if not isinstance(target_features, list):
                        target_features = [target_features]
                    target_features = [F.layer_norm(h, (h.size(-1),)) for h in target_features]
                
                # Context encoder + predictor forward  
                context_features = encoder(clips, masks_enc)
                if not isinstance(context_features, list):
                    context_features = [context_features]
                
                predicted_features = predictor(context_features, masks_enc, masks_pred)
                if not isinstance(predicted_features, list):
                    predicted_features = [predicted_features]
                
                # Compute loss
                loss = 0.0
                for pred_feats, target_feats, mask_pred in zip(predicted_features, target_features, masks_pred):
                    target_masked = apply_masks(target_feats, [mask_pred], concat=False)[0]
                    loss += F.l1_loss(pred_feats, target_masked)
                
                loss = loss / len(predicted_features)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
            
            # EMA update of target encoder
            momentum = momentum_schedule[epoch]
            with torch.no_grad():
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
            
            epoch_loss += batch_loss
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {batch_loss:.4f}")
            
            # Step scheduler
            scheduler.step()
        
        # Print epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(epoch + 1, checkpoint_path)
    
    print("Training completed!")
    
    # Save final checkpoint
    save_checkpoint(num_epochs, "final_checkpoint.pt")


def test_training_small():
    """Test training with very small parameters for quick verification"""
    print("Running small parameter test...")
    
    # Create minimal test configuration
    test_args = {
        "meta": {
            "seed": 42,
            "dtype": "float32",
            "use_sdpa": False
        },
        "mask": [
            {
                "enc_mask_scale": (0.2, 0.8),
                "pred_mask_scale": (0.2, 0.8),
                "aspect_ratio": (0.3, 3.0),
                "nenc": 0.2,
                "npred": 0.3
            }
        ],
        "model": {
            "model_name": "vit_tiny",
            "compile_model": False,
            "use_activation_checkpointing": False,
            "pred_depth": 2,
            "pred_num_heads": 4,
            "pred_embed_dim": 192,
            "uniform_power": False,
            "use_mask_tokens": True,
            "zero_init_mask_tokens": True,
            "use_rope": False,
            "use_silu": False,
            "use_pred_silu": False,
            "wide_silu": False
        },
        "data": {
            "batch_size": 2,
            "tubelet_size": 2,
            "crop_size": 64,
            "patch_size": 8,
            "dataset_fpcs": [8]
        },
        "optimization": {
            "ipe": None,
            "weight_decay": 0.01,
            "epochs": 2,
            "warmup_epochs": 1,
            "start_lr": 1e-6,
            "lr": 1e-4,
            "final_lr": 1e-6
        }
    }
    
    # Run the test
    try:
        main(test_args, resume_preempt=False)
        print("✓ Small parameter test completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Small parameter test failed: {e}")
        return False


if __name__ == "__main__":
    # Uncomment the line below to run the test
    test_training_small()
    pass
