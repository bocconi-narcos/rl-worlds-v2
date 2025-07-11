import copy
import random
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.setups import init_opt, init_video_model, init_mask_generator
from src.scripts.collect_load_data import DataCollectionPipeline
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
    input_size = cfgs_mask.get("input_size", (6, 224, 224))  # Default input size
    patch_size_mask = cfgs_mask.get("patch_size", (2, 16, 16))  # Default patch size
    masking_ratio = cfgs_mask.get("masking_ratio", 0.5)  # Use first mask config for ratio


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
    crop_size = cfgs_data.get("crop_size")
    patch_size = cfgs_data.get("patch_size")
    
    # Placeholder values for removed distributed params
    dataset_fpcs = cfgs_data.get("dataset_fpcs", [16])  # Default fallback
    max_num_frames = max(dataset_fpcs)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
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

    print("Pred embedding dimension:", pred_embed_dim)
    print("patch size:", patch_size)
    print("tubelet size:", tubelet_size)

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=1,
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

    print(f"Training dataloader length: {len(train_dataloader)}")
    

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
    momentum_scheduler = [
        ema_start + (ema_end - ema_start) * (epoch / num_epochs) 
        for epoch in range(num_epochs)
    ]

    print(f'Patch size: {patch_size}')
    # -- Initialize mask collator
    mask_generator = init_mask_generator(
        input_size=input_size,
        patch_size=patch_size_mask,
        num_blocks=1,
        masking_ratio=cfgs_mask.get("nenc", 0.5),  # Use first mask config for ratio
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

        print('Starting epoch:', epoch + 1)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            print('')
            print(f'Batch index: {batch_idx}, Batch length: {len(batch)}')


            state, next_state, action, reward = batch
            clip = torch.cat((state, next_state), dim=2).to(device)
            
            print('clip shape:', clip.shape)
            masks_enc = mask_generator(batch_size).to(device)  # Generate masks for encoder
            print('masks_enc shape:', masks_enc.shape)
            masks_pred =  ~masks_enc  # Using the inverse of masks_enc for prediction

            def train_step():
                def forward_target(c):
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                        return h

                def forward_context(c):
                    print('Encoding context...')
                    z = encoder(c, masks_enc)
                    print('')
                    print('Predicting with context...')
                    z = predictor(z, masks_enc, masks_pred)
                    print('')
                    return z

                def loss_fn(z, h):
                    # Assumption: predictor will have returned only masked tokens for z
                    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]

                    loss, n = 0, 0
                    for zi, hi in zip(z, h):
                        for zij, hij in zip(zi, hi):
                            loss += torch.mean(torch.abs(zij - hij))
                            n += 1
                    loss /= n
                    return loss

                # Step 1. Forward
                print('Forwarding target')
                h = forward_target(clip)
                print('')
                print('Forwarding context')
                z = forward_context(clip)
                loss = loss_fn(z, h)  # jepa prediction loss

                # Step 2. Backward & step
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                _new_lr = scheduler.step()

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)
                print('-' * 20)

                return float(loss)

            # Run the training step
            loss = train_step()
            epoch_loss += loss
            num_batches += 1
            print(f"Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {loss:.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / num_batches if num_batches > 0 else 0:.4f}")

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
        "mask":
            {
                "input_size": (6, 64, 64),  # (frames, height, width)
                "patch_size": (2, 8, 8),    # (temporal, spatial, spatial)
                "masking_ratio": 0.5,
                "enc_mask_scale": (0.2, 0.8),
                "pred_mask_scale": (0.2, 0.8),
                "aspect_ratio": (0.3, 3.0),
                "nenc": 0.2,
                "npred": 0.3
            },
        "model": {
            "model_name": "vit_tiny",
            "compile_model": False,
            "use_activation_checkpointing": False,
            "pred_depth": 2,
            "pred_num_heads": 4,
            "pred_embed_dim": 16,
            "uniform_power": False,
            "use_mask_tokens": True,
            "zero_init_mask_tokens": True,
            "use_rope": True,
            "use_silu": False,
            "use_pred_silu": False,
            "wide_silu": False
        },
        "data": {
            "batch_size": 32,
            "tubelet_size": 2,
            "crop_size": 64,
            "patch_size": 8,
            "dataset_fpcs": [6]
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
    
    main(test_args, resume_preempt=False)


if __name__ == "__main__":
    # Uncomment the line below to run the test
    test_training_small()
    pass
