import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
import src.models.predictor_first_stage as vit_pred
import src.models.vision_transformer as video_vit

def init_opt(
    encoder,
    predictor,
    start_lr: float,
    ref_lr: float,
    warmup_epochs: int,
    num_epochs: int,
    wd: float = 1e-6,
    final_lr: float = 0.0,
):
    """
    Create AdamW optimizer and standard PyTorch LR schedulers.

    Args:
        encoder: feature extractor module
        predictor: prediction head module
        start_lr: initial learning rate at epoch 0
        ref_lr: peak learning rate after warmup
        warmup_epochs: number of epochs for linear warmup
        num_epochs: total training epochs
        wd: weight decay for optimizer
        final_lr: minimum learning rate after cosine annealing
    Returns:
        optim: AdamW optimizer
        scheduler: SequentialLR combining warmup and cosine annealing
    """
    # parameter grouping for weight decay
    decay, no_decay = [], []
    for module in (encoder, predictor):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or param.ndim == 1:
                no_decay.append(param)
            else:
                decay.append(param)

    optim = torch.optim.AdamW(
        [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=ref_lr,
        weight_decay=wd,
    )

    # Warmup: linearly increase LR from start_lr to ref_lr
    warmup_sched = LinearLR(
        optim,
        start_factor=start_lr / ref_lr,
        total_iters=warmup_epochs,
    )
    # Cosine annealing: decay from ref_lr to final_lr over remaining epochs
    cosine_sched = CosineAnnealingLR(
        optim,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=final_lr,
    )
    # Combine schedulers: warmup then cosine
    scheduler = SequentialLR(
        optim,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    return optim, scheduler


def init_video_model(  
    device,  
    patch_size=16,  
    max_num_frames=16,  
    tubelet_size=2,  
    model_name="vit_base",  
    crop_size=224,  
    pred_depth=6,  
    pred_num_heads=None,  
    pred_embed_dim=384,  
    uniform_power=False,  
    use_mask_tokens=False,  
    num_mask_tokens=2,  
    zero_init_mask_tokens=True,  
    use_sdpa=False,  
    use_rope=False,  
    use_silu=False,  
    use_pred_silu=False,  
    wide_silu=False,  
    use_activation_checkpointing=False,  
):  
    print('[setups] Initialising video model with max_num_frames:', max_num_frames,)
    encoder = video_vit.__dict__[model_name](  
        img_size=crop_size,  
        patch_size=patch_size,  
        num_frames=max_num_frames,  
        tubelet_size=tubelet_size,  
        uniform_power=uniform_power,  
        use_sdpa=use_sdpa,  
        use_silu=use_silu,  
        wide_silu=wide_silu,  
        use_activation_checkpointing=use_activation_checkpointing,  
        use_rope=use_rope,  
    )  
      
    predictor = vit_pred.__dict__["vit_predictor"](  
        img_size=crop_size,  
        use_mask_tokens=use_mask_tokens,  
        patch_size=patch_size,  
        num_frames=max_num_frames,  
        tubelet_size=tubelet_size,  
        embed_dim=encoder.embed_dim,  # Remove .backbone  
        predictor_embed_dim=pred_embed_dim,  
        depth=pred_depth,  
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,  # Remove .backbone  
        uniform_power=uniform_power,  
        num_mask_tokens=num_mask_tokens,  
        zero_init_mask_tokens=zero_init_mask_tokens,  
        use_rope=use_rope,  
        use_sdpa=use_sdpa,  
        use_silu=use_pred_silu,  
        wide_silu=wide_silu,  
        use_activation_checkpointing=use_activation_checkpointing,  
    )  
  
    encoder.to(device)  
    predictor.to(device)  
  
    def count_parameters(model):  
        return sum(p.numel() for p in model.parameters() if p.requires_grad)  
      
    print(f"Encoder parameters: {count_parameters(encoder)}")  
    print(f"Predictor parameters: {count_parameters(predictor)}")  
  
    return encoder, predictor


def init_mask_generator(
    input_size,
    patch_size,
    num_blocks=1,
    masking_ratio=0.5,
):
    from src.masks.multiseq_multiblock3d import MaskGenerator
    return MaskGenerator(
        input_size=input_size,
        patch_size=patch_size,
        num_blocks=num_blocks,
        masking_ratio=masking_ratio,
    )