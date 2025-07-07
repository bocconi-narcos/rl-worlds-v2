import torch
import os # For path joining and checking existence
import wandb
import time

from src.utils.config_utils import load_config
from src.utils.env_utils import get_env_details
from src.data_handling import prepare_dataloaders
from src.model_setup import initialize_models
from src.loss_setup import initialize_loss_functions
from src.optimizer_setup import initialize_optimizers
from src.training_engine import run_training_epochs
from src.training_loops.vjepa2_world_model_loop import VJEPA2WorldModelTrainer

def main():
    # 1. Load Configuration
    config = load_config()

    # Initialize wandb
    wandb_cfg = config.get('wandb', {})
    wandb_run = None
    if wandb_cfg.get('enabled', False):
        try:
            wandb_run = wandb.init(
                project=wandb_cfg.get('project'),
                entity=wandb_cfg.get('entity'),
                name=f"{wandb_cfg.get('run_name_prefix', 'exp')}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=config
            )
            print("Weights & Biases initialized successfully.")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}. Proceeding without W&B.")
            wandb_run = None
    else:
        print("Weights & Biases is disabled in the configuration.")

    # 2. Setup Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Get directories from the new config structure
    model_dir = config.get('model_loading', {}).get('dir', 'trained_models/')
    dataset_dir = config.get('data', {}).get('dataset', {}).get('dir', 'datasets/')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Ensured model directory exists: {model_dir}")
    print(f"Ensured dataset directory exists: {dataset_dir}")
    print('--' * 40)

    # 3. Get Environment Details
    print("\nFetching environment details...")
    env_config = config.get('environment', {})
    action_dim, action_type, observation_space = get_env_details(env_config.get('name'))
    print('--' * 40)

    # 4. Prepare Dataloaders
    print("\nPreparing dataloaders...")
    data_config = config.get('data', {})
    validation_split = data_config.get('validation_split', 0.2)
    dataloaders_map = {}
    train_dataloader, val_dataloader = prepare_dataloaders(config, validation_split)
    if train_dataloader is None:
        print("Exiting due to no training data.")
        return
    dataloaders_map['train'] = train_dataloader
    if val_dataloader:
        dataloaders_map['val'] = val_dataloader
    print('--' * 40)

    # 5. Train VJEPA2 (Stage 1 & 2)
    print("\nTraining VJEPA2 (Stage 1 & 2)...")
    vjepa2_trainer = VJEPA2WorldModelTrainer(
        config=config,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    vjepa2_trainer.train()  # Trains both stages
    vjepa2_models = vjepa2_trainer.get_trained_models()
    vjepa2_encoder = vjepa2_models['encoder']
    vjepa2_world_model = vjepa2_models['world_model']
    print('--' * 40)

    # Freeze VJEPA2 encoder and world model parameters for downstream use
    for param in vjepa2_encoder.parameters():
        param.requires_grad = False
    for param in vjepa2_world_model.parameters():
        param.requires_grad = False
    vjepa2_encoder.eval()
    vjepa2_world_model.eval()

    # 6. Initialize the rest of the pipeline (reward MLPs, LARP, etc.)
    # Use the VJEPA2 encoder and world model as the "JEPA" model for downstream tasks
    # We'll create a wrapper to mimic the JEPA interface for downstream code
    class VJEPA2JEPAWrapper(torch.nn.Module):
        def __init__(self, encoder, world_model, action_type):
            super().__init__()
            self.encoder = encoder
            self.world_model = world_model
            self.action_type = action_type
            # For compatibility with downstream code
            self.action_embedding = getattr(world_model, 'action_embedding', None)
        def forward(self, s_t, a_t, s_t_plus_1):
            # Flatten s_t and s_t_plus_1 if they are 5D (B, F, C, H, W)
            if s_t.dim() == 5:
                B, F, C, H, W = s_t.shape
                s_t = s_t.view(B, F * C, H, W)
            if s_t_plus_1.dim() == 5:
                B, F, C, H, W = s_t_plus_1.shape
                s_t_plus_1 = s_t_plus_1.view(B, F * C, H, W)
            z_t = self.encoder(s_t)
            z_t_plus_1 = self.encoder(s_t_plus_1)
            z_t_seq = z_t.unsqueeze(1)  # [B, 1, D]
            a_t_seq = a_t.unsqueeze(1) if a_t.dim() == 1 else a_t.unsqueeze(1)  # [B, 1] or [B, 1, A]
            pred_z_t_plus_1_seq = self.world_model(z_t_seq, a_t_seq)  # [B, 1, D]
            pred_z_t_plus_1 = pred_z_t_plus_1_seq.squeeze(1)  # [B, D]
            # Return in the same tuple format as JEPA: (pred, target, online, target_online)
            return pred_z_t_plus_1, z_t_plus_1.detach(), z_t, z_t_plus_1

    # Replace JEPA model in models_map with VJEPA2 wrapper
    # Initialize other models as before
    image_h = env_config.get('image_height')
    image_w = env_config.get('image_width')
    image_h_w = (image_h, image_w)
    from src.utils.config_utils import get_effective_input_channels, validate_environment_config
    validate_environment_config(config)
    input_channels = config.get('environment', {}).get('input_channels_per_frame', 3)
    if config.get('environment', {}).get('grayscale_conversion', False):
        input_channels = 1
    models_map = initialize_models(config, action_dim, action_type, device, image_h_w, input_channels)
    models_map['jepa'] = VJEPA2JEPAWrapper(vjepa2_encoder, vjepa2_world_model, action_type)

    # Losses and optimizers as before
    jepa_model_latent_dim_for_dino = None
    aux_loss_config = config.get('models', {}).get('auxiliary_loss', {})
    if aux_loss_config.get('type') == 'dino':
        jepa_model_latent_dim_for_dino = config.get('models', {}).get('shared_latent_dim')
    losses_map = initialize_loss_functions(config, device, jepa_model_latent_dim=jepa_model_latent_dim_for_dino)
    optimizers_map = initialize_optimizers(models_map, config)

    # 7. Run Training Epochs for downstream tasks (reward MLPs, LARP, etc.)
    training_results = run_training_epochs(
        models_map=models_map,
        optimizers_map=optimizers_map,
        losses_map=losses_map,
        dataloaders_map=dataloaders_map,
        device=device,
        config=config,
        action_dim=action_dim,
        action_type=action_type,
        image_h_w=image_h_w,
        input_channels=input_channels,
        std_enc_dec_loaded_successfully=False,
        jepa_loaded_successfully=True,  # VJEPA2 is trained
        wandb_run=wandb_run
    )

    # 8. Post-Training: Load best models and set to eval mode (same as before)
    print("\nLoading best models (if available) after training and setting to eval mode...")

    # Standard Encoder/Decoder
    std_enc_dec = models_map.get('std_enc_dec')
    if std_enc_dec:
        best_checkpoint_enc_dec_path = training_results.get("best_checkpoint_enc_dec")
        if best_checkpoint_enc_dec_path and os.path.exists(best_checkpoint_enc_dec_path):
            print(f"Loading best Encoder/Decoder model from {best_checkpoint_enc_dec_path}")
            std_enc_dec.load_state_dict(torch.load(best_checkpoint_enc_dec_path, map_location=device))
        std_enc_dec.eval()

    # JEPA Model
    jepa_model = models_map.get('jepa')
    if jepa_model:
        best_checkpoint_jepa_path = training_results.get("best_checkpoint_jepa")
        if best_checkpoint_jepa_path and os.path.exists(best_checkpoint_jepa_path):
            print(f"Loading best JEPA model from {best_checkpoint_jepa_path}")
            jepa_model.load_state_dict(torch.load(best_checkpoint_jepa_path, map_location=device))
        jepa_model.eval()

    # JEPA State Decoder
    jepa_decoder = models_map.get('jepa_decoder')
    jepa_decoder_training_config = config.get('models', {}).get('jepa', {}).get('decoder_training', {})
    if jepa_decoder and jepa_decoder_training_config.get('enabled', False):
        best_checkpoint_jepa_decoder_path = training_results.get("best_checkpoint_jepa_decoder")
        print(f"Attempting to load best JEPA State Decoder (if available) after training...")
        if best_checkpoint_jepa_decoder_path and os.path.exists(best_checkpoint_jepa_decoder_path):
            print(f"Loading best JEPA State Decoder model from {best_checkpoint_jepa_decoder_path}")
            jepa_decoder.load_state_dict(torch.load(best_checkpoint_jepa_decoder_path, map_location=device))
        else:
            print(f"No best checkpoint found for JEPA State Decoder at expected path. Model remains in its last training state (if any training occurred).")
        jepa_decoder.eval()
    elif jepa_decoder :
        print("JEPA State Decoder was initialized but not enabled for training. Setting to eval mode.")
        jepa_decoder.eval()

    # Reward MLPs
    if models_map.get('reward_mlp_enc_dec'):
        models_map['reward_mlp_enc_dec'].eval()
        print("Encoder-Decoder Reward MLP set to eval mode.")
    if models_map.get('reward_mlp_jepa'):
        models_map['reward_mlp_jepa'].eval()
        print("JEPA Reward MLP set to eval mode.")

    # LARP Models eval mode
    if models_map.get('larp_enc_dec'):
        models_map['larp_enc_dec'].eval()
        print("Encoder-Decoder LARP set to eval mode.")
    if models_map.get('larp_jepa'):
        models_map['larp_jepa'].eval()
        print("JEPA LARP set to eval mode.")

    if losses_map.get('aux_fn') and hasattr(losses_map['aux_fn'], 'eval'):
        losses_map['aux_fn'].eval()
        print("Auxiliary loss function (if DINO) set to eval mode.")

    print("\nProcess complete. Relevant models are in eval mode.")

    if wandb_run:
        wandb_run.finish()
        print("Weights & Biases run finished.")

if __name__ == '__main__':
    main() 