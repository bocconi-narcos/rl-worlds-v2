# Contents for src/model_setup.py
from src.models.encoder_decoder import StandardEncoderDecoder
from src.models.jepa import JEPA
from src.models.mlp import RewardPredictorMLP
from src.models.decoder import StateDecoder
from src.models.encoder_decoder_jepa_style import EncoderDecoderJEPAStyle
from copy import deepcopy
from src.models.larp_mlp import LookAheadRewardPredictorMLP # Added for LARP
from src.utils.larp_utils import calculate_larp_input_dim_enc_dec, calculate_larp_input_dim_jepa # Added for LARP
from src.utils.weight_init import print_num_parameters
from src.utils.config_utils import get_single_frame_channels

def initialize_models(config, action_dim, action_type, device, image_h_w, input_channels): # Added action_type
    models = {}

    # Calculate output channels for single frame (next state)
    output_channels = input_channels
    print(f"Model initialization: input_channels={input_channels} (stacked frames), output_channels={output_channels} (single frame)")

    # Load Reward Predictor Configurations
    reward_pred_config = config.get('models', {}).get('reward_predictors', {})
    enc_dec_mlp_config = reward_pred_config.get('reward_mlp', {})
    jepa_mlp_config = reward_pred_config.get('reward_mlp', {})
    num_frames = config.get('environment', {}).get('frame_stack_size')  # Default to 1 if not set

    # Encoder configuration
    models_config = config.get('models', {})
    encoder_config = models_config.get('encoder', {})

    encoder_type = encoder_config.get('type')
    all_encoder_params_from_config = encoder_config.get('params', {})
    specific_encoder_params = all_encoder_params_from_config.get(encoder_type, {})
    if specific_encoder_params is None:
        specific_encoder_params = {}

    global_patch_size = models_config.get('shared_patch_size')
    shared_latent_dim = models_config.get('shared_latent_dim')

    if encoder_type == 'vit':
        if 'patch_size' not in specific_encoder_params or specific_encoder_params['patch_size'] is None:
            specific_encoder_params['patch_size'] = global_patch_size
        # Use the specific encoder params from the new config structure
        specific_encoder_params.setdefault('depth', specific_encoder_params.get('depth', 6))
        specific_encoder_params.setdefault('heads', specific_encoder_params.get('heads', 8))
        specific_encoder_params.setdefault('mlp_dim', specific_encoder_params.get('mlp_dim', 1024))
        #specific_encoder_params.setdefault('pool', specific_encoder_params.get('pool', 'cls'))
        specific_encoder_params.setdefault('dropout', specific_encoder_params.get('dropout', 0.0))
        specific_encoder_params.setdefault('emb_dropout', specific_encoder_params.get('emb_dropout', 0.0))
        specific_encoder_params.setdefault('num_frames', num_frames)  # Ensure frames is set for ViT

    print(f"Initializing Standard Encoder-Decoder Model with {encoder_type.upper()} encoder...")
    std_enc_dec_config = models_config.get('standard_encoder_decoder', {})

    # Read the variant from the config
    encoder_decoder_variant = std_enc_dec_config.get('variant', 'standard')

    jepa_config = models_config.get('jepa', {}) # Used by both variants for some params
    
    jepa_state_decoder_arch_config = std_enc_dec_config # We want the JEPAStateDecoder to have the same architecture as the standard encoder-decoder
    jepa_state_decoder_arch_config.pop('variant', None)  # Remove variant if it exists, to avoid confusion
    jepa_state_decoder_arch_config['input_latent_dim'] = shared_latent_dim  # Ensure input_latent_dim is set for JEPAStateDecoder

    if encoder_decoder_variant == 'jepa_style':
        print(f"Selected Encoder-Decoder Variant: JEPA-Style with {encoder_type.upper()} encoder.")
        # Parameters for EncoderDecoderJEPAStyle, sourcing from various config sections as per comments in config.yaml
        # Encoder params are already prepared (encoder_type, specific_encoder_params, shared_latent_dim)
        # Action embedding dim from standard_encoder_decoder config (convention)
        action_emb_dim_jepa_style = std_enc_dec_config.get('action_emb_dim', shared_latent_dim)
        
        # Predictor params from jepa config
        predictor_hidden_dims_jepa_style = jepa_config.get('predictor_hidden_dims', [256, 256])
        predictor_dropout_rate_jepa_style = jepa_config.get('predictor_dropout_rate')
        # Predictor output_dim for enc_dec_jepa_style is the input_latent_dim of its internal JEPAStateDecoder
        # This comes from jepa_state_decoder_arch.input_latent_dim
        predictor_output_dim_jepa_style = jepa_state_decoder_arch_config.get('input_latent_dim', shared_latent_dim)

        # Internal JEPAStateDecoder architectural params from jepa_state_decoder_arch config
        # Note: input_latent_dim for this internal decoder is predictor_output_dim_jepa_style
        jepa_decoder_dim_internal = jepa_state_decoder_arch_config.get('decoder_dim')
        jepa_decoder_depth_internal = jepa_state_decoder_arch_config.get('decoder_depth')
        jepa_decoder_heads_internal = jepa_state_decoder_arch_config.get('decoder_heads')
        jepa_decoder_mlp_dim_internal = jepa_state_decoder_arch_config.get('decoder_mlp_dim')
        jepa_decoder_dropout_internal = jepa_state_decoder_arch_config.get('decoder_dropout')
        
        # Decoder patch size: from jepa_state_decoder_arch, fallback to global_patch_size
        jepa_decoder_patch_size_internal = jepa_state_decoder_arch_config.get('decoder_patch_size', global_patch_size)

        # Output channels and image size for the final output (from environment config, passed as args)
        # output_channels_internal = input_channels (function arg)
        # output_image_size_internal = image_h_w (function arg)

        std_enc_dec = EncoderDecoderJEPAStyle(
            image_size=image_h_w,
            patch_size=global_patch_size, # For ViT encoder and default for decoder patch size
            input_channels=input_channels,
            action_dim=action_dim,
            action_emb_dim=action_emb_dim_jepa_style,
            action_type=action_type, # Added action_type
            latent_dim=shared_latent_dim, # Encoder output latent dim

            predictor_hidden_dims=predictor_hidden_dims_jepa_style,
            predictor_output_dim=predictor_output_dim_jepa_style, # This is input to internal JEPAStateDecoder
            predictor_dropout_rate=predictor_dropout_rate_jepa_style,

            # Internal JEPAStateDecoder architecture params
            jepa_decoder_dim=jepa_decoder_dim_internal,
            jepa_decoder_depth=jepa_decoder_depth_internal,
            jepa_decoder_heads=jepa_decoder_heads_internal,
            jepa_decoder_mlp_dim=jepa_decoder_mlp_dim_internal,
            jepa_decoder_dropout=jepa_decoder_dropout_internal,
            jepa_decoder_patch_size=jepa_decoder_patch_size_internal,

            output_channels=output_channels, # Final output channels (single frame)
            output_image_size=image_h_w,

            encoder_type=encoder_type,
            encoder_params=specific_encoder_params
        ).to(device)

    elif encoder_decoder_variant == 'standard':
        print(f"Selected Encoder-Decoder Variant: Standard with {encoder_type.upper()} encoder.")
        std_enc_dec = StandardEncoderDecoder(
            image_size=image_h_w,
            patch_size=global_patch_size, # For ViT encoder and default for decoder patch size
            input_channels=input_channels,
            action_dim=action_dim,
            # StandardEncoderDecoder might also need action_type if it uses actions.
            # Assuming it does for now, or this will be unused but harmless.
            action_type=action_type, # Added action_type
            action_emb_dim=std_enc_dec_config.get('action_emb_dim', shared_latent_dim),
            latent_dim=shared_latent_dim,
            decoder_dim=std_enc_dec_config.get('decoder_dim', 128),
            decoder_depth=std_enc_dec_config.get('decoder_depth', 3),
            decoder_heads=std_enc_dec_config.get('decoder_heads', 6),
            decoder_mlp_dim=std_enc_dec_config.get('decoder_mlp_dim', 256),
            output_channels=output_channels,
            output_image_size=image_h_w,
            decoder_dropout=std_enc_dec_config.get('decoder_dropout', 0.0),
            encoder_type=encoder_type,
            encoder_params=specific_encoder_params,
            decoder_patch_size=std_enc_dec_config.get('decoder_patch_size', global_patch_size)
        ).to(device)
    else:
        raise ValueError(f"Unknown encoder_decoder_variant: {encoder_decoder_variant} in config.yaml")

    models['std_enc_dec'] = std_enc_dec # Store the selected model
    print('--' * 40)

    print(f"\nInitializing JEPA Model with {encoder_type.upper()} encoder...")
    # jepa_config already fetched
    jepa_model = JEPA(
        image_size=image_h_w,
        patch_size=global_patch_size,
        input_channels=input_channels,
        action_dim=action_dim,
        action_emb_dim=std_enc_dec_config.get('action_emb_dim', shared_latent_dim), # Or jepa_config's action_emb_dim if different
        action_type=action_type, # Added action_type
        latent_dim=shared_latent_dim,
        predictor_hidden_dims=jepa_config.get('predictor_hidden_dims'),
        ema_decay=jepa_config.get('ema_decay', 0.996),
        encoder_type=encoder_type,
        encoder_params=specific_encoder_params,
        predictor_dropout_rate=jepa_config.get('predictor_dropout_rate', 0.0)
    ).to(device)
    models['jepa'] = jepa_model

    vicreg_param_config = models_config.get('auxiliary_loss', {}).get('params', {}).get('vicreg', {})
    vicreg_proj_hidden_dim = vicreg_param_config.get('proj_hidden_dim')
    vicreg_proj_output_dim = vicreg_param_config.get('proj_output_dim')
    num_params_first_layer = shared_latent_dim * vicreg_proj_hidden_dim + vicreg_proj_hidden_dim
    num_params_second_layer = vicreg_proj_hidden_dim * vicreg_proj_output_dim + vicreg_proj_output_dim
    total_vicreg_proj_params = num_params_first_layer + num_params_second_layer
    print(f'{"Additional parameters in VICReg Projector:":<65}{total_vicreg_proj_params:,}')

    print('--' * 40)
    reward_mlp_enc_dec = None
    if enc_dec_mlp_config.get('enabled', False):
        print("\nInitializing Reward MLP for Encoder-Decoder...")
        if enc_dec_mlp_config.get('input_type') == "flatten":
            input_dim_enc_dec = shared_latent_dim + action_emb_dim_jepa_style # Flattened input: latent_dim + action_dim
        else:
            print(f"Warning: encoder_decoder_reward_mlp input_type is '{enc_dec_mlp_config.get('input_type')}'. Defaulting to flattened image dim.")
            input_dim_enc_dec = shared_latent_dim + action_emb_dim_jepa_style
        
        reward_mlp_enc_dec = RewardPredictorMLP(
            input_dim=input_dim_enc_dec,
            hidden_dims=enc_dec_mlp_config.get('hidden_dims', [128, 64]),
            activation_fn_str=enc_dec_mlp_config.get('activation', 'relu'),
            use_batch_norm=enc_dec_mlp_config.get('use_batch_norm', False),
            dropout_rate=enc_dec_mlp_config.get('dropout_rate', 0.0) # Added
        ).to(device)
    models['reward_mlp_enc_dec'] = reward_mlp_enc_dec
    print_num_parameters(reward_mlp_enc_dec, check_total=False)

    print('--' * 40)
    reward_mlp_jepa = None
    if jepa_mlp_config.get('enabled', False):
        print("\nInitializing Reward MLP for JEPA...")
        # copy the reward predictor for encoder-decoder
        reward_mlp_jepa = deepcopy(reward_mlp_enc_dec)
        print("Using deepcopy of reward predictor from Encoder-Decoder.")

    models['reward_mlp_jepa'] = reward_mlp_jepa
    print('--' * 40)

    # --- Initialize LARP Models ---
    larp_config_main = reward_pred_config.get('larp', {})
    enc_dec_larp_config = larp_config_main
    jepa_larp_config = larp_config_main

    larp_enc_dec = None
    if enc_dec_larp_config.get('enabled', False) and std_enc_dec:
        print("\nInitializing Look Ahead Reward Predictor (LARP) for Encoder-Decoder...")
        # Determine if the std_enc_dec is 'standard' or 'jepa_style' for correct dim calculation
        # The 'encoder_decoder_variant' variable holds this information from earlier in the function
        
        larp_input_dim_enc_dec = calculate_larp_input_dim_enc_dec(
            config=config, # Full config
            encoder_decoder_variant=encoder_decoder_variant,
            image_h_w=image_h_w,
            input_channels=input_channels
        )
        larp_enc_dec = LookAheadRewardPredictorMLP(
            input_dim=larp_input_dim_enc_dec,
            hidden_dims=enc_dec_larp_config.get('hidden_dims'),
            activation_fn_str=enc_dec_larp_config.get('activation', 'relu'),
            use_batch_norm=enc_dec_larp_config.get('use_batch_norm', False),
            dropout_rate=enc_dec_larp_config.get('dropout_rate', 0.0)
        ).to(device)
    models['larp_enc_dec'] = larp_enc_dec
    print('--' * 40)

    larp_jepa = None
    if jepa_larp_config.get('enabled', False) and jepa_model:
        print("\nInitializing Look Ahead Reward Predictor (LARP) for JEPA...")
        larp_input_dim_jepa = calculate_larp_input_dim_jepa(config=config) # Full config
        larp_jepa = LookAheadRewardPredictorMLP(
            input_dim=larp_input_dim_jepa,
            hidden_dims=jepa_larp_config.get('hidden_dims'),
            activation_fn_str=jepa_larp_config.get('activation', 'relu'),
            use_batch_norm=jepa_larp_config.get('use_batch_norm', False),
            dropout_rate=jepa_larp_config.get('dropout_rate', 0.0)
        ).to(device)
    models['larp_jepa'] = larp_jepa
    print('--' * 40)

    # Initialize JEPA State Decoder
    jepa_decoder_config = jepa_config.get('decoder_training', {})
    if jepa_decoder_config.get('enabled', False):
        print("\nInitializing JEPA State Decoder...")

        # Ensure image_h_w is a tuple for JEPAStateDecoder
        current_image_h_w = image_h_w

        jepa_decoder = StateDecoder(
            input_latent_dim=shared_latent_dim, # JEPA's predictor output dim
            decoder_dim=std_enc_dec_config.get('decoder_dim', 128),
            decoder_depth=std_enc_dec_config.get('decoder_depth', 3),
            decoder_heads=std_enc_dec_config.get('decoder_heads', 4),
            decoder_mlp_dim=std_enc_dec_config.get('decoder_mlp_dim', 256),
            output_channels=output_channels,
            output_image_size=current_image_h_w,
            decoder_dropout=std_enc_dec_config.get('decoder_dropout', 0.0),
            decoder_patch_size=std_enc_dec_config.get('decoder_patch_size', global_patch_size) # Default to global patch_size if specific not found
        ).to(device)
        models['jepa_decoder'] = jepa_decoder
        print_num_parameters(jepa_decoder, check_total=False)
    else:
        models['jepa_decoder'] = None
        print("JEPA State Decoder is disabled in the configuration.")

    # The EncoderDecoderJEPAStyle model is now initialized conditionally above,
    # and stored in models['std_enc_dec'] if selected.
    # The old models['enc_dec_jepa_style'] key is no longer used for this purpose.
    # If a separate instance for comparison is ever needed, it would be re-added here,
    # but for now, the logic handles selecting one or the other as the primary 'std_enc_dec'.

    

    return models
