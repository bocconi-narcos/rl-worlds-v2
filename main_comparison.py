import torch
import os
import wandb
import time
import copy
import json
import sys
from pathlib import Path


# Import functions from the src directory
from src.utils.config_utils import load_config
from src.utils.env_utils import get_env_details
from src.data_handling import prepare_dataloaders
from src.model_setup import initialize_models
from src.loss_setup import initialize_loss_functions
from src.optimizer_setup import initialize_optimizers
from src.training_engine import run_training_epochs

def create_aux_loss_config(base_config, aux_loss_type, aux_params, base_overrides=None):
    """Create a copy of the config with the specified auxiliary loss type and parameters."""
    config = copy.deepcopy(base_config)
    
    # Apply base overrides if provided
    if base_overrides:
        config = apply_config_overrides(config, base_overrides)
    
    # Update auxiliary loss configuration
    aux_loss_config = config.get('models', {}).get('auxiliary_loss', {})
    aux_loss_config['type'] = aux_loss_type
    aux_loss_config['params'] = aux_params
    
    return config

def apply_config_overrides(config, overrides):
    """Recursively apply overrides to the configuration."""
    for key, value in overrides.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key] = apply_config_overrides(config[key], value)
        else:
            config[key] = value
    return config

def run_single_experiment(config, experiment_name, device, wandb_run=None):
    """Run a single experiment with the given configuration."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Create experiment-specific directories
    model_dir = config.get('model_loading', {}).get('dir', 'trained_models/')
    experiment_model_dir = os.path.join(model_dir, experiment_name)
    os.makedirs(experiment_model_dir, exist_ok=True)
    
    # Update config to use experiment-specific model directory
    config['model_loading']['dir'] = experiment_model_dir
    
    # Initialize wandb for this experiment if enabled
    experiment_wandb_run = None
    if wandb_run and config.get('wandb', {}).get('enabled', False):
        try:
            experiment_wandb_run = wandb.init(
                project=config.get('wandb', {}).get('project'),
                entity=config.get('wandb', {}).get('entity'),
                name=f"{experiment_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=config,
                group="aux_loss_comparison"
            )
            print(f"Weights & Biases initialized for {experiment_name}.")
        except Exception as e:
            print(f"Error initializing Weights & Biases for {experiment_name}: {e}")
    
    # Get environment details
    env_config = config.get('environment', {})
    action_dim, action_type, observation_space = get_env_details(env_config.get('name'))
    
    # Prepare dataloaders
    data_config = config.get('data', {})
    validation_split = data_config.get('validation_split', 0.2)
    train_dataloader, val_dataloader = prepare_dataloaders(config, validation_split)
    if train_dataloader is None:
        print(f"Exiting {experiment_name} due to no training data.")
        return None
    
    dataloaders_map = {'train': train_dataloader}
    if val_dataloader:
        dataloaders_map['val'] = val_dataloader
    
    # Initialize models
    image_h = env_config.get('image_height')
    image_w = env_config.get('image_width')
    image_h_w = (image_h, image_w)
    
    from src.utils.config_utils import validate_environment_config
    validate_environment_config(config)
    
    input_channels = config.get('environment', {}).get('input_channels_per_frame', 3)
    if config.get('environment', {}).get('grayscale_conversion', False):
        input_channels = 1
    
    models_map = initialize_models(config, action_dim, action_type, device, image_h_w, input_channels)
    
    # Initialize losses and optimizers
    aux_loss_config = config.get('models', {}).get('auxiliary_loss', {})
    jepa_model_latent_dim_for_dino = None
    if models_map.get('jepa') and aux_loss_config.get('type') == 'dino':
        jepa_model_latent_dim_for_dino = config.get('models', {}).get('shared_latent_dim')
    
    losses_map = initialize_loss_functions(config, device, jepa_model_latent_dim=jepa_model_latent_dim_for_dino)
    optimizers_map = initialize_optimizers(models_map, config)
    
    # Run training
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
        jepa_loaded_successfully=False,
        wandb_run=experiment_wandb_run
    )
    
    # Load best models and set to eval mode
    jepa_model = models_map.get('jepa')
    if jepa_model:
        best_checkpoint_jepa = training_results.get("best_checkpoint_jepa")
        if best_checkpoint_jepa and os.path.exists(best_checkpoint_jepa):
            print(f"Loading best JEPA model for {experiment_name}")
            jepa_model.load_state_dict(torch.load(best_checkpoint_jepa, map_location=device))
        jepa_model.eval()
    
    # Finish wandb run
    if experiment_wandb_run:
        experiment_wandb_run.finish()
    
    return {
        'experiment_name': experiment_name,
        'aux_loss_type': aux_loss_config.get('type'),
        'training_results': training_results,
        'config': config
    }

def compare_experiments(results, comparison_config=None):
    """Compare and summarize the results from different experiments."""
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_data = []
    
    for result in results:
        if result is None:
            continue
            
        experiment_name = result['experiment_name']
        aux_loss_type = result['aux_loss_type']
        training_results = result['training_results']
        
        # Extract key metrics - training engine returns checkpoint paths, not loss values
        best_checkpoint_jepa = training_results.get('best_checkpoint_jepa')
        best_checkpoint_enc_dec = training_results.get('best_checkpoint_enc_dec')
        best_checkpoint_jepa_decoder = training_results.get('best_checkpoint_jepa_decoder')
        
        # Check if models were successfully saved (indicates training completed)
        jepa_trained = best_checkpoint_jepa is not None and os.path.exists(best_checkpoint_jepa)
        enc_dec_trained = best_checkpoint_enc_dec is not None and os.path.exists(best_checkpoint_enc_dec)
        decoder_trained = best_checkpoint_jepa_decoder is not None and os.path.exists(best_checkpoint_jepa_decoder)
        
        # Create a simple success metric
        training_success = "SUCCESS" if jepa_trained else "FAILED"
        
        comparison_data.append({
            'experiment': experiment_name,
            'aux_loss_type': aux_loss_type,
            'jepa_trained': jepa_trained,
            'enc_dec_trained': enc_dec_trained,
            'decoder_trained': decoder_trained,
            'training_success': training_success,
            'best_checkpoint_jepa': best_checkpoint_jepa,
            'best_checkpoint_enc_dec': best_checkpoint_enc_dec,
            'best_checkpoint_jepa_decoder': best_checkpoint_jepa_decoder
        })
        
        print(f"\n{experiment_name}:")
        print(f"  Auxiliary Loss: {aux_loss_type}")
        print(f"  Training Status: {training_success}")
        print(f"  JEPA Model Trained: {jepa_trained}")
        print(f"  Encoder-Decoder Trained: {enc_dec_trained}")
        print(f"  JEPA Decoder Trained: {decoder_trained}")
        if jepa_trained:
            print(f"  JEPA Checkpoint: {best_checkpoint_jepa}")
        if enc_dec_trained:
            print(f"  Enc-Dec Checkpoint: {best_checkpoint_enc_dec}")
        if decoder_trained:
            print(f"  Decoder Checkpoint: {best_checkpoint_jepa_decoder}")
    
    # Find best performing experiment (successful training)
    successful_experiments = [exp for exp in comparison_data if exp['jepa_trained']]
    if successful_experiments:
        best_experiment = successful_experiments[0]  # First successful one
        print(f"\n{'='*40}")
        print(f"BEST PERFORMING EXPERIMENT: {best_experiment['experiment']}")
        print(f"Auxiliary Loss: {best_experiment['aux_loss_type']}")
        print(f"Training Status: {best_experiment['training_success']}")
        print(f"{'='*40}")
    else:
        print(f"\n{'='*40}")
        print("NO SUCCESSFUL EXPERIMENTS")
        print("All experiments failed to train properly")
        print(f"{'='*40}")
    
    # Create output directory if specified
    output_dir = "comparison_results"
    if comparison_config and 'comparison' in comparison_config:
        output_dir = comparison_config['comparison'].get('output_dir', 'comparison_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, "experiment_comparison_results.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nComparison results saved to: {comparison_file}")
    
    # Create comparison summary text file
    summary_file = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("AUXILIARY LOSS COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for data in comparison_data:
            f.write(f"{data['experiment']}:\n")
            f.write(f"  Auxiliary Loss: {data['aux_loss_type']}\n")
            f.write(f"  Training Status: {data['training_success']}\n")
            f.write(f"  JEPA Model Trained: {data['jepa_trained']}\n")
            f.write(f"  Encoder-Decoder Trained: {data['enc_dec_trained']}\n")
            f.write(f"  JEPA Decoder Trained: {data['decoder_trained']}\n")
            if data['jepa_trained']:
                f.write(f"  JEPA Checkpoint: {data['best_checkpoint_jepa']}\n")
            f.write("\n")
        
        if successful_experiments:
            f.write(f"BEST PERFORMING: {best_experiment['experiment']}\n")
            f.write(f"Auxiliary Loss: {best_experiment['aux_loss_type']}\n")
            f.write(f"Training Status: {best_experiment['training_success']}\n")
        else:
            f.write("NO SUCCESSFUL EXPERIMENTS\n")
            f.write("All experiments failed to train properly\n")
    
    print(f"Comparison summary saved to: {summary_file}")
    
    return comparison_data

def main():
    """Main function to run the auxiliary loss comparison."""
    print("JEPA Auxiliary Loss Comparison")
    print("Comparing different auxiliary loss configurations")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("\nUsage:")
            print("  python main_comparison.py [config_file]")
            print("  python main_comparison.py config_comparison.yaml")
            print("  python main_comparison.py config_comparison_simple.yaml")
            print("\nAvailable config files:")
            print("  config_comparison.yaml - Full comparison with multiple experiments")
            print("  config_comparison_simple.yaml - Quick comparison (VICReg vs Barlow Twins)")
            return
        config_file = sys.argv[1]
    else:
        config_file = 'config_comparison.yaml'
    
    print(f"Using comparison config: {config_file}")
    
    # Load comparison configuration
    comparison_config = load_config(config_file)
    
    # Load base configuration
    base_config_path = comparison_config.get('base_config_path', 'config.yaml')
    base_config = load_config(base_config_path)
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # Initialize main wandb run for the comparison
    wandb_run = None
    wandb_cfg = base_config.get('wandb', {})
    if wandb_cfg.get('enabled', False):
        try:
            wandb_run = wandb.init(
                project=wandb_cfg.get('project'),
                entity=wandb_cfg.get('entity'),
                name=f"aux_loss_comparison-{time.strftime('%Y%m%d-%H%M%S')}",
                config=base_config
            )
            print("Main Weights & Biases run initialized for comparison.")
        except Exception as e:
            print(f"Error initializing main Weights & Biases run: {e}")
    
    # Get experiments from comparison config
    experiments = comparison_config.get('experiments', [])
    
    if not experiments:
        print("No experiments defined in config_comparison.yaml")
        return
    
    # Get base overrides from comparison config
    base_overrides = comparison_config.get('base_overrides', {})
    
    # Run experiments
    results = []
    for exp_config in experiments:
        experiment_name = exp_config['name']
        aux_loss_type = exp_config['aux_loss_type']
        aux_params = exp_config['aux_params']
        
        print(f"\nPreparing experiment: {experiment_name}")
        print(f"Auxiliary Loss Type: {aux_loss_type}")
        
        config = create_aux_loss_config(base_config, aux_loss_type, aux_params, base_overrides)
        result = run_single_experiment(config, experiment_name, device, wandb_run)
        results.append(result)
    
    # Compare results
    compare_experiments(results, comparison_config)
    
    # Finish main wandb run
    if wandb_run:
        wandb_run.finish()
        print("Main Weights & Biases run finished.")

if __name__ == '__main__':
    main() 