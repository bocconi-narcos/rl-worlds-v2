# JEPA Auxiliary Loss Comparison

This directory contains functionality to compare JEPA models trained with different auxiliary losses (VICReg, Barlow Twins, DINO).

## Quick Start

### Option 1: Using the comparison script directly
```bash
python main_comparison.py
```

### Option 2: Using the main script with comparison flag
```bash
python main.py --compare
```

## Configuration

The comparison is configured through `config_comparison.yaml`. This file allows you to:

1. **Define experiments**: Specify different auxiliary loss types and their parameters
2. **Override base config**: Modify training parameters for faster comparison
3. **Control output**: Specify where results are saved

### Example Configuration

```yaml
# config_comparison.yaml
base_config_path: "config.yaml"

experiments:
  - name: "jepa_vicreg"
    aux_loss_type: "vicreg"
    aux_params:
      vicreg:
        sim_coeff: 1.0
        std_coeff: 1.0
        cov_coeff: 0.8
        eps: 0.0001
        proj_hidden_dim: 256
        proj_output_dim: 256

  - name: "jepa_barlow_twins"
    aux_loss_type: "barlow_twins"
    aux_params:
      barlow_twins:
        lambda_param: 0.0051
        eps: 0.00001
        scale_loss: 1.0

base_overrides:
  training:
    num_epochs: 10  # Reduced for faster comparison
  
  models:
    standard_encoder_decoder:
      enabled: false  # Focus only on JEPA
    reward_predictors:
      reward_mlp:
        enabled: false
      larp:
        enabled: false
    jepa:
      decoder_training:
        enabled: false
```

## Available Auxiliary Losses

### VICReg
- **Purpose**: Variance-Invariance-Covariance Regularization
- **Parameters**:
  - `sim_coeff`: Similarity coefficient (invariance term)
  - `std_coeff`: Standard deviation coefficient (variance term)
  - `cov_coeff`: Covariance coefficient (covariance term)
  - `eps`: Numerical stability epsilon
  - `proj_hidden_dim`: Projector hidden dimension
  - `proj_output_dim`: Projector output dimension

### Barlow Twins
- **Purpose**: Cross-correlation matrix regularization
- **Parameters**:
  - `lambda_param`: Weight for redundancy reduction term
  - `eps`: Numerical stability epsilon
  - `scale_loss`: Overall loss scaling factor

### DINO
- **Purpose**: Centering mechanism to prevent collapse
- **Parameters**:
  - `center_ema_decay`: EMA decay rate for center updates
  - `eps`: Numerical stability epsilon

## Output

The comparison generates:

1. **Individual experiment results**: Each experiment gets its own model directory
2. **Comparison summary**: JSON and text files with comparison results
3. **WandB logs**: Separate runs for each experiment (if enabled)

### Output Structure
```
comparison_results/
├── experiment_comparison_results.json
├── comparison_summary.txt
└── trained_models/
    ├── jepa_vicreg/
    ├── jepa_barlow_twins/
    └── ...
```

## Customizing Experiments

To add new experiments:

1. **Add to config_comparison.yaml**:
```yaml
experiments:
  - name: "your_experiment_name"
    aux_loss_type: "vicreg"  # or "barlow_twins", "dino"
    aux_params:
      vicreg:
        # Your custom parameters
```

2. **Modify base overrides** if needed:
```yaml
base_overrides:
  training:
    num_epochs: 5  # Even faster comparison
```

## Tips

1. **Start small**: Use fewer epochs and disable unnecessary components for quick testing
2. **Monitor resources**: Each experiment runs independently, so ensure you have enough memory/GPU
3. **Use WandB**: Enable WandB logging to track training curves and compare performance
4. **Check results**: Review the comparison summary to identify the best performing configuration

## Troubleshooting

- **No experiments defined**: Ensure `config_comparison.yaml` has an `experiments` section
- **Missing base config**: Check that `base_config_path` points to a valid config file
- **Memory issues**: Reduce batch size or number of experiments in the base config 