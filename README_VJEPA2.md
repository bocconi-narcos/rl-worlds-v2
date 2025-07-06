# V-JEPA2 World Model Implementation

This repository contains a complete implementation of a two-stage V-JEPA2-inspired world model for model-based reinforcement learning.

## Architecture Overview

### Stage 1: Self-Supervised Encoder Pretraining
- **MaskedViT**: Vision Transformer with random masking support
- **MaskedPredictionModel**: Complete model with online encoder, predictor, and EMA target encoder
- **MaskedPredictionLoss**: L1 loss between predicted and target embeddings

### Stage 2: Action-Conditioned World Model Training
- **WorldModelTransformer**: Transformer that processes (z_t, a_t) sequences
- **WorldModelLoss**: Combined teacher forcing and rollout loss
- **Frozen Encoder**: Pretrained encoder from Stage 1

## Usage

### Training
```bash
# Train both stages
python train_vjepa2_world_model.py

# Train only Stage 1
python train_vjepa2_world_model.py --stage 1

# Train only Stage 2
python train_vjepa2_world_model.py --stage 2
```

### Evaluation
```bash
python evaluate_vjepa2_world_model.py
```

## Key Features
- Random masking for self-supervised pretraining
- EMA target encoder for stable training
- Learnable mask tokens
- Transformer-based world model
- Teacher forcing + rollout training
- Full Weights & Biases integration
- Comprehensive evaluation and visualization

## Configuration
Edit `config.yaml` to customize training parameters:
```yaml
vjepa2_world_model:
  stage1:
    masking_ratio: 0.75
    num_epochs: 100
    learning_rate: 0.0003
    
  stage2:
    sequence_length: 8
    teacher_forcing_ratio: 0.8
    rollout_steps: 4
    num_epochs: 50
```

## Model Architecture
- **MaskedViT**: Extends ViT with random patch masking
- **MaskedPredictionModel**: Implements V-JEPA2 Stage 1 architecture
- **WorldModelTransformer**: Action-conditioned sequence modeling
- **Loss Functions**: L1 loss for Stage 1, combined loss for Stage 2

## Evaluation Metrics
- Stage 1: Masked prediction accuracy across different masking ratios
- Stage 2: World model prediction loss and rollout accuracy
- Visualization: Latent space trajectories and prediction errors

## Integration
Seamlessly integrates with existing codebase:
- Reuses ViT, MLP, and training infrastructure
- Compatible with existing data pipeline
- Full configuration system support
- Weights & Biases logging integration 