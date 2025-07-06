# V-JEPA2 World Model Implementation

This repository contains a complete implementation of a two-stage V-JEPA2-inspired world model for model-based reinforcement learning. The implementation follows the V-JEPA2 architecture with masked prediction pretraining followed by action-conditioned world model training.

## 🎯 Overview

V-JEPA2 (Video Joint Embedding Predictive Architecture) is a self-supervised learning approach that learns world models through two distinct stages:

1. **Stage 1**: Self-supervised encoder pretraining with masked prediction
2. **Stage 2**: Action-conditioned world model training with temporal sequences

## 🏗️ Stage 1: Self-Supervised Encoder Pretraining

### **Objective**
Learn meaningful visual representations by predicting masked regions in images using only visible context.

### **Architecture Components**

```
Input: s_t (4 stacked frames)
    ↓
┌─────────────────┬─────────────────┐
│   Online Path   │   Target Path   │
│                 │                 │
│ s_t (masked)    │ s_t (unmasked)  │
│     ↓           │     ↓           │
│ MaskedViT       │ MaskedViT       │
│     ↓           │     ↓           │
│ masked_latent   │ target_latent   │
│     ↓           │                 │
│ Predictor       │                 │
│     ↓           │                 │
│ predicted_latent│                 │
└─────────────────┴─────────────────┘
                ↓
        L1 Loss Comparison
```

### **Detailed Process**

1. **Input Processing**:
   - Take current state `s_t` (4 stacked frames: [4, 1, 64, 64])
   - Apply random masking to 40% of image patches
   - Replace masked patches with learnable mask tokens

2. **Dual Encoder Setup**:
   - **Online Encoder**: Processes masked image, outputs `masked_latent` [B, D]
   - **Target Encoder**: Processes unmasked image, outputs `target_latent` [B, D]
   - Target encoder is EMA-updated from online encoder (frozen during training)

3. **Prediction**:
   - **Predictor**: MLP that takes `masked_latent` and predicts `predicted_latent`
   - Goal: Reconstruct the full latent representation from partial information

4. **Loss Computation**:
   ```
   Loss = L1(predicted_latent, target_latent)
   ```

### **Key Features**
- **Random Masking**: 40% of patches randomly masked (configurable)
- **Learnable Mask Tokens**: Replaces masked patches with learned embeddings
- **EMA Target**: Exponential moving average of online encoder for stable training
- **Self-supervised**: No external labels required

### **Training Parameters**
```yaml
stage1:
  masking_ratio: 0.4          # 40% of patches masked
  num_epochs: 10              # Training epochs
  learning_rate: 0.0001       # Learning rate
  ema_decay: 0.996           # EMA update rate
  weight_decay: 0.05         # Weight decay
```

## 🚀 Stage 2: Action-Conditioned World Model Training

### **Objective**
Learn temporal dynamics by predicting future latent states given current states and actions.

### **Architecture Components**

```
Temporal Sequence: (s_0...s_7, a_0...a_7, r_0...r_7, s_1...s_8)
    ↓
┌─────────────────────────────────────┐
│           Frozen Encoder            │
│  (from Stage 1, no gradients)       │
│                                     │
│ s_t → z_t [B, T, D]                │
│ s_t+1 → z_t+1 [B, T, D]            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      WorldModelTransformer          │
│                                     │
│ Input: (z_t, a_t) [B, T, D+A]      │
│ Output: predicted_z_t+1 [B, T, D]   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         WorldModelLoss              │
│                                     │
│ Teacher Forcing: z_t+1 prediction   │
│ Rollout: Multi-step prediction      │
└─────────────────────────────────────┘
```

### **Detailed Process**

1. **Temporal Sequence Creation**:
   - Create sequences of 8 consecutive transitions
   - Input: `(s_0...s_7, a_0...a_7, r_0...r_7, s_1...s_8)`
   - Each sequence contains real temporal dynamics

2. **Encoding**:
   - **Frozen Encoder**: Encode all frames in sequence to latents
   - Input: `[B, T, F*C, H, W]` → Output: `[B, T, D]`
   - No gradients computed (encoder frozen from Stage 1)

3. **World Model Prediction**:
   - **WorldModelTransformer**: Takes `(latents, actions)` and predicts next latents
   - Input: `[B, T, D]` latents + `[B, T]` actions
   - Output: `[B, T, D]` predicted next latents

4. **Loss Computation**:
   ```
   Teacher Forcing Loss = L1(predicted_z_t+1, target_z_t+1)
   Rollout Loss = L1(predicted_z_t+1, target_z_t+1)  # Multi-step
   Total Loss = 0.7 * TF_Loss + 0.3 * Rollout_Loss
   ```

### **Key Features**
- **Temporal Sequences**: Real 8-step sequences from dataset
- **Frozen Encoder**: Uses Stage 1 encoder without fine-tuning
- **Teacher Forcing**: Mix of supervised and rollout training
- **Action Conditioning**: Actions influence predictions

### **Training Parameters**
```yaml
stage2:
  sequence_length: 8          # Temporal sequence length
  teacher_forcing_ratio: 0.8  # 80% teacher forcing
  rollout_steps: 4           # Multi-step prediction
  num_epochs: 30             # Training epochs
  learning_rate: 0.0001      # Learning rate
  weight_decay: 0.05         # Weight decay
```

## 📊 Performance Benchmarks

### **Stage 1 Benchmarks**
| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **Train Loss** | < 0.01 | 0.01-0.05 | 0.05-0.1 | > 0.1 |
| **Val Loss** | < 0.015 | 0.015-0.06 | 0.06-0.12 | > 0.12 |
| **Val/Train Ratio** | < 1.2 | 1.2-1.5 | 1.5-2.0 | > 2.0 |

### **Stage 2 Benchmarks**
| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **Total Loss** | < 0.1 | 0.1-0.3 | 0.3-0.5 | > 0.5 |
| **Teacher Forcing Loss** | < 0.08 | 0.08-0.2 | 0.2-0.4 | > 0.4 |
| **Rollout Loss** | < 0.15 | 0.15-0.4 | 0.4-0.6 | > 0.6 |
| **Val/Train Ratio** | < 1.2 | 1.2-1.5 | 1.5-2.0 | > 2.0 |

### **Expected Learning Curves**
```
Stage 1:
Epoch 1:  Train Loss: 0.4-0.6, Val Loss: 0.3-0.5
Epoch 5:  Train Loss: 0.1-0.2, Val Loss: 0.08-0.15
Epoch 10: Train Loss: 0.05-0.1, Val Loss: 0.04-0.08

Stage 2:
Epoch 1:  Total Loss: 0.4-0.6, TF: 0.3-0.5, Rollout: 0.5-0.8
Epoch 10: Total Loss: 0.2-0.3, TF: 0.15-0.25, Rollout: 0.25-0.4
Epoch 20: Total Loss: 0.15-0.25, TF: 0.1-0.2, Rollout: 0.2-0.3
```

## 🚀 Usage

### **Training Commands**
```bash
# Train both stages sequentially
python train_vjepa2_world_model.py

# Train only Stage 1
python train_vjepa2_world_model.py --stage 1

# Train only Stage 2 (requires Stage 1 checkpoint)
python train_vjepa2_world_model.py --stage 2 --stage1-checkpoint trained_models/best_stage1_model.pth
```

### **Evaluation**
```bash
# Evaluate both stages
python evaluate_vjepa2_world_model.py

# Evaluate specific stage
python evaluate_vjepa2_world_model.py --stage 1
python evaluate_vjepa2_world_model.py --stage 2
```

## ⚙️ Configuration

### **Environment Settings**
```yaml
environment:
  name: "ALE/Breakout-v5"
  frame_stack_size: 4              # Number of stacked frames
  grayscale_conversion: true       # Convert to grayscale
  image_height: 64                 # Target image height
  image_width: 64                  # Target image width
```

### **Model Settings**
```yaml
models:
  shared_latent_dim: 64            # Latent dimension
  shared_patch_size: 8             # ViT patch size
  encoder:
    params:
      vit:
        depth: 4                   # Transformer depth
        heads: 4                   # Number of attention heads
        mlp_dim: 512               # MLP dimension
        dropout: 0.2               # Dropout rate
```

### **Training Settings**
```yaml
vjepa2_world_model:
  stage1:
    masking_ratio: 0.4             # 40% masking
    num_epochs: 10                 # Training epochs
    learning_rate: 0.0001          # Learning rate
    ema_decay: 0.996              # EMA update rate
    weight_decay: 0.05            # Weight decay
    early_stopping_patience: 3     # Early stopping
    
  stage2:
    sequence_length: 8             # Temporal sequence length
    teacher_forcing_ratio: 0.8     # Teacher forcing ratio
    rollout_steps: 4              # Multi-step prediction
    num_epochs: 30                # Training epochs
    learning_rate: 0.0001         # Learning rate
    weight_decay: 0.05            # Weight decay
    early_stopping_patience: 3     # Early stopping
```

## 🔧 Key Components

### **Models**
- **MaskedViT**: Vision Transformer with random masking support
- **MaskedPredictionModel**: Complete Stage 1 model with online/target encoders
- **WorldModelTransformer**: Action-conditioned temporal transformer
- **TemporalSequenceDataset**: Dataset for temporal sequence training

### **Loss Functions**
- **MaskedPredictionLoss**: L1 loss for Stage 1 masked prediction
- **WorldModelLoss**: Combined teacher forcing and rollout loss for Stage 2

### **Training Features**
- **EMA Target Encoder**: Stable learning target for Stage 1
- **Frozen Encoder**: Stage 1 encoder frozen during Stage 2
- **Temporal Sequences**: Real 8-step sequences for Stage 2
- **Teacher Forcing**: Mix of supervised and rollout training
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Stable training

## 📈 Integration

### **Weights & Biases**
- Automatic experiment tracking
- Loss curves and metrics visualization
- Model checkpointing
- Hyperparameter logging

### **Existing Codebase**
- Reuses ViT, MLP, and training infrastructure
- Compatible with existing data pipeline
- Full configuration system support
- Seamless integration with main training loop

## 🎯 Success Criteria

### **Stage 1 Success**
- ✅ Train loss < 0.05
- ✅ Val/Train ratio < 1.5
- ✅ No overfitting (val loss stable)
- ✅ Encoder learns meaningful representations

### **Stage 2 Success**
- ✅ Total loss reaches 0.1-0.3 range
- ✅ Rollout loss > Teacher forcing loss
- ✅ Training runs for 15+ epochs
- ✅ Validation loss stable and reasonable
- ✅ World model learns temporal dynamics

This implementation provides a complete V-JEPA2 pipeline for learning world models from visual observations and actions! 🎉 