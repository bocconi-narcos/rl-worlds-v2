# RL World Models: Encoder-Decoder vs. JEPA vs. VJEPA2

This project implements and compares three world model architectures for learning from visual observations in Gymnasium environments:
1.  A **Standard Encoder-Decoder** model.
2.  A **Joint Embedding Predictive Architecture (JEPA)**.
3.  A **Two-Stage VJEPA2 World Model** (NEW, see below).

Both JEPA and VJEPA2 are designed to learn representations of the environment and predict future states or state embeddings. The VJEPA2 pipeline is now the recommended approach for advanced latent world modeling.

## VJEPA2 Pipeline (NEW)

VJEPA2 is a two-stage world model:
- **Stage 1:** Self-supervised encoder pretraining (MaskedViT, MaskedPredictionModel)
- **Stage 2:** Action-conditioned world model transformer in latent space

After both stages, the encoder and world model are **frozen** and used as a drop-in replacement for JEPA in downstream tasks (reward MLPs, LARP, etc.).

- The new entry point is `main_vjepa2.py`.
- The JEPA model in the pipeline is now a wrapper around the frozen VJEPA2 encoder/world model.
- Only downstream predictors (reward MLPs, LARP, etc.) are trained after VJEPA2 stages.

For full details, see [`README_VJEPA2.md`](README_VJEPA2.md).

## Quick Start (VJEPA2)

1. **Configure your experiment:** Edit `config.yaml` as usual.
2. **Run the VJEPA2 pipeline:**
   ```bash
   python main_vjepa2.py
   ```
   This will train VJEPA2 (both stages), freeze the encoder/world model, and train all downstream predictors as before.

## Other Architectures

(Standard Encoder-Decoder and JEPA remain available; see below for details.)

## Full Documentation

For detailed information on the project, including in-depth explanations of the models, data collection, configuration, and evaluation methodologies, please refer to our comprehensive documentation:

**[View Full Documentation in `docs/`](docs/)**

We recommend starting with `docs/index.md` (if available) or browsing the individual Markdown files in the `docs/` directory for specific topics. The `docs/06_usage_guide.md` is particularly helpful for getting started.

## Project Structure

```
.
├── .gitignore
├── README.md           # This file (high-level overview)
├── config.yaml         # Central configuration file
├── main.py             # Main training and execution script
├── requirements.txt    # Python dependencies
├── docs/               # Detailed project documentation
│   ├── index.md
│   ├── ... (other .md files)
│   └── 06_usage_guide.md
├── datasets/           # Stores collected trajectory datasets
├── trained_models/     # Stores trained model checkpoints
├── validation_plots/   # Stores validation image outputs (e.g., JEPA decoder reconstructions)
├── src/                # Source code (models, utilities, modular training loops, etc.)
│   ├── __init__.py
│   ├── data_handling.py
│   ├── loss_setup.py
│   ├── model_setup.py
│   ├── optimizer_setup.py
│   ├── training_engine.py # Orchestrates training 
│   ├── training_loops/   # Contains individual epoch-level training logic
│   ├── models/           # Model architectures (Enc-Dec, JEPA, ViT, CNN, MLP, etc.)
│   ├── losses/           # Loss implementations (VICReg, Barlow Twins, DINO)
│   ├── utils/            # General utilities, config handling, data processing
│   │   ├── __init__.py
│   │   ├── config_utils.py
│   │   ├── data_utils.py   # Data collection and dataset management
│   │   ├── env_utils.py
│   │   └── env_wrappers.py # Environment wrappers like ActionRepeat
│   └── rl_agent.py       # PPO agent setup and training for data collection
└── tests/              # Unit and integration tests
```

## Quick Start

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    For specific Gymnasium environments (e.g., Atari, Box2D), additional installations might be needed.

For detailed setup instructions, including environment-specific installations, please see **[`docs/06_usage_guide.md`](docs/06_usage_guide.md)**.

### Running the Training

1.  **Configure your experiment:** Edit `config.yaml` to define the environment, model architectures, data collection methods, and training parameters.
2.  **Run the main script:**
    ```bash
    python main.py
    ```

For a comprehensive explanation of all configuration options in `config.yaml` and details on the execution flow, refer to **[`docs/06_usage_guide.md`](docs/06_usage_guide.md)**.

## Experiment Tracking with Weights & Biases

This project integrates Weights & Biases (wandb) for experiment tracking, visualization, and management. Wandb helps you log metrics, configurations, and even model outputs during training runs.

### Configuration

To use wandb, you need to configure its settings in the `config.yaml` file under the `wandb` top-level key:

```yaml
wandb:
  project: "my_ai_project"  # Your wandb project name
  entity: null              # Your wandb entity (username or team name). Set to null or omit if logging to default entity.
  run_name_prefix: "exp"    # A prefix for automatically generated run names
  enabled: true             # Set to false to disable wandb logging
```

*   `project`: The name of the project in wandb where runs will be logged.
*   `entity`: Your wandb username or team name. If you are working in a team, this will be your team's entity. If left as `null` or omitted, the run will be logged to your default wandb entity.
*   `run_name_prefix`: A prefix used to generate run names automatically (e.g., "exp-YYYYMMDD-HHMMSS").
*   `enabled`: Set this to `true` to enable wandb logging, or `false` to disable it.

### Prerequisites

*   **Wandb Account**: You need an account at [wandb.ai](https://wandb.ai).
*   **Login**: If you specify an `entity` or want to ensure runs are associated with your account, you must be logged into wandb in your environment. You can do this by running:
    ```bash
    wandb login
    ```
    Follow the prompts to authenticate. If `enabled` is `true` but you are not logged in and no `entity` is specified, wandb might run in anonymous mode or prompt you to log in.


## Implemented Architectures

This project provides implementations of the following world model architectures:

### 1. Standard Encoder-Decoder
*   **Encoder**: A configurable encoder (ViT, CNN, or MLP) processes the input state image `s_t` into a latent representation.
*   **Decoder**: A Transformer-based decoder takes the latent state and an embedded action `a_t` to predict the next state image `s_t+1` in pixel space.
*   **Loss**: Typically Mean Squared Error (MSE) between the predicted `s_t+1` and the actual `s_t+1`.

For a detailed description, see **[`docs/03_encoder_decoder_model.md`](docs/03_encoder_decoder_model.md)**.

### 2. JEPA (Joint Embedding Predictive Architecture)
*   **Encoders**: Utilizes an *Online Encoder* and an EMA-updated *Target Encoder* (ViT, CNN, or MLP) to process states into embeddings.
*   **Predictor**: An MLP (or Transformer) predicts the target encoder's embedding of the next state `s_{t+1}` based on the target-encoded current state `s_t` and action `a_t`.
*   **Loss**: Combines a primary *Prediction Loss* (MSE in embedding space) with an *Auxiliary Loss* (e.g., VICReg, Barlow Twins, DINO) applied to the online encoder's outputs to encourage informative representations.

For a detailed description, see **[`docs/04_jepa_model.md`](docs/04_jepa_model.md)**.

### 3. Encoder-Decoder JEPA-Style (Fair JEPA Baseline)
*   **Encoder:** Configurable (ViT, CNN, or MLP), same as other models.
*   **Predictor:** A JEPA-style MLP takes the concatenated encoder output and action embedding, producing a latent vector.
*   **Decoder:** A Transformer-based decoder reconstructs the next state image from the predictor's output.
*   **Purpose:** This model is designed for apples-to-apples comparison with JEPA, matching parameter count and architectural complexity. It allows rigorous evaluation of whether JEPA's performance gains are due to its modeling principle or simply model capacity.
*   **Output:** Predicts the next state in pixel space, just like the standard Encoder-Decoder, but with a JEPA-style predictor in the pipeline.

## Contributing

Contributions to this project are welcome. Please refer to the documentation and existing code structure for guidance. (Further details on contributing can be added here or in a separate `CONTRIBUTING.md` file).