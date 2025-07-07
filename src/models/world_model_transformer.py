import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from .vit import Transformer, PreNorm, Attention, FeedForward
from src.utils.weight_init import initialize_weights


class WorldModelTransformer(nn.Module):
    """
    Stage 2: Action-Conditioned World Model
    Transformer that models world dynamics in latent space, conditioned on actions.
    
    Takes sequences of (z_t, a_t) and predicts z_{t+1}.
    """
    
    def __init__(self,
                 latent_dim,
                 action_dim,
                 action_emb_dim,
                 action_type,  # 'discrete' or 'continuous'
                 sequence_length=8,
                 # Transformer parameters
                 transformer_depth=4,
                 transformer_heads=8,
                 transformer_mlp_dim=512,
                 transformer_dim_head=64,
                 transformer_dropout=0.1,
                 # Positional encoding
                 use_pos_encoding=True,
                 pos_encoding_dim=None):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.sequence_length = sequence_length
        self.use_pos_encoding = use_pos_encoding
        
        # Use pos_encoding_dim if provided, otherwise use latent_dim + action_emb_dim
        if pos_encoding_dim is None:
            pos_encoding_dim = latent_dim + action_emb_dim
        
        # Action embedding
        if self.action_type == 'discrete':
            self.action_embedding = nn.Embedding(action_dim, action_emb_dim)
        elif self.action_type == 'continuous':
            self.action_embedding = nn.Linear(action_dim, action_emb_dim)
        else:
            raise ValueError(f"Unsupported action_type: {action_type}")
        
        # Input projection: concatenate latent and action embeddings
        self.input_projection = nn.Linear(latent_dim + action_emb_dim, pos_encoding_dim)
        
        # Positional encoding
        if self.use_pos_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, pos_encoding_dim) * 0.02)
        
        # Transformer for processing sequences
        self.transformer = Transformer(
            dim=pos_encoding_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=transformer_dropout
        )
        
        # Output projection: back to latent dimension
        self.output_projection = nn.Linear(pos_encoding_dim, latent_dim)
        
        # Layer normalization for input and output
        self.input_norm = nn.LayerNorm(pos_encoding_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        
        self.apply(initialize_weights)
    
    def forward(self, latents, actions):
        """
        Forward pass for world model prediction.
        
        Args:
            latents: Latent representations [B, T, D] where T is sequence length
            actions: Actions [B, T, A] or [B, T] for discrete actions
        
        Returns:
            predicted_latents: Predicted next latent representations [B, T, D]
        """
        B, T, D = latents.shape
        
        # Process actions
        if self.action_type == 'discrete':
            # Ensure actions are long and squeezed if needed
            if actions.ndim == 3 and actions.shape[2] == 1:
                actions = actions.squeeze(2)
            if actions.dtype != torch.long:
                actions = actions.long()
            action_embeddings = self.action_embedding(actions)  # [B, T, action_emb_dim]
        elif self.action_type == 'continuous':
            # Ensure actions are float
            if actions.dtype != torch.float32:
                actions = actions.float()
            action_embeddings = self.action_embedding(actions)  # [B, T, action_emb_dim]
        else:
            raise ValueError(f"Unsupported action_type in forward pass: {self.action_type}")
        
        # Concatenate latents and action embeddings
        combined_input = torch.cat([latents, action_embeddings], dim=-1)  # [B, T, D + action_emb_dim]
        
        # Project to transformer dimension
        x = self.input_projection(combined_input)  # [B, T, pos_encoding_dim]
        x = self.input_norm(x)
        
        # Add positional encoding
        if self.use_pos_encoding:
            x = x + self.pos_embedding[:, :T, :]
        
        # Apply transformer
        x = self.transformer(x)  # [B, T, pos_encoding_dim]
        
        # Project back to latent dimension
        predicted_latents = self.output_projection(x)  # [B, T, latent_dim]
        predicted_latents = self.output_norm(predicted_latents)
        
        return predicted_latents
    
    def predict_next_step(self, latents, actions):
        """
        Predict the next latent representation given current latents and actions.
        
        Args:
            latents: Current latent representations [B, T, D]
            actions: Current actions [B, T, A] or [B, T] for discrete
        
        Returns:
            next_latents: Predicted next latent representations [B, T, D]
        """
        return self.forward(latents, actions)
    
    def rollout_prediction(self, initial_latent, actions, num_steps):
        """
        Perform multi-step rollout prediction.
        
        Args:
            initial_latent: Initial latent representation [B, D]
            actions: Actions for rollout [B, num_steps, A] or [B, num_steps] for discrete
            num_steps: Number of steps to predict
        
        Returns:
            predicted_latents: Predicted latent representations [B, num_steps, D]
        """
        B = initial_latent.shape[0]
        device = initial_latent.device
        
        # Initialize output tensor
        predicted_latents = torch.zeros(B, num_steps, self.latent_dim, device=device)
        
        # Current latent for rolling prediction
        current_latent = initial_latent.unsqueeze(1)  # [B, 1, D]
        
        for step in range(num_steps):
            # Get current action
            current_action = actions[:, step:step+1]  # [B, 1, A] or [B, 1]
            
            # Predict next latent
            next_latent = self.forward(current_latent, current_action)  # [B, 1, D]
            
            # Store prediction
            predicted_latents[:, step:step+1] = next_latent
            
            # Update current latent for next step
            current_latent = next_latent
        
        return predicted_latents 