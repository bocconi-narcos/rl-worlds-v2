import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPredictionLoss(nn.Module):
    """
    Loss function for masked prediction training (Stage 1).
    
    Computes L1 loss between predicted and target embeddings in latent space.
    """
    
    def __init__(self, loss_type='l1', reduction='mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    def forward(self, predicted_embeddings, target_embeddings, mask=None):
        """
        Compute masked prediction loss.
        
        Args:
            predicted_embeddings: Predicted embeddings from predictor [B, D]
            target_embeddings: Target embeddings from EMA encoder [B, D]
            mask: Binary mask indicating which patches were masked [B, N] (optional)
        
        Returns:
            loss: Scalar loss value
        """
        # Basic loss computation
        loss = self.loss_fn(predicted_embeddings, target_embeddings)
        
        return loss
    
    def compute_loss_with_mask(self, predicted_embeddings, target_embeddings, mask):
        """
        Compute loss only for masked patches (if we have patch-level predictions).
        
        Args:
            predicted_embeddings: Predicted embeddings [B, N, D] or [B, D]
            target_embeddings: Target embeddings [B, N, D] or [B, D]
            mask: Binary mask indicating which patches were masked [B, N]
        
        Returns:
            loss: Scalar loss value
        """
        if predicted_embeddings.dim() == 2:
            # Global prediction (CLS token), use standard loss
            return self.forward(predicted_embeddings, target_embeddings)
        
        # Patch-level prediction
        B, N, D = predicted_embeddings.shape
        
        # Apply mask to get only masked patches
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
        
        masked_pred = predicted_embeddings[mask_expanded.bool()].view(-1, D)
        masked_target = target_embeddings[mask_expanded.bool()].view(-1, D)
        
        if masked_pred.numel() == 0:
            # No masked patches, return zero loss
            return torch.tensor(0.0, device=predicted_embeddings.device, requires_grad=True)
        
        # Compute loss only on masked patches
        loss = self.loss_fn(masked_pred, masked_target)
        
        return loss


class WorldModelLoss(nn.Module):
    """
    Loss function for world model training (Stage 2).
    
    Combines teacher forcing loss and rollout loss.
    """
    
    def __init__(self, 
                 teacher_forcing_weight=1.0,
                 rollout_weight=0.5,
                 loss_type='l1',
                 reduction='mean'):
        super().__init__()
        self.teacher_forcing_weight = teacher_forcing_weight
        self.rollout_weight = rollout_weight
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    def forward(self, 
                predicted_latents, 
                target_latents, 
                teacher_forcing_mask=None,
                rollout_mask=None):
        """
        Compute world model loss.
        
        Args:
            predicted_latents: Predicted latent representations [B, T, D]
            target_latents: Target latent representations [B, T, D]
            teacher_forcing_mask: Mask for teacher forcing steps [B, T] (optional)
            rollout_mask: Mask for rollout steps [B, T] (optional)
        
        Returns:
            total_loss: Combined loss value
            teacher_forcing_loss: Teacher forcing loss component
            rollout_loss: Rollout loss component
        """
        B, T, D = predicted_latents.shape
        
        # Teacher forcing loss (next-step prediction)
        if teacher_forcing_mask is not None:
            # Apply mask to get only teacher forcing steps
            tf_pred = predicted_latents[teacher_forcing_mask.bool()].view(-1, D)
            tf_target = target_latents[teacher_forcing_mask.bool()].view(-1, D)
            
            if tf_pred.numel() > 0:
                teacher_forcing_loss = self.loss_fn(tf_pred, tf_target)
            else:
                teacher_forcing_loss = torch.tensor(0.0, device=predicted_latents.device)
        else:
            # Use all steps for teacher forcing
            teacher_forcing_loss = self.loss_fn(predicted_latents, target_latents)
        
        # Rollout loss (multi-step prediction)
        if rollout_mask is not None:
            # Apply mask to get only rollout steps
            rollout_pred = predicted_latents[rollout_mask.bool()].view(-1, D)
            rollout_target = target_latents[rollout_mask.bool()].view(-1, D)
            
            if rollout_pred.numel() > 0:
                rollout_loss = self.loss_fn(rollout_pred, rollout_target)
            else:
                rollout_loss = torch.tensor(0.0, device=predicted_latents.device)
        else:
            # Use all steps for rollout
            rollout_loss = self.loss_fn(predicted_latents, target_latents)
        
        # Combine losses
        total_loss = (self.teacher_forcing_weight * teacher_forcing_loss + 
                     self.rollout_weight * rollout_loss)
        
        return total_loss, teacher_forcing_loss, rollout_loss
    
    def compute_sequence_loss(self, predicted_sequence, target_sequence):
        """
        Compute loss for a sequence of predictions.
        
        Args:
            predicted_sequence: Predicted sequence [B, T, D]
            target_sequence: Target sequence [B, T, D]
        
        Returns:
            loss: Average loss across the sequence
        """
        return self.loss_fn(predicted_sequence, target_sequence) 