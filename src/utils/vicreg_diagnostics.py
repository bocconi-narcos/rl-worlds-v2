"""
Specialized VICReg Training Diagnostics

This module provides targeted diagnostics for VICReg loss plateau issues,
focusing on the specific characteristics of VICReg training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

class VICRegDiagnostics:
    """
    Specialized diagnostics for VICReg training issues
    """
    
    def __init__(self, save_dir: str = "vicreg_diagnostics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.history = {
            'total_loss': [],
            'sim_loss': [],
            'std_loss': [],
            'cov_loss': [],
            'grad_norms': [],
            'param_norms': [],
            'representation_stats': [],
            'projector_stats': []
        }
        self.step_count = 0
        
    def diagnose_vicreg_step(self, 
                           vicreg_loss_fn: nn.Module,
                           embeddings: torch.Tensor,
                           target_embeddings: torch.Tensor,
                           model: nn.Module,
                           loss: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive VICReg step diagnosis
        
        Args:
            vicreg_loss_fn: VICReg loss function
            embeddings: Online embeddings
            target_embeddings: Target embeddings
            model: Full model
            loss: Current loss
            
        Returns:
            Dict with detailed diagnosis
        """
        self.step_count += 1
        diagnosis = {}
        
        # 1. VICReg Loss Component Analysis
        diagnosis['loss_components'] = self._analyze_vicreg_loss_components(
            vicreg_loss_fn, embeddings, target_embeddings)
        
        # 2. Projector Analysis
        diagnosis['projector'] = self._analyze_projector(vicreg_loss_fn, embeddings)
        
        # 3. Representation Analysis
        diagnosis['representations'] = self._analyze_representations(embeddings, target_embeddings)
        
        # 4. Gradient Analysis
        diagnosis['gradients'] = self._analyze_gradients(model, loss)
        
        # 5. Parameter Analysis
        diagnosis['parameters'] = self._analyze_parameters(model)
        
        # 6. Plateau Detection
        diagnosis['plateau'] = self._detect_plateau()
        
        # 7. Recommendations
        diagnosis['recommendations'] = self._generate_recommendations(diagnosis)
        
        # Store history
        self._update_history(diagnosis)
        
        # Save periodically
        if self.step_count % 50 == 0:
            self._save_diagnosis(diagnosis)
        
        return diagnosis
    
    def _analyze_vicreg_loss_components(self, vicreg_loss_fn: nn.Module,
                                      embeddings: torch.Tensor,
                                      target_embeddings: torch.Tensor) -> Dict[str, Any]:
        """Analyze VICReg loss components in detail"""
        
        # Get loss components
        with torch.no_grad():
            if hasattr(vicreg_loss_fn, 'calculate_reg_terms'):
                # JEPA-style VICReg
                total_loss, std_loss, cov_loss = vicreg_loss_fn.calculate_reg_terms(embeddings)
                sim_loss = torch.tensor(0.0, device=embeddings.device)
            else:
                # Full VICReg
                total_loss, sim_loss, std_loss, cov_loss = vicreg_loss_fn(embeddings, target_embeddings)
        
        # Analyze component ratios
        components = {
            'total': total_loss.item(),
            'sim': sim_loss.item(),
            'std': std_loss.item(),
            'cov': cov_loss.item()
        }
        
        # Check for component dominance
        issues = []
        if components['total'] > 0:
            sim_ratio = components['sim'] / components['total']
            std_ratio = components['std'] / components['total']
            cov_ratio = components['cov'] / components['total']
            
            if sim_ratio > 0.8:
                issues.append(f"SIM_DOMINANT: {sim_ratio:.3f}")
            if std_ratio > 0.8:
                issues.append(f"STD_DOMINANT: {std_ratio:.3f}")
            if cov_ratio > 0.8:
                issues.append(f"COV_DOMINANT: {cov_ratio:.3f}")
        
        # Check for component collapse
        if components['std'] < 1e-6:
            issues.append("STD_COLLAPSE")
        if components['cov'] < 1e-6:
            issues.append("COV_COLLAPSE")
        
        return {
            'components': components,
            'ratios': {
                'sim_ratio': sim_ratio if components['total'] > 0 else 0,
                'std_ratio': std_ratio if components['total'] > 0 else 0,
                'cov_ratio': cov_ratio if components['total'] > 0 else 0
            },
            'issues': issues
        }
    
    def _analyze_projector(self, vicreg_loss_fn: nn.Module, embeddings: torch.Tensor) -> Dict[str, Any]:
        """Analyze VICReg projector behavior"""
        
        if not hasattr(vicreg_loss_fn, 'projector') or vicreg_loss_fn.projector is None:
            return {'has_projector': False, 'issues': []}
        
        projector = vicreg_loss_fn.projector
        
        # Analyze projector layers
        layer_stats = {}
        issues = []
        
        x = embeddings
        for i, layer in enumerate(projector):
            if isinstance(layer, nn.Linear):
                # Check weight statistics
                weight_norm = layer.weight.norm().item()
                weight_std = layer.weight.std().item()
                
                if weight_norm > 100:
                    issues.append(f"PROJ_WEIGHT_EXPLOSION_L{i}: {weight_norm:.2e}")
                if weight_std < 1e-6:
                    issues.append(f"PROJ_WEIGHT_COLLAPSE_L{i}: {weight_std:.2e}")
                
                layer_stats[f'linear_{i}'] = {
                    'weight_norm': weight_norm,
                    'weight_std': weight_std,
                    'bias_norm': layer.bias.norm().item() if layer.bias is not None else 0
                }
            
            elif isinstance(layer, nn.BatchNorm1d):
                # Check batch norm statistics
                running_mean = layer.running_mean
                running_var = layer.running_var
                
                if running_mean is not None:
                    mean_norm = running_mean.norm().item()
                    var_mean = running_var.mean().item()
                    
                    if mean_norm > 10:
                        issues.append(f"PROJ_BN_MEAN_EXPLOSION_L{i}: {mean_norm:.2e}")
                    if var_mean < 1e-6:
                        issues.append(f"PROJ_BN_VAR_COLLAPSE_L{i}: {var_mean:.2e}")
                
                layer_stats[f'batchnorm_{i}'] = {
                    'mean_norm': mean_norm if running_mean is not None else 0,
                    'var_mean': var_mean if running_var is not None else 0
                }
            
            # Forward pass through layer
            with torch.no_grad():
                x = layer(x)
                
                # Check for saturation (ReLU)
                if isinstance(layer, nn.ReLU):
                    dead_ratio = (x == 0).float().mean().item()
                    if dead_ratio > 0.5:
                        issues.append(f"PROJ_RELU_DEAD_L{i}: {dead_ratio:.3f}")
                
                # Check for extreme values
                x_norm = x.norm().item()
                if x_norm > 100:
                    issues.append(f"PROJ_ACTIVATION_EXPLOSION_L{i}: {x_norm:.2e}")
        
        return {
            'has_projector': True,
            'layer_stats': layer_stats,
            'final_output_norm': x.norm().item(),
            'issues': issues
        }
    
    def _analyze_representations(self, embeddings: torch.Tensor, 
                               target_embeddings: torch.Tensor) -> Dict[str, Any]:
        """Analyze representation quality"""
        
        # Basic statistics
        emb_mean = embeddings.mean().item()
        emb_std = embeddings.std().item()
        emb_norm = embeddings.norm().item()
        
        # Feature-wise analysis
        feature_std = embeddings.std(dim=0)
        feature_mean = embeddings.mean(dim=0)
        
        dead_features = (feature_std < 1e-6).sum().item()
        saturated_features = (torch.abs(feature_mean) > 10).sum().item()
        
        # Correlation analysis
        if embeddings.shape[1] > 1:
            corr_matrix = torch.corrcoef(embeddings.T)
            off_diag_corr = corr_matrix.fill_diagonal_(0)
            max_corr = off_diag_corr.abs().max().item()
            mean_corr = off_diag_corr.abs().mean().item()
        else:
            max_corr = mean_corr = 0
        
        # Similarity with target
        if target_embeddings is not None:
            similarity = F.cosine_similarity(embeddings.flatten(), 
                                           target_embeddings.flatten(), dim=0).item()
        else:
            similarity = None
        
        issues = []
        if emb_std < 0.01:
            issues.append("REP_COLLAPSE")
        if emb_norm > 100:
            issues.append("REP_EXPLOSION")
        if dead_features > embeddings.shape[1] * 0.1:  # More than 10% dead
            issues.append(f"MANY_DEAD_FEATURES: {dead_features}")
        if max_corr > 0.9:
            issues.append(f"HIGH_CORRELATION: {max_corr:.3f}")
        
        return {
            'mean': emb_mean,
            'std': emb_std,
            'norm': emb_norm,
            'dead_features': dead_features,
            'saturated_features': saturated_features,
            'max_correlation': max_corr,
            'mean_correlation': mean_corr,
            'similarity': similarity,
            'issues': issues
        }
    
    def _analyze_gradients(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Analyze gradient flow"""
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Analyze gradients by component
        grad_stats = {}
        total_norm = 0
        issues = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm ** 2
                
                # Group by component
                component = name.split('.')[0] if '.' in name else 'other'
                if component not in grad_stats:
                    grad_stats[component] = []
                grad_stats[component].append(grad_norm)
        
        total_norm = total_norm ** 0.5
        
        # Check for gradient issues
        if total_norm < 1e-6:
            issues.append("VANISHING_GRADIENTS")
        elif total_norm > 10:
            issues.append("EXPLODING_GRADIENTS")
        
        # Component-specific issues
        for component, norms in grad_stats.items():
            if len(norms) > 0:
                avg_norm = np.mean(norms)
                if avg_norm < 1e-8:
                    issues.append(f"COMPONENT_VANISHING: {component}")
                elif avg_norm > 10:
                    issues.append(f"COMPONENT_EXPLODING: {component}")
        
        return {
            'total_norm': total_norm,
            'component_norms': {k: np.mean(v) for k, v in grad_stats.items()},
            'issues': issues
        }
    
    def _analyze_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model parameters"""
        
        param_stats = {}
        issues = []
        
        for name, param in model.named_parameters():
            param_norm = param.norm().item()
            param_std = param.std().item()
            
            param_stats[name] = {
                'norm': param_norm,
                'std': param_std
            }
            
            if not np.isfinite(param_norm):
                issues.append(f"PARAM_NAN_INF: {name}")
            elif param_norm > 100:
                issues.append(f"PARAM_EXPLOSION: {name}")
            elif param_std < 1e-6:
                issues.append(f"PARAM_COLLAPSE: {name}")
        
        return {
            'param_stats': param_stats,
            'issues': issues
        }
    
    def _detect_plateau(self) -> Dict[str, Any]:
        """Detect loss plateau"""
        
        if len(self.history['total_loss']) < 20:
            return {'plateau_detected': False, 'reason': 'insufficient_history'}
        
        # Get recent loss values
        recent_losses = self.history['total_loss'][-20:]
        
        # Calculate trend
        x = np.arange(len(recent_losses))
        trend = np.polyfit(x, recent_losses, 1)[0]
        
        # Calculate variance
        loss_variance = np.var(recent_losses)
        
        # Plateau detection criteria
        plateau_detected = False
        reasons = []
        
        if abs(trend) < 1e-6:
            plateau_detected = True
            reasons.append("ZERO_TREND")
        
        if loss_variance < 1e-8:
            plateau_detected = True
            reasons.append("ZERO_VARIANCE")
        
        # Check if loss is oscillating around a value
        if len(recent_losses) >= 10:
            first_half = recent_losses[:10]
            second_half = recent_losses[10:]
            if abs(np.mean(first_half) - np.mean(second_half)) < 1e-6:
                plateau_detected = True
                reasons.append("STABLE_OSCILLATION")
        
        return {
            'plateau_detected': plateau_detected,
            'trend': trend,
            'variance': loss_variance,
            'reasons': reasons
        }
    
    def _generate_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Loss component issues
        if 'loss_components' in diagnosis:
            loss_issues = diagnosis['loss_components']['issues']
            if any('STD_DOMINANT' in issue for issue in loss_issues):
                recommendations.append("Reduce std_coeff in VICReg configuration")
            if any('COV_DOMINANT' in issue for issue in loss_issues):
                recommendations.append("Reduce cov_coeff in VICReg configuration")
            if any('STD_COLLAPSE' in issue for issue in loss_issues):
                recommendations.append("Increase std_coeff or check data normalization")
            if any('COV_COLLAPSE' in issue for issue in loss_issues):
                recommendations.append("Increase cov_coeff or check representation diversity")
        
        # Projector issues
        if 'projector' in diagnosis:
            proj_issues = diagnosis['projector']['issues']
            if any('PROJ_WEIGHT_EXPLOSION' in issue for issue in proj_issues):
                recommendations.append("Reduce learning rate or add weight decay to projector")
            if any('PROJ_RELU_DEAD' in issue for issue in proj_issues):
                recommendations.append("Use different activation function or adjust initialization")
        
        # Representation issues
        if 'representations' in diagnosis:
            rep_issues = diagnosis['representations']['issues']
            if any('REP_COLLAPSE' in issue for issue in rep_issues):
                recommendations.append("Increase std_coeff or check encoder training")
            if any('HIGH_CORRELATION' in issue for issue in rep_issues):
                recommendations.append("Increase cov_coeff or check feature diversity")
        
        # Gradient issues
        if 'gradients' in diagnosis:
            grad_issues = diagnosis['gradients']['issues']
            if any('VANISHING_GRADIENTS' in issue for issue in grad_issues):
                recommendations.append("Increase learning rate or reduce loss coefficients")
            if any('EXPLODING_GRADIENTS' in issue for issue in grad_issues):
                recommendations.append("Reduce learning rate or apply gradient clipping")
        
        # Plateau issues
        if 'plateau' in diagnosis and diagnosis['plateau']['plateau_detected']:
            recommendations.append("Loss plateau detected - consider learning rate scheduling")
            recommendations.append("Check if loss components are balanced")
        
        return recommendations
    
    def _update_history(self, diagnosis: Dict[str, Any]):
        """Update history with current diagnosis"""
        
        if 'loss_components' in diagnosis:
            components = diagnosis['loss_components']['components']
            self.history['total_loss'].append(components['total'])
            self.history['sim_loss'].append(components['sim'])
            self.history['std_loss'].append(components['std'])
            self.history['cov_loss'].append(components['cov'])
        
        if 'gradients' in diagnosis:
            self.history['grad_norms'].append(diagnosis['gradients']['total_norm'])
        
        if 'representations' in diagnosis:
            self.history['representation_stats'].append({
                'mean': diagnosis['representations']['mean'],
                'std': diagnosis['representations']['std'],
                'norm': diagnosis['representations']['norm']
            })
    
    def _save_diagnosis(self, diagnosis: Dict[str, Any]):
        """Save diagnosis to disk"""
        
        # Save detailed diagnosis
        with open(self.save_dir / f"vicreg_diagnosis_step_{self.step_count}.json", 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_diagnosis = self._convert_to_json_serializable(diagnosis)
            json.dump(json_diagnosis, f, indent=2)
        
        # Save history
        with open(self.save_dir / f"vicreg_history_step_{self.step_count}.json", 'w') as f:
            json_history = self._convert_to_json_serializable(self.history)
            json.dump(json_history, f, indent=2)
        
        # Generate plots
        self._generate_plots()
    
    def _convert_to_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_plots(self):
        """Generate diagnostic plots"""
        
        # Loss components plot
        if len(self.history['total_loss']) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Total loss
            axes[0, 0].plot(self.history['total_loss'])
            axes[0, 0].set_title('Total VICReg Loss')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # Component breakdown
            if len(self.history['std_loss']) > 0:
                axes[0, 1].plot(self.history['std_loss'], label='Std Loss')
                axes[0, 1].plot(self.history['cov_loss'], label='Cov Loss')
                axes[0, 1].set_title('VICReg Components')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Gradient norms
            if len(self.history['grad_norms']) > 0:
                axes[1, 0].plot(self.history['grad_norms'])
                axes[1, 0].set_title('Gradient Norms')
                axes[1, 0].set_ylabel('Norm')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True)
            
            # Representation stats
            if len(self.history['representation_stats']) > 0:
                reps = self.history['representation_stats']
                stds = [r['std'] for r in reps]
                axes[1, 1].plot(stds)
                axes[1, 1].set_title('Representation Std')
                axes[1, 1].set_ylabel('Std')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / f'vicreg_diagnostics_step_{self.step_count}.png')
            plt.close()

# Convenience function for quick VICReg diagnosis
def quick_vicreg_diagnostic(vicreg_loss_fn: nn.Module,
                           embeddings: torch.Tensor,
                           target_embeddings: torch.Tensor,
                           model: nn.Module,
                           loss: torch.Tensor) -> Dict[str, Any]:
    """
    Quick VICReg diagnostic for immediate issues
    
    Returns:
        Dict with critical issues and recommendations
    """
    diagnostics = VICRegDiagnostics()
    results = diagnostics.diagnose_vicreg_step(vicreg_loss_fn, embeddings, 
                                              target_embeddings, model, loss)
    
    # Extract critical information
    critical_issues = []
    for analysis in results.values():
        if 'issues' in analysis:
            critical_issues.extend(analysis['issues'])
    
    return {
        'critical_issues': critical_issues,
        'recommendations': results['recommendations'],
        'plateau_detected': results['plateau']['plateau_detected'],
        'loss_components': results['loss_components']['components'],
        'detailed_results': results
    } 