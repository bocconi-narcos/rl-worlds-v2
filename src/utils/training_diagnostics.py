"""
Comprehensive Training Diagnostics for VICReg and Auxiliary Loss Training

This module provides SOTA diagnostic tools for debugging training issues,
especially focused on VICReg and auxiliary loss training problems.

Features:
- Gradient flow analysis
- Loss component breakdown
- Representation quality metrics
- Training dynamics monitoring
- Numerical stability checks
- Performance profiling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
from collections import defaultdict, deque
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiagnosticConfig:
    """Configuration for training diagnostics"""
    # Gradient analysis
    check_gradients: bool = True
    gradient_norm_threshold: float = 1e-6
    max_gradient_norm: float = 10.0
    
    # Loss analysis
    track_loss_components: bool = True
    loss_history_length: int = 1000
    
    # Representation analysis
    check_representations: bool = True
    representation_similarity_threshold: float = 0.95
    
    # Numerical stability
    check_numerical_stability: bool = True
    nan_inf_threshold: float = 1e6
    
    # Performance profiling
    enable_profiling: bool = False
    profile_frequency: int = 100
    
    # Output settings
    save_diagnostics: bool = True
    output_dir: str = "training_diagnostics"
    plot_frequency: int = 50

class TrainingDiagnostics:
    """
    SOTA Training Diagnostics for VICReg and Auxiliary Loss Training
    
    This class provides comprehensive diagnostic tools to identify and resolve
    training issues, especially for VICReg loss plateau problems.
    """
    
    def __init__(self, config: DiagnosticConfig = None):
        self.config = config or DiagnosticConfig()
        self.diagnostics = defaultdict(list)
        self.history = defaultdict(deque)
        self.step_count = 0
        self.epoch_count = 0
        
        # Initialize output directory
        if self.config.save_diagnostics:
            self.output_dir = Path(self.config.output_dir)
            self.output_dir.mkdir(exist_ok=True)
            
        # Performance tracking
        self.timing_stats = defaultdict(list)
        self.memory_stats = defaultdict(list)
        
        logger.info("Training Diagnostics initialized")
    
    def analyze_gradient_flow(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive gradient flow analysis
        
        Returns:
            Dict containing gradient statistics and potential issues
        """
        if not self.config.check_gradients:
            return {}
            
        grad_stats = {}
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Analyze gradients by layer
        total_norm = 0
        param_grads = {}
        layer_stats = defaultdict(list)
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm += grad_norm ** 2
                param_grads[name] = grad_norm
                
                # Group by layer type
                layer_type = name.split('.')[0] if '.' in name else 'other'
                layer_stats[layer_type].append(grad_norm)
        
        total_norm = total_norm ** 0.5
        
        # Detect issues
        issues = []
        
        # Vanishing gradients
        if total_norm < self.config.gradient_norm_threshold:
            issues.append(f"VANISHING_GRADIENTS: Total norm {total_norm:.2e} < {self.config.gradient_norm_threshold}")
        
        # Exploding gradients
        if total_norm > self.config.max_gradient_norm:
            issues.append(f"EXPLODING_GRADIENTS: Total norm {total_norm:.2e} > {self.config.max_gradient_norm}")
        
        # Layer-specific issues
        for layer_type, norms in layer_stats.items():
            if len(norms) > 0:
                avg_norm = np.mean(norms)
                if avg_norm < 1e-8:
                    issues.append(f"LAYER_VANISHING: {layer_type} avg norm {avg_norm:.2e}")
                elif avg_norm > 10:
                    issues.append(f"LAYER_EXPLODING: {layer_type} avg norm {avg_norm:.2e}")
        
        grad_stats.update({
            'total_norm': total_norm,
            'param_grads': param_grads,
            'layer_stats': dict(layer_stats),
            'issues': issues,
            'step': self.step_count
        })
        
        return grad_stats
    
    def analyze_loss_components(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Detailed loss component analysis
        
        Args:
            loss_dict: Dictionary containing loss components
            
        Returns:
            Dict containing loss analysis and potential issues
        """
        if not self.config.track_loss_components:
            return {}
            
        loss_analysis = {}
        issues = []
        
        # Convert to scalar values
        loss_scalars = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        # Check for NaN/Inf
        for name, value in loss_scalars.items():
            if not np.isfinite(value):
                issues.append(f"LOSS_NAN_INF: {name} = {value}")
            elif abs(value) > self.config.nan_inf_threshold:
                issues.append(f"LOSS_EXPLOSION: {name} = {value:.2e}")
        
        # Analyze loss ratios
        if 'total_loss' in loss_scalars and 'primary_loss' in loss_scalars:
            aux_ratio = (loss_scalars['total_loss'] - loss_scalars['primary_loss']) / loss_scalars['primary_loss']
            if aux_ratio > 10:
                issues.append(f"AUX_LOSS_DOMINANT: Auxiliary loss {aux_ratio:.1f}x primary loss")
            elif aux_ratio < 0.01:
                issues.append(f"AUX_LOSS_NEGLIGIBLE: Auxiliary loss {aux_ratio:.3f}x primary loss")
        
        # Track loss history
        for name, value in loss_scalars.items():
            if name not in self.history:
                self.history[name] = deque(maxlen=self.config.loss_history_length)
            self.history[name].append(value)
        
        # Analyze loss trends
        trend_analysis = {}
        for name, values in self.history.items():
            if len(values) >= 10:
                recent_values = list(values)[-10:]
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                trend_analysis[name] = trend
                
                if trend > 0 and name != 'learning_rate':
                    issues.append(f"LOSS_INCREASING: {name} trend {trend:.2e}")
        
        loss_analysis.update({
            'loss_scalars': loss_scalars,
            'aux_ratio': aux_ratio if 'total_loss' in loss_scalars and 'primary_loss' in loss_scalars else None,
            'trend_analysis': trend_analysis,
            'issues': issues,
            'step': self.step_count
        })
        
        return loss_analysis
    
    def analyze_representations(self, embeddings: torch.Tensor, 
                              target_embeddings: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Comprehensive representation quality analysis
        
        Args:
            embeddings: Model embeddings
            target_embeddings: Target embeddings (if available)
            
        Returns:
            Dict containing representation analysis
        """
        if not self.config.check_representations:
            return {}
            
        rep_analysis = {}
        issues = []
        
        # Basic statistics
        rep_mean = embeddings.mean().item()
        rep_std = embeddings.std().item()
        rep_norm = embeddings.norm().item()
        
        # Check for representation collapse
        if rep_std < 0.01:
            issues.append(f"REP_COLLAPSE: Std {rep_std:.2e} very low")
        
        # Check for representation explosion
        if rep_norm > 100:
            issues.append(f"REP_EXPLOSION: Norm {rep_norm:.2e} very high")
        
        # Feature-wise analysis
        feature_std = embeddings.std(dim=0)
        feature_mean = embeddings.mean(dim=0)
        
        # Check for dead features
        dead_features = (feature_std < 1e-6).sum().item()
        if dead_features > 0:
            issues.append(f"DEAD_FEATURES: {dead_features} features with std < 1e-6")
        
        # Check for saturated features
        saturated_features = (torch.abs(feature_mean) > 10).sum().item()
        if saturated_features > 0:
            issues.append(f"SATURATED_FEATURES: {saturated_features} features with mean > 10")
        
        # Similarity analysis (if target available)
        if target_embeddings is not None:
            similarity = F.cosine_similarity(embeddings.flatten(), target_embeddings.flatten(), dim=0).item()
            if similarity > self.config.representation_similarity_threshold:
                issues.append(f"REP_TOO_SIMILAR: Similarity {similarity:.3f} > {self.config.representation_similarity_threshold}")
        
        # Correlation analysis
        if embeddings.shape[1] > 1:
            corr_matrix = torch.corrcoef(embeddings.T)
            off_diag_corr = corr_matrix.fill_diagonal_(0)
            max_corr = off_diag_corr.abs().max().item()
            if max_corr > 0.9:
                issues.append(f"HIGH_CORRELATION: Max off-diagonal correlation {max_corr:.3f}")
        
        rep_analysis.update({
            'mean': rep_mean,
            'std': rep_std,
            'norm': rep_norm,
            'dead_features': dead_features,
            'saturated_features': saturated_features,
            'max_correlation': max_corr if embeddings.shape[1] > 1 else None,
            'similarity': similarity if target_embeddings is not None else None,
            'issues': issues,
            'step': self.step_count
        })
        
        return rep_analysis
    
    def check_numerical_stability(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive numerical stability analysis
        """
        if not self.config.check_numerical_stability:
            return {}
            
        stability_analysis = {}
        issues = []
        
        # Check model parameters
        param_stats = {}
        for name, param in model.named_parameters():
            param_norm = param.norm().item()
            param_std = param.std().item()
            param_stats[name] = {'norm': param_norm, 'std': param_std}
            
            if not np.isfinite(param_norm):
                issues.append(f"PARAM_NAN_INF: {name} norm {param_norm}")
            elif param_norm > self.config.nan_inf_threshold:
                issues.append(f"PARAM_EXPLOSION: {name} norm {param_norm:.2e}")
        
        # Check loss value
        loss_value = loss.item()
        if not np.isfinite(loss_value):
            issues.append(f"LOSS_NAN_INF: Loss value {loss_value}")
        elif abs(loss_value) > self.config.nan_inf_threshold:
            issues.append(f"LOSS_EXPLOSION: Loss value {loss_value:.2e}")
        
        # Check gradients
        loss.backward(retain_graph=True)
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm
                
                if not np.isfinite(grad_norm):
                    issues.append(f"GRAD_NAN_INF: {name} grad norm {grad_norm}")
                elif grad_norm > self.config.nan_inf_threshold:
                    issues.append(f"GRAD_EXPLOSION: {name} grad norm {grad_norm:.2e}")
        
        stability_analysis.update({
            'param_stats': param_stats,
            'grad_stats': grad_stats,
            'loss_value': loss_value,
            'issues': issues,
            'step': self.step_count
        })
        
        return stability_analysis
    
    def profile_performance(self, func, *args, **kwargs):
        """
        Performance profiling wrapper
        """
        if not self.config.enable_profiling:
            return func(*args, **kwargs)
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        self.timing_stats[func.__name__].append(end_time - start_time)
        self.memory_stats[func.__name__].append(end_memory - start_memory)
        
        return result
    
    def step(self, model: nn.Module, loss: torch.Tensor, 
             loss_dict: Dict[str, torch.Tensor] = None,
             embeddings: torch.Tensor = None,
             target_embeddings: torch.Tensor = None) -> Dict[str, Any]:
        """
        Perform comprehensive diagnostic step
        
        Returns:
            Dict containing all diagnostic information
        """
        self.step_count += 1
        
        step_diagnostics = {}
        
        # Gradient analysis
        step_diagnostics['gradients'] = self.analyze_gradient_flow(model, loss)
        
        # Loss analysis
        if loss_dict:
            step_diagnostics['loss'] = self.analyze_loss_components(loss_dict)
        
        # Representation analysis
        if embeddings is not None:
            step_diagnostics['representations'] = self.analyze_representations(embeddings, target_embeddings)
        
        # Numerical stability
        step_diagnostics['stability'] = self.check_numerical_stability(model, loss)
        
        # Store diagnostics
        self.diagnostics[self.step_count] = step_diagnostics
        
        # Log issues
        all_issues = []
        for analysis in step_diagnostics.values():
            if 'issues' in analysis:
                all_issues.extend(analysis['issues'])
        
        if all_issues:
            logger.warning(f"Step {self.step_count} issues: {all_issues}")
        
        # Save diagnostics periodically
        if self.config.save_diagnostics and self.step_count % self.config.plot_frequency == 0:
            self.save_diagnostics()
        
        return step_diagnostics
    
    def save_diagnostics(self):
        """Save diagnostic data to disk"""
        if not self.config.save_diagnostics:
            return
        
        # Save raw data
        with open(self.output_dir / f"diagnostics_step_{self.step_count}.pkl", 'wb') as f:
            pickle.dump(self.diagnostics, f)
        
        # Save summary
        summary = self.generate_summary()
        with open(self.output_dir / f"summary_step_{self.step_count}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate plots
        self.generate_plots()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate diagnostic summary"""
        summary = {
            'total_steps': self.step_count,
            'total_epochs': self.epoch_count,
            'critical_issues': [],
            'performance_stats': {}
        }
        
        # Collect all issues
        all_issues = []
        for step_diag in self.diagnostics.values():
            for analysis in step_diag.values():
                if 'issues' in analysis:
                    all_issues.extend(analysis['issues'])
        
        # Count issue types
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_type = issue.split(':')[0]
            issue_counts[issue_type] += 1
        
        summary['issue_counts'] = dict(issue_counts)
        
        # Performance stats
        if self.timing_stats:
            summary['performance_stats']['timing'] = {
                func: {'mean': np.mean(times), 'std': np.std(times)}
                for func, times in self.timing_stats.items()
            }
        
        if self.memory_stats:
            summary['performance_stats']['memory'] = {
                func: {'mean': np.mean(mem), 'std': np.std(mem)}
                for func, mem in self.memory_stats.items()
            }
        
        return summary
    
    def generate_plots(self):
        """Generate diagnostic plots"""
        if not self.config.save_diagnostics:
            return
        
        # Loss trends
        if self.history:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss components
            for i, (name, values) in enumerate(self.history.items()):
                if i < 4:  # Plot first 4 components
                    ax = axes[i // 2, i % 2]
                    ax.plot(list(values))
                    ax.set_title(f'{name} over time')
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Loss')
                    ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'loss_trends_step_{self.step_count}.png')
            plt.close()
        
        # Gradient norms
        if 'gradients' in self.diagnostics:
            grad_norms = [diag['gradients']['total_norm'] 
                         for diag in self.diagnostics.values() 
                         if 'gradients' in diag]
            
            if grad_norms:
                plt.figure(figsize=(10, 6))
                plt.plot(grad_norms)
                plt.title('Gradient Norms over Time')
                plt.xlabel('Step')
                plt.ylabel('Gradient Norm')
                plt.yscale('log')
                plt.grid(True)
                plt.savefig(self.output_dir / f'gradient_norms_step_{self.step_count}.png')
                plt.close()
    
    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on diagnostics"""
        recommendations = []
        
        # Analyze recent diagnostics
        recent_steps = list(self.diagnostics.keys())[-10:]
        recent_issues = []
        
        for step in recent_steps:
            for analysis in self.diagnostics[step].values():
                if 'issues' in analysis:
                    recent_issues.extend(analysis['issues'])
        
        # Generate recommendations
        if any('VANISHING_GRADIENTS' in issue for issue in recent_issues):
            recommendations.append("Increase learning rate or reduce loss coefficients")
        
        if any('EXPLODING_GRADIENTS' in issue for issue in recent_issues):
            recommendations.append("Reduce learning rate or apply gradient clipping")
        
        if any('REP_COLLAPSE' in issue for issue in recent_issues):
            recommendations.append("Increase std_coeff in VICReg or adjust lambda in Barlow Twins")
        
        if any('AUX_LOSS_DOMINANT' in issue for issue in recent_issues):
            recommendations.append("Reduce auxiliary loss weight or coefficients")
        
        if any('LOSS_INCREASING' in issue for issue in recent_issues):
            recommendations.append("Check learning rate and loss function implementation")
        
        return recommendations

# Convenience function for quick diagnostics
def quick_diagnostic(model: nn.Module, loss: torch.Tensor, 
                    loss_dict: Dict[str, torch.Tensor] = None,
                    embeddings: torch.Tensor = None) -> Dict[str, Any]:
    """
    Quick diagnostic for immediate issues
    
    Args:
        model: The model being trained
        loss: Current loss tensor
        loss_dict: Dictionary of loss components
        embeddings: Current embeddings
        
    Returns:
        Dict with critical issues and recommendations
    """
    config = DiagnosticConfig(
        check_gradients=True,
        track_loss_components=True,
        check_representations=True,
        check_numerical_stability=True,
        save_diagnostics=False
    )
    
    diagnostics = TrainingDiagnostics(config)
    results = diagnostics.step(model, loss, loss_dict, embeddings)
    
    # Extract critical issues
    critical_issues = []
    for analysis in results.values():
        if 'issues' in analysis:
            critical_issues.extend(analysis['issues'])
    
    recommendations = diagnostics.get_recommendations()
    
    return {
        'critical_issues': critical_issues,
        'recommendations': recommendations,
        'detailed_results': results
    } 