#!/usr/bin/env python3
"""
VICReg Diagnostics Test Script

This script runs comprehensive diagnostics on VICReg training to identify
the causes of loss plateau issues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# Import our diagnostic tools
from src.utils.vicreg_diagnostics import VICRegDiagnostics, quick_vicreg_diagnostic
from src.losses.vicreg import VICRegLoss
from src.losses.barlow_twins import BarlowTwinsLoss

def create_test_data(batch_size=32, seq_len=10, embed_dim=256, num_batches=100):
    """Create synthetic test data"""
    print("Creating synthetic test data...")
    
    # Create synthetic embeddings
    all_embeddings = []
    all_targets = []
    
    for _ in range(num_batches):
        # Create embeddings with some structure
        embeddings = torch.randn(batch_size, embed_dim)
        
        # Add some correlation structure
        embeddings[:, :embed_dim//4] += 0.5 * embeddings[:, embed_dim//4:embed_dim//2]
        
        # Create target embeddings (slightly different)
        targets = embeddings + 0.1 * torch.randn_like(embeddings)
        
        all_embeddings.append(embeddings)
        all_targets.append(targets)
    
    return all_embeddings, all_targets

def test_vicreg_configurations():
    """Test different VICReg configurations to identify issues"""
    
    print("=== VICReg Configuration Diagnostics ===")
    
    # Test configurations
    configs = [
        {
            'name': 'vicreg_standard',
            'sim_coeff': 1.0,
            'std_coeff': 1.0,
            'cov_coeff': 0.8,
            'proj_hidden_dim': 256,
            'proj_output_dim': 256
        },
        {
            'name': 'vicreg_high_weight',
            'sim_coeff': 25.0,
            'std_coeff': 25.0,
            'cov_coeff': 1.0,
            'proj_hidden_dim': 512,
            'proj_output_dim': 512
        },
        {
            'name': 'vicreg_no_projector',
            'sim_coeff': 1.0,
            'std_coeff': 1.0,
            'cov_coeff': 0.8,
            'proj_hidden_dim': None,
            'proj_output_dim': None
        }
    ]
    
    # Create test data
    test_embeddings, test_targets = create_test_data()
    
    results = {}
    
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")
        
        # Initialize VICReg
        if config['proj_hidden_dim'] is not None:
            vicreg = VICRegLoss(
                sim_coeff=config['sim_coeff'],
                std_coeff=config['std_coeff'],
                cov_coeff=config['cov_coeff'],
                proj_hidden_dim=config['proj_hidden_dim'],
                proj_output_dim=config['proj_output_dim'],
                representation_dim=256
            )
        else:
            vicreg = VICRegLoss(
                sim_coeff=config['sim_coeff'],
                std_coeff=config['std_coeff'],
                cov_coeff=config['cov_coeff']
            )
        
        # Initialize diagnostics
        diagnostics = VICRegDiagnostics(save_dir=f"vicreg_diagnostics_{config['name']}")
        
        # Test on first few batches
        config_results = []
        for i, (embeddings, targets) in enumerate(zip(test_embeddings[:10], test_targets[:10])):
            print(f"  Batch {i+1}/10")
            
            # Create dummy model and loss for diagnostics
            dummy_model = nn.Linear(256, 256)
            dummy_loss = torch.tensor(1.0, requires_grad=True)
            
            # Run diagnostics
            diagnosis = diagnostics.diagnose_vicreg_step(
                vicreg, embeddings, targets, dummy_model, dummy_loss)
            
            config_results.append(diagnosis)
            
            # Check for immediate issues
            if diagnosis['recommendations']:
                print(f"    Issues detected: {diagnosis['recommendations']}")
        
        results[config['name']] = config_results
    
    return results

def test_training_dynamics():
    """Test training dynamics to identify plateau causes"""
    
    print("\n=== Training Dynamics Test ===")
    
    # Create test data
    test_embeddings, test_targets = create_test_data()
    
    # Test different learning rates
    lrs = [1e-3, 1e-4, 1e-5]
    
    for lr in lrs:
        print(f"\n--- Testing Learning Rate {lr} ---")
        
        # Initialize VICReg with high weights (problematic config)
        vicreg = VICRegLoss(
            sim_coeff=25.0,
            std_coeff=25.0,
            cov_coeff=1.0,
            proj_hidden_dim=512,
            proj_output_dim=512,
            representation_dim=256
        )
        
        # Create optimizer
        optimizer = optim.Adam(vicreg.parameters(), lr=lr)
        
        # Training loop
        losses = []
        for i, (embeddings, targets) in enumerate(zip(test_embeddings[:20], test_targets[:20])):
            optimizer.zero_grad()
            
            # Compute loss
            loss_tuple = vicreg(embeddings, targets)
            total_loss = loss_tuple[0]  # Get the total loss (first element)
            losses.append(total_loss.item())
            
            # Backward pass
            total_loss.backward()
            
            # Check gradients
            total_norm = 0
            for param in vicreg.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            
            print(f"  Step {i+1}: Loss={total_loss.item():.6f}, GradNorm={total_norm:.6f}")
            
            # Check for issues
            if total_norm < 1e-6:
                print(f"    WARNING: Vanishing gradients detected!")
            elif total_norm > 10:
                print(f"    WARNING: Exploding gradients detected!")
            
            optimizer.step()
        
        # Analyze loss curve
        if len(losses) >= 10:
            recent_losses = losses[-10:]
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            variance = np.var(recent_losses)
            
            print(f"  Loss trend: {trend:.2e}")
            print(f"  Loss variance: {variance:.2e}")
            
            if abs(trend) < 1e-6:
                print(f"  PLATEAU DETECTED: Zero trend")
            if variance < 1e-8:
                print(f"  PLATEAU DETECTED: Zero variance")

def test_representation_quality():
    """Test representation quality and identify collapse issues"""
    
    print("\n=== Representation Quality Test ===")
    
    # Create test data with different characteristics
    batch_size, embed_dim = 32, 256
    
    # Test case 1: Normal embeddings
    normal_embeddings = torch.randn(batch_size, embed_dim)
    
    # Test case 2: Collapsed embeddings (low variance)
    collapsed_embeddings = torch.randn(batch_size, embed_dim) * 0.01
    
    # Test case 3: Correlated embeddings
    correlated_embeddings = torch.randn(batch_size, embed_dim)
    correlated_embeddings[:, :embed_dim//2] += 0.8 * correlated_embeddings[:, embed_dim//2:]
    
    # Test case 4: Saturated embeddings
    saturated_embeddings = torch.randn(batch_size, embed_dim) * 10
    
    test_cases = [
        ('normal', normal_embeddings),
        ('collapsed', collapsed_embeddings),
        ('correlated', correlated_embeddings),
        ('saturated', saturated_embeddings)
    ]
    
    for case_name, embeddings in test_cases:
        print(f"\n--- Testing {case_name} embeddings ---")
        
        # Analyze representation quality
        mean_val = embeddings.mean().item()
        std_val = embeddings.std().item()
        norm_val = embeddings.norm().item()
        
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std: {std_val:.6f}")
        print(f"  Norm: {norm_val:.6f}")
        
        # Check for issues
        issues = []
        if std_val < 0.01:
            issues.append("REPRESENTATION_COLLAPSE")
        if norm_val > 100:
            issues.append("REPRESENTATION_EXPLOSION")
        
        # Correlation analysis
        if embed_dim > 1:
            corr_matrix = torch.corrcoef(embeddings.T)
            off_diag_corr = corr_matrix.fill_diagonal_(0)
            max_corr = off_diag_corr.abs().max().item()
            
            print(f"  Max correlation: {max_corr:.6f}")
            
            if max_corr > 0.9:
                issues.append("HIGH_CORRELATION")
        
        if issues:
            print(f"  ISSUES DETECTED: {issues}")
        else:
            print(f"  No issues detected")

def generate_diagnostic_report():
    """Generate comprehensive diagnostic report"""
    
    print("\n=== Generating Diagnostic Report ===")
    
    # Run all tests
    config_results = test_vicreg_configurations()
    test_training_dynamics()
    test_representation_quality()
    
    # Generate summary report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'configs_tested': len(config_results),
            'critical_issues_found': [],
            'recommendations': []
        }
    }
    
    # Analyze results
    for config_name, results in config_results.items():
        for result in results:
            if result['recommendations']:
                report['summary']['critical_issues_found'].extend(result['recommendations'])
    
    # Remove duplicates
    report['summary']['critical_issues_found'] = list(set(report['summary']['critical_issues_found']))
    
    # Generate recommendations
    if any('VANISHING_GRADIENTS' in issue for issue in report['summary']['critical_issues_found']):
        report['summary']['recommendations'].append("Increase learning rate or reduce loss coefficients")
    
    if any('EXPLODING_GRADIENTS' in issue for issue in report['summary']['critical_issues_found']):
        report['summary']['recommendations'].append("Reduce learning rate or apply gradient clipping")
    
    if any('REPRESENTATION_COLLAPSE' in issue for issue in report['summary']['critical_issues_found']):
        report['summary']['recommendations'].append("Increase std_coeff or check data normalization")
    
    if any('HIGH_CORRELATION' in issue for issue in report['summary']['critical_issues_found']):
        report['summary']['recommendations'].append("Increase cov_coeff or check feature diversity")
    
    # Save report
    with open('vicreg_diagnostic_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDiagnostic report saved to 'vicreg_diagnostic_report.json'")
    print(f"Critical issues found: {len(report['summary']['critical_issues_found'])}")
    print(f"Recommendations: {len(report['summary']['recommendations'])}")
    
    return report

if __name__ == "__main__":
    print("VICReg Diagnostics Test Suite")
    print("=" * 50)
    
    # Run comprehensive diagnostics
    report = generate_diagnostic_report()
    
    print("\n=== DIAGNOSTIC COMPLETE ===")
    print("Check the generated files for detailed analysis:")
    print("- vicreg_diagnostic_report.json: Summary report")
    print("- vicreg_diagnostics_*/: Detailed diagnostics for each config")
    print("- vicreg_diagnostics_*/vicreg_diagnostics_step_*.png: Diagnostic plots") 