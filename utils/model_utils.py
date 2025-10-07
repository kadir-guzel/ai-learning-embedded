#!/usr/bin/env python3
"""
Shared utilities for AI learning examples
=========================================

Common functions used across different examples for benchmarking,
visualization, and model analysis.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from pathlib import Path

def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def estimate_model_size(model, precision='fp32'):
    """Estimate model size in MB based on parameter count"""
    param_count = count_parameters(model)['total']
    
    if precision == 'fp32':
        bytes_per_param = 4
    elif precision == 'fp16':
        bytes_per_param = 2
    elif precision == 'int8':
        bytes_per_param = 1
    else:
        bytes_per_param = 4  # Default to fp32
    
    size_bytes = param_count * bytes_per_param
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb

def benchmark_inference_speed(model, input_tensor, device='cpu', num_runs=100, warmup_runs=10):
    """Benchmark model inference speed"""
    
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Benchmark runs
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

def plot_training_curves(train_losses, val_losses=None, train_accs=None, val_accs=None):
    """Plot training curves"""
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    if train_accs:
        axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy')
    if val_accs:
        axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    if train_accs or val_accs:
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    return fig

def create_model_summary(model, input_shape=(1, 3, 32, 32)):
    """Create a comprehensive model summary"""
    
    # Parameter count
    params = count_parameters(model)
    
    # Model size estimates
    fp32_size = estimate_model_size(model, 'fp32')
    int8_size = estimate_model_size(model, 'int8')
    
    # Create dummy input for shape inference
    dummy_input = torch.randn(*input_shape)
    
    summary = {
        'architecture': model.__class__.__name__,
        'parameters': params,
        'model_size_mb': {
            'fp32': fp32_size,
            'int8': int8_size
        },
        'input_shape': input_shape,
        'compression_ratio': fp32_size / int8_size
    }
    
    return summary

def print_model_summary(model, input_shape=(1, 3, 32, 32)):
    """Print a formatted model summary"""
    
    summary = create_model_summary(model, input_shape)
    
    print("📋 MODEL SUMMARY")
    print("=" * 40)
    print(f"Architecture: {summary['architecture']}")
    print(f"Input Shape: {summary['input_shape']}")
    print(f"")
    print(f"Parameters:")
    print(f"  Total: {summary['parameters']['total']:,}")
    print(f"  Trainable: {summary['parameters']['trainable']:,}")
    print(f"  Non-trainable: {summary['parameters']['non_trainable']:,}")
    print(f"")
    print(f"Model Size:")
    print(f"  FP32: {summary['model_size_mb']['fp32']:.2f} MB")
    print(f"  INT8: {summary['model_size_mb']['int8']:.2f} MB")
    print(f"  Compression: {summary['compression_ratio']:.1f}x")
    print("=" * 40)

def save_model_safely(model, filepath, include_optimizer=None):
    """Save model with error handling"""
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': time.time()
    }
    
    if include_optimizer:
        save_dict['optimizer_state_dict'] = include_optimizer.state_dict()
    
    try:
        torch.save(save_dict, filepath)
        print(f"✅ Model saved to: {filepath}")
        
        # Report file size
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"💾 File size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def load_model_safely(filepath, model_class):
    """Load model with error handling"""
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create model instance
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Model loaded from: {filepath}")
        return model, checkpoint
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def compare_models(models_dict, input_tensor, device='cpu'):
    """Compare multiple models on the same input"""
    
    print("📊 MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Parameters':<12} {'Size (MB)':<10} {'Speed (ms)':<12}")
    print("-" * 60)
    
    results = {}
    
    for name, model in models_dict.items():
        # Parameters
        params = count_parameters(model)['total']
        
        # Size
        size_mb = estimate_model_size(model)
        
        # Speed
        speed_stats = benchmark_inference_speed(model, input_tensor, device, num_runs=50)
        avg_speed = speed_stats['mean']
        
        results[name] = {
            'parameters': params,
            'size_mb': size_mb,
            'speed_ms': avg_speed
        }
        
        print(f"{name:<20} {params:<12,} {size_mb:<10.2f} {avg_speed:<12.2f}")
    
    print("-" * 60)
    
    return results

def create_confusion_matrix_plot(y_true, y_pred, class_names=None):
    """Create a confusion matrix plot"""
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    return plt.gcf()

def analyze_model_complexity(model):
    """Analyze model complexity and provide recommendations"""
    
    params = count_parameters(model)
    size_mb = estimate_model_size(model)
    
    # Complexity categories
    if params['total'] < 100_000:
        complexity = "Low"
        recommendations = [
            "✅ Suitable for microcontrollers",
            "✅ Fast inference on mobile devices",
            "✅ Minimal memory requirements"
        ]
    elif params['total'] < 1_000_000:
        complexity = "Medium"
        recommendations = [
            "✅ Good for mobile applications",
            "⚠️  Consider quantization for embedded use",
            "✅ Balanced accuracy/efficiency trade-off"
        ]
    else:
        complexity = "High"
        recommendations = [
            "⚠️  May be too large for mobile deployment",
            "🔧 Consider model pruning or distillation",
            "💾 Quantization strongly recommended"
        ]
    
    analysis = {
        'complexity_level': complexity,
        'parameter_count': params['total'],
        'size_mb': size_mb,
        'recommendations': recommendations
    }
    
    return analysis

def print_complexity_analysis(model):
    """Print formatted complexity analysis"""
    
    analysis = analyze_model_complexity(model)
    
    print(f"\n🔍 MODEL COMPLEXITY ANALYSIS")
    print("=" * 40)
    print(f"Complexity Level: {analysis['complexity_level']}")
    print(f"Parameters: {analysis['parameter_count']:,}")
    print(f"Estimated Size: {analysis['size_mb']:.2f} MB")
    print(f"\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
    print("=" * 40)

if __name__ == "__main__":
    print("🛠️  AI Learning Utilities")
    print("This module provides common utilities for the AI learning examples.")
    print("Import functions as needed in your experiments.")