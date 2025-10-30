#!/usr/bin/env python3
"""
Custom Loss Functions for Edge AI - Version with Issues for PyTrim Testing
PyTrim: https://github.com/TrimTeam/PyTrim
=========================================================================

This example demonstrates PyTorch's flexibility in creating custom loss functions
specifically designed for edge AI deployment. We'll implement loss functions that
balance accuracy with model efficiency constraints.

This version includes unused imports and dependencies for PyTrim testing.

Key Concepts:
- Custom loss functions that consider model complexity
- Sparsity regularization for better quantization
- Mobile-friendly architectures (DepthwiseSeparable convolutions)
- Dynamic loss weighting during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# UNUSED IMPORTS - PyTrim should detect and remove these
import os
import sys
import json
import pickle
import datetime
import random
import logging
import argparse
import collections
import itertools
import functools
import pathlib
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import csv
import sqlite3
import threading
import multiprocessing
import asyncio
import socket
import hashlib
import base64
import re
import math
import statistics
import decimal
import fractions

# More unused scientific computing imports
import scipy
import scipy.stats
import scipy.optimize
import sklearn
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFilter
import cv2
import requests
import urllib3
import boto3
import tensorflow as tf
from tensorflow import keras
import jax
import jax.numpy as jnp
from transformers import pipeline
import datasets
from datasets import load_dataset

class EfficiencyAwareLoss(nn.Module):
    """
    Custom loss function that balances classification accuracy with model efficiency.
    Perfect for edge AI where model size and complexity matter.
    """
    
    def __init__(self, 
                 complexity_weight=0.001, 
                 sparsity_weight=0.0001,
                 efficiency_schedule='static'):
        super(EfficiencyAwareLoss, self).__init__()
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.complexity_weight = complexity_weight
        self.sparsity_weight = sparsity_weight
        self.efficiency_schedule = efficiency_schedule
        self.current_epoch = 0
        
    def forward(self, outputs, targets, model):
        """
        Compute multi-objective loss for edge AI optimization
        
        Args:
            outputs: Model predictions
            targets: Ground truth labels
            model: The neural network model
            
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary of individual loss components
        """
        
        # Standard classification loss
        classification_loss = self.classification_loss(outputs, targets)
        
        # Model complexity penalty (parameter count)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        complexity_penalty = total_params * self.complexity_weight
        
        # Sparsity regularization (L1 norm) - promotes prunable models
        sparsity_penalty = 0
        for name, param in model.named_parameters():
            if 'weight' in name:  # Only apply to weight parameters
                sparsity_penalty += torch.sum(torch.abs(param))
        sparsity_penalty *= self.sparsity_weight
        
        # Dynamic weighting based on training progress
        efficiency_weight = self._get_efficiency_weight()
        
        # Combined loss
        total_loss = (classification_loss + 
                     efficiency_weight * complexity_penalty + 
                     efficiency_weight * sparsity_penalty)
        
        # Return detailed breakdown for monitoring
        loss_components = {
            'classification': classification_loss.item(),
            'complexity': complexity_penalty.item(),
            'sparsity': sparsity_penalty.item(),
            'efficiency_weight': efficiency_weight,
            'total': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _get_efficiency_weight(self):
        """Dynamic efficiency weighting during training"""
        if self.efficiency_schedule == 'increasing':
            # Gradually increase efficiency importance
            return min(1.0, self.current_epoch / 10.0)
        elif self.efficiency_schedule == 'decreasing':
            # Start high, then focus on accuracy
            return max(0.1, 1.0 - self.current_epoch / 10.0)
        else:  # static
            return 1.0
    
    def step_epoch(self):
        """Call this at the end of each epoch"""
        self.current_epoch += 1

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution - much more efficient than standard convolution.
    Used in MobileNets and other mobile-friendly architectures.
    
    Standard Conv: O(H * W * C_in * C_out * K^2)
    Depthwise Sep: O(H * W * C_in * K^2 + H * W * C_in * C_out)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution (spatial filtering)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=False
        )
        
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class EfficientMobileNet(nn.Module):
    """
    Mobile-friendly CNN using depthwise separable convolutions.
    Designed for edge AI deployment with custom loss function.
    """
    
    def __init__(self, num_classes=10):
        super(EfficientMobileNet, self).__init__()
        
        # Initial standard convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Depthwise separable blocks
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2, 2),
            
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2, 2),
            
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2, 2),
            
            DepthwiseSeparableConv(256, 512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def count_operations(model, input_size=(1, 3, 32, 32)):
    """Estimate FLOPs for the model"""
    def conv_flop_count(input_shape, output_shape, kernel_size, groups=1):
        batch_size, input_channels, input_height, input_width = input_shape
        output_batch_size, output_channels, output_height, output_width = output_shape
        
        kernel_flops = kernel_size * kernel_size
        output_elements = output_batch_size * output_channels * output_height * output_width
        
        if groups != 1:  # Depthwise convolution
            flops = kernel_flops * output_elements
        else:  # Standard convolution
            flops = kernel_flops * input_channels * output_elements
        
        return flops
    
    # This is a simplified FLOP counter - in practice you'd use tools like torchprofile
    total_flops = 0
    
    # Rough estimate based on model structure
    # In practice, use specialized tools for accurate FLOP counting
    dummy_input = torch.randn(input_size)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Count parameters as a proxy for complexity
    total_params = sum(p.numel() for p in model.parameters())
    
    return total_params

def setup_data():
    """Setup CIFAR-10 data with optimized transforms for mobile deployment"""
    
    # Lighter data augmentation for faster training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=False, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_with_custom_loss(model, train_loader, test_loader, device, num_epochs=5):
    """Train model using custom efficiency-aware loss function"""
    
    print(f"🚀 Training with custom loss on {device}...")
    
    # Custom loss function with efficiency constraints
    custom_loss = EfficiencyAwareLoss(
        complexity_weight=0.0001,
        sparsity_weight=0.0001,
        efficiency_schedule='increasing'
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.to(device)
    
    train_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0,
            'classification': 0,
            'complexity': 0,
            'sparsity': 0
        }
        
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            
            # Use custom loss function
            loss, loss_components = custom_loss(outputs, targets, model)
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            for key in epoch_losses:
                if key in loss_components:
                    epoch_losses[key] += loss_components[key]
                else:
                    epoch_losses[key] += loss_components['total']
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{100.*correct/total:.1f}%',
                    'Sparse': f'{loss_components["sparsity"]:.4f}'
                })
        
        # End of epoch
        custom_loss.step_epoch()
        scheduler.step()
        
        # Calculate averages
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        train_acc = 100. * correct / total
        val_acc = evaluate_model(model, test_loader, device)
        
        train_history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            **epoch_losses
        })
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"  Classification Loss: {epoch_losses['classification']:.4f}")
        print(f"  Complexity Penalty: {epoch_losses['complexity']:.6f}")
        print(f"  Sparsity Penalty: {epoch_losses['sparsity']:.6f}")
        print(f"  Total Loss: {epoch_losses['total']:.4f}")
    
    return model, train_history

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

def analyze_model_sparsity(model):
    """Analyze how sparse the model has become"""
    total_params = 0
    zero_params = 0
    
    sparsity_by_layer = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_total = param.numel()
            layer_zeros = (torch.abs(param) < 1e-6).sum().item()
            
            total_params += layer_total
            zero_params += layer_zeros
            
            layer_sparsity = layer_zeros / layer_total * 100
            sparsity_by_layer[name] = layer_sparsity
    
    overall_sparsity = zero_params / total_params * 100
    
    print(f"\n🔍 Model Sparsity Analysis:")
    print(f"📊 Overall sparsity: {overall_sparsity:.2f}%")
    print(f"📊 Zero parameters: {zero_params:,} / {total_params:,}")
    
    print("\n📋 Layer-wise sparsity:")
    for name, sparsity in sparsity_by_layer.items():
        print(f"  {name}: {sparsity:.1f}%")
    
    return overall_sparsity

def compare_models():
    """Compare standard model vs efficiency-optimized model"""
    
    print("\n📊 Model Comparison:")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Standard model (for comparison)
    standard_model = EfficientMobileNet(num_classes=10)
    efficient_model = EfficientMobileNet(num_classes=10)
    
    # Count parameters
    standard_params = sum(p.numel() for p in standard_model.parameters())
    efficient_params = sum(p.numel() for p in efficient_model.parameters())
    
    print(f"📱 MobileNet Parameters: {standard_params:,}")
    print(f"📱 Efficient MobileNet Parameters: {efficient_params:,}")
    
    # Estimate model sizes
    param_bytes = 4  # 32-bit floats
    standard_size = standard_params * param_bytes / (1024 * 1024)  # MB
    efficient_size = efficient_params * param_bytes / (1024 * 1024)  # MB
    
    print(f"💾 Standard model size: {standard_size:.2f} MB")
    print(f"💾 Efficient model size: {efficient_size:.2f} MB")
    
    return standard_model, efficient_model

def plot_training_history(history):
    """Plot training metrics over time"""
    
    epochs = [h['epoch'] for h in history]
    train_accs = [h['train_acc'] for h in history]
    val_accs = [h['val_acc'] for h in history]
    total_losses = [h['total'] for h in history]
    sparsity_losses = [h['sparsity'] for h in history]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy plot
    ax1.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax1.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(epochs, total_losses, 'g-', label='Total Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Sparsity plot
    ax3.plot(epochs, sparsity_losses, 'm-', label='Sparsity Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Sparsity Loss')
    ax3.set_title('Sparsity Regularization')
    ax3.legend()
    ax3.grid(True)
    
    # Loss components
    classification_losses = [h['classification'] for h in history]
    complexity_losses = [h['complexity'] for h in history]
    
    ax4.plot(epochs, classification_losses, 'b-', label='Classification')
    ax4.plot(epochs, complexity_losses, 'r-', label='Complexity')
    ax4.plot(epochs, sparsity_losses, 'm-', label='Sparsity')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Component')
    ax4.set_title('Loss Components')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution flow"""
    print("🚀 Custom Loss Functions for Edge AI")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Setup data
    train_loader, test_loader = setup_data()
    
    # Create efficient model
    print("\n🤖 Creating EfficientMobileNet...")
    model = EfficientMobileNet(num_classes=10)
    
    # Model analysis
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # Train with custom loss
    model, history = train_with_custom_loss(
        model, train_loader, test_loader, device, num_epochs=3
    )
    
    # Analyze sparsity
    sparsity = analyze_model_sparsity(model)
    
    # Final evaluation
    final_accuracy = evaluate_model(model, test_loader, device)
    
    # Plot training history (optional - comment out if running headless)
    # plot_training_history(history)
    
    # Model comparison
    standard_model, efficient_model = compare_models()
    
    print("\n" + "=" * 50)
    print("📋 CUSTOM LOSS EXAMPLE SUMMARY")
    print("=" * 50)
    print(f"✅ EfficientMobileNet trained successfully!")
    print(f"📈 Final Accuracy: {final_accuracy:.2f}%")
    print(f"🎯 Model Sparsity: {sparsity:.2f}%")
    print(f"🔧 Total Parameters: {total_params:,}")
    print(f"💾 Approximate Size: {total_params * 4 / (1024*1024):.2f} MB")
    
    print("\n💡 KEY ADVANTAGES OF CUSTOM LOSS:")
    print("- ✅ Balances accuracy with efficiency")
    print("- ✅ Encourages sparse, quantization-ready models")
    print("- ✅ Optimizes for edge deployment constraints")
    print("- ✅ Flexible loss weighting strategies")
    print("- ✅ Impossible to achieve with FastAI alone!")
    
    print("\n🎉 This demonstrates PyTorch's power for cutting-edge research!")

if __name__ == "__main__":
    main()