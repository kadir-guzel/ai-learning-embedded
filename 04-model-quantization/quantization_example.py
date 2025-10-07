#!/usr/bin/env python3
"""
Model Quantization for Edge Deployment
=====================================

This example demonstrates different quantization techniques to optimize models
for edge deployment. Essential for embedded systems with resource constraints.

Key Techniques:
- Post-Training Quantization (PTQ)
- Quantization-Aware Training (QAT)
- Dynamic Quantization
- Static Quantization with calibration

Perfect for embedded engineers moving into AI!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quantization
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
import copy
from tqdm import tqdm

# For calibration dataset
import random

class SimpleClassifier(nn.Module):
    """Simple CNN for demonstrating quantization techniques"""
    
    def __init__(self, num_classes=10):
        super(SimpleClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Add QuantStub and DeQuantStub for quantization
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)  # Quantize input
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)  # Dequantize output
        return x

def setup_data(subset_size=10000):
    """Setup CIFAR-10 data with option for smaller calibration set"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Full datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=False, transform=transform
    )
    
    # Create smaller calibration dataset
    calibration_indices = random.sample(range(len(train_dataset)), min(subset_size, len(train_dataset)))
    calibration_dataset = Subset(train_dataset, calibration_indices)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, calibration_loader

def train_baseline_model(model, train_loader, test_loader, device, num_epochs=3):
    """Train a baseline FP32 model"""
    
    print("🚀 Training baseline FP32 model...")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if len(pbar) % 100 == 0:
                accuracy = 100. * correct / total
                pbar.set_postfix({'Acc': f'{accuracy:.2f}%', 'Loss': f'{running_loss/100:.3f}'})
                running_loss = 0.0
        
        # Validation
        val_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}: Validation Accuracy: {val_acc:.2f}%")
    
    return model

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

def measure_inference_time(model, test_loader, device, num_batches=10):
    """Measure average inference time"""
    model.eval()
    
    times = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            data = data.to(device)
            
            # Warm up
            if i == 0:
                for _ in range(5):
                    _ = model(data)
            
            # Measure time
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            _ = model(data)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(batch_time)
    
    return np.mean(times), np.std(times)

def get_model_size(model, temp_path='temp_model.pth'):
    """Get model size in MB"""
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb

# 1. POST-TRAINING QUANTIZATION (PTQ)
def post_training_quantization(model, calibration_loader, device):
    """Apply post-training quantization"""
    
    print("\n🔧 Applying Post-Training Quantization...")
    
    # Prepare model for quantization
    model.eval()
    model_fp32 = copy.deepcopy(model)
    
    # Set quantization configuration
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model (inserts observers)
    model_fp32_prepared = torch.quantization.prepare(model_fp32)
    
    # Calibrate with representative data
    print("📊 Calibrating with representative data...")
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(calibration_loader, desc="Calibrating")):
            if i > 50:  # Limit calibration for speed
                break
            data = data.to(device)
            _ = model_fp32_prepared(data)
    
    # Convert to quantized model
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    
    print("✅ Post-training quantization complete!")
    return model_int8

# 2. QUANTIZATION-AWARE TRAINING (QAT)
def quantization_aware_training(model, train_loader, test_loader, device, num_epochs=2):
    """Apply quantization-aware training"""
    
    print("\n🚀 Starting Quantization-Aware Training...")
    
    # Prepare model for QAT
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_qat = torch.quantization.prepare_qat(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_qat.parameters(), lr=0.0001)  # Lower LR for QAT
    
    for epoch in range(num_epochs):
        model_qat.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'QAT Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model_qat(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                pbar.set_postfix({'Loss': f'{running_loss/100:.3f}'})
                running_loss = 0.0
            
            # Limit training for demo purposes
            if batch_idx > 200:
                break
        
        # Evaluate QAT model
        val_acc = evaluate_model(model_qat, test_loader, device)
        print(f"QAT Epoch {epoch+1}: Validation Accuracy: {val_acc:.2f}%")
    
    # Convert QAT model to quantized model
    model_qat.eval()
    model_qat_int8 = torch.quantization.convert(model_qat)
    
    print("✅ Quantization-aware training complete!")
    return model_qat_int8

# 3. DYNAMIC QUANTIZATION
def dynamic_quantization(model):
    """Apply dynamic quantization (weights only)"""
    
    print("\n⚡ Applying Dynamic Quantization...")
    
    # Dynamic quantization - only weights are quantized
    model_dynamic = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    print("✅ Dynamic quantization complete!")
    return model_dynamic

def compare_quantization_methods():
    """Compare different quantization approaches"""
    
    print("🚀 Model Quantization Comparison")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Setup data
    train_loader, test_loader, calibration_loader = setup_data()
    
    # Train baseline model
    print("\n📚 Training baseline model...")
    baseline_model = SimpleClassifier(num_classes=10)
    baseline_model = train_baseline_model(baseline_model, train_loader, test_loader, device)
    
    # Move to CPU for quantization (required for most quantization ops)
    baseline_model = baseline_model.to('cpu')
    test_loader_cpu = DataLoader(
        torchvision.datasets.CIFAR10(
            root='../data', train=False, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        ),
        batch_size=32, shuffle=False
    )
    
    calibration_loader_cpu = DataLoader(
        torchvision.datasets.CIFAR10(
            root='../data', train=True, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        ),
        batch_size=32, shuffle=False
    )
    
    # Results storage
    results = {}
    
    # 1. Baseline FP32 model
    print("\n📊 Evaluating baseline FP32 model...")
    fp32_accuracy = evaluate_model(baseline_model, test_loader_cpu, 'cpu')
    fp32_size = get_model_size(baseline_model)
    fp32_time, _ = measure_inference_time(baseline_model, test_loader_cpu, 'cpu')
    
    results['FP32 Baseline'] = {
        'accuracy': fp32_accuracy,
        'size_mb': fp32_size,
        'inference_time_ms': fp32_time
    }
    
    # 2. Post-Training Quantization
    try:
        ptq_model = post_training_quantization(
            copy.deepcopy(baseline_model), calibration_loader_cpu, 'cpu'
        )
        ptq_accuracy = evaluate_model(ptq_model, test_loader_cpu, 'cpu')
        ptq_size = get_model_size(ptq_model)
        ptq_time, _ = measure_inference_time(ptq_model, test_loader_cpu, 'cpu')
        
        results['Post-Training Quantization'] = {
            'accuracy': ptq_accuracy,
            'size_mb': ptq_size,
            'inference_time_ms': ptq_time
        }
    except Exception as e:
        print(f"❌ PTQ failed: {e}")
        results['Post-Training Quantization'] = None
    
    # 3. Dynamic Quantization
    try:
        dynamic_model = dynamic_quantization(copy.deepcopy(baseline_model))
        dynamic_accuracy = evaluate_model(dynamic_model, test_loader_cpu, 'cpu')
        dynamic_size = get_model_size(dynamic_model)
        dynamic_time, _ = measure_inference_time(dynamic_model, test_loader_cpu, 'cpu')
        
        results['Dynamic Quantization'] = {
            'accuracy': dynamic_accuracy,
            'size_mb': dynamic_size,
            'inference_time_ms': dynamic_time
        }
    except Exception as e:
        print(f"❌ Dynamic quantization failed: {e}")
        results['Dynamic Quantization'] = None
    
    # 4. Quantization-Aware Training (commented out due to time constraints)
    # try:
    #     qat_model = quantization_aware_training(
    #         copy.deepcopy(baseline_model), train_loader, test_loader_cpu, 'cpu'
    #     )
    #     qat_accuracy = evaluate_model(qat_model, test_loader_cpu, 'cpu')
    #     qat_size = get_model_size(qat_model)
    #     qat_time, _ = measure_inference_time(qat_model, test_loader_cpu, 'cpu')
    #     
    #     results['Quantization-Aware Training'] = {
    #         'accuracy': qat_accuracy,
    #         'size_mb': qat_size,
    #         'inference_time_ms': qat_time
    #     }
    # except Exception as e:
    #     print(f"❌ QAT failed: {e}")
    #     results['Quantization-Aware Training'] = None
    
    return results

def print_comparison_table(results):
    """Print formatted comparison table"""
    
    print("\n" + "=" * 80)
    print("📊 QUANTIZATION COMPARISON RESULTS")
    print("=" * 80)
    
    # Table header
    print(f"{'Method':<25} {'Accuracy (%)':<12} {'Size (MB)':<10} {'Speed (ms)':<12} {'Size Reduction':<15}")
    print("-" * 80)
    
    baseline_size = results['FP32 Baseline']['size_mb']
    
    for method, data in results.items():
        if data is None:
            print(f"{method:<25} {'Failed':<12} {'N/A':<10} {'N/A':<12} {'N/A':<15}")
            continue
            
        accuracy = data['accuracy']
        size = data['size_mb']
        speed = data['inference_time_ms']
        size_reduction = f"{baseline_size/size:.1f}x" if size > 0 else "N/A"
        
        print(f"{method:<25} {accuracy:<12.2f} {size:<10.2f} {speed:<12.2f} {size_reduction:<15}")
    
    print("-" * 80)
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print("- Post-Training Quantization: Quick, no retraining needed")
    print("- Dynamic Quantization: Good for CPU inference")  
    print("- Quantization-Aware Training: Best accuracy retention")
    print("- All methods reduce model size significantly")
    print("- Speed improvements depend on hardware support")

def demonstrate_edge_deployment():
    """Demonstrate practical edge deployment considerations"""
    
    print("\n🚀 Edge Deployment Considerations")
    print("=" * 50)
    
    print("📱 Typical Edge Device Constraints:")
    print("  • Memory: 1-4 GB RAM")
    print("  • Storage: 8-32 GB")
    print("  • Compute: ARM Cortex-A, limited GPU")
    print("  • Power: Battery-powered")
    
    print("\n🎯 Quantization Benefits for Edge:")
    print("  • 4x smaller models (FP32 → INT8)")
    print("  • 2-4x faster inference")
    print("  • Lower power consumption")
    print("  • Better cache utilization")
    
    print("\n⚙️  Hardware Accelerator Support:")
    print("  • ARM NEON: Optimized for INT8")
    print("  • Qualcomm Hexagon DSP: Native INT8")
    print("  • Google Edge TPU: INT8 only")
    print("  • Intel Neural Compute Stick: INT8 preferred")
    
    print("\n🔧 Implementation Tips:")
    print("  • Use representative calibration data")
    print("  • Test accuracy degradation thoroughly")
    print("  • Consider per-channel quantization")
    print("  • Profile on actual target hardware")

def main():
    """Main execution flow"""
    
    print("🚀 Model Quantization for Edge Deployment")
    print("=" * 60)
    
    # Compare quantization methods
    results = compare_quantization_methods()
    
    # Print results table
    print_comparison_table(results)
    
    # Edge deployment insights
    demonstrate_edge_deployment()
    
    print("\n" + "=" * 60)
    print("✅ QUANTIZATION EXAMPLE COMPLETE!")
    print("=" * 60)
    
    print("\n🎉 Key Takeaways:")
    print("- Quantization is essential for edge AI deployment")
    print("- 4x size reduction with minimal accuracy loss")
    print("- Multiple techniques available (PTQ, QAT, Dynamic)")
    print("- Hardware support enables significant speedups")
    print("- Critical for embedded systems with resource constraints")
    
    print("\n💼 Perfect for your embedded engineering background!")
    print("🚀 Now you can optimize AI models for any edge device!")

if __name__ == "__main__":
    main()