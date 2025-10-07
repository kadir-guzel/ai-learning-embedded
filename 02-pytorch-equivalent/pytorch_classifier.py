#!/usr/bin/env python3
"""
PyTorch Image Classification Example
===================================

This example implements the same CIFAR-10 classifier as the FastAI version,
but using pure PyTorch. Notice the increased complexity but also the increased control.

Key Differences from FastAI:
- Manual training loops
- Explicit data loading and preprocessing  
- Custom model architecture definition
- Manual optimization and learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

class CIFAR10Net(nn.Module):
    """Custom CNN architecture for CIFAR-10"""
    
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def setup_data():
    """Setup CIFAR-10 data loaders with manual preprocessing"""
    print("📦 Setting up CIFAR-10 dataset...")
    
    # Define transforms manually (vs FastAI's automatic transforms)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=False, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    
    # CIFAR-10 classes
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"📊 Classes: {classes}")
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader, classes

def train_model(model, train_loader, test_loader, device, num_epochs=3):
    """Train the model with manual training loop"""
    print(f"\n🚀 Training model on {device}...")
    
    # Define loss function and optimizer manually
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model.to(device)
    
    train_losses = []
    train_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, targets) in enumerate(train_pbar):
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            if batch_idx % 100 == 99:
                avg_loss = running_loss / 100
                accuracy = 100. * correct / total
                train_pbar.set_postfix({
                    'Loss': f'{avg_loss:.3f}',
                    'Acc': f'{accuracy:.2f}%'
                })
                running_loss = 0.0
        
        # Epoch statistics
        epoch_acc = 100. * correct / total
        train_accuracies.append(epoch_acc)
        
        # Validation
        val_acc = evaluate_model(model, test_loader, device)
        
        print(f"Epoch {epoch+1}: Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.2f} seconds")
    
    return model, train_accuracies

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
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
    
    accuracy = 100. * correct / total
    return accuracy

def benchmark_model(model, device):
    """Benchmark model size and inference speed"""
    print("\n⚡ Performance Benchmarks:")
    
    # Save model and check size
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/pytorch_classifier.pth'
    torch.save(model.state_dict(), model_path)
    
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"💾 Model size: {model_size:.2f} MB")
    
    # Benchmark inference speed
    model.eval()
    test_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    
    # Measure inference time
    start_time = time.time()
    num_runs = 100
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_input)
    
    avg_inference_time = (time.time() - start_time) / num_runs * 1000  # ms
    print(f"⚡ Average inference time: {avg_inference_time:.2f} ms")
    
    return model_size, avg_inference_time

def visualize_predictions(model, test_loader, classes, device, num_samples=8):
    """Visualize model predictions"""
    model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images[:num_samples].to(device))
        _, predicted = torch.max(outputs, 1)
    
    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Denormalize image for display
        img = images[i]
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution flow"""
    print("🚀 PyTorch Image Classification Example")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Setup data
    train_loader, test_loader, classes = setup_data()
    
    # Create model
    print("\n🤖 Creating custom CNN model...")
    model = CIFAR10Net(num_classes=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Train model
    model, train_accuracies = train_model(model, train_loader, test_loader, device)
    
    # Final evaluation
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f"\n📈 Final Test Accuracy: {final_accuracy:.2f}%")
    
    # Benchmark performance
    model_size, inference_time = benchmark_model(model, device)
    
    # Visualize some predictions (optional - comment out if running headless)
    # visualize_predictions(model, test_loader, classes, device)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 PYTORCH EXAMPLE SUMMARY")
    print("=" * 50)
    print(f"✅ Model trained successfully!")
    print(f"📈 Final Accuracy: {final_accuracy:.2f}%")
    print(f"💾 Model Size: {model_size:.2f} MB")
    print(f"⚡ Inference Time: {inference_time:.2f} ms")
    print(f"🔧 Total Parameters: {total_params:,}")
    print("\n💡 COMPARISON WITH FASTAI:")
    print("- ✅ Full control over every aspect")
    print("- ✅ Custom architecture design")
    print("- ✅ Manual optimization tuning")
    print("- ❌ Much more code required")
    print("- ❌ Need to implement best practices manually")

if __name__ == "__main__":
    main()