#!/usr/bin/env python3
"""
Quick Test Script - Verify Installation
=======================================

Run this script to verify that all dependencies are properly installed
and the examples are ready to run.
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check")
    print(f"   Version: {sys.version}")
    
    if sys.version_info >= (3, 8):
        print("   ✅ Python version is suitable")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally check version"""
    
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        if min_version and hasattr(module, '__version__'):
            installed_version = module.__version__
            print(f"   {package_name}: {installed_version}")
        else:
            print(f"   {package_name}: ✅ Installed")
        
        return True
        
    except ImportError:
        print(f"   {package_name}: ❌ Not installed")
        return False

def check_dependencies():
    """Check all required dependencies"""
    
    print("\n📦 Dependency Check")
    
    # Core dependencies
    core_deps = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('PIL', 'PIL'),
        ('tqdm', 'tqdm')
    ]
    
    all_core = True
    for pkg, import_name in core_deps:
        if not check_package(pkg, import_name):
            all_core = False
    
    print(f"\n   Core dependencies: {'✅ All installed' if all_core else '❌ Missing packages'}")
    
    # Optional dependencies
    print("\n   Optional dependencies:")
    optional_deps = [
        ('fastai', 'fastai'),
        ('onnx', 'onnx'),
        ('onnxruntime', 'onnxruntime'),
        ('tensorflow', 'tensorflow'),
        ('sklearn', 'sklearn')
    ]
    
    for pkg, import_name in optional_deps:
        check_package(pkg, import_name)
    
    return all_core

def test_torch_functionality():
    """Test basic PyTorch functionality"""
    
    print("\n🔥 PyTorch Functionality Test")
    
    try:
        import torch
        
        # Create tensors
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        print(f"   ✅ Tensor operations work")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   ⚠️  CUDA not available (CPU only)")
        
        # Test neural network
        model = torch.nn.Linear(10, 5)
        x = torch.randn(1, 10)
        y = model(x)
        
        print(f"   ✅ Neural network operations work")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PyTorch test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    
    print("\n📊 Data Loading Test")
    
    try:
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        # Create simple dataset
        transform = transforms.Compose([transforms.ToTensor()])
        
        # This won't download, just test the API
        try:
            dataset = torchvision.datasets.FakeData(
                size=100, 
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform
            )
            
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            # Test loading one batch
            for batch in dataloader:
                break
            
            print(f"   ✅ Data loading works")
            return True
            
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            return False
            
    except ImportError:
        print(f"   ❌ TorchVision not available")
        return False

def suggest_installation():
    """Suggest installation commands for missing packages"""
    
    print("\n💡 Installation Suggestions")
    print("=" * 40)
    
    print("If packages are missing, install them with:")
    print()
    print("🔥 Core PyTorch (CPU):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("🚀 Core PyTorch (CUDA 11.8):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("📦 All dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("🤖 FastAI:")
    print("   pip install fastai")
    print()
    print("🔄 ONNX and deployment:")
    print("   pip install onnx onnxruntime tensorflow")

def create_test_model():
    """Create a simple test model"""
    
    print("\n🤖 Model Creation Test")
    
    try:
        import torch
        import torch.nn as nn
        
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleNet()
        
        # Test forward pass
        x = torch.randn(1, 10)
        y = model(x)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"   ✅ Model created successfully")
        print(f"   📊 Parameters: {param_count}")
        print(f"   📋 Output shape: {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    
    print("🧪 AI Learning Environment Test")
    print("=" * 50)
    
    results = []
    
    # Run all tests
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("PyTorch Functionality", test_torch_functionality()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Creation", create_test_model()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your environment is ready for AI learning!")
        print("\n🚀 Next steps:")
        print("1. Start with: cd 01-fastai-basics && python fastai_classifier.py")
        print("2. Compare with: cd 02-pytorch-equivalent && python pytorch_classifier.py")
        print("3. Explore advanced examples in other directories")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❗ Please install missing dependencies before proceeding")
        suggest_installation()
    
    return all_passed

if __name__ == "__main__":
    run_comprehensive_test()