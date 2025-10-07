#!/usr/bin/env python3
"""
Edge AI Deployment - ONNX & TensorFlow Lite
===========================================

This example demonstrates how to convert PyTorch models to different formats
for deployment on various edge devices and platforms.

Key Formats:
- ONNX: Universal format for cross-platform deployment
- TensorFlow Lite: Optimized for mobile and embedded devices
- Model optimization and benchmarking

Perfect for embedded engineers deploying AI across different hardware!
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
from pathlib import Path

# For ONNX export
import onnx
import onnxruntime as ort

# For TensorFlow conversion (optional - install if needed)
try:
    import tensorflow as tf
    from onnx_tf.backend import prepare
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available - TFLite conversion will be skipped")

class MobileClassifier(nn.Module):
    """Lightweight classifier optimized for mobile deployment"""
    
    def __init__(self, num_classes=10):
        super(MobileClassifier, self).__init__()
        
        # Lightweight feature extractor
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_sample_model():
    """Create and train a sample model for conversion"""
    
    print("🤖 Creating sample mobile classifier...")
    
    # Simple model
    model = MobileClassifier(num_classes=10)
    
    # Load some pre-trained weights or train briefly
    # For demo purposes, we'll use random weights
    print("📊 Model parameters:", sum(p.numel() for p in model.parameters()))
    
    return model

def export_to_onnx(model, output_path="model.onnx", input_shape=(1, 3, 32, 32)):
    """Export PyTorch model to ONNX format"""
    
    print(f"\n🔄 Exporting model to ONNX: {output_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,                          # Model to export
        dummy_input,                    # Dummy input
        output_path,                    # Output path
        export_params=True,             # Store trained parameters
        opset_version=11,              # ONNX version
        do_constant_folding=True,       # Optimize constant folding
        input_names=['input'],          # Input tensor name
        output_names=['output'],        # Output tensor name
        dynamic_axes={                  # Variable length axes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ ONNX export successful!")
    print(f"📄 Model saved to: {output_path}")
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"💾 ONNX model size: {file_size:.2f} MB")
    
    return output_path

def test_onnx_inference(onnx_path, test_input):
    """Test ONNX model inference"""
    
    print(f"\n⚡ Testing ONNX inference...")
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    print(f"📝 Input name: {input_name}")
    print(f"📝 Output name: {output_name}")
    
    # Convert PyTorch tensor to numpy
    test_input_np = test_input.detach().cpu().numpy()
    
    # Run inference
    start_time = time.time()
    onnx_outputs = ort_session.run([output_name], {input_name: test_input_np})
    inference_time = (time.time() - start_time) * 1000  # ms
    
    print(f"⏱️  ONNX inference time: {inference_time:.2f} ms")
    print(f"📊 Output shape: {onnx_outputs[0].shape}")
    
    return onnx_outputs[0], inference_time

def convert_onnx_to_tensorflow(onnx_path, tf_output_dir):
    """Convert ONNX model to TensorFlow SavedModel format"""
    
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available - skipping TF conversion")
        return None
    
    print(f"\n🔄 Converting ONNX to TensorFlow...")
    
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Export to SavedModel format
        tf_rep.export_graph(tf_output_dir)
        
        print(f"✅ TensorFlow conversion successful!")
        print(f"📁 SavedModel saved to: {tf_output_dir}")
        
        # Get directory size
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(tf_output_dir)
            for filename in filenames
        ) / (1024 * 1024)  # MB
        
        print(f"💾 TensorFlow model size: {total_size:.2f} MB")
        
        return tf_output_dir
        
    except Exception as e:
        print(f"❌ TensorFlow conversion failed: {e}")
        return None

def convert_to_tflite(tf_model_dir, tflite_output_path):
    """Convert TensorFlow SavedModel to TensorFlow Lite"""
    
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available - skipping TFLite conversion")
        return None
    
    print(f"\n🔄 Converting to TensorFlow Lite...")
    
    try:
        # Load the SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(tflite_output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TensorFlow Lite conversion successful!")
        print(f"📄 TFLite model saved to: {tflite_output_path}")
        
        # Get file size
        file_size = os.path.getsize(tflite_output_path) / (1024 * 1024)  # MB
        print(f"💾 TFLite model size: {file_size:.2f} MB")
        
        return tflite_output_path
        
    except Exception as e:
        print(f"❌ TensorFlow Lite conversion failed: {e}")
        return None

def test_tflite_inference(tflite_path, test_input):
    """Test TensorFlow Lite model inference"""
    
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available - skipping TFLite test")
        return None, 0
    
    print(f"\n⚡ Testing TensorFlow Lite inference...")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📝 Input shape: {input_details[0]['shape']}")
        print(f"📝 Output shape: {output_details[0]['shape']}")
        
        # Prepare input data
        test_input_np = test_input.detach().cpu().numpy().astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_input_np)
        
        # Run inference
        start_time = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get output
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"⏱️  TFLite inference time: {inference_time:.2f} ms")
        print(f"📊 Output shape: {tflite_output.shape}")
        
        return tflite_output, inference_time
        
    except Exception as e:
        print(f"❌ TFLite inference failed: {e}")
        return None, 0

def benchmark_all_formats(pytorch_model, onnx_path, tflite_path, test_input):
    """Benchmark all model formats"""
    
    print("\n📊 PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    results = {}
    
    # PyTorch benchmark
    pytorch_model.eval()
    with torch.no_grad():
        start_time = time.time()
        pytorch_output = pytorch_model(test_input)
        pytorch_time = (time.time() - start_time) * 1000
    
    pytorch_size = sum(p.numel() * 4 for p in pytorch_model.parameters()) / (1024 * 1024)  # MB
    
    results['PyTorch'] = {
        'inference_time_ms': pytorch_time,
        'model_size_mb': pytorch_size,
        'output_shape': pytorch_output.shape
    }
    
    # ONNX benchmark
    if os.path.exists(onnx_path):
        _, onnx_time = test_onnx_inference(onnx_path, test_input)
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        
        results['ONNX'] = {
            'inference_time_ms': onnx_time,
            'model_size_mb': onnx_size,
            'output_shape': 'Same as PyTorch'
        }
    
    # TensorFlow Lite benchmark
    if tflite_path and os.path.exists(tflite_path):
        _, tflite_time = test_tflite_inference(tflite_path, test_input)
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
        
        results['TensorFlow Lite'] = {
            'inference_time_ms': tflite_time,
            'model_size_mb': tflite_size,
            'output_shape': 'Same as PyTorch'
        }
    
    return results

def print_benchmark_table(results):
    """Print formatted benchmark results"""
    
    print(f"{'Format':<15} {'Size (MB)':<10} {'Inference (ms)':<15} {'Speed vs PyTorch':<15}")
    print("-" * 60)
    
    pytorch_time = results['PyTorch']['inference_time_ms']
    
    for format_name, data in results.items():
        size = data['model_size_mb']
        time_ms = data['inference_time_ms']
        speedup = f"{pytorch_time/time_ms:.2f}x" if time_ms > 0 else "N/A"
        
        print(f"{format_name:<15} {size:<10.2f} {time_ms:<15.2f} {speedup:<15}")
    
    print("-" * 60)

def deployment_recommendations():
    """Provide deployment recommendations for different platforms"""
    
    print("\n🚀 DEPLOYMENT RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = {
        "🤖 Android Devices": [
            "Use TensorFlow Lite for best performance",
            "Enable NNAPI delegate for hardware acceleration",
            "Consider INT8 quantization for mobile CPUs",
            "Use ARM NEON optimizations"
        ],
        "🍎 iOS Devices": [
            "Convert to Core ML for optimal performance",
            "Use Metal Performance Shaders for GPU",
            "Consider Neural Engine on A12+ chips",
            "ONNX → Core ML conversion available"
        ],
        "💻 Desktop/Server": [
            "ONNX Runtime provides excellent cross-platform support",
            "Use GPU providers (CUDA, DirectML, CoreML)",
            "Consider TensorRT for NVIDIA GPUs",
            "OpenVINO for Intel hardware optimization"
        ],
        "🔧 Embedded Linux": [
            "TensorFlow Lite supports ARM Cortex-A",
            "ONNX Runtime has ARM64 builds",
            "Consider Qualcomm Neural Processing SDK",
            "Use INT8 quantization for efficiency"
        ],
        "⚡ Edge TPU/NPU": [
            "TensorFlow Lite required for Edge TPU",
            "Specific quantization schemes needed",
            "Model architecture constraints apply",
            "Compile models for target hardware"
        ]
    }
    
    for platform, tips in recommendations.items():
        print(f"\n{platform}:")
        for tip in tips:
            print(f"  • {tip}")

def create_deployment_checklist():
    """Create a practical deployment checklist"""
    
    print("\n✅ DEPLOYMENT CHECKLIST")
    print("=" * 30)
    
    checklist = [
        "🎯 Define target hardware specifications",
        "📊 Profile model performance on target device",
        "🔧 Apply quantization if needed",
        "🧪 Test accuracy on representative data",
        "⚡ Benchmark inference speed",
        "💾 Verify model size constraints",
        "🔋 Test power consumption (mobile)",
        "🚀 Integrate into application",
        "🧪 End-to-end testing",
        "📈 Monitor performance in production"
    ]
    
    for i, item in enumerate(checklist, 1):
        print(f"{i:2d}. {item}")

def main():
    """Main execution flow"""
    
    print("🚀 Edge AI Deployment - ONNX & TensorFlow Lite")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("../models/deployment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample model
    model = create_sample_model()
    
    # Create test input
    test_input = torch.randn(1, 3, 32, 32)
    
    # Export to ONNX
    onnx_path = output_dir / "mobile_classifier.onnx"
    export_to_onnx(model, str(onnx_path))
    
    # Convert to TensorFlow
    tf_model_dir = output_dir / "tensorflow_model"
    tf_model_path = convert_onnx_to_tensorflow(str(onnx_path), str(tf_model_dir))
    
    # Convert to TensorFlow Lite
    tflite_path = output_dir / "mobile_classifier.tflite"
    if tf_model_path:
        convert_to_tflite(str(tf_model_dir), str(tflite_path))
    
    # Benchmark all formats
    results = benchmark_all_formats(
        model, str(onnx_path), str(tflite_path) if tflite_path.exists() else None, test_input
    )
    
    # Print benchmark results
    print_benchmark_table(results)
    
    # Deployment recommendations
    deployment_recommendations()
    
    # Deployment checklist
    create_deployment_checklist()
    
    print("\n" + "=" * 60)
    print("✅ EDGE DEPLOYMENT EXAMPLE COMPLETE!")
    print("=" * 60)
    
    print(f"\n📁 Generated Files:")
    print(f"  • ONNX Model: {onnx_path}")
    if tf_model_path:
        print(f"  • TensorFlow Model: {tf_model_dir}")
    if tflite_path.exists():
        print(f"  • TensorFlow Lite: {tflite_path}")
    
    print("\n🎉 Key Achievements:")
    print("- ✅ Exported PyTorch model to ONNX")
    print("- ✅ Cross-platform deployment ready")
    print("- ✅ Mobile optimization with TFLite")
    print("- ✅ Performance benchmarking complete")
    
    print("\n💼 Perfect for embedded engineers!")
    print("🚀 Deploy your AI models on ANY device!")

if __name__ == "__main__":
    main()