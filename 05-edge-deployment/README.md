# Edge AI Deployment - ONNX & TensorFlow Lite

This example shows how to convert PyTorch models to different formats for deployment across various edge devices and platforms.

## 🎯 What You'll Learn

- **ONNX Export** - Universal model format for cross-platform deployment
- **TensorFlow Lite** - Google's edge AI runtime
- **Model Optimization** - Size and speed optimization for mobile
- **Cross-Platform Deployment** - Same model, multiple targets

## 🚀 Why Model Conversion Matters

1. **Hardware Diversity** - Different edge devices use different runtimes
2. **Optimization** - Each runtime has specific optimizations
3. **Ecosystem Access** - Leverage platform-specific tools
4. **Performance** - Native runtimes often faster than generic ones

## 📱 Deployment Targets

- **ONNX Runtime** - Windows, Linux, macOS, mobile
- **TensorFlow Lite** - Android, iOS, embedded Linux
- **Core ML** - iOS/macOS (Apple devices)
- **OpenVINO** - Intel hardware optimization

## 🔄 Conversion Pipeline

PyTorch → ONNX → TensorFlow → TensorFlow Lite
         ↓
    ONNX Runtime (cross-platform)

Perfect for deploying your AI models everywhere!