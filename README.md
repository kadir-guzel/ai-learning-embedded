# AI Learning Workspace for Embedded Engineers

Welcome to your comprehensive AI learning journey! This workspace contains practical examples to help you understand the differences between FastAI and PyTorch, and how to optimize models for edge deployment.

## 🎯 Learning Objectives

1. **Understand FastAI vs PyTorch trade-offs**
2. **Learn when to use each framework**
3. **Master edge AI optimization techniques**
4. **Practice model quantization and deployment**

## 📁 Project Structure

```
ai-learning-workspace/
├── 01-fastai-basics/           # Quick start with FastAI
├── 02-pytorch-equivalent/      # Same task in PyTorch
├── 03-custom-loss-functions/   # Advanced PyTorch features
├── 04-model-quantization/      # Edge optimization
├── 05-edge-deployment/         # ONNX & TensorFlow Lite
├── data/                       # Sample datasets (auto-downloaded)
├── models/                     # Saved models (auto-created)
├── utils/                      # Shared utilities
├── requirements.txt           # Dependencies
├── setup.sh                   # Automated setup script
└── test_setup.py             # Environment verification
```

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
cd /home/kguzel/proj/secopera/ai
./setup.sh
```

### Option 2: Manual Setup
```bash
cd /home/kguzel/proj/secopera/ai
pip install -r requirements.txt
python test_setup.py  # Verify installation
```

### Option 3: With Virtual Environment
```bash
cd /home/kguzel/proj/secopera/ai
python -m venv ai_env
source ai_env/bin/activate
pip install -r requirements.txt
python test_setup.py
```

## 📚 Learning Path

### 1. Start with FastAI (Beginner-Friendly)
```bash
cd 01-fastai-basics
python simple_example.py        # Quick demo
python fastai_classifier.py     # Full example
```

### 2. Compare with PyTorch (More Control)
```bash
cd 02-pytorch-equivalent
python pytorch_classifier.py    # Same task, more code
```

### 3. Advanced PyTorch Features (Research-Level)
```bash
cd 03-custom-loss-functions
python custom_loss_example.py   # Custom loss for edge AI
```

### 4. Model Optimization (Edge AI)
```bash
cd 04-model-quantization
python quantization_example.py  # Reduce model size 4x
```

### 5. Cross-Platform Deployment (Production)
```bash
cd 05-edge-deployment
python deployment_example.py    # ONNX, TensorFlow Lite
```

## 🔍 Key Concepts You'll Learn

### FastAI Advantages ⚡
- **Rapid prototyping** - Get results in ~10 lines of code
- **Best practices built-in** - Transfer learning, data augmentation
- **High-level abstractions** - Less boilerplate, more results
- **Beginner-friendly** - Gentle learning curve

### PyTorch Advantages 🔧
- **Full control** - Custom architectures, loss functions, training loops
- **Production flexibility** - Fine-grained optimization
- **Research capabilities** - Implement cutting-edge techniques
- **Industry standard** - Most companies use PyTorch in production

### Edge AI Essentials 📱
- **Model quantization** - 4x smaller models (FP32 → INT8)
- **ONNX portability** - Deploy across different hardware
- **TensorFlow Lite** - Mobile and embedded deployment
- **Hardware acceleration** - Leverage NPUs, TPUs, DSPs

## 📊 What You'll Build

| Example | FastAI | PyTorch | Edge AI |
|---------|--------|---------|---------|
| **Image Classifier** | ✅ 10 lines | ✅ 150 lines | ❌ Limited |
| **Custom Loss Function** | ❌ Difficult | ✅ Easy | ✅ Perfect |
| **Model Quantization** | ❌ No support | ✅ Full control | ✅ Essential |
| **Cross-Platform Deploy** | ❌ Limited | ✅ ONNX/TFLite | ✅ Required |

## 🏆 Performance Comparisons

Each example includes real benchmarks:
- **Model size** (MB) - FP32 vs INT8 quantization
- **Inference speed** (ms) - CPU vs GPU vs edge devices
- **Memory usage** (MB) - Runtime memory requirements
- **Accuracy trade-offs** - Performance vs efficiency

## 🛠️ Embedded Engineer Advantages

This curriculum is specifically designed for your background:

### 1. **Hardware-Aware AI** 🔩
- Memory constraints understanding
- Real-time performance requirements
- Power efficiency considerations
- Hardware accelerator utilization

### 2. **System-Level Thinking** ⚙️
- End-to-end deployment pipelines
- Cross-compilation for embedded targets
- Resource optimization techniques
- Performance profiling and analysis

### 3. **Practical Applications** 🎯
- Computer vision for industrial automation
- Sensor data processing with AI
- Edge inference optimization
- IoT device AI integration

## 🚀 Next Steps After This Course

1. **Specialized Hardware**
   - NVIDIA Jetson development
   - Google Coral Edge TPU programming
   - Qualcomm Neural Processing SDK
   - Intel OpenVINO optimization

2. **Advanced Techniques**
   - Neural Architecture Search (NAS)
   - Knowledge distillation
   - Pruning and sparsity
   - Hardware-software co-design

3. **Industry Applications**
   - Autonomous vehicle perception
   - Industrial quality control
   - Medical device AI
   - Smart home automation

## 🤝 Getting Help

- **Documentation**: Each directory has detailed README files
- **Comments**: All code is extensively commented
- **Error Handling**: Scripts include helpful error messages
- **Testing**: Use `test_setup.py` to verify your environment

## 📈 Success Metrics

By the end of this course, you'll be able to:

- ✅ Choose the right framework for any AI project
- ✅ Optimize models for edge deployment (4x size reduction)
- ✅ Deploy models across multiple platforms (ONNX, TFLite)
- ✅ Implement custom loss functions for embedded constraints
- ✅ Benchmark and profile AI model performance
- ✅ Apply quantization techniques effectively

**Perfect for embedded engineers entering the AI field!** 🎉

Let's start learning! 🚀