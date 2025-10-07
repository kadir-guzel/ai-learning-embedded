#!/usr/bin/env python3
"""
FastAI Image Classification Example
==================================

This example shows how FastAI makes deep learning accessible with minimal code.
We'll build an image classifier using transfer learning in just a few lines.

Key Concepts:
- Transfer learning with pre-trained models
- FastAI's data loading utilities
- Automatic best practices (data augmentation, learning rates)
- Built-in visualization and metrics
"""

import torch
import matplotlib.pyplot as plt
import time
import os

# FastAI imports
try:
    from fastai.vision.all import *
    print("✅ FastAI imported successfully")
except ImportError as e:
    print(f"❌ FastAI import error: {e}")
    print("💡 Please install FastAI: pip install fastai")
    exit(1)

def setup_data():
    """Download and prepare CIFAR-10 dataset using FastAI"""
    print("📦 Setting up CIFAR-10 dataset...")
    
    # FastAI makes data loading incredibly simple
    path = untar_data(URLs.CIFAR_10)
    print(f"✅ Data downloaded to: {path}")
    
    # Create data loaders with automatic augmentation
    dls = ImageDataLoaders.from_folder(
        path, 
        train='train', 
        valid='test',
        item_tfms=Resize(224),  # Resize for pre-trained model
        batch_tfms=aug_transforms(),  # Automatic data augmentation
        bs=32  # Batch size
    )
    
    print(f"📊 Classes: {dls.vocab}")
    print(f"📊 Training samples: {len(dls.train_ds)}")
    print(f"📊 Validation samples: {len(dls.valid_ds)}")
    
    return dls

def show_data_samples(dls):
    """Visualize some training samples"""
    print("\n🖼️  Sample training images:")
    dls.show_batch(max_n=8, nrows=2)
    plt.show()

def create_and_train_model(dls):
    """Create and train model using FastAI's vision_learner"""
    print("\n🤖 Creating model with transfer learning...")
    
    # This one line creates a model with:
    # - Pre-trained ResNet34 backbone
    # - Appropriate head for our classes
    # - Best practice optimizers and learning rates
    learn = vision_learner(dls, resnet34, metrics=accuracy)
    
    print("📋 Model architecture:")
    print(learn.model)
    
    # Fine-tune the model (transfer learning)
    print("\n🚀 Starting training...")
    start_time = time.time()
    
    # FastAI automatically finds good learning rates
    learn.fine_tune(3)  # 3 epochs of fine-tuning
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.2f} seconds")
    
    return learn

def evaluate_model(learn):
    """Evaluate the trained model"""
    print("\n📊 Model Evaluation:")
    
    # Show training results
    learn.show_results(max_n=6, nrows=2)
    plt.show()
    
    # Confusion matrix
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(8, 8))
    plt.show()
    
    # Get validation accuracy
    val_loss, val_acc = learn.validate()
    print(f"📈 Validation Accuracy: {val_acc:.4f}")
    print(f"📉 Validation Loss: {val_loss:.4f}")
    
    return val_acc

def benchmark_model(learn):
    """Benchmark model size and inference speed"""
    print("\n⚡ Performance Benchmarks:")
    
    # Save model and check size
    model_path = '../models/fastai_classifier.pkl'
    os.makedirs('../models', exist_ok=True)
    learn.export(model_path)
    
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"💾 Model size: {model_size:.2f} MB")
    
    # Benchmark inference speed
    test_input = torch.randn(1, 3, 224, 224)
    
    # Warm up
    for _ in range(10):
        _ = learn.model(test_input)
    
    # Measure inference time
    start_time = time.time()
    num_runs = 100
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = learn.model(test_input)
    
    avg_inference_time = (time.time() - start_time) / num_runs * 1000  # ms
    print(f"⚡ Average inference time: {avg_inference_time:.2f} ms")
    
    return model_size, avg_inference_time

def make_predictions(learn):
    """Make predictions on new images"""
    print("\n🔮 Making predictions on sample images...")
    
    # You can make predictions on individual images
    # For demo, we'll use a sample from the validation set
    val_dl = learn.dls.valid
    batch = val_dl.one_batch()
    imgs, labels = batch
    
    # Predict on first image
    img = imgs[0]
    actual_label = learn.dls.vocab[labels[0]]
    
    pred_class, pred_idx, outputs = learn.predict(img)
    confidence = outputs.max().item()
    
    print(f"🎯 Actual: {actual_label}")
    print(f"🤖 Predicted: {pred_class}")
    print(f"📊 Confidence: {confidence:.2f}")
    
    return pred_class, confidence

def main():
    """Main execution flow"""
    print("🚀 FastAI Image Classification Example")
    print("=" * 50)
    
    # Set up data
    dls = setup_data()
    
    # Show sample data (optional - comment out if running headless)
    # show_data_samples(dls)
    
    # Create and train model
    learn = create_and_train_model(dls)
    
    # Evaluate model
    accuracy = evaluate_model(learn)
    
    # Benchmark performance
    model_size, inference_time = benchmark_model(learn)
    
    # Make sample predictions
    pred_class, confidence = make_predictions(learn)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 FASTAI EXAMPLE SUMMARY")
    print("=" * 50)
    print(f"✅ Model trained successfully!")
    print(f"📈 Validation Accuracy: {accuracy:.2f}")
    print(f"💾 Model Size: {model_size:.2f} MB")
    print(f"⚡ Inference Time: {inference_time:.2f} ms")
    print(f"🤖 Sample Prediction: {pred_class} (confidence: {confidence:.2f})")
    print("\n🎉 FastAI makes deep learning incredibly accessible!")
    print("💡 Notice how little code was needed for a complete ML pipeline!")

if __name__ == "__main__":
    main()