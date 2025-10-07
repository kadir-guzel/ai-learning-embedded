#!/usr/bin/env python3
"""
FastAI Image Classification Example - Simple Version
===================================================

This example shows FastAI's power for rapid prototyping with a simple CIFAR-10 classifier.
"""

def main():
    """Simple FastAI example that demonstrates the power of high-level APIs"""
    
    print("🚀 FastAI Image Classification Example")
    print("=" * 50)
    
    try:
        # Import FastAI - this will work when installed
        from fastai.vision.all import *
        import time
        
        print("✅ FastAI imported successfully")
        
        # 1. Load data with minimal code
        print("\n📦 Loading CIFAR-10 dataset...")
        path = untar_data(URLs.CIFAR_10)
        
        # Create data loaders with automatic best practices
        dls = ImageDataLoaders.from_folder(
            path, 
            train='train', 
            valid='test',
            item_tfms=Resize(224),
            batch_tfms=aug_transforms(),
            bs=32
        )
        
        print(f"📊 Classes: {dls.vocab}")
        print(f"📊 Training samples: {len(dls.train_ds)}")
        
        # 2. Create model with transfer learning (1 line!)
        print("\n🤖 Creating model with ResNet34 backbone...")
        learn = vision_learner(dls, resnet34, metrics=accuracy)
        
        # 3. Train the model (1 line!)
        print("\n🚀 Training model...")
        start_time = time.time()
        learn.fine_tune(2)  # Just 2 epochs for demo
        training_time = time.time() - start_time
        
        # 4. Evaluate
        print(f"\n⏱️  Training time: {training_time:.2f} seconds")
        val_loss, val_acc = learn.validate()
        print(f"📈 Validation Accuracy: {val_acc:.4f}")
        
        # 5. Make a prediction
        pred_class, pred_idx, outputs = learn.predict(dls.valid_ds[0][0])
        print(f"🔮 Sample prediction: {pred_class}")
        
        print("\n" + "=" * 50)
        print("✅ FastAI Example Complete!")
        print("💡 Notice how little code was needed!")
        print("🎉 This is the power of FastAI's high-level API")
        
    except ImportError:
        print("❌ FastAI not installed")
        print("💡 Install with: pip install fastai")
        print("\n📚 What this example would show:")
        print("- Load CIFAR-10 dataset in 1 line")
        print("- Create transfer learning model in 1 line") 
        print("- Train with automatic best practices in 1 line")
        print("- Achieve ~90%+ accuracy in minutes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 This might happen if running without proper setup")

if __name__ == "__main__":
    main()