#!/usr/bin/env python3
"""
GPU Test Script for Plant Disease Detection
This script tests if your GPU setup is working correctly for YOLO training.
"""

import sys

def test_gpu_setup():
    """Test GPU setup for YOLO training"""
    print("="*50)
    print("GPU Setup Verification")
    print("="*50)
    
    # Test 1: Check if PyTorch is installed
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Test 2: Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("✗ CUDA not available - training will use CPU")
        print("  Run 'setup_gpu.bat' to install CUDA-enabled PyTorch")
        return False
    
    # Test 3: Test GPU tensor operations
    try:
        print("\nTesting GPU operations...")
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print("✓ GPU tensor operations working")
    except Exception as e:
        print(f"✗ GPU tensor operations failed: {e}")
        return False
    
    # Test 4: Test Ultralytics
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO available")
        
        # Test model loading
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 model loading successful")
        
        # Check if model can use GPU
        if next(model.model.parameters()).is_cuda:
            print("✓ YOLO model on GPU")
        else:
            print("⚠ YOLO model on CPU")
            
    except ImportError:
        print("✗ Ultralytics not installed")
        return False
    except Exception as e:
        print(f"⚠ Ultralytics test warning: {e}")
    
    print("\n" + "="*50)
    print("🚀 GPU setup verification complete!")
    print("Your system is ready for GPU-accelerated training.")
    print("="*50)
    return True

def benchmark_gpu():
    """Run a simple benchmark to test GPU performance"""
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("GPU not available for benchmarking")
            return
        
        print("\nRunning GPU benchmark...")
        device = torch.device('cuda:0')
        
        # Warm up
        for _ in range(10):
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"GPU benchmark: {elapsed:.3f} seconds for 100 matrix multiplications")
        
        # CPU comparison
        print("Running CPU comparison...")
        start_time = time.time()
        
        for _ in range(10):  # Fewer iterations for CPU
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            z = torch.mm(x, y)
        
        end_time = time.time()
        cpu_time = (end_time - start_time) * 10  # Scale to 100 iterations
        
        speedup = cpu_time / elapsed
        print(f"CPU benchmark: {cpu_time:.3f} seconds (estimated for 100 iterations)")
        print(f"GPU speedup: {speedup:.1f}x faster than CPU")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

if __name__ == "__main__":
    success = test_gpu_setup()
    
    if success:
        benchmark_gpu()
        
        print("\n🎯 Ready to start training with GPU acceleration!")
        print("Run: python main.py --model n --epochs 50 --batch-size 32")
    else:
        print("\n❌ GPU setup issues detected.")
        print("Please run 'setup_gpu.bat' to fix GPU setup.")
        sys.exit(1)
