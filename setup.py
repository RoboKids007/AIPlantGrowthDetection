#!/usr/bin/env python3
"""
Setup script for Plant Disease Detection project
This script sets up the environment and installs required packages.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible. Python 3.8+ required.")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Upgrade pip first
    print("Upgrading pip...")
    success, output = run_command(f"{sys.executable} -m pip install --upgrade pip")
    if not success:
        print(f"Warning: Failed to upgrade pip: {output}")
    
    # Check for GPU and install appropriate PyTorch version
    print("Checking for GPU support...")
    gpu_available = check_nvidia_gpu()
    
    if gpu_available:
        print("NVIDIA GPU detected! Installing CUDA-enabled PyTorch...")
        # Try CUDA 12.4 first
        success, output = run_command(f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        if not success:
            print("CUDA 12.4 installation failed, trying CUDA 12.1...")
            success, output = run_command(f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        if success:
            print("✓ CUDA-enabled PyTorch installed successfully")
        else:
            print("⚠ CUDA installation failed, installing CPU version...")
            success, output = run_command(f"{sys.executable} -m pip install torch torchvision torchaudio")
    else:
        print("No NVIDIA GPU detected, installing CPU-only PyTorch...")
        success, output = run_command(f"{sys.executable} -m pip install torch torchvision torchaudio")
    
    # Install other requirements
    print("Installing other packages from requirements.txt...")
    success, output = run_command(f"{sys.executable} -m pip install ultralytics numpy opencv-python Pillow PyYAML matplotlib seaborn pandas tqdm tensorboard psutil thop")
    if success:
        print("✓ All packages installed successfully")
        return True
    else:
        print(f"✗ Failed to install packages: {output}")
        return False

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_gpu_availability():
    """Check if GPU is available for training"""
    # First check if NVIDIA GPU hardware is available
    nvidia_available = check_nvidia_gpu()
    
    if not nvidia_available:
        print("⚠ No NVIDIA GPU detected. Training will use CPU (slower).")
        return
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU available: {gpu_name}")
            print(f"  - GPU count: {gpu_count}")
            print(f"  - GPU memory: {gpu_memory:.1f} GB")
            print(f"  - CUDA version: {torch.version.cuda}")
            print("🚀 GPU training is ready for faster training times!")
        else:
            print("⚠ NVIDIA GPU detected but PyTorch cannot access it.")
            print("  This usually means CUDA-enabled PyTorch is not installed.")
            print("  Run 'setup_gpu.bat' to install CUDA-enabled PyTorch.")
    except ImportError:
        print("PyTorch not installed yet. GPU check will be performed after installation.")

def create_directories():
    """Create necessary directories"""
    directories = [
        "runs/train",
        "runs/detect", 
        "models",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def verify_dataset():
    """Verify that the dataset is properly structured"""
    data_path = Path("archive")
    required_files = [
        "data.yaml",
        "train/images",
        "train/labels", 
        "valid/images",
        "valid/labels",
        "test/images",
        "test/labels"
    ]
    
    print("Verifying dataset structure...")
    all_exist = True
    
    for file_path in required_files:
        full_path = data_path / file_path
        if full_path.exists():
            if full_path.is_dir():
                file_count = len(list(full_path.glob("*")))
                print(f"✓ {file_path}: {file_count} files")
            else:
                print(f"✓ {file_path}: exists")
        else:
            print(f"✗ {file_path}: missing")
            all_exist = False
    
    return all_exist

def main():
    print("="*60)
    print("Plant Disease Detection - Setup Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Verify dataset
    print("\nVerifying dataset...")
    if not verify_dataset():
        print("⚠ Dataset structure issues detected. Please ensure all files are in place.")
    
    # Install requirements
    print("\nInstalling dependencies...")
    if not install_requirements():
        return False
    
    # Check GPU
    print("\nChecking GPU availability...")
    check_gpu_availability()
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run training: python main.py --model n --epochs 50")
    print("2. Or with custom settings: python main.py --model s --epochs 100 --batch-size 16")
    print("3. For inference: python inference.py --model <model_path> --source <image_path>")
    print("\nFor help: python main.py --help")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
