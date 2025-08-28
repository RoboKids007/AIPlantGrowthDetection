@echo off
echo ===============================================
echo GPU Setup for Plant Disease Detection
echo ===============================================
echo.

echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: No NVIDIA GPU detected or NVIDIA drivers not installed
    echo.
    echo Please ensure you have:
    echo 1. NVIDIA GPU installed
    echo 2. Latest NVIDIA drivers installed
    echo 3. CUDA toolkit installed (optional but recommended)
    echo.
    echo Download drivers from: https://www.nvidia.com/drivers
    pause
    exit /b 1
)

echo GPU detected! Getting GPU information...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
echo.

echo Checking current PyTorch installation...
python -c "import torch; print('Current PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())" 2>nul
if %errorlevel% neq 0 (
    echo PyTorch not installed yet.
) else (
    python -c "import torch; cuda_available = torch.cuda.is_available(); print('GPU name:', torch.cuda.get_device_name(0) if cuda_available else 'No GPU detected by PyTorch')"
)

echo.
set /p choice="Do you want to install/reinstall CUDA-enabled PyTorch? (y/n): "
if /i not "%choice%"=="y" (
    echo Setup cancelled.
    pause
    exit /b 0
)

echo.
echo Uninstalling existing PyTorch installations...
pip uninstall torch torchvision torchaudio -y

echo.
echo Installing CUDA-enabled PyTorch...
echo This may take several minutes depending on your internet connection...
echo.

rem Try CUDA 12.4 first (latest stable)
echo Trying CUDA 12.4 installation...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if %errorlevel% neq 0 (
    echo CUDA 12.4 installation failed, trying CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if %errorlevel% neq 0 (
        echo CUDA installation failed, installing CPU version...
        pip install torch torchvision torchaudio
        echo WARNING: Only CPU version installed. GPU acceleration not available.
        pause
        exit /b 1
    )
)

echo.
echo Verifying installation...
python -c "import torch; print('=== PyTorch Installation Verification ==='); print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'Not available'); print('GPU count:', torch.cuda.device_count()); print('Current GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); print('GPU memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else 'N/A')"

if %errorlevel% equ 0 (
    echo.
    echo ===============================================
    echo GPU Setup completed successfully!
    echo Your system is ready for GPU-accelerated training.
    echo ===============================================
) else (
    echo.
    echo ERROR: Verification failed. Please check the installation.
)

echo.
pause
