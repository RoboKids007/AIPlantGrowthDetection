@echo off
echo ===================================================
echo Plant Disease Detection - Setup and Training
echo ===================================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo GPU detected! Setting up CUDA-enabled PyTorch...
    echo.
    
    echo Checking current PyTorch installation...
    python -c "import torch; print('Current PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>nul
    if %errorlevel% neq 0 (
        echo PyTorch not installed yet.
    )
    
    echo.
    echo Installing/upgrading to CUDA-enabled PyTorch...
    echo This may take a few minutes...
    
    rem Uninstall CPU version if exists
    pip uninstall torch torchvision torchaudio -y >nul 2>&1
    
    rem Install CUDA version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if %errorlevel% neq 0 (
        echo WARNING: Failed to install CUDA PyTorch, falling back to CPU version
        pip install torch torchvision torchaudio
    )
    
    echo.
    echo Verifying GPU setup...
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
    
) else (
    echo No NVIDIA GPU detected. Training will use CPU.
    echo For faster training, consider using a system with NVIDIA GPU.
)

echo.
echo Running setup script...
python setup.py
if %errorlevel% neq 0 (
    echo ERROR: Setup failed
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.

echo Starting training with default settings...
echo This will train a YOLOv8 nano model for 50 epochs
echo You can stop training anytime with Ctrl+C
echo.

set /p choice="Do you want to start training now? (y/n): "
if /i "%choice%"=="y" (
    echo.
    echo Starting training...
    python main.py --model n --epochs 50 --batch-size 16
) else (
    echo.
    echo Training skipped. You can start training later with:
    echo python main.py --model n --epochs 50 --batch-size 16
)

echo.
echo Script completed!
pause
