@echo off
setlocal enabledelayedexpansion

:: ==============================================================================
::  🌱 GOWTHAM'S PLANT DISEASE DETECTION - AUTO SETUP SCRIPT 🌱
:: ==============================================================================
::  This script will automatically set up everything needed to run the
::  Plant Disease Detection AI system. Just run it and everything will be ready!
:: ==============================================================================

echo.
echo ================================================================================
echo                    🌱 PLANT DISEASE DETECTION - AUTO SETUP 🌱
echo ================================================================================
echo.
echo Welcome! This script will automatically set up your Plant Disease Detection AI.
echo.
echo What this script will do:
echo   ✅ Check and install Python
echo   ✅ Create virtual environment
echo   ✅ Install all AI dependencies
echo   ✅ Download YOLO model if needed
echo   ✅ Set up directories
echo   ✅ Start the web server
echo.
echo ⏱️  This may take 5-10 minutes depending on your internet speed...
echo.
pause

:: ==============================================================================
:: STEP 1: Check Python Installation
:: ==============================================================================
echo.
echo 🔍 STEP 1: Checking Python installation...
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%a in ('python --version') do set PYTHON_VERSION=%%a
echo ✅ Python %PYTHON_VERSION% is installed

:: Check if Python version is adequate
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python version is too old! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: ==============================================================================
:: STEP 2: Set up project directory
:: ==============================================================================
echo.
echo 📁 STEP 2: Setting up project directory...
echo.

:: Get the directory where this script is located
cd /d "%~dp0"
echo Current directory: %CD%

:: Create required directories
if not exist "static" mkdir "static"
if not exist "static\uploads" mkdir "static\uploads"
if not exist "static\results" mkdir "static\results"
if not exist "templates" mkdir "templates"
if not exist "runs" mkdir "runs"

echo ✅ Project directories created

:: ==============================================================================
:: STEP 3: Create Virtual Environment
:: ==============================================================================
echo.
echo 🔧 STEP 3: Creating virtual environment...
echo.

:: Check if virtual environment already exists
if exist "plant_ai_env\Scripts\activate.bat" (
    echo ✅ Virtual environment already exists
    goto :activate_env
)

:: Create new virtual environment
echo Creating new virtual environment...
python -m venv plant_ai_env
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment!
    echo.
    echo Trying with alternative method...
    python -m pip install --user virtualenv
    python -m virtualenv plant_ai_env
    if %errorlevel% neq 0 (
        echo ❌ Still failed! Please check your Python installation.
        pause
        exit /b 1
    )
)

echo ✅ Virtual environment created successfully

:activate_env
:: Activate virtual environment
echo.
echo 🔋 Activating virtual environment...
call plant_ai_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment!
    pause
    exit /b 1
)

echo ✅ Virtual environment activated

:: ==============================================================================
:: STEP 4: Upgrade pip and install dependencies
:: ==============================================================================
echo.
echo 📦 STEP 4: Installing AI dependencies...
echo.

:: Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ⚠️  Warning: Failed to upgrade pip, continuing anyway...
)

:: Install core dependencies
echo.
echo Installing FastAPI and web framework...
python -m pip install fastapi==0.104.1
python -m pip install uvicorn[standard]==0.24.0
python -m pip install jinja2==3.1.2
python -m pip install python-multipart==0.0.6

if %errorlevel% neq 0 (
    echo ❌ Failed to install web framework dependencies!
    echo.
    echo Trying alternative installation method...
    python -m pip install --no-cache-dir fastapi uvicorn jinja2 python-multipart
    if %errorlevel% neq 0 (
        echo ❌ Installation failed! Check your internet connection.
        pause
        exit /b 1
    )
)

echo ✅ Web framework installed

echo.
echo Installing AI and image processing libraries...
python -m pip install ultralytics==8.0.206
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install opencv-python==4.8.1.78
python -m pip install pillow==10.0.1
python -m pip install numpy==1.24.3

if %errorlevel% neq 0 (
    echo ⚠️  GPU version failed, trying CPU version...
    python -m pip install torch torchvision
    python -m pip install opencv-python pillow numpy
    if %errorlevel% neq 0 (
        echo ❌ Failed to install AI dependencies!
        pause
        exit /b 1
    )
)

echo ✅ AI libraries installed

echo.
echo Installing additional dependencies...
python -m pip install pyyaml matplotlib seaborn pandas tqdm psutil

echo ✅ All dependencies installed successfully!

:: ==============================================================================
:: STEP 5: Download and verify model
:: ==============================================================================
echo.
echo 🤖 STEP 5: Setting up AI model...
echo.

:: Check if model file exists
if exist "best.pt" (
    echo ✅ AI model found: best.pt
) else (
    echo ⚠️  Custom model not found, will use YOLOv8 nano model
    echo Downloading YOLOv8 nano model...
    python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model downloaded successfully')"
    if %errorlevel% neq 0 (
        echo ❌ Failed to download model!
        pause
        exit /b 1
    )
    echo ✅ YOLOv8 model ready
)

:: ==============================================================================
:: STEP 6: Verify installation
:: ==============================================================================
echo.
echo 🔍 STEP 6: Verifying installation...
echo.

:: Test Python imports
python -c "import fastapi, uvicorn, ultralytics, cv2, PIL; print('All imports successful')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Some dependencies are missing! Attempting to fix...
    python -m pip install --force-reinstall fastapi uvicorn ultralytics opencv-python pillow
    
    python -c "import fastapi, uvicorn, ultralytics, cv2, PIL; print('All imports successful')" >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Installation verification failed!
        echo Please check the error messages above and try running the script again.
        pause
        exit /b 1
    )
)

echo ✅ Installation verified successfully!

:: ==============================================================================
:: STEP 7: Create convenience scripts
:: ==============================================================================
echo.
echo 📝 STEP 7: Creating convenience scripts...
echo.

:: Create start server script
echo @echo off > start_server.bat
echo call plant_ai_env\Scripts\activate.bat >> start_server.bat
echo echo Starting Plant Disease Detection Server... >> start_server.bat
echo echo Open http://localhost:8000 in your browser >> start_server.bat
echo python app.py >> start_server.bat
echo pause >> start_server.bat

:: Create quick start script
echo @echo off > quick_start.bat
echo call plant_ai_env\Scripts\activate.bat >> quick_start.bat
echo start http://localhost:8000 >> quick_start.bat
echo python app.py >> quick_start.bat

echo ✅ Convenience scripts created

:: ==============================================================================
:: STEP 8: Launch the application
:: ==============================================================================
echo.
echo 🚀 STEP 8: Starting Plant Disease Detection Server...
echo.

echo ================================================================================
echo                            🎉 SETUP COMPLETE! 🎉
echo ================================================================================
echo.
echo Your Plant Disease Detection AI is ready to use!
echo.
echo 📱 The web interface will open automatically at: http://localhost:8000
echo.
echo Features available:
echo   🌱 Upload & Analyze plant images
echo   📷 Webcam capture for real-time analysis  
echo   🖼️  Test with sample images
echo   🔬 AI-powered disease detection
echo   💊 Treatment recommendations
echo.
echo 💡 Tip: Keep this window open while using the application
echo.
echo ⚡ Quick restart: Run 'quick_start.bat' anytime
echo.

:: Wait a moment then open browser
timeout /t 3 /nobreak >nul
start http://localhost:8000

:: Start the server
echo Starting server now...
echo.
python app.py

:: If we get here, the server stopped
echo.
echo ================================================================================
echo Server stopped. You can restart it anytime by running:
echo   • quick_start.bat (opens browser automatically)
echo   • start_server.bat (manual browser opening)
echo   • python app.py (after activating environment)
echo ================================================================================
echo.
pause
