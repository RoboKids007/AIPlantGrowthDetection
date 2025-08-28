@echo off
echo ===============================================
echo Plant Disease Detection - Web Application
echo ===============================================
echo.

echo Checking if model exists...
if not exist "plant_disease_yolov8n_trained.pt" (
    echo WARNING: Trained model not found!
    echo Please run training first: python main.py
    echo The web app will use the default YOLOv8n model.
    echo.
)

echo Installing web application dependencies...
pip install fastapi uvicorn python-multipart jinja2 aiofiles pillow opencv-python
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Creating required directories...
if not exist "static" mkdir static
if not exist "static\uploads" mkdir static\uploads  
if not exist "static\results" mkdir static\results
if not exist "templates" mkdir templates

echo.
echo Starting Plant Disease Detection Web App...
echo.
echo The web application will be available at:
echo   http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
