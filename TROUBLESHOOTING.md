# 🌱 Plant Disease Detection AI - Troubleshooting Guide

## Quick Fixes

### 🚫 "Python not found" Error
**Solution:**
1. Install Python 3.8+ from https://python.org/downloads/
2. During installation, CHECK "Add Python to PATH"
3. Restart your computer
4. Run GowthamSetup.bat again

### 🔧 "Failed to create virtual environment"
**Solution:**
1. Open Command Prompt as Administrator
2. Run: `python -m pip install --upgrade pip`
3. Run: `python -m pip install virtualenv`
4. Run GowthamSetup.bat again

### 🌐 "Can't access localhost:8000"
**Solutions:**
- Wait 30 seconds for server to fully start
- Try: http://127.0.0.1:8000
- Check if antivirus is blocking the connection
- Restart the server using quick_start.bat

### 🤖 "Model not found" Error
**Solution:**
1. Make sure you have internet connection
2. The script will auto-download YOLOv8 model
3. If you have a custom model, name it "best.pt"

### 📱 Webcam not working
**Solutions:**
- Allow camera permissions in your browser
- Try a different browser (Chrome/Edge recommended)
- Check if another app is using the camera

### 💾 Installation takes too long
**Normal behavior:**
- First run: 5-10 minutes (downloading AI models)
- Subsequent runs: 30 seconds

## Manual Installation Steps

If automatic setup fails, follow these steps:

1. **Create virtual environment:**
   ```
   python -m venv plant_ai_env
   plant_ai_env\Scripts\activate.bat
   ```

2. **Install dependencies:**
   ```
   pip install -r webapp_requirements.txt
   ```

3. **Start server:**
   ```
   python app.py
   ```

## System Requirements

- **OS:** Windows 10/11
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space
- **Internet:** Required for initial setup

## Support

If you still have issues:
1. Check that all files are in the same folder
2. Run as Administrator if needed
3. Temporarily disable antivirus during setup
4. Make sure you have a stable internet connection

## File Structure
Your folder should contain:
```
PlantLeafDetections/
├── GowthamSetup.bat          # Main setup script
├── quick_start.bat           # Quick launcher
├── app.py                    # Main application
├── templates/index.html      # Web interface
├── plant_ai_env/            # Virtual environment (created by setup)
├── static/                  # Web assets
└── archive/                 # Dataset (if available)
```
