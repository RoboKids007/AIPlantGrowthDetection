@echo off 
call plant_ai_env\Scripts\activate.bat 
start http://localhost:8000 
python app.py 
