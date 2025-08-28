@echo off 
call plant_ai_env\Scripts\activate.bat 
echo Starting Plant Disease Detection Server... 
echo Open http://localhost:8000 in your browser 
python app.py 
pause 
