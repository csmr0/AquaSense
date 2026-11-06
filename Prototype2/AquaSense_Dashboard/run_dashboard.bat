@echo off
cd /d "%~dp0"
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Could not start AquaSense. Install dependencies first:
    echo python -m pip install -r requirements.txt
    pause
)
