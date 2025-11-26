@echo off
REM Setup script for Windows

echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Initializing database...
python manage.py init-db

echo.
echo Setup complete!
echo.
echo To activate the environment, run: venv\Scripts\activate.bat
echo To start the web app, run: python app.py
echo.
