#!/bin/bash
# Setup script for Unix/Linux/Mac

echo "Creating virtual environment..."
python3 -m venv venv

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Initializing database..."
python manage.py init-db

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run: source venv/bin/activate"
echo "To start the web app, run: python app.py"
echo ""
