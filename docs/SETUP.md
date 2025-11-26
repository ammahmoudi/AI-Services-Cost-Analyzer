# AI Cost Manager - Virtual Environment Setup

## Quick Setup (Recommended)

### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Initialize database
python manage.py init-db
```

### Windows (Command Prompt)
```cmd
# Run the setup script
setup.bat
```

### Linux/Mac
```bash
# Make setup script executable
chmod +x setup.sh

# Run it
./setup.sh
```

## Manual Setup

### 1. Create Virtual Environment

**Using venv (built-in):**
```bash
python -m venv venv
```

**Using uv (faster alternative):**
```bash
# Install uv first
pip install uv

# Create environment with uv
uv venv
```

### 2. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using uv (faster):**
```bash
uv pip install -r requirements.txt
```

### 4. Initialize Database
```bash
python manage.py init-db
```

## Verifying Installation

Check that everything is installed:
```bash
pip list
```

You should see packages like:
- Flask
- SQLAlchemy
- requests
- beautifulsoup4

## Running the Application

### Start the Web Interface
```bash
python app.py
```

Open http://localhost:5000 in your browser.

### Use CLI Commands
```bash
# Add fal.ai source
python manage.py add-source --name "Fal.ai" --url "https://fal.ai/api/trpc/models.list" --extractor "fal"

# Extract models
python manage.py extract

# List models
python manage.py list-models
```

## Deactivating Virtual Environment

When you're done:
```bash
deactivate
```

## Updating Dependencies

To update all packages:
```bash
pip install --upgrade -r requirements.txt
```

## Troubleshooting

**PowerShell execution policy error?**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Python not found?**
Make sure Python 3.8+ is installed and in your PATH.

**Port 5000 already in use?**
Edit `app.py` and change the port number in the last line.
