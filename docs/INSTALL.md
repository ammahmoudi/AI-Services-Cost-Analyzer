# Installation Guide

## Requirements

- Python 3.8+
- pip

## Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-costs
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows (CMD):
     ```cmd
     venv\Scripts\activate.bat
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Playwright browsers**
   
   After installing the Python packages, you need to install the browser binaries:
   ```bash
   playwright install chromium
   ```
   
   This will download and install Chromium (~200MB). This is required for authenticated fal.ai pricing extraction.

6. **Initialize database**
   ```bash
   python -c "from ai_cost_manager.database import init_db; init_db()"
   ```

7. **Run the application**
   ```bash
   python app.py
   ```

8. **Access the web interface**
   
   Open your browser to: http://localhost:5000

## Configuration

### API Keys

For certain AI providers, you may need to add API keys:
1. Go to http://localhost:5000/api-keys
2. Add your API keys for relevant services

### Authentication (fal.ai)

To get accurate pricing for fal.ai models:
1. Go to http://localhost:5000/auth-settings
2. Follow the instructions to extract your `wos-session` cookie
3. Save the authentication settings
4. Re-fetch fal.ai models to get real pricing

## Troubleshooting

### Playwright Installation Issues

If `playwright install chromium` fails:
- Make sure you have enough disk space (~200MB)
- Try: `playwright install --with-deps chromium` to include system dependencies
- On Linux, you may need to install additional libraries

### Database Issues

If you get database errors:
```bash
# Reset the database
rm -f ai_costs.db  # or del ai_costs.db on Windows
python -c "from ai_cost_manager.database import init_db; init_db()"
```

### Port Already in Use

If port 5000 is already in use:
```bash
# Run on a different port
python app.py --port 5001
```
