# Runware Authentication Setup

To extract **all available models** from Runware (not just the 81 examples on the public pricing page), you need to set up authentication.

## Why Authentication?

- **Public page**: 81 example models with pricing
- **Authenticated access**: Hundreds of models from the full catalog
- **Playground pricing**: Get approximate costs for each model configuration

## Setup Steps

### 1. Get Your Cookies

1. Open https://my.runware.ai in your browser and log in
2. Open Developer Tools (Press `F12`)
3. Go to the **Application** (Chrome) or **Storage** (Firefox) tab
4. Navigate to **Cookies** → `https://my.runware.ai`
5. Copy all cookies in this format:
   ```
   BEARER=your_token_here; RUNWARE_APP-AUTH_TOKEN=your_auth_token; other_cookie=value; ...
   ```

### 2. Configure Authentication

Edit `setup_runware_auth.py` and replace `PASTE YOUR COOKIES HERE` with your cookie string:

```python
COOKIE_STRING = """
BEARER=eyJhbGc...; RUNWARE_APP-AUTH_TOKEN=abc123...; session_id=xyz...
"""
```

### 3. Run Setup Script

```bash
python setup_runware_auth.py
```

You should see:
```
✅ Parsed X cookies
✅ Created new authentication
✅ Runware authentication configured successfully!
```

### 4. Test Extraction

```bash
python test_runware.py
```

With authentication, you should see:
- **Many more models** (hundreds instead of 81)
- Models extracted from authenticated pages
- Playground pricing for each model

## How It Works

The authenticated extraction:

1. **Loads cookies** from your database (`AuthSettings` table)
2. **Visits** `https://my.runware.ai/models/all` with authentication
3. **Extracts** all model AIR identifiers (e.g., `vidu:1@1`, `flux:1@dev`)
4. **For each model**:
   - Constructs playground URL: `https://my.runware.ai/playground/videoInference?modelAIR=vidu:1@1`
   - Visits the page with Playwright
   - Scrapes the approximate cost from the UI
5. **Falls back** to public pricing page if authentication fails

## Cookie Format

Your cookies should include at minimum:
- `BEARER` - Main authentication token
- `RUNWARE_APP-AUTH_TOKEN` - App-specific token

The setup script automatically converts cookie strings into Playwright-compatible format.

## Troubleshooting

**"No Runware authentication found"**
- Run `python setup_runware_auth.py` first
- Make sure you pasted valid cookies

**"Authenticated extraction returned no models"**
- Check if your cookies are still valid (they expire)
- Re-login to Runware and get fresh cookies

**Rate limiting**
- The extractor adds 0.5s delay between models
- For hundreds of models, extraction may take a few minutes

## Security Note

Cookies are stored in your local database (`ai_costs.db`). Keep this file secure as it contains your authentication tokens.

Never commit `setup_runware_auth.py` with real cookies to version control.
