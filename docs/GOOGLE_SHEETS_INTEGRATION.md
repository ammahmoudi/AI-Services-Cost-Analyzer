# Google Sheets Integration Guide

This guide shows how to integrate the AI Cost Manager with Google Sheets to search for models and display pricing information.

## Features

- **Automatic Model Search**: Search for models based on name in your sheet
- **Dropdown Selection**: Show all matching models in a dropdown list
- **Cheapest Option Display**: Automatically show the cheapest option for each model search
- **Real-time Pricing**: Get up-to-date pricing information from your AI Cost Manager database

## Setup Instructions

### 1. Deploy Your AI Cost Manager API

First, ensure your AI Cost Manager is running and accessible via a URL:

- **Local Development**: `http://localhost:5000`
- **Production**: Use ngrok, render.com, or any hosting service

For local testing with ngrok:
```bash
# Install ngrok: https://ngrok.com/
ngrok http 5000
```

This will give you a public URL like `https://abc123.ngrok.io`

### 2. Set Up Your Google Sheet

Create a Google Sheet with the following structure:

| Column A: Model Name | Column B: Results Dropdown | Column C: Cheapest Option | Column D: Price |
|---------------------|---------------------------|---------------------------|---------|
| Recraft V3 | (dropdown) | Recraft V3 | $0.0050 |
| Flux Pro 1.1 Ultra | (dropdown) | FLUX1.1 [pro] ultra | $0.0500 |
| Flux Dev LoRA | (dropdown) | Flux.1 [dev] lora | $0.0000 |

### 3. Add Google Apps Script

1. Open your Google Sheet
2. Click **Extensions > Apps Script**
3. Delete any existing code
4. Paste the code from `google-sheets-script.gs` (see below)
5. Update the `API_BASE_URL` variable with your API URL
6. Save the script (Ctrl+S or Cmd+S)

### 4. Authorize the Script

1. Run the `onOpen` function (click the ▶️ button)
2. Authorize the script when prompted
3. Refresh your Google Sheet - you should see a new "AI Models" menu

### 5. Use the Integration

**Option 1: Menu-Based Search**
- Select the range with model names (Column A)
- Click **AI Models > Search Models in Selection**
- Wait for results to populate

**Option 2: Custom Functions**
Use these formulas in your sheet:
- `=SEARCH_MODEL(A2)` - Returns JSON with all matches
- `=GET_CHEAPEST(A2)` - Returns name of cheapest model
- `=GET_CHEAPEST_PRICE(A2)` - Returns price of cheapest model

## API Endpoints Used

### Search Models
- **Endpoint**: `/api/search-model?name={model_name}`
- **Method**: GET
- **Response**:
```json
{
  "success": true,
  "query": "Flux Pro 1.1",
  "total_found": 8,
  "models": [
    {
      "id": 123,
      "name": "FLUX1.1 [pro]",
      "model_id": "black-forest-labs/FLUX.1.1-pro",
      "provider": "BFL",
      "cost_per_call": 0.05,
      "pricing_type": "per_image",
      "parsed_company": "BFL",
      "parsed_version": "1.1"
    }
  ],
  "cheapest": {
    "name": "FLUX1.1 [pro]",
    "cost_per_call": 0.04
  }
}
```

## Troubleshooting

### CORS Errors
If you get CORS errors, add CORS headers to your Flask app:

```python
# Add to app.py
from flask_cors import CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

Install flask-cors:
```bash
pip install flask-cors
```

### Rate Limiting
Google Apps Script has quotas:
- URL Fetch calls: 20,000/day
- Script runtime: 6 min/execution

For large sheets, process in batches.

### Script Doesn't Run
1. Check API URL is correct
2. Check API is accessible (test in browser)
3. Check Google Apps Script permissions
4. Check execution logs (View > Logs)

## Advanced Usage

### Batch Processing
For sheets with 100+ models, use batch processing:
```javascript
// Process 10 models at a time
for (let i = 0; i < modelNames.length; i += 10) {
  const batch = modelNames.slice(i, i + 10);
  // Process batch
  Utilities.sleep(1000); // Wait 1 second between batches
}
```

### Custom Filtering
Modify the script to filter results:
```javascript
// Only show models from specific providers
const filteredModels = models.filter(m => 
  m.provider === "OpenAI" || m.provider === "Anthropic"
);
```

### Price Formatting
Customize price display:
```javascript
function formatPrice(price) {
  if (!price || price === 0) return "N/A";
  return "$" + price.toFixed(4);
}
```

## Example Use Cases

1. **Cost Comparison**: Compare pricing across different providers
2. **Budget Planning**: Calculate total costs for your AI workflow
3. **Provider Selection**: Find the cheapest provider for each model type
4. **Price Tracking**: Monitor price changes over time
5. **Procurement**: Generate cost reports for procurement teams

## Support

For issues or questions:
- Check the API is running: `http://your-api-url/models`
- Check Apps Script logs: **View > Logs** in Script Editor
- Test API endpoint in browser: `http://your-api-url/api/search-model?name=flux`
