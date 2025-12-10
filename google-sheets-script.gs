/**
 * AI Cost Manager - Google Sheets Integration
 * 
 * This script integrates your Google Sheet with the AI Cost Manager API
 * to search for AI models and display pricing information.
 * 
 * Setup:
 * 1. Update API_BASE_URL with your API URL
 * 2. Run onOpen() to authorize
 * 3. Use "AI Models" menu or custom functions
 */

// ============================================
// CONFIGURATION
// ============================================

// Update this with your API URL (no trailing slash)
const API_BASE_URL = "http://localhost:5000";  // Change to your API URL
// For production: "https://your-domain.com"
// For ngrok: "https://abc123.ngrok.io"

// ============================================
// MENU FUNCTIONS
// ============================================

/**
 * Creates custom menu when spreadsheet opens
 */
function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('AI Models')
    .addItem('Search Models in Selection', 'searchModelsInSelection')
    .addItem('Update All Prices', 'updateAllPrices')
    .addItem('Clear Results', 'clearResults')
    .addSeparator()
    .addItem('About', 'showAbout')
    .addToUi();
}

/**
 * Shows information dialog
 */
function showAbout() {
  const ui = SpreadsheetApp.getUi();
  ui.alert(
    'AI Cost Manager Integration',
    'Version 1.0\n\n' +
    'This integration searches for AI models and displays pricing.\n\n' +
    'Usage:\n' +
    '1. Enter model names in Column A\n' +
    '2. Select the range\n' +
    '3. Click "AI Models > Search Models"\n\n' +
    'API URL: ' + API_BASE_URL,
    ui.ButtonSet.OK
  );
}

// ============================================
// MAIN FUNCTIONS
// ============================================

/**
 * Searches for models in the selected range and populates results
 */
function searchModelsInSelection() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const range = sheet.getActiveRange();
  const values = range.getValues();
  
  const ui = SpreadsheetApp.getUi();
  const response = ui.alert(
    'Search Models',
    `Search for ${values.length} model(s)?\n\n` +
    'This will populate columns B (dropdown), C (cheapest), and D (price).',
    ui.ButtonSet.YES_NO
  );
  
  if (response !== ui.Button.YES) {
    return;
  }
  
  // Show progress
  const startRow = range.getRow();
  const startCol = range.getColumn();
  
  for (let i = 0; i < values.length; i++) {
    const modelName = values[i][0];
    
    if (!modelName || modelName.toString().trim() === '') {
      continue;
    }
    
    try {
      // Search for model
      const result = searchModel(modelName);
      
      if (result && result.success) {
        const currentRow = startRow + i;
        
        // Column B: Create dropdown with all model options
        if (result.models && result.models.length > 0) {
          const modelOptions = result.models.map(m => {
            const price = formatPrice(m.cost_per_call);
            return `${m.name} [${m.provider}] - ${price}`;
          });
          
          // Set data validation (dropdown)
          const dropdownCell = sheet.getRange(currentRow, startCol + 1);
          const rule = SpreadsheetApp.newDataValidation()
            .requireValueInList(modelOptions, true)
            .setAllowInvalid(false)
            .build();
          dropdownCell.setDataValidation(rule);
          
          // Set default to first (best match)
          dropdownCell.setValue(modelOptions[0]);
        }
        
        // Column C: Cheapest model name
        if (result.cheapest) {
          sheet.getRange(currentRow, startCol + 2)
            .setValue(result.cheapest.name);
        }
        
        // Column D: Cheapest price
        if (result.cheapest && result.cheapest.cost_per_call) {
          sheet.getRange(currentRow, startCol + 3)
            .setValue(result.cheapest.cost_per_call)
            .setNumberFormat('$0.0000');
        }
      } else {
        // No results found
        sheet.getRange(startRow + i, startCol + 1)
          .setValue('No results found')
          .setBackground('#ffebee');
      }
      
      // Small delay to avoid rate limiting
      Utilities.sleep(100);
      
    } catch (e) {
      Logger.log(`Error processing ${modelName}: ${e.message}`);
      sheet.getRange(startRow + i, startCol + 1)
        .setValue('Error: ' + e.message)
        .setBackground('#ffebee');
    }
  }
  
  ui.alert('Complete', `Processed ${values.length} model(s)`, ui.ButtonSet.OK);
}

/**
 * Updates all prices in the sheet
 */
function updateAllPrices() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const lastRow = sheet.getLastRow();
  
  if (lastRow < 2) {
    SpreadsheetApp.getUi().alert('No data found');
    return;
  }
  
  const range = sheet.getRange(2, 1, lastRow - 1, 1); // Start from row 2, column A
  const selection = sheet.setActiveRange(range);
  searchModelsInSelection();
}

/**
 * Clears results from columns B, C, D
 */
function clearResults() {
  const sheet = SpreadsheetApp.getActiveSheet();
  const range = sheet.getActiveRange();
  const startRow = range.getRow();
  const numRows = range.getNumRows();
  
  // Clear columns B, C, D
  sheet.getRange(startRow, 2, numRows, 3).clearContent().clearDataValidations().setBackground(null);
  
  SpreadsheetApp.getUi().alert('Results cleared');
}

// ============================================
// API FUNCTIONS
// ============================================

/**
 * Searches for a model using the API
 * @param {string} modelName - Name of the model to search
 * @returns {Object} Search results
 */
function searchModel(modelName) {
  const url = `${API_BASE_URL}/api/search-model?name=${encodeURIComponent(modelName)}`;
  
  try {
    const response = UrlFetchApp.fetch(url, {
      method: 'get',
      muteHttpExceptions: true,
      headers: {
        'Accept': 'application/json'
      }
    });
    
    const statusCode = response.getResponseCode();
    
    if (statusCode === 200) {
      return JSON.parse(response.getContentText());
    } else {
      Logger.log(`API error (${statusCode}): ${response.getContentText()}`);
      throw new Error(`API returned status ${statusCode}`);
    }
  } catch (e) {
    Logger.log(`Error calling API: ${e.message}`);
    throw new Error(`Failed to connect to API: ${e.message}`);
  }
}

/**
 * Formats price for display
 * @param {number} price - Price value
 * @returns {string} Formatted price
 */
function formatPrice(price) {
  if (!price || price === 0) {
    return 'N/A';
  }
  return '$' + price.toFixed(4);
}

// ============================================
// CUSTOM SHEET FUNCTIONS
// ============================================

/**
 * Searches for a model and returns JSON results
 * Usage: =SEARCH_MODEL(A2)
 * 
 * @param {string} modelName - Name of the model
 * @return {string} JSON string with results
 * @customfunction
 */
function SEARCH_MODEL(modelName) {
  if (!modelName) {
    return "Error: Model name required";
  }
  
  try {
    const result = searchModel(modelName);
    return JSON.stringify(result, null, 2);
  } catch (e) {
    return `Error: ${e.message}`;
  }
}

/**
 * Gets the cheapest model name for a search
 * Usage: =GET_CHEAPEST(A2)
 * 
 * @param {string} modelName - Name of the model
 * @return {string} Name of cheapest model
 * @customfunction
 */
function GET_CHEAPEST(modelName) {
  if (!modelName) {
    return "N/A";
  }
  
  try {
    const result = searchModel(modelName);
    
    if (result && result.success && result.cheapest) {
      return result.cheapest.name;
    }
    
    return "No results";
  } catch (e) {
    return `Error: ${e.message}`;
  }
}

/**
 * Gets the price of the cheapest model
 * Usage: =GET_CHEAPEST_PRICE(A2)
 * 
 * @param {string} modelName - Name of the model
 * @return {number} Price of cheapest model
 * @customfunction
 */
function GET_CHEAPEST_PRICE(modelName) {
  if (!modelName) {
    return 0;
  }
  
  try {
    const result = searchModel(modelName);
    
    if (result && result.success && result.cheapest && result.cheapest.cost_per_call) {
      return result.cheapest.cost_per_call;
    }
    
    return 0;
  } catch (e) {
    Logger.log(`Error: ${e.message}`);
    return 0;
  }
}

/**
 * Gets all matching models as a list
 * Usage: =GET_ALL_MODELS(A2)
 * 
 * @param {string} modelName - Name of the model
 * @return {string} Comma-separated list of models
 * @customfunction
 */
function GET_ALL_MODELS(modelName) {
  if (!modelName) {
    return "N/A";
  }
  
  try {
    const result = searchModel(modelName);
    
    if (result && result.success && result.models) {
      return result.models.map(m => m.name).join(', ');
    }
    
    return "No results";
  } catch (e) {
    return `Error: ${e.message}`;
  }
}

/**
 * Gets model count for a search
 * Usage: =GET_MODEL_COUNT(A2)
 * 
 * @param {string} modelName - Name of the model
 * @return {number} Number of matching models
 * @customfunction
 */
function GET_MODEL_COUNT(modelName) {
  if (!modelName) {
    return 0;
  }
  
  try {
    const result = searchModel(modelName);
    
    if (result && result.success) {
      return result.total_found;
    }
    
    return 0;
  } catch (e) {
    Logger.log(`Error: ${e.message}`);
    return 0;
  }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Tests the API connection
 */
function testAPIConnection() {
  try {
    const result = searchModel("test");
    Logger.log("API connection successful");
    Logger.log(JSON.stringify(result, null, 2));
    SpreadsheetApp.getUi().alert(
      'API Connection Test',
      'Connection successful!\n\n' +
      `Found ${result.total_found} models for "test"`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );
  } catch (e) {
    Logger.log("API connection failed: " + e.message);
    SpreadsheetApp.getUi().alert(
      'API Connection Test',
      'Connection failed!\n\n' +
      'Error: ' + e.message + '\n\n' +
      'Please check:\n' +
      '1. API_BASE_URL is correct\n' +
      '2. API is running\n' +
      '3. API is accessible from internet (if using ngrok)',
      SpreadsheetApp.getUi().ButtonSet.OK
    );
  }
}
