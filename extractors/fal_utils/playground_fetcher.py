"""
Fal.ai Playground Data Fetcher

Functions for fetching pricing data from fal.ai playground pages.
"""
import re
import requests
from typing import Dict, Any, Optional


def fetch_playground_data(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch pricing data from the playground page.
    
    The playground page contains embedded endpointBilling data in JavaScript.
    We extract: endpoint, billing_unit, price, and pricing_text.
    
    Args:
        model_id: Model identifier (e.g., 'bria/reimagine/3.2')
        
    Returns:
        Dictionary with playground pricing data or None if failed
    """
    try:
        # Construct playground endpoint URL
        playground_url = f"https://fal.ai/models/{model_id}/playground"
        
        # GET the playground page (contains embedded billing data in JavaScript)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        
        response = requests.get(
            playground_url,
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            return None
        
        html = response.text
        
        # Extract individual billing fields from escaped JSON in JavaScript
        endpoints = re.findall(r'\\"endpoint\\":\\"([^"]*' + re.escape(model_id) + r'[^"]*)\\"', html)
        units = re.findall(r'\\"billing_unit\\":\\"([^"]*)\\"', html)
        prices = re.findall(r'\\"price\\":([0-9.]+)', html)
        
        # Extract the full pricing text for LLM analysis
        # This captures complex pricing like "For 5s video your request will cost $0.35..."
        pricing_text = _extract_pricing_text(html)
        
        if endpoints and units and prices:
            billing_data = {
                'endpoint': endpoints[0],
                'billing_unit': units[0],
                'price': float(prices[0]),
                'pricing_text': pricing_text  # Full text for LLM
            }
            
            return billing_data
        else:
            return None
            
    except Exception:
        # Silently fail - playground might not exist for all models
        return None


def _extract_pricing_text(html: str) -> Optional[str]:
    """
    Extract human-readable pricing text from HTML for LLM analysis.
    
    Examples:
        - "Your request will cost $0.04 per image."
        - "For 5s video your request will cost $0.35. For every additional second..."
    
    Args:
        html: Raw HTML content
        
    Returns:
        Cleaned pricing text or None
    """
    # Find "will cost" and extract surrounding context
    will_cost_idx = html.lower().find("will cost")
    if will_cost_idx == -1:
        return None
    
    # Extract a section around "will cost"
    start = max(0, will_cost_idx - 100)
    end = min(len(html), will_cost_idx + 400)
    raw_section = html[start:end]
    
    # Clean HTML comments and tags
    cleaned = re.sub(r'<!--[^>]*-->', ' ', raw_section)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Extract sentence with pricing information
    # Note: cleaned text has "$ 0.04" with space between $ and number
    sentence_patterns = [
        r'((?:Your|For|This|The)[^.]*?will cost[^.]*?\$\s*[0-9.]+[^.]*?\.)',
        r'(will cost[^.]*?\$\s*[0-9.]+[^.]*?\.)',  # Fallback without capital start
    ]
    
    for pattern in sentence_patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        if matches:
            pricing_text = matches[0].strip()
            if len(pricing_text) > 15:  # Must be reasonable length
                return pricing_text
    
    return None
