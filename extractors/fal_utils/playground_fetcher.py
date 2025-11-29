"""
Fal.ai Playground Data Fetcher

Functions for fetching pricing data from fal.ai playground pages.
"""
import re
import requests
from typing import Dict, Any, Optional
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def get_auth_cookies() -> Optional[Dict[str, str]]:
    """
    Get authentication cookies from database if available.
    
    Returns:
        Dictionary of cookies or None
    """
    try:
        from ai_cost_manager.database import get_session
        from ai_cost_manager.models import AuthSettings
        import json
        
        session = get_session()
        try:
            auth = session.query(AuthSettings).filter_by(
                source_name='fal.ai',
                is_active=True
            ).first()
            
            if auth and auth.cookies:
                return json.loads(auth.cookies)
        finally:
            session.close()
    except Exception:
        pass
    
    return None


def get_auth_headers() -> Optional[Dict[str, str]]:
    """
    Get authentication headers from database if available.
    
    Returns:
        Dictionary of headers or None
    """
    try:
        from ai_cost_manager.database import get_session
        from ai_cost_manager.models import AuthSettings
        import json
        
        session = get_session()
        try:
            auth = session.query(AuthSettings).filter_by(
                source_name='fal.ai',
                is_active=True
            ).first()
            
            if auth and auth.headers:
                return json.loads(auth.headers)
        finally:
            session.close()
    except Exception:
        pass
    
    return None


def fetch_playground_data(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch pricing data from the playground page.
    
    The playground page contains embedded endpointBilling data in JavaScript.
    We extract: endpoint, billing_unit, price, and pricing_text.
    
    If authentication is available in database, uses it to get authenticated pricing.
    
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
        
        # Add custom headers if available
        auth_headers = get_auth_headers()
        if auth_headers:
            headers.update(auth_headers)
        
        # Get cookies from database
        cookies = get_auth_cookies()
        
        response = requests.get(
            playground_url,
            headers=headers,
            cookies=cookies,
            timeout=10
        )
        
        if response.status_code != 200:
            return None
        
        html = response.text
        
        # Extract individual billing fields from escaped JSON in JavaScript
        endpoints = re.findall(r'\\"endpoint\\":\\"([^"]*' + re.escape(model_id) + r'[^"]*)\\"', html)
        units = re.findall(r'\\"billing_unit\\":\\"([^"]*)\\"', html)
        prices = re.findall(r'\\"price\\":([0-9.]+)', html)
        
        # Extract machine type (useful for understanding pricing)
        machine_types = re.findall(r'\\"machineType\\":\\"([^"]*)\\"', html)
        machine_type = machine_types[0] if machine_types else None
        
        # Extract the full pricing text for LLM analysis
        # This captures complex pricing like "For 5s video your request will cost $0.35..."
        pricing_text = _extract_pricing_text(html)
        
        if endpoints and units and prices:
            price = float(prices[0])
            
            # If price is 0, try to get dynamic price using Playwright
            if price == 0.0 and PLAYWRIGHT_AVAILABLE:
                dynamic_price = _fetch_dynamic_price(playground_url)
                if dynamic_price is not None:
                    price = dynamic_price
                    pricing_text = f"Your request will cost ${price} per {units[0]}."
            
            billing_data = {
                'endpoint': endpoints[0],
                'billing_unit': units[0],
                'price': price,
                'pricing_text': pricing_text,  # Full text for LLM
                'machine_type': machine_type,  # Hardware used (e.g., GPU-A100)
                'note': 'Price may be calculated dynamically based on compute resources' if price == 0.0 and machine_type else None
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


def _fetch_dynamic_price(playground_url: str) -> Optional[float]:
    """
    Use Playwright to fetch dynamically rendered price from playground page.
    
    Some fal.ai models show price: 0 in static HTML but calculate the actual
    price client-side based on machine costs (e.g., GPU-A100 at $0.00111/sec).
    
    Args:
        playground_url: Full playground URL
        
    Returns:
        Dynamic price as float or None if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return None
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate to playground
            page.goto(playground_url, wait_until='networkidle', timeout=15000)
            
            # Wait for the price to be rendered (look for the "will cost" text with actual price)
            # The price appears in format: "Your request will cost $0.00111 per compute second"
            try:
                # Wait for element containing "will cost" to have a non-zero price
                page.wait_for_function(
                    """() => {
                        const text = document.body.innerText;
                        const match = text.match(/will cost.*?\\$\\s*([0-9.]+)/i);
                        return match && parseFloat(match[1]) > 0;
                    }""",
                    timeout=10000
                )
            except PlaywrightTimeout:
                # If timeout, price might still be 0
                pass
            
            # Extract the price from rendered page
            page_text = page.inner_text('body')
            price_match = re.search(r'will cost.*?\$\s*([0-9.]+)', page_text, re.IGNORECASE)
            
            browser.close()
            
            if price_match:
                price = float(price_match.group(1))
                if price > 0:
                    return price
            
            return None
            
    except Exception as e:
        # Silently fail - Playwright might not be installed or configured
        return None
