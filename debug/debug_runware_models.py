"""
Debug script to inspect the authenticated models page
"""

import json
import sys
import io
from playwright.sync_api import sync_playwright
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AuthSettings

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def debug_models_page():
    """Debug the models page to see what's actually there"""
    
    # Get cookies
    session_db = get_session()
    auth = session_db.query(AuthSettings).filter_by(
        source_name='runware',
        is_active=True
    ).first()
    
    if not auth or not hasattr(auth, 'cookies') or not auth.cookies:
        print("❌ No authentication found")
        return
    
    cookies = json.loads(str(auth.cookies))
    print(f"✅ Loaded {len(cookies)} cookies\n")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set to False to see the browser
        context = browser.new_context()
        context.add_cookies(cookies)
        
        page = context.new_page()
        
        print("Navigating to models page...")
        page.goto('https://my.runware.ai/models/all', timeout=60000)
        
        print("Waiting for page to load...")
        try:
            page.wait_for_load_state('domcontentloaded', timeout=30000)
        except:
            pass
        
        input("\nPress Enter after the page has fully loaded in the browser window...")
        
        # Save page content
        content = page.content()
        with open('debug_page.html', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Saved page HTML to debug_page.html")
        
        # Try to find models
        print("\n" + "="*60)
        print("Searching for model elements...")
        print("="*60)
        
        # Strategy 1: Look for AIR patterns in text
        print("\n1. Searching for AIR patterns (provider:version@variant)...")
        import re
        air_matches = re.findall(r'[a-z]+:\d+@[a-z0-9]+', content, re.IGNORECASE)
        if air_matches:
            print(f"   Found {len(set(air_matches))} unique AIR patterns:")
            for air in sorted(set(air_matches))[:10]:
                print(f"   - {air}")
        else:
            print("   No AIR patterns found")
        
        # Strategy 2: Look for links to playground
        print("\n2. Searching for playground links...")
        links = page.query_selector_all('a[href*="playground"]')
        print(f"   Found {len(links)} playground links")
        if links:
            for link in links[:5]:
                href = link.get_attribute('href')
                print(f"   - {href}")
        
        # Strategy 3: Look for model cards or containers
        print("\n3. Searching for model containers...")
        selectors_to_try = [
            '[data-testid*="model"]',
            '[class*="model"]',
            '[class*="Model"]',
            '[class*="card"]',
            '[class*="Card"]',
        ]
        
        for selector in selectors_to_try:
            elements = page.query_selector_all(selector)
            if elements:
                print(f"   Found {len(elements)} elements matching '{selector}'")
                # Show first element's attributes
                if elements:
                    first = elements[0]
                    classes = first.get_attribute('class') or ''
                    data_attrs = {k: v for k, v in first.evaluate('el => Object.fromEntries(Object.entries(el.dataset))').items()}
                    print(f"     First element classes: {classes}")
                    if data_attrs:
                        print(f"     Data attributes: {data_attrs}")
        
        # Strategy 4: Look for buttons or interactive elements
        print("\n4. Searching for model-related buttons...")
        buttons = page.query_selector_all('button, [role="button"]')
        print(f"   Found {len(buttons)} buttons")
        
        # Strategy 5: Check for API calls in network
        print("\n5. Checking page title and URL...")
        print(f"   Title: {page.title()}")
        print(f"   URL: {page.url}")
        
        print("\n" + "="*60)
        print("Debug complete! Check debug_page.html for full page content")
        print("="*60)
        
        input("\nPress Enter to close browser...")
        
        page.close()
        context.close()
        browser.close()

if __name__ == '__main__':
    debug_models_page()
