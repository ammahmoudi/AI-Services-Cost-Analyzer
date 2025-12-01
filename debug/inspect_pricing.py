"""
Debug script to inspect pricing on Runware playground pages
"""

from playwright.sync_api import sync_playwright
import time
import re

def inspect_pricing():
    """Inspect pricing structure on a sample playground page"""
    
    email = "rzrd2024@gmail.com"
    password = "REZVANI@rzrd2024"
    
    # Sample AIRs to test (only 2 for quick testing)
    test_airs = [
        "bfl:4@1",  # Black Forest Labs (known model)
        "klingai:1@1",  # Kling AI
    ]
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible for inspection
        page = browser.new_page()
        
        # Login first
        print("Logging in...")
        page.goto('https://my.runware.ai/login')
        page.wait_for_load_state('networkidle')
        time.sleep(2)
        
        # Try multiple selectors for email field
        email_selectors = ['input[type="email"]', '[name="email"]', '#email']
        for selector in email_selectors:
            try:
                page.fill(selector, email, timeout=5000)
                break
            except Exception:
                continue
        
        # Try multiple selectors for password field
        password_selectors = ['input[type="password"]', '[name="password"]', '#password']
        for selector in password_selectors:
            try:
                page.fill(selector, password, timeout=5000)
                break
            except Exception:
                continue
        
        # Try multiple selectors for submit button
        submit_selectors = ['button[type="submit"]', '[type="submit"]', 'button:has-text("Sign in")', 'button:has-text("Log in")']
        for selector in submit_selectors:
            try:
                page.click(selector, timeout=5000)
                break
            except Exception:
                continue
        
        page.wait_for_url('**/home**', timeout=30000)
        print("✅ Login successful")
        
        # Navigate to models page
        page.goto('https://my.runware.ai/models/all')
        page.wait_for_load_state('networkidle')
        time.sleep(2)
        print("✅ On models page")
        
        for air in test_airs:
            print(f"\n{'='*60}")
            print(f"Testing AIR: {air}")
            print(f"{'='*60}")
            
            # Navigate to playground
            playground_url = f'https://my.runware.ai/playground?modelAIR={air}'
            page.goto(playground_url, wait_until='networkidle', timeout=15000)
            time.sleep(3)  # Wait for pricing to load
            
            # Save the page HTML for inspection
            filename = f"debug/playground_{air.replace(':', '_').replace('@', '_')}.html"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(page.content())
            print(f"Saved HTML to {filename}")
            
            # Take screenshot of the playground
            screenshot_path = f"debug/playground_{air.replace(':', '_').replace('@', '_')}.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved to {screenshot_path}")
            
            # Try to find model name/title
            print("\n🔍 Looking for model name...")
            name_selectors = ['h1', 'h2', '[class*="title"]', '[class*="model"]', '[class*="name"]']
            for selector in name_selectors:
                try:
                    elem = page.query_selector(selector)
                    if elem:
                        text = elem.text_content()
                        if text and len(text.strip()) > 0:
                            print(f"  - {selector}: {text.strip()[:100]}")
                except Exception:
                    pass
            
            # Try to find pricing/cost elements
            print("\n💰 Looking for pricing elements...")
            
            # Look for elements that might contain cost information
            cost_selectors = [
                '[data-testid*="cost"]',
                '[data-testid*="price"]',
                '[class*="cost"]',
                '[class*="price"]',
                '[class*="total"]',
                '[id*="cost"]',
                '[id*="price"]',
                'text=/\\$[\\d.]+/',
                'text=/cost/i',
                'text=/price/i',
            ]
            
            found_costs = []
            for selector in cost_selectors:
                try:
                    elems = page.query_selector_all(selector)
                    for elem in elems:
                        text = elem.text_content()
                        if text:
                            text = text.strip()
                            if text and (('$' in text) or ('cost' in text.lower()) or ('price' in text.lower()) or text.replace('.', '').isdigit()):
                                if text not in found_costs:
                                    found_costs.append(text)
                                    print(f"  - {selector}: {text}")
                except Exception:
                    pass
            
            # Look for any inputs/dropdowns that might affect pricing
            print("\n⚙️ Looking for settings/parameters that affect pricing...")
            
            # Common parameter types
            param_selectors = [
                'select',  # Dropdowns
                'input[type="number"]',  # Number inputs
                'input[type="range"]',  # Sliders
                '[role="slider"]',  # Custom sliders
                '[class*="slider"]',
                '[class*="select"]',
                '[class*="dropdown"]',
            ]
            
            for selector in param_selectors:
                try:
                    elems = page.query_selector_all(selector)
                    for i, elem in enumerate(elems):
                        # Get the label or nearby text
                        try:
                            label_elem = page.query_selector(f'{selector}:nth-of-type({i+1}) ~ label')
                            if label_elem:
                                label_text = label_elem.text_content()
                                value = elem.get_attribute('value') or elem.text_content()
                                print(f"  - {selector}: {label_text} = {value}")
                        except Exception:
                            pass
                except Exception:
                    pass
            
            # Search entire page content for dollar amounts
            print("\n🔎 All dollar amounts on page:")
            content = page.content()
            dollar_matches = re.findall(r'\$[\d.]+', content)
            unique_amounts = sorted(set(dollar_matches), key=lambda x: float(x.replace('$', '')))
            print(f"  Found: {unique_amounts}")
        
        print("\n✅ Inspection complete!")
        browser.close()

if __name__ == "__main__":
    inspect_pricing()
