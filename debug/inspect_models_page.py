"""
Script to inspect the models page and see how it loads
"""

import sys
import io
import json
import time
from playwright.sync_api import sync_playwright

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ai_cost_manager.database import get_session
from ai_cost_manager.models import AuthSettings

def inspect_page():
    """Open the page with Playwright and inspect how it loads"""
    
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
        # Launch headless
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        
        # Add cookies
        context.add_cookies(cookies)
        
        page = context.new_page()
        
        # Collect console messages
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))
        
        # Collect requests/responses
        api_calls = []
        
        def log_response(response):
            if 'model' in response.url.lower() or 'api' in response.url.lower():
                api_calls.append({
                    'status': response.status,
                    'url': response.url,
                    'content_type': response.headers.get('content-type', '')
                })
                if response.status == 200 and 'json' in response.headers.get('content-type', ''):
                    try:
                        data = response.json()
                        api_calls[-1]['data_keys'] = list(data.keys()) if isinstance(data, dict) else 'array'
                        api_calls[-1]['data'] = data
                    except Exception:
                        pass
        
        page.on("response", log_response)
        
        print("Navigating to https://my.runware.ai/models/all...")
        try:
            page.goto('https://my.runware.ai/models/all', wait_until='domcontentloaded', timeout=60000)
        except Exception as e:
            print(f"Navigation error: {e}")
            page.close()
            context.close()
            browser.close()
            return
        
        print(f"\nPage URL: {page.url}")
        print(f"Page Title: {page.title()}")
        
        # Wait for network idle
        print("\nWaiting for network to be idle...")
        try:
            page.wait_for_load_state('networkidle', timeout=30000)
        except Exception as e:
            print(f"Network idle timeout: {e}")
        
        time.sleep(3)
        
        # Print API calls captured
        print("\n" + "="*60)
        print(f"Captured {len(api_calls)} API calls with 'model' or 'api' in URL:")
        print("="*60)
        for call in api_calls:
            print(f"\n[{call['status']}] {call['url']}")
            if 'data_keys' in call:
                print(f"  Keys: {call['data_keys']}")
                if 'data' in call:
                    print(f"  Data sample: {str(call['data'])[:200]}...")
        
        # Check for model data in page
        print("\n" + "="*60)
        print("Analyzing page content...")
        print("="*60)
        
        # 1. Check for AIR patterns in visible text
        print("\n1. Searching visible text for model AIRs...")
        visible_text = page.evaluate("() => document.body.innerText")
        import re
        airs = re.findall(r'[a-z]+:\d+@[a-z0-9-]+', visible_text, re.IGNORECASE)
        if airs:
            print(f"   Found {len(set(airs))} unique AIRs in visible text:")
            for air in sorted(set(airs))[:10]:
                print(f"   - {air}")
        else:
            print("   No AIRs found in visible text")
        
        # 2. Check localStorage
        print("\n2. Checking localStorage for model data...")
        try:
            local_storage = page.evaluate("() => JSON.stringify(localStorage)")
            ls_data = json.loads(local_storage)
            if ls_data:
                print(f"   localStorage keys: {list(ls_data.keys())}")
                for key in ls_data:
                    if 'model' in key.lower():
                        print(f"   - {key}: {ls_data[key][:100]}...")
            else:
                print("   localStorage is empty")
        except Exception as e:
            print(f"   Error reading localStorage: {e}")
        
        # 3. Check window object for data
        print("\n3. Checking window object for model data...")
        try:
            window_keys = page.evaluate("() => Object.keys(window).filter(k => k.toLowerCase().includes('model'))")
            if window_keys:
                print(f"   Window keys with 'model': {window_keys}")
            else:
                print("   No model-related keys in window")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 4. Look for React root data
        print("\n4. Looking for React data...")
        try:
            react_data = page.evaluate("""() => {
                const root = document.querySelector('#root, #__next, [data-reactroot]');
                if (root) {
                    return root.innerHTML.substring(0, 500);
                }
                return null;
            }""")
            if react_data:
                print(f"   React root found, content sample:")
                print(f"   {react_data[:200]}...")
            else:
                print("   No React root found")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 5. Save full page HTML
        print("\n5. Saving full page HTML...")
        content = page.content()
        with open('models_page_full.html', 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   Saved to models_page_full.html ({len(content)} bytes)")
        
        # 6. Take a screenshot
        print("\n6. Taking screenshot...")
        page.screenshot(path='models_page_screenshot.png', full_page=True)
        print("   Saved to models_page_screenshot.png")
        
        print("\n" + "="*60)
        print("Inspection complete!")
        print("="*60)
        
        page.close()
        context.close()
        browser.close()

if __name__ == '__main__':
    inspect_page()
