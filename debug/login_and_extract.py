"""
Login to Runware and extract models with authentication
"""

import sys
import io
import json
import time
from playwright.sync_api import sync_playwright

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def login_and_extract():
    """Login to Runware and get authenticated access to models"""
    
    email = "rzrd2024@gmail.com"
    password = "REZVANI@rzrd2024"
    
    with sync_playwright() as p:
        # Launch with non-headless to see what's happening
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Go to login page
            print("Navigating to login page...")
            page.goto('https://my.runware.ai/login', timeout=60000)
            
            print("Waiting for page to load...")
            page.wait_for_load_state('domcontentloaded', timeout=30000)
            time.sleep(3)
            
            # Take screenshot of login page
            page.screenshot(path='login_page.png')
            print("Login page screenshot saved")
            
            # Try to find email input
            print("\nLooking for email input field...")
            email_selectors = [
                'input[type="email"]',
                'input[name="email"]',
                'input[placeholder*="email" i]',
                'input#email',
                '[data-testid="email"]',
                'input[autocomplete="email"]',
            ]
            
            email_input = None
            for selector in email_selectors:
                try:
                    if page.query_selector(selector):
                        email_input = selector
                        print(f"Found email input: {selector}")
                        break
                except Exception:
                    continue
            
            if not email_input:
                print("❌ Could not find email input field")
                print("Available inputs:")
                inputs = page.query_selector_all('input')
                for inp in inputs:
                    inp_type = inp.get_attribute('type')
                    inp_name = inp.get_attribute('name')
                    inp_placeholder = inp.get_attribute('placeholder')
                    print(f"  - type={inp_type}, name={inp_name}, placeholder={inp_placeholder}")
                return
            
            # Fill in email
            print(f"Entering email into {email_input}...")
            page.fill(email_input, email)
            time.sleep(1)
            
            # Try to find password input
            print("\nLooking for password input field...")
            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[placeholder*="password" i]',
                'input#password',
                '[data-testid="password"]',
                'input[autocomplete="current-password"]',
            ]
            
            password_input = None
            for selector in password_selectors:
                try:
                    if page.query_selector(selector):
                        password_input = selector
                        print(f"Found password input: {selector}")
                        break
                except Exception:
                    continue
            
            if not password_input:
                print("❌ Could not find password input field")
                return
            
            # Fill in password
            print(f"Entering password into {password_input}...")
            page.fill(password_input, password)
            time.sleep(1)
            
            # Take screenshot before clicking
            page.screenshot(path='before_login_click.png')
            print("Screenshot before login saved")
            
            # Try to find and click login button
            print("\nLooking for login button...")
            button_selectors = [
                'button[type="submit"]',
                'button:has-text("Login")',
                'button:has-text("Sign in")',
                'button:has-text("Log in")',
                '[data-testid="login"]',
                'input[type="submit"]',
            ]
            
            login_button = None
            for selector in button_selectors:
                try:
                    if page.query_selector(selector):
                        login_button = selector
                        print(f"Found login button: {selector}")
                        break
                except Exception:
                    continue
            
            if not login_button:
                print("❌ Could not find login button")
                print("Available buttons:")
                buttons = page.query_selector_all('button')
                for btn in buttons:
                    btn_text = btn.text_content()
                    btn_type = btn.get_attribute('type')
                    print(f"  - type={btn_type}, text={btn_text}")
                return
            
            print(f"Clicking login button: {login_button}...")
            page.click(login_button)
            
            # Wait for navigation or error message
            print("Waiting for response...")
            time.sleep(3)
            
            # Check for error messages
            error_selectors = [
                '[class*="error"]',
                '[class*="Error"]',
                '[role="alert"]',
                '.error-message',
                '#error',
            ]
            
            for selector in error_selectors:
                try:
                    error_elem = page.query_selector(selector)
                    if error_elem:
                        error_text = error_elem.text_content()
                        if error_text and error_text.strip():
                            print(f"⚠️  Error message found: {error_text}")
                except Exception:
                    continue
            
            # Wait longer for potential redirect
            time.sleep(5)
            
            current_url = page.url
            print(f"\nCurrent URL after login: {current_url}")
            print(f"Page title: {page.title()}")
            
            if 'login' in current_url.lower():
                print("❌ Still on login page - login may have failed")
                page.screenshot(path='login_failed.png')
                print("Screenshot saved to login_failed.png")
                return
            
            print("✅ Login successful!")
            
            # Get cookies after successful login
            cookies = context.cookies()
            print(f"\n✅ Retrieved {len(cookies)} cookies")
            
            # Save cookies to file
            with open('runware_cookies.json', 'w') as f:
                json.dump(cookies, f, indent=2)
            print("Saved cookies to runware_cookies.json")
            
            # Now navigate to models page
            print("\nNavigating to models page...")
            page.goto('https://my.runware.ai/models/all', timeout=60000)
            time.sleep(3)
            
            print(f"Current URL: {page.url}")
            print(f"Page title: {page.title()}")
            
            # Wait for network to be idle
            try:
                page.wait_for_load_state('networkidle', timeout=30000)
            except Exception as e:
                print(f"Network idle timeout: {e}")
            
            time.sleep(3)
            
            # Save page content
            content = page.content()
            with open('models_page_authenticated.html', 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Saved page HTML to models_page_authenticated.html ({len(content)} bytes)")
            
            # Take screenshot
            page.screenshot(path='models_page_authenticated.png', full_page=True)
            print("Saved screenshot to models_page_authenticated.png")
            
            # Search for model AIRs in the page
            print("\nSearching for model AIRs...")
            import re
            
            # Get visible text
            visible_text = page.evaluate("() => document.body.innerText")
            airs = re.findall(r'[a-z]+:\d+@[a-z0-9-]+', visible_text, re.IGNORECASE)
            
            if airs:
                unique_airs = list(set(airs))
                print(f"✅ Found {len(unique_airs)} unique model AIRs:")
                for air in sorted(unique_airs)[:20]:
                    print(f"  - {air}")
                
                # Save AIRs to file
                with open('model_airs.txt', 'w') as f:
                    for air in sorted(unique_airs):
                        f.write(air + '\n')
                print(f"\nSaved all {len(unique_airs)} AIRs to model_airs.txt")
            else:
                print("❌ No model AIRs found in page text")
                
                # Try to find any API calls that might contain model data
                print("\nChecking for model data in page HTML...")
                if 'model' in content.lower():
                    print("Found 'model' mentions in HTML")
                    
                    # Try to extract from JSON in HTML
                    json_matches = re.findall(r'\{[^}]*"air"[^}]*:[^}]*"([^"]+)"[^}]*\}', content, re.IGNORECASE)
                    if json_matches:
                        print(f"Found {len(json_matches)} potential AIRs in JSON data")
                        for match in json_matches[:10]:
                            print(f"  - {match}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            page.screenshot(path='error_screenshot.png')
            print("Error screenshot saved to error_screenshot.png")
        
        finally:
            page.close()
            context.close()
            browser.close()
    
    print("\n" + "="*60)
    print("Login and extraction complete!")
    print("="*60)

if __name__ == '__main__':
    login_and_extract()
