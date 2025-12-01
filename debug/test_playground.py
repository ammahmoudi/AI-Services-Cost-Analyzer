"""
Test accessing a Runware playground page directly
"""
from playwright.sync_api import sync_playwright
import time

def test_playground():
    email = "rzrd2024@gmail.com"
    password = "REZVANI@rzrd2024"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()
        
        try:
            # Login first
            print("Logging in...")
            page.goto('https://my.runware.ai/login')
            page.fill('input[type="email"]', email)
            page.fill('input[type="password"]', password)
            page.click('button[type="submit"]')
            time.sleep(8)
            
            # Try accessing playground
            test_air = "bfl:4@1"  # FLUX.4 [schnell]
            playground_url = f'https://my.runware.ai/playground/imageInference?modelAIR={test_air}'
            
            print(f"\nAccessing playground: {playground_url}")
            page.goto(playground_url, timeout=30000)
            time.sleep(3)
            
            print(f"Current URL: {page.url}")
            print(f"Page title: {page.title()}")
            
            # Take screenshot
            page.screenshot(path='playground_test.png')
            print("Screenshot saved to playground_test.png")
            
            # Get page text
            text = page.evaluate("() => document.body.innerText")
            print(f"\nPage text ({len(text)} chars):")
            print(text[:1000])
            
            # Save HTML
            with open('playground_test.html', 'w', encoding='utf-8') as f:
                f.write(page.content())
            print("\nSaved HTML to playground_test.html")
            
            input("\nPress Enter to close browser...")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            page.close()
            context.close()
            browser.close()

if __name__ == '__main__':
    test_playground()
