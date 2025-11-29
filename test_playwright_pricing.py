"""Test Playwright dynamic pricing extraction"""
from playwright.sync_api import sync_playwright
import re
import time

url = 'https://fal.ai/models/fal-ai/animatediff-sparsectrl-lcm/playground'

print(f"Testing: {url}\n")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    
    # Track API requests
    api_requests = []
    def log_request(request):
        if 'api' in request.url or 'pricing' in request.url or 'billing' in request.url:
            api_requests.append({
                'url': request.url,
                'method': request.method
            })
    
    page.on('request', log_request)
    
    print("Navigating to page...")
    page.goto(url, wait_until='domcontentloaded', timeout=15000)
    
    print("Waiting for dynamic content...")
    time.sleep(5)  # Give extra time for JavaScript to calculate price
    
    # Try to trigger any lazy-loaded pricing calculations
    print("Scrolling page...")
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(2)
    
    print(f"\nAPI Requests captured: {len(api_requests)}")
    for req in api_requests[:10]:
        print(f"  {req['method']} {req['url']}")
    
    print("Extracting HTML...")
    page_html = page.content()
    
    # Look for machine type and pricing data
    machine_match = re.search(r'machineType["\']:\s*["\']([^"\']+)["\']', page_html)
    if machine_match:
        print(f"\nMachine Type: {machine_match.group(1)}")
    
    # Look for any price data in JavaScript
    price_patterns = [
        r'"price":\s*([0-9.]+)',
        r'price:\s*([0-9.]+)',
        r'\$\s*([0-9.]+)\s*per'
    ]
    
    print("\nSearching for price patterns in HTML:")
    for pattern in price_patterns:
        matches = re.findall(pattern, page_html)
        if matches:
            print(f"  Pattern '{pattern}': {matches[:5]}")
    
    print("\nExtracting visible text...")
    page_text = page.inner_text('body')
    
    # Find the pricing text
    price_match = re.search(r'will cost.*?\$\s*([0-9.]+)', page_text, re.IGNORECASE)
    
    if price_match:
        print(f"\n✓ Found pricing text: {price_match.group(0)}")
        print(f"✓ Extracted price: ${price_match.group(1)}")
    else:
        print("\n✗ No pricing in visible text")
    
    browser.close()
