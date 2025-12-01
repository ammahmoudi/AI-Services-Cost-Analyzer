"""
Quick test to verify pricing extraction with parameter setting
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractors.runware_extractor import RunwareExtractor
import json

# Test with a couple of models
extractor = RunwareExtractor(use_llm=False, fetch_schemas=False)

# Manually test the authenticated extraction with just a few models
test_airs = ["vidu:1@0", "bfl:4@1", "klingai:1@1"]

print("Testing pricing extraction with parameter configuration...")
print("=" * 60)

from playwright.sync_api import sync_playwright

email = "rzrd2024@gmail.com"
password = "REZVANI@rzrd2024"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    
    # Login
    print("Logging in...")
    page.goto('https://my.runware.ai/login')
    page.wait_for_load_state('networkidle')
    
    email_selectors = ['input[type="email"]', '[name="email"]', '#email']
    for selector in email_selectors:
        try:
            page.fill(selector, email, timeout=5000)
            break
        except:
            continue
    
    password_selectors = ['input[type="password"]', '[name="password"]', '#password']
    for selector in password_selectors:
        try:
            page.fill(selector, password, timeout=5000)
            break
        except:
            continue
    
    submit_selectors = ['button[type="submit"]', '[type="submit"]']
    for selector in submit_selectors:
        try:
            page.click(selector, timeout=5000)
            break
        except:
            continue
    
    page.wait_for_url('**/home**', timeout=30000)
    print("✅ Logged in\n")
    
    # Test each AIR
    for air in test_airs:
        print(f"\nTesting: {air}")
        print("-" * 60)
        
        result = extractor._extract_model_pricing_from_playground(page, air)
        
        if result:
            print(f"✅ Successfully extracted:")
            print(f"   Model: {result.get('model_name')}")
            print(f"   Provider: {result.get('provider')}")
            print(f"   Type: {result.get('model_type')}")
            
            # Show all pricing fields
            price_display = None
            if result.get('output_price_per_image'):
                price_display = f"${result.get('output_price_per_image')} per image"
            elif result.get('output_price_per_second'):
                price_display = f"${result.get('output_price_per_second')} per second"
            elif result.get('output_price_per_request'):
                price_display = f"${result.get('output_price_per_request')} per request"
            else:
                price_display = "N/A"
            
            print(f"   Price: {price_display}")
            print(f"   Tags: {result.get('tags')}")
        else:
            print(f"❌ Failed to extract pricing")
    
    browser.close()

print("\n" + "=" * 60)
print("Test complete!")
