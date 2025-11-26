"""
Debug pricing text extraction.
"""
import requests
import re

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

response = requests.get(playground_url, timeout=10)
html = response.text

# Find "will cost" and extract surrounding section
will_cost_idx = html.lower().find("will cost")
if will_cost_idx != -1:
    print(f"Found 'will cost' at index {will_cost_idx}\n")
    
    # Extract section
    start = max(0, will_cost_idx - 100)
    end = min(len(html), will_cost_idx + 400)
    raw_section = html[start:end]
    
    print("=== Raw section ===")
    print(raw_section[:300])
    print("\n")
    
    # Clean HTML
    cleaned = re.sub(r'<!--[^>]*-->', ' ', raw_section)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    print("=== Cleaned section ===")
    print(cleaned[:300])
    print("\n")
    
    # Try pattern
    pattern = r'([^.]*?will cost[^.]*?\$[0-9.]+[^.]*?\.)'
    matches = re.findall(pattern, cleaned, re.IGNORECASE)
    
    print(f"=== Pattern matches ({len(matches)}) ===")
    for m in matches:
        print(f"  - {m}")
