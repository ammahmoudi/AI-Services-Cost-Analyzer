"""
Test complex pricing text extraction.
"""
import requests
import re

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

response = requests.get(playground_url, timeout=10)
html = response.text

# Find the actual pricing text
will_cost_idx = html.find("will cost")
if will_cost_idx != -1:
    print("=== Raw HTML around 'will cost' ===")
    snippet = html[will_cost_idx-50:will_cost_idx + 300]
    print(snippet)
    print("\n")

# Try to extract just the visible text from that section
# Remove HTML tags
text_only = re.sub(r'<[^>]+>', ' ', snippet)
text_only = re.sub(r'<!--[^>]*-->', '', text_only)
text_only = re.sub(r'\s+', ' ', text_only).strip()
print("=== Cleaned text ===")
print(text_only)
print("\n")

# Now test patterns on the cleaned text
patterns = [
    r'will cost[^.]+\$[0-9.]+[^.]+\.',
    r'will cost.*?\$[0-9.]+.*?per.*?[a-z]+',
]

for i, pattern in enumerate(patterns, 1):
    matches = re.findall(pattern, text_only, re.IGNORECASE)
    print(f"Pattern {i}: {pattern}")
    print(f"  Matches: {matches if matches else 'None'}")
    print()
