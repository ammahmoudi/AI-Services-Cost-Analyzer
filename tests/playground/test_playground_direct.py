"""
Direct test of playground endpoint.
"""
import requests
import json

model_id = "bria/reimagine/3.2"
playground_url = f"https://fal.ai/models/{model_id}/playground"

print(f"Testing URL: {playground_url}")

headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

print("\nAttempting POST with empty body...")
response = requests.post(
    playground_url,
    json={},
    headers=headers,
    timeout=10
)

print(f"Status: {response.status_code}")
print(f"Headers: {dict(response.headers)}")
print(f"\nFirst 1000 chars of response:")
print(response.text[:1000])

# Try to parse JSON
try:
    data = response.json()
    print("\n=== Parsed JSON ===")
    
    # Look for billing info
    if 'endpointBilling' in data:
        print("\nFound endpointBilling:")
        print(json.dumps(data['endpointBilling'], indent=2))
    
    if 'publicEndpointBilling' in data:
        print("\nFound publicEndpointBilling:")
        print(json.dumps(data['publicEndpointBilling'], indent=2))
        
except:
    print("\nNot valid JSON or error parsing")
