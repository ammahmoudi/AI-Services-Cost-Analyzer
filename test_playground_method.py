"""
Test _fetch_playground_data directly.
"""
from extractors.fal_extractor import FalAIExtractor

extractor = FalAIExtractor()

model_id = "bria/reimagine/3.2"
print(f"Testing playground fetch for: {model_id}\n")

playground_data = extractor._fetch_playground_data(model_id)

if playground_data:
    print("Success! Playground data:")
    print(f"  Endpoint: {playground_data.get('endpoint')}")
    print(f"  Unit: {playground_data.get('billing_unit')}")
    print(f"  Price: ${playground_data.get('price')}")
else:
    print("Failed to fetch playground data")
