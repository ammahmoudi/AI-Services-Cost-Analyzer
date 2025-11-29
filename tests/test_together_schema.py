"""
Quick test to verify schema fetching for Together AI
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extractors.together_utils.pricing_scraper import fetch_model_page

def test_schema_fetch():
    """Test fetching schema from a single model page."""
    print("Testing schema fetch for Together AI model...\n")
    
    # Test URL - using a known model
    test_url = "https://www.together.ai/models/meta-llama-Llama-3.3-70B-Instruct-Turbo"
    
    print(f"Fetching: {test_url}\n")
    
    schema = fetch_model_page(test_url)
    
    if schema:
        print("Schema data fetched successfully!")
        print("\nSchema keys:", list(schema.keys()))
        print("\nClean schema data:")
        print(json.dumps(schema, indent=2))
    else:
        print("Failed to fetch schema")

if __name__ == "__main__":
    test_schema_fetch()

