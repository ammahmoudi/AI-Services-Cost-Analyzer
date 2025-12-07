"""
Test Runware Extractor - extracts pricing with authentication
"""

import sys
import os
import io

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from extractors.runware_extractor import RunwareExtractor
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AuthSettings

def check_auth():
    """Check if authentication is configured"""
    session = get_session()
    try:
        auth = session.query(AuthSettings).filter_by(
            source_name='runware',
            is_active=True
        ).first()
        
        if auth and auth.cookies:
            print("✅ Runware authentication found")
            return True
        else:
            print("⚠️  No Runware authentication found")
            print("   Run: python setup_runware_auth.py")
            print("   The extractor will use public pricing page as fallback\n")
            return False
    finally:
        session.close()

def test_runware():
    """Test extracting Runware pricing"""
    print("\n" + "="*60)
    print("Testing Runware Extractor")
    print("="*60 + "\n")
    
    has_auth = check_auth()
    
    extractor = RunwareExtractor()
    
    # Extract without LLM
    models = extractor.extract(progress_tracker=None)
    
    if not models:
        print("❌ No models extracted")
        return
    
    print(f"\n✅ Successfully extracted {len(models)} models\n")
    
    # Group by category
    categories = {}
    for model in models:
        cat = model['model_type']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("Distribution by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} models")
    
    # Group by provider
    providers = {}
    for model in models:
        prov = model['provider']
        providers[prov] = providers.get(prov, 0) + 1
    
    print("\nTop providers:")
    for prov, count in sorted(providers.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {prov}: {count} models")
    
    # Show sample models
    print("\nSample models:")
    
    # Image model
    image_models = [m for m in models if m['model_type'] == 'image']
    if image_models:
        img = image_models[0]
        price = img.get('output_price_per_image') or img.get('input_price_per_image') or 0
        print(f"\n  [Image] {img['model_name']}")
        print(f"    Provider: {img['provider']}")
        print(f"    Price: ${price} per image")
        print(f"    Model ID: {img['model_id']}")
        if img.get('description'):
            print(f"    Info: {img['description'][:100]}")
        if img.get('tags'):
            print(f"    Tags: {', '.join(img['tags'][:3])}")
    
    # Video model
    video_models = [m for m in models if m['model_type'] == 'video']
    if video_models:
        vid = video_models[0]
        price = vid.get('output_price_per_request') or vid.get('output_price_per_second') or 0
        unit = "per second" if vid.get('output_price_per_second') else "per video"
        print(f"\n  [Video] {vid['model_name']}")
        print(f"    Provider: {vid['provider']}")
        print(f"    Price: ${price} {unit}")
        print(f"    Model ID: {vid['model_id']}")
        if vid.get('description'):
            print(f"    Info: {vid['description'][:100]}")
        if vid.get('tags'):
            print(f"    Tags: {', '.join(vid['tags'][:3])}")
    
    # Tool model
    tool_models = [m for m in models if m['model_type'] == 'other']
    if tool_models:
        tool = tool_models[0]
        price = tool.get('output_price_per_request') or 0
        print(f"\n  [Tool] {tool['model_name']}")
        print(f"    Provider: {tool['provider']}")
        print(f"    Price: ${price} per call")
        print(f"    Model ID: {tool['model_id']}")
        if tool.get('description'):
            print(f"    Info: {tool['description'][:100]}")
        if tool.get('tags'):
            print(f"    Tags: {', '.join(tool['tags'][:3])}")
    
    print("\n" + "="*60)
    if has_auth:
        print("Note: Authenticated extraction should provide many more models")
    else:
        print("Note: Using public pricing page (81 models)")
        print("      Set up authentication for full model catalog")
    print("="*60)

if __name__ == '__main__':
    test_runware()
