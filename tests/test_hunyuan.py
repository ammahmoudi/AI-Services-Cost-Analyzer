from app import app
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AIModel

session = get_session()

# Search for Hunyuan models
models = session.query(AIModel).filter(AIModel.name.ilike('%hunyuan%')).all()
print(f'Found {len(models)} Hunyuan models:')
for m in models[:10]:
    print(f'  - {m.name} (ID: {m.model_id})')

print("\n" + "="*80)
print("Testing search term tokenization:")

import re

def extract_tokens(text):
    """Extract tokens handling cases like 'flux1.1' -> ['flux', '1', '1'] and 'hunyuan3d' -> ['hunyuan', '3d']"""
    # First split on word boundaries
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    # Then split tokens that have letters followed by numbers (e.g., "flux1" -> "flux")
    expanded_tokens = set()
    for token in tokens:
        # Extract alphabetic prefix (e.g., "flux1" -> "flux", "hunyuan3d" -> "hunyuan")
        alpha_match = re.match(r'^([a-z]+)', token)
        if alpha_match:
            expanded_tokens.add(alpha_match.group(1))
        
        # Extract numeric+letter suffix (e.g., "hunyuan3d" -> "3d", "sd3" -> "3")
        # This handles cases like "3D", "2d", etc. in concatenated text
        numeric_suffix = re.search(r'(\d+[a-z]*)', token)
        if numeric_suffix:
            expanded_tokens.add(numeric_suffix.group(1))
        
        # Also keep the original token
        expanded_tokens.add(token)
    return expanded_tokens

search = "Hunyuan 3D V2.0"
search_tokens = extract_tokens(search)
search_tokens_filtered = {t for t in search_tokens if not re.match(r'^\d+\.?\d*$', t)}
search_tokens_for_overlap = {t for t in search_tokens_filtered 
                             if not re.match(r'^v\d*\.?\d*$', t) 
                             and len(t) > 1}

print(f"Search: '{search}'")
print(f"All tokens: {search_tokens}")
print(f"After number filter: {search_tokens_filtered}")
print(f"For overlap check: {search_tokens_for_overlap}")

print("\n" + "="*80)
print("Checking model tokenization:")

for m in models[:3]:
    model_tokens = extract_tokens(f"{m.name} {m.model_id}")
    print(f"\nModel: {m.name}")
    print(f"  Model tokens: {model_tokens}")
    print(f"  Matching tokens: {search_tokens_for_overlap & model_tokens}")
