import re

def extract_tokens(text):
    """Extract tokens handling cases like 'flux1.1' -> ['flux', '1', '1']"""
    # First split on word boundaries
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    # Then split tokens that have letters followed by numbers (e.g., "flux1" -> "flux")
    expanded_tokens = set()
    for token in tokens:
        # Extract alphabetic prefix (e.g., "flux1" -> "flux")
        alpha_match = re.match(r'^([a-z]+)', token)
        if alpha_match:
            expanded_tokens.add(alpha_match.group(1))
        # Also keep the original token
        expanded_tokens.add(token)
    return expanded_tokens

# Test cases
search = 'flux 1.1'
models = [
    'FLUX1.1 [pro]',
    'BFL flux-1.1-pro',
    'FLUX1.1 [pro] ultra',
    'black-forest-labs/FLUX.1.1-pro'
]

print("TOKEN EXTRACTION TEST")
print("=" * 60)

search_tokens = extract_tokens(search)
search_tokens = {t for t in search_tokens if not re.match(r'^\d+\.?\d*$', t)}
print(f"\nSearch query: '{search}'")
print(f"Search tokens: {search_tokens}")

for model in models:
    model_tokens = extract_tokens(model)
    matching = search_tokens & model_tokens
    print(f"\nModel: {model}")
    print(f"  Model tokens: {model_tokens}")
    print(f"  Matching: {matching}")
    print(f"  ✅ MATCH" if matching else "  ❌ NO MATCH")
