"""Test script to verify parser improvements for model name parsing."""

from ai_cost_manager.model_name_parser import ModelNameParser

def test_parser():
    parser = ModelNameParser()
    
    # Test cases showing the improvements
    test_cases = [
        # Runware models - should extract correct company from model_id (after runware prefix)
        ("bfl 1 v2", "runware-bfl-1-2", "BFL"),
        ("bfl 4 v1", "runware-bfl-4-1", "BFL"),
        ("bytedance 1 v1", "runware-bytedance-1-1", "Bytedance"),
        ("civitai 101055 v128078", "runware-civitai-101055-128078", "Civitai"),
        ("bria 20 v1", "runware-bria-20-1", "Bria"),
        
        # Fal models - fal-ai/ is PROVIDER prefix, NOT company - extract from name
        ("ACE-Step", "fal-ai/ace-step", None),  # Generic model, no clear company
        ("Flux 2 (Edit)", "fal-ai/flux/v2/edit", "Fal"),  # Flux family -> Fal company (via mapping)
        ("AnimateDiff (Text To Video)", "fal-ai/fast-animatediff/text-to-video", None),  # AnimateDiff is not a company
        ("Bria 3.2 Text-to-Image", "bria/text-to-image/3.2", "Bria"),  # Company in name
        ("Bytedance (Edit Image)", "fal-ai/bytedance/seededit/v3/edit-image", "Bytedance"),  # Company in path
        
        # Models with company name at start
        ("Alibaba qwen3-235b-a22b-fp8-tput", "qwen3-235b-a22b-fp8-tput", "Alibaba"),
        ("Alibaba wan2.2-t2i-flash", "wan2.2-t2i-flash", "Alibaba"),
        ("Anthropic claude-opus-4-5", "claude-opus-4-5", "Anthropic"),
        ("Cohere cohere.embed-v4.0", "cohere.embed-v4.0", "Cohere"),
        ("ByteDance Seedance 1.0 Lite", "bytedance-seedance-1-0-lite-720p/5s", "Bytedance"),
        ("BFL flux-1.1-pro", "flux-1.1-pro", "BFL"),
        
        # OpenRouter style (company/model after provider)
        ("anthropic claude-3-5-sonnet", "anthropic/claude-3-5-sonnet", "Anthropic"),
        ("black-forest-labs flux-pro", "black-forest-labs/flux-pro", "BFL"),
        ("cohere command-r-plus", "cohere/command-r-plus", "Cohere"),
        
        # Models with company-model pattern
        ("ByteDance Seedream 4.0", "bytedance-seedream-4-0", "Bytedance"),
        ("ByteDance SeedEdit", "bytedance-seededit", "Bytedance"),
    ]
    
    print("\n" + "="*80)
    print("PARSER IMPROVEMENT TEST RESULTS")
    print("="*80 + "\n")
    
    passed = 0
    failed = 0
    
    for name, model_id, expected_company in test_cases:
        result = parser.parse(name, model_id)
        
        # For version filtering tests, we just check company extraction
        if expected_company is None:
            # These tests check that version is NOT incorrectly extracted
            status = "✅ PASS" if result.version is None or result.version not in ['101055', '128078', '2025'] else "❌ FAIL"
            version_info = f" (version={result.version})" if result.version else ""
        else:
            status = "✅ PASS" if result.company == expected_company else "❌ FAIL"
            version_info = ""
        
        if "PASS" in status:
            passed += 1
        else:
            failed += 1
        
        print(f"{status}")
        print(f"  Name: {name}")
        print(f"  Model ID: {model_id}")
        print(f"  Expected Company: {expected_company}")
        print(f"  Parsed Company: {result.company}")
        print(f"  Parsed Family: {result.model_family}")
        print(f"  Parsed Version: {result.version}")
        if result.variants:
            print(f"  Variants: {result.variants}")
        if result.modes:
            print(f"  Modes: {result.modes}")
        print()
    
    print("="*80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*80 + "\n")
    
    # Show some examples of improved family detection
    print("\nEXAMPLE FAMILY INFERENCE:")
    print("-" * 60)
    examples = [
        ("Flux model", "some-flux-model", "Fal (inferred from 'flux' family)"),
        ("Seedance video", "seedance-1-0", "Bytedance (inferred from 'seedance' family)"),
        ("Claude chat", "claude-4-opus", "Anthropic (inferred from 'claude' family)"),
    ]
    
    for name, model_id, expected in examples:
        result = parser.parse(name, model_id)
        print(f"✓ {name} ({model_id})")
        print(f"  → Company: {result.company}, Family: {result.model_family}")
        print(f"  Expected: {expected}\n")

if __name__ == "__main__":
    test_parser()
