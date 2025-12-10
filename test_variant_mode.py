from ai_cost_manager.model_name_parser import ModelNameParser

parser = ModelNameParser()

# Test models with turbo, lite, fast, pro
test_cases = [
    ("Vidu (Turbo)", "fal-ai/vidu/q2/image-to-video/turbo"),
    ("Virtueguard Text Lite", "Virtue-AI/VirtueGuard-Text-Lite"),
    ("Wan (Fast Wan)", "fal-ai/wan/v2.2-5b/text-to-video/fast-wan"),
    ("Wan-2.1 Pro Image-to-Video", "fal-ai/wan-pro/image-to-video"),
    ("XAI grok-3-fast", "grok-3-fast"),
]

print("VARIANT vs MODE CLASSIFICATION TEST")
print("=" * 80)

for name, model_id in test_cases:
    result = parser.parse(name, model_id)
    print(f"\n{name}")
    print(f"  Model ID: {model_id}")
    print(f"  Variants: {result.variants}")
    print(f"  Modes: {result.modes}")
    
    # Check for duplicates
    if result.variants and result.modes:
        duplicates = set(result.variants) & set(result.modes)
        if duplicates:
            print(f"  ❌ FAIL: {duplicates} appears in BOTH variants and modes")
        else:
            print(f"  ✅ PASS: No duplicates between variants and modes")
    else:
        print(f"  ✅ PASS: No overlap possible")

print("\n" + "=" * 80)
print("SUMMARY: Check that Turbo/Lite/Fast/Pro appear ONLY in Variants, NOT in Modes")
