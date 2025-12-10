from ai_cost_manager.model_name_parser import ModelNameParser

parser = ModelNameParser()

# Test Wan models with different modes
test_cases = [
    ("Wan v2.2 5B (Text To Video)", "fal-ai/wan/v2.2-5b/text-to-video"),
    ("Wan v2.2 5B (Image To Video)", "fal-ai/wan/v2.2-5b/image-to-video"),
    ("Wan (Image To Image)", "fal-ai/wan/v2.2-a14b/image-to-image"),
    ("Wan (Video To Video)", "fal-ai/wan/v2.2-a14b/video-to-video"),
    ("Wan-2.1 Pro Image-to-Video", "fal-ai/wan-pro/image-to-video"),
    ("Wan (Text To Image)", "fal-ai/wan/v2.2-a14b/text-to-image"),
    ("Wan (Turbo)", "fal-ai/wan/v2.2-a14b/text-to-video/turbo"),
    ("Wan (Turbo)", "fal-ai/wan/v2.2-a14b/image-to-video/turbo"),
]

print("WAN MODE EXTRACTION TEST")
print("=" * 80)

for name, model_id in test_cases:
    result = parser.parse(name, model_id)
    print(f"\n{name}")
    print(f"  Model ID: {model_id}")
    print(f"  Company: {result.company}")
    print(f"  Family: {result.model_family}")
    print(f"  Version: {result.version}")
    print(f"  Variants: {result.variants}")
    print(f"  Modes: {result.modes}")
    
    # Check if mode is extracted correctly from path
    if 'text-to-video' in model_id and 'Text-To-Video' in result.modes:
        print("  ✅ Text-To-Video mode extracted from path")
    elif 'image-to-video' in model_id and 'Image-To-Video' in result.modes:
        print("  ✅ Image-To-Video mode extracted from path")
    elif 'video-to-video' in model_id and 'Video-To-Video' in result.modes:
        print("  ✅ Video-To-Video mode extracted from path")
    elif 'image-to-image' in model_id and 'Image-To-Image' in result.modes:
        print("  ✅ Image-To-Image mode extracted from path")
    elif 'text-to-image' in model_id and 'Text-To-Image' in result.modes:
        print("  ✅ Text-To-Image mode extracted from path")

print("\n" + "=" * 80)
