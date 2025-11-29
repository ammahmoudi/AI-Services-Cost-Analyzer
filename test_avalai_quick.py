from extractors.avalai_extractor import AvalAIExtractor

e = AvalAIExtractor()
m = e.extract()
print(f'Total models: {len(m)}')

# Check for bad models
bad = [x for x in m if 'هزینه' in x.get('model_id', '') or 'unit' in x.get('model_id', '').lower() or 'usd' in x.get('model_id', '').lower()]
print(f'Bad models found: {len(bad)}')
if bad:
    print('Sample bad models:')
    for x in bad[:5]:
        print(f"  - {x.get('model_id')}")
else:
    print('✅ No bad models found!')
