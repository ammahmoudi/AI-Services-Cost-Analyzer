#!/usr/bin/env python3
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AIModel
from sqlalchemy import or_

session = get_session()

names = [
    'Qwen Image Trainer',
    'Qwen Image Edit Trainer',
    'Qwen Image Edit Plus Trainer',
    'Hyper3d (V2)',
    'Hyper3d',
    'NSFW Filter',
    'NSFW Checker',
    'SAM',
]

print('\nInspecting sample models in DB:\n')
for name in names:
    models = session.query(AIModel).filter(AIModel.name.ilike(f'%{name}%')).all()
    if not models:
        print(f"No rows found matching '{name}'")
        continue
    for m in models:
        print('---')
        print(f"ID: {m.id}")
        print(f"Name: {m.name}")
        print(f"Model ID: {m.model_id}")
        print(f"Model Type: {m.model_type}")
        print(f"Category: {m.category}")
        print(f"Tags: {m.tags}")
        print(f"Cost per call: {m.cost_per_call}")
        print(f"LLM extracted (summary): {bool(m.llm_extracted)}")
        try:
            import json
            if m.llm_extracted:
                # try to pretty print keys
                if isinstance(m.llm_extracted, dict):
                    print('LLM keys:', list(m.llm_extracted.keys()))
                else:
                    print('LLM raw:', str(m.llm_extracted)[:200])
        except Exception as e:
            print('Error reading llm_extracted:', e)

session.close()
print('\nDone.')
