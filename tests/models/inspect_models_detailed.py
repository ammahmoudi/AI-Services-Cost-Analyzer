#!/usr/bin/env python3
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AIModel

session = get_session()

# List of model IDs to inspect (change if needed)
ids = [311, 130, 129, 212, 754, 869]

print('\nDetailed inspection for specific model IDs:\n')
for mid in ids:
    m = session.query(AIModel).filter_by(id=mid).first()
    if not m:
        print(f'ID {mid}: not found')
        continue
    print('---')
    print(f'ID: {m.id}')
    print(f'Name: {m.name}')
    print(f'Model Type (db): {m.model_type}')
    print(f'Category (db): {m.category}')
    print(f'LLM extracted exists: {bool(m.llm_extracted)}')
    if m.llm_extracted:
        try:
            mt = m.llm_extracted.get('model_type') if isinstance(m.llm_extracted, dict) else None
            print(f"LLM extracted model_type: {mt}")
            print('LLM keys:', list(m.llm_extracted.keys()) if isinstance(m.llm_extracted, dict) else str(m.llm_extracted)[:200])
        except Exception as e:
            print('Error reading llm_extracted:', e)
    print(f'Updated at: {m.updated_at}')

session.close()
print('\nDone.')
