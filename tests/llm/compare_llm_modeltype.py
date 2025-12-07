#!/usr/bin/env python3
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AIModel

session = get_session()

print('\nChecking models where llm_extracted.model_type != model_type (showing up to 50)\n')
count = 0
for m in session.query(AIModel).filter(AIModel.llm_extracted.isnot(None)).all():
    try:
        llm = m.llm_extracted
        if isinstance(llm, dict):
            llm_type = llm.get('model_type')
            if llm_type and str(llm_type) != str(m.model_type):
                print('---')
                print(f'ID: {m.id} name: {m.name}')
                print(f' DB model_type: {m.model_type}  |  LLM model_type: {llm_type}')
                print(f' Updated at: {m.updated_at}')
                count += 1
                if count >= 50:
                    break
    except Exception as e:
        print('Error for id', m.id, e)

print(f'\nTotal mismatches found: {count}\n')
session.close()
