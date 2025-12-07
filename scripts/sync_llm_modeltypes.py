"""Normalize and sync AIModel.model_type from existing llm_extracted JSON.

This script finds models where `llm_extracted` exists and its `model_type`
differs from the stored `model_type`, normalizes the LLM value using
`ai_cost_manager.model_types.normalize_model_type`, and updates the DB row.
"""
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel
from ai_cost_manager.model_types import normalize_model_type

if __name__ == '__main__':
    session = get_session()
    try:
        query = session.query(AIModel).filter(AIModel.llm_extracted.isnot(None))
        models = query.all()
        changed = 0
        for m in models:
            try:
                llm = m.llm_extracted or {}
                llm_type = None
                if isinstance(llm, dict):
                    llm_type = llm.get('model_type')
                if not llm_type:
                    continue
                normalized = normalize_model_type(llm_type)
                if normalized != (m.model_type or 'other'):
                    print(f"Updating ID {m.id} '{m.name}': {m.model_type} -> {normalized} (raw: {llm_type})")
                    m.model_type = normalized
                    changed += 1
            except Exception as e:
                print(f"Error processing {m.id}: {e}")
                session.rollback()
                continue
        if changed > 0:
            session.commit()
            print(f"Committed {changed} updates.")
        else:
            print("No changes needed.")
    finally:
        close_session()
