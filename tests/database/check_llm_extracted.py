"""Check what's in llm_extracted for these models."""
import json
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel

if __name__ == '__main__':
    session = get_session()
    try:
        test_ids = [19, 311, 212, 87, 354]
        models = session.query(AIModel).filter(AIModel.id.in_(test_ids)).all()
        
        for m in models:
            print(f"\nID {m.id}: {m.name}")
            print(f"  DB model_type: {m.model_type}")
            if m.llm_extracted:
                if isinstance(m.llm_extracted, dict):
                    llm_type = m.llm_extracted.get('model_type')
                    print(f"  LLM model_type: {llm_type}")
                elif isinstance(m.llm_extracted, str):
                    try:
                        data = json.loads(m.llm_extracted)
                        print(f"  LLM (parsed): {data.get('model_type')}")
                    except:
                        print(f"  LLM (raw string): {m.llm_extracted[:100]}")
                else:
                    print(f"  LLM (other type): {type(m.llm_extracted)}")
            else:
                print(f"  LLM extracted: None")
    finally:
        close_session()
