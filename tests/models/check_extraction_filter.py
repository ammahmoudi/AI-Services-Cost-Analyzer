"""Check which models the LLM extraction targeted."""
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel

if __name__ == '__main__':
    session = get_session()
    try:
        # Rebuild the filter from the extraction route
        # POST /api/run-llm-extraction with likely default params (no filter)
        query = session.query(AIModel).filter(AIModel.is_active == True)
        
        models = query.all()
        print(f"Total models matching filter: {len(models)}\n")
        
        # Check if our test models are in this list
        test_ids = [19, 311, 212, 87, 354]
        test_models = session.query(AIModel).filter(AIModel.id.in_(test_ids)).all()
        
        print("Test models in filtered query:")
        for m in test_models:
            print(f"  ID {m.id}: {m.name} → model_type={m.model_type}")
    finally:
        close_session()
