"""Verify that DB actually has the updated model types after extraction."""
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel

if __name__ == '__main__':
    session = get_session()
    try:
        # Check a few specific models that should have been updated
        test_models = [
            'Train Flux Krea LoRA',
            'Qwen Image Trainer',
            'Hyper3d (V2)',
            'Sam 3 (Embed)',
            'NSFW Checker (Nsfw)',
        ]
        
        print("Checking DB for updated model types after LLM extraction:\n")
        for name_part in test_models:
            models = session.query(AIModel).filter(AIModel.name.ilike(f'%{name_part}%')).all()
            for m in models:
                print(f"ID {m.id}: {m.name}")
                print(f"  DB model_type: {m.model_type}")
                print(f"  Expected: training|detection|3d-generation (NOT other)")
                if m.llm_extracted and isinstance(m.llm_extracted, dict):
                    print(f"  LLM extracted: {m.llm_extracted.get('model_type', 'N/A')}")
                print()
    finally:
        close_session()
