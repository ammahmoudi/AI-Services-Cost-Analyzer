"""Check what model data is being sent to LLM for a training model."""
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel

if __name__ == '__main__':
    session = get_session()
    try:
        model = session.query(AIModel).filter_by(id=311).first()  # Qwen Image Trainer
        if model:
            print(f"Model: {model.name} (ID {model.id})")
            print(f"Description: {model.description}")
            print(f"Category: {model.category}")
            print(f"Tags: {model.tags}")
            print(f"Current type: {model.model_type}")
            print(f"\nThis should be detected as: TRAINING (has 'Trainer' in name, has 'image' in name/tags)")
    finally:
        close_session()
