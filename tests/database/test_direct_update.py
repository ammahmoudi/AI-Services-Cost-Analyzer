"""Direct test: update a model in DB and verify it persists."""
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel

if __name__ == '__main__':
    session = get_session()
    try:
        model = session.query(AIModel).filter_by(id=19).first()  # Train Flux Krea LoRA
        if model:
            print(f"Before: {model.name} → model_type={model.model_type}")
            model.model_type = "training"
            print(f"Set to: model_type={model.model_type}")
            session.flush()
            session.commit()
            print(f"After commit: model_type={model.model_type}")
            
            # Re-query to verify
            session.expire_all()
            model2 = session.query(AIModel).filter_by(id=19).first()
            print(f"After re-query: {model2.name} → model_type={model2.model_type}")
        else:
            print("Model not found")
    finally:
        close_session()
