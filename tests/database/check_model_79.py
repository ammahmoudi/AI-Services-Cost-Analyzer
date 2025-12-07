"""Check model 79 data."""
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel

if __name__ == '__main__':
    session = get_session()
    try:
        model = session.query(AIModel).filter_by(id=79).first()
        if model:
            print(f"Model 79: {model.name}")
            print(f"  Description: {model.description}")
            print(f"  Category: {model.category}")
            print(f"  Tags: {model.tags}")
    finally:
        close_session()
