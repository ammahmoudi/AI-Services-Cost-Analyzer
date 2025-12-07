"""Test LLM extraction for model 79 (Flux 2 Trainer)."""
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel
from ai_cost_manager.llm_extractor import LLMPricingExtractor

if __name__ == '__main__':
    session = get_session()
    try:
        model = session.query(AIModel).filter_by(id=79).first()
        if model:
            print(f"Model: {model.name} (ID {model.id})")
            print(f"Description: {model.description}")
            print(f"Category: {model.category}")
            print(f"Tags: {model.tags}")
            print(f"Current type: {model.model_type}")
            print(f"Raw metadata keys: {model.raw_metadata.keys() if model.raw_metadata else 'None'}")
            
            # Prepare model data exactly as extraction does
            model_data = {
                'name': model.name,
                'model_id': model.model_id,
                'description': model.description,
                'pricing_info': model.pricing_formula or '',
                'creditsRequired': model.credits_required or 0,
                'model_type': model.model_type,
                'category': model.category,
                'tags': model.tags or [],
                'raw_metadata': model.raw_metadata or {},
                'input_schema': model.input_schema or {},
                'output_schema': model.output_schema or {},
            }
            
            print("\n" + "="*60)
            print("Testing LLM extraction...")
            print("="*60)
            
            extractor = LLMPricingExtractor()
            result = extractor.extract_pricing(model_data)
            
            print(f"\nLLM Result:")
            print(f"  model_type: {result.get('model_type')}")
            print(f"  category: {result.get('category')}")
            print(f"  description: {result.get('description', 'N/A')[:100]}")
            
        else:
            print("Model 79 not found")
    finally:
        close_session()
