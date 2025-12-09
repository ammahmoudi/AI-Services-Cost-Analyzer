"""
Script to parse and update all existing models with structured name components
Run this after migration 010 to populate parsed fields
"""

from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel
from ai_cost_manager.model_name_parser import parse_model_name


def update_all_models():
    """Parse and update all models with structured components"""
    session = get_session()
    
    try:
        models = session.query(AIModel).all()
        total = len(models)
        
        print(f"Parsing {total} models...")
        
        updated = 0
        for i, model in enumerate(models, 1):
            try:
                # Parse the model name
                parsed = parse_model_name(model.name, model.model_id)
                
                # Update model with parsed data
                model.parsed_company = parsed.company
                model.parsed_model_family = parsed.model_family
                model.parsed_version = parsed.version
                model.parsed_size = parsed.size
                model.parsed_variants = parsed.variants
                model.parsed_modes = parsed.modes
                model.parsed_tokens = list(parsed.tokens)
                
                updated += 1
                
                if i % 100 == 0:
                    print(f"  Processed {i}/{total}...")
                    session.commit()
                    
            except Exception as e:
                print(f"  Error parsing model {model.id} ({model.name}): {e}")
                continue
        
        session.commit()
        print(f"✓ Successfully parsed and updated {updated}/{total} models")
        
        # Show some examples
        print("\nExample parsed models:")
        examples = session.query(AIModel).filter(
            AIModel.parsed_version.isnot(None)
        ).limit(10).all()
        
        for model in examples:
            print(f"\n  {model.name}")
            print(f"    Company: {model.parsed_company}")
            print(f"    Family: {model.parsed_model_family}")
            print(f"    Version: {model.parsed_version}")
            print(f"    Size: {model.parsed_size}")
            print(f"    Variants: {model.parsed_variants}")
            print(f"    Modes: {model.parsed_modes}")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error: {e}")
        raise
    finally:
        close_session()


if __name__ == '__main__':
    update_all_models()
