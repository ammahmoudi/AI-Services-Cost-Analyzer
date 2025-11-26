"""Test saving extracted models to database"""
from extractors.fal_extractor import FalAIExtractor
from ai_cost_manager.database import get_session, init_db
from ai_cost_manager.models import APISource, AIModel
from datetime import datetime

# Initialize database
init_db()

print("Extracting 10 models from fal.ai...\n")

# Create extractor
extractor = FalAIExtractor(fetch_schemas=False, use_llm=False)

# Extract models
all_models = extractor.extract()
models = all_models[:10]  # Only first 10 for testing

print(f"\n{'='*80}")
print("SAVING TO DATABASE")
print(f"{'='*80}\n")

session = get_session()

try:
    # Get or create fal.ai source
    source = session.query(APISource).filter_by(name="fal.ai").first()
    if not source:
        source = APISource(
            name="fal.ai",
            url="https://fal.ai/api/trpc/models.list",
            extractor_name="fal",
            is_active=True
        )
        session.add(source)
        session.commit()
        print("✓ Created fal.ai source\n")
    
    # Save models
    new_count = 0
    updated_count = 0
    
    for model_data in models:
        # Filter to only include columns that exist in AIModel
        filtered_data = {
            k: v for k, v in model_data.items()
            if hasattr(AIModel, k)
        }
        
        print(f"Processing: {model_data['name']}")
        print(f"  Columns: {', '.join(filtered_data.keys())}")
        
        existing = session.query(AIModel).filter_by(
            source_id=source.id,
            model_id=model_data['model_id']
        ).first()
        
        if existing:
            # Update existing
            for key, value in filtered_data.items():
                if key != 'id':
                    setattr(existing, key, value)
            updated_count += 1
            print(f"  Status: Updated")
        else:
            # Create new
            model = AIModel(source_id=source.id, **filtered_data)
            session.add(model)
            new_count += 1
            print(f"  Status: Created")
        print()
    
    source.last_extracted = datetime.utcnow()
    session.commit()
    
    print(f"\n✅ Saved {len(models)} models")
    print(f"   New: {new_count}, Updated: {updated_count}\n")
    
    # Verify by querying back
    print(f"{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}\n")
    
    saved = session.query(AIModel).filter_by(source_id=source.id).limit(5).all()
    
    for model in saved:
        print(f"• {model.name} ({model.model_id})")
        print(f"  Cost: ${model.cost_per_call:.4f}")
        print(f"  Raw fetched: {model.last_raw_fetched}")
        print(f"  Schema fetched: {model.last_schema_fetched}")
        print(f"  Playground fetched: {model.last_playground_fetched}")
        print(f"  LLM fetched: {model.last_llm_fetched}")
        print()
    
    # Statistics
    total = session.query(AIModel).filter_by(source_id=source.id).count()
    print(f"Total models in database: {total}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    session.rollback()
finally:
    session.close()

print("\n✅ Test completed!")
