"""Test improved extraction with progress bar and timestamps"""
from extractors.fal_extractor import FalAIExtractor
from ai_cost_manager.database import get_session, init_db
from ai_cost_manager.models import APISource, AIModel
from datetime import datetime

# Initialize database
init_db()

# Create extractor - extract only first 10 models for testing
print("Creating FalAIExtractor...\n")
extractor = FalAIExtractor(fetch_schemas=False, use_llm=False)

# Extract models
print("Starting extraction...\n")
models = extractor.extract()

# Limit to first 10 for testing
models = models[:10]

print(f"\n\n{'='*80}")
print(f"EXTRACTION SUMMARY")
print(f"{'='*80}\n")

# Show sample with full names and timestamps
for i, model in enumerate(models[:5], 1):
    print(f"{i}. {model['name']} (ID: {model['model_id']})")
    print(f"   Type: {model['model_type']}")
    print(f"   Cost: ${model['cost_per_call']:.4f}")
    
    # Show timestamps
    if model.get('last_raw_fetched'):
        print(f"   📦 Raw data fetched: {model['last_raw_fetched'].strftime('%Y-%m-%d %H:%M:%S')}")
    if model.get('last_schema_fetched'):
        print(f"   📋 Schema fetched: {model['last_schema_fetched'].strftime('%Y-%m-%d %H:%M:%S')}")
    if model.get('last_playground_fetched'):
        print(f"   🎮 Playground fetched: {model['last_playground_fetched'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()

print(f"{'='*80}\n")

# Save to database
print("Saving to database...\n")
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
        existing = session.query(AIModel).filter_by(
            source_id=source.id,
            model_id=model_data['model_id']
        ).first()
        
        if existing:
            # Update existing
            for key, value in model_data.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            updated_count += 1
        else:
            # Create new
            model = AIModel(source_id=source.id, **model_data)
            session.add(model)
            new_count += 1
    
    source.last_extracted = datetime.utcnow()
    session.commit()
    
    print(f"✅ Saved {len(models)} models to database")
    print(f"   - New: {new_count}")
    print(f"   - Updated: {updated_count}\n")
    
    # Query back to verify timestamps
    print("Verifying timestamps in database...\n")
    saved_models = session.query(AIModel).filter_by(source_id=source.id).limit(3).all()
    
    for model in saved_models:
        print(f"• {model.name}")
        print(f"  Raw: {model.last_raw_fetched}")
        print(f"  Schema: {model.last_schema_fetched}")
        print(f"  Playground: {model.last_playground_fetched}")
        print(f"  LLM: {model.last_llm_fetched}")
        print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    session.rollback()
finally:
    session.close()

print("\n✅ Test completed!")
