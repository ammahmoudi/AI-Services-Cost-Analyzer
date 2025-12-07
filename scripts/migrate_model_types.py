#!/usr/bin/env python3
"""
Migration script to standardize model_type values across the database.
Updates old type names to match VALID_MODEL_TYPES.
"""

from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import AIModel, CanonicalModel
from sqlalchemy import text

# Mapping of old types to new standardized types
TYPE_MIGRATIONS = {
    'audio': 'audio-generation',
    'chat': 'text-generation',
    'embedding': 'embeddings',
    'image': 'image-generation',
    'rerank': 'reranking',
    'video': 'video-generation',
    'text': 'text-generation',
    'code': 'code-generation',
    '3d': '3d-generation',
}

def migrate_model_types():
    """Migrate old model_type values to standardized types"""
    session = get_session()
    
    try:
        print("🔄 Starting model type migration...\n")
        
        # Migrate AI Models
        print("📦 Migrating AI Models...")
        for old_type, new_type in TYPE_MIGRATIONS.items():
            result = session.execute(
                text("UPDATE ai_models SET model_type = :new_type WHERE model_type = :old_type"),
                {'new_type': new_type, 'old_type': old_type}
            )
            if result.rowcount > 0:
                print(f"  ✅ Updated {result.rowcount} models: '{old_type}' → '{new_type}'")
        
        # Migrate Canonical Models
        print("\n🔄 Migrating Canonical Models...")
        for old_type, new_type in TYPE_MIGRATIONS.items():
            result = session.execute(
                text("UPDATE canonical_models SET model_type = :new_type WHERE model_type = :old_type"),
                {'new_type': new_type, 'old_type': old_type}
            )
            if result.rowcount > 0:
                print(f"  ✅ Updated {result.rowcount} canonical models: '{old_type}' → '{new_type}'")
        
        session.commit()
        
        # Show final type distribution
        print("\n📊 Final model type distribution:")
        print("\nAI Models:")
        results = session.execute(
            text("SELECT model_type, COUNT(*) as count FROM ai_models WHERE model_type IS NOT NULL GROUP BY model_type ORDER BY model_type")
        )
        for row in results:
            print(f"  {row.model_type}: {row.count}")
        
        print("\nCanonical Models:")
        results = session.execute(
            text("SELECT model_type, COUNT(*) as count FROM canonical_models WHERE model_type IS NOT NULL GROUP BY model_type ORDER BY model_type")
        )
        for row in results:
            print(f"  {row.model_type}: {row.count}")
        
        print("\n✅ Migration completed successfully!")
        
    except Exception as e:
        session.rollback()
        print(f"\n❌ Migration failed: {e}")
        raise
    finally:
        close_session()

if __name__ == '__main__':
    migrate_model_types()
