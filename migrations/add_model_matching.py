"""
Migration: Add model matching tables

Adds canonical_models and model_matches tables for cross-provider model comparison.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from ai_cost_manager.database import DATABASE_URL

def upgrade():
    """Add model matching tables"""
    engine = create_engine(DATABASE_URL)
    
    # Detect database type
    is_postgres = DATABASE_URL.startswith('postgresql')
    
    # Use appropriate SQL syntax for database type
    if is_postgres:
        pk_type = "SERIAL PRIMARY KEY"
        timestamp_default = "CURRENT_TIMESTAMP"
    else:
        pk_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
        timestamp_default = "CURRENT_TIMESTAMP"
    
    with engine.connect() as conn:
        # Create canonical_models table
        try:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS canonical_models (
                    id {pk_type},
                    canonical_name VARCHAR(255) NOT NULL UNIQUE,
                    display_name VARCHAR(255),
                    description TEXT,
                    model_type VARCHAR(50),
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT {timestamp_default},
                    updated_at TIMESTAMP DEFAULT {timestamp_default}
                )
            """))
            conn.commit()
            print("✓ Created canonical_models table")
        except Exception as e:
            print(f"  canonical_models table error: {e}")
        
        # Create model_matches table
        try:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS model_matches (
                    id {pk_type},
                    canonical_model_id INTEGER NOT NULL,
                    ai_model_id INTEGER NOT NULL,
                    confidence FLOAT DEFAULT 0.0,
                    matched_by VARCHAR(50) DEFAULT 'llm',
                    matched_at TIMESTAMP DEFAULT {timestamp_default},
                    FOREIGN KEY (canonical_model_id) REFERENCES canonical_models(id) ON DELETE CASCADE,
                    FOREIGN KEY (ai_model_id) REFERENCES ai_models(id) ON DELETE CASCADE,
                    UNIQUE(canonical_model_id, ai_model_id)
                )
            """))
            conn.commit()
            print("✓ Created model_matches table")
        except Exception as e:
            print(f"  model_matches table error: {e}")
        
        # Create indexes for better performance
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_matches_canonical 
                ON model_matches(canonical_model_id)
            """))
            conn.commit()
            print("✓ Created index on model_matches.canonical_model_id")
        except Exception as e:
            print(f"  Index error: {e}")
        
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_matches_ai_model 
                ON model_matches(ai_model_id)
            """))
            conn.commit()
            print("✓ Created index on model_matches.ai_model_id")
        except Exception as e:
            print(f"  Index error: {e}")
        
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_canonical_models_type 
                ON canonical_models(model_type)
            """))
            conn.commit()
            print("✓ Created index on canonical_models.model_type")
        except Exception as e:
            print(f"  Index error: {e}")
    
    print("\n✅ Migration completed!")
    print("\nNext steps:")
    print("  1. Run: python run_matching.py")
    print("  2. Visit: http://localhost:5000/canonical-models")

def downgrade():
    """Remove model matching tables"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        try:
            conn.execute(text("DROP TABLE IF EXISTS model_matches"))
            conn.execute(text("DROP TABLE IF EXISTS canonical_models"))
            conn.commit()
            print("✓ Removed model matching tables")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n✅ Downgrade completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model matching migration')
    parser.add_argument('--down', action='store_true', help='Run downgrade (remove tables)')
    args = parser.parse_args()
    
    if args.down:
        print("Running migration downgrade: Remove model matching tables\n")
        downgrade()
    else:
        print("Running migration: Add model matching tables\n")
        upgrade()
