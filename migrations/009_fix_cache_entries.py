"""
Migration 009: Redesign cache_entries to be independent of AIModel

Changes:
- Drop foreign key constraint and model_id column
- Add cache_key (unique), source_name, model_identifier columns
- Add indexes for efficient lookups
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from ai_cost_manager.database import DATABASE_URL

def upgrade():
    """Redesign cache_entries to be independent"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Drop the old table (since we're fundamentally changing structure)
        conn.execute(text("DROP TABLE IF EXISTS cache_entries"))
        conn.commit()
        print("✓ Dropped old cache_entries table")
        
        # Recreate with new structure for PostgreSQL
        conn.execute(text("""
            CREATE TABLE cache_entries (
                id SERIAL PRIMARY KEY,
                cache_key VARCHAR(500) NOT NULL UNIQUE,
                source_name VARCHAR(200) NOT NULL,
                model_identifier VARCHAR(200) NOT NULL,
                cache_type VARCHAR(50) NOT NULL,
                data JSONB NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
        print("✓ Created new cache_entries table")
        
        # Add indexes for efficient lookups
        conn.execute(text("CREATE INDEX idx_cache_key ON cache_entries(cache_key)"))
        conn.execute(text("CREATE INDEX idx_source_name ON cache_entries(source_name)"))
        conn.execute(text("CREATE INDEX idx_model_identifier ON cache_entries(model_identifier)"))
        conn.execute(text("CREATE INDEX idx_cache_type ON cache_entries(cache_type)"))
        conn.commit()
        print("✓ Created indexes")
    
    print("\n✅ Cache entries redesigned - now independent of AIModel!")
    print("Cache can now store data before models exist in database")

if __name__ == "__main__":
    print("Running migration: Redesign cache_entries\n")
    upgrade()
