"""
Migration: Add Together.ai API Source

Adds Together.ai as an available API source in the database.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from ai_cost_manager.database import DATABASE_URL


def upgrade():
    """Add Together.ai source to the database"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        try:
            # Check if Together.ai source already exists
            result = conn.execute(text(
                "SELECT COUNT(*) FROM api_sources WHERE name = 'Together AI'"
            ))
            count = result.scalar()
            
            if count > 0:
                print("✓ Together AI source already exists")
                return
            
            # Insert Together.ai source
            conn.execute(text("""
                INSERT INTO api_sources (name, url, extractor_name, is_active, created_at, updated_at)
                VALUES ('Together AI', 'https://api.together.xyz/v1/models', 'together', true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """))
            conn.commit()
            
            print("✓ Added Together AI source")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n✅ Migration completed!")


if __name__ == "__main__":
    print("Running migration: Add Together AI source\n")
    upgrade()

