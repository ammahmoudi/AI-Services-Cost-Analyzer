"""
Migration: Add timestamp columns for tracking data fetching

Adds last_raw_fetched, last_schema_fetched, last_playground_fetched, last_llm_fetched
columns to the ai_models table.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from ai_cost_manager.database import DATABASE_URL

def upgrade():
    """Add timestamp columns"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Add columns if they don't exist
        try:
            conn.execute(text("""
                ALTER TABLE ai_models 
                ADD COLUMN last_raw_fetched DATETIME
            """))
            conn.commit()
            print("✓ Added last_raw_fetched column")
        except Exception as e:
            print(f"  last_raw_fetched already exists or error: {e}")
        
        try:
            conn.execute(text("""
                ALTER TABLE ai_models 
                ADD COLUMN last_schema_fetched DATETIME
            """))
            conn.commit()
            print("✓ Added last_schema_fetched column")
        except Exception as e:
            print(f"  last_schema_fetched already exists or error: {e}")
        
        try:
            conn.execute(text("""
                ALTER TABLE ai_models 
                ADD COLUMN last_playground_fetched DATETIME
            """))
            conn.commit()
            print("✓ Added last_playground_fetched column")
        except Exception as e:
            print(f"  last_playground_fetched already exists or error: {e}")
        
        try:
            conn.execute(text("""
                ALTER TABLE ai_models 
                ADD COLUMN last_llm_fetched DATETIME
            """))
            conn.commit()
            print("✓ Added last_llm_fetched column")
        except Exception as e:
            print(f"  last_llm_fetched already exists or error: {e}")
    
    print("\n✅ Migration completed!")

if __name__ == "__main__":
    print("Running migration: Add timestamp columns\n")
    upgrade()
