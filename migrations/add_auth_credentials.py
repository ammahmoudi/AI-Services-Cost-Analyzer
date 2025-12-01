"""
Migration: Add username/password columns to auth_settings

Adds username and password columns for credential-based authentication (e.g., Runware).
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from ai_cost_manager.database import DATABASE_URL

def upgrade():
    """Add username and password columns"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Add username column
        try:
            conn.execute(text("""
                ALTER TABLE auth_settings 
                ADD COLUMN username VARCHAR(255)
            """))
            conn.commit()
            print("✓ Added username column")
        except Exception as e:
            print(f"  username column already exists or error: {e}")
        
        # Add password column
        try:
            conn.execute(text("""
                ALTER TABLE auth_settings 
                ADD COLUMN password TEXT
            """))
            conn.commit()
            print("✓ Added password column")
        except Exception as e:
            print(f"  password column already exists or error: {e}")
    
    print("\n✅ Migration completed!")
    print("\nNow you can configure Runware credentials in Settings → Authentication")

if __name__ == "__main__":
    print("Running migration: Add username/password to auth_settings\n")
    upgrade()
