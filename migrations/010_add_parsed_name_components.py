"""
Migration: Add parsed model name components
Adds columns for structured model name parsing
"""

from ai_cost_manager.database import get_session, close_session
from sqlalchemy import text


def upgrade():
    """Add parsed name component columns to ai_models table"""
    session = get_session()
    
    try:
        print("Adding parsed name component columns...")
        
        # Add columns for parsed components
        alterations = [
            "ALTER TABLE ai_models ADD COLUMN IF NOT EXISTS parsed_company VARCHAR(100)",
            "ALTER TABLE ai_models ADD COLUMN IF NOT EXISTS parsed_model_family VARCHAR(100)",
            "ALTER TABLE ai_models ADD COLUMN IF NOT EXISTS parsed_version VARCHAR(50)",
            "ALTER TABLE ai_models ADD COLUMN IF NOT EXISTS parsed_size VARCHAR(50)",
            "ALTER TABLE ai_models ADD COLUMN IF NOT EXISTS parsed_variants JSON",
            "ALTER TABLE ai_models ADD COLUMN IF NOT EXISTS parsed_modes JSON",
            "ALTER TABLE ai_models ADD COLUMN IF NOT EXISTS parsed_tokens JSON"
        ]
        
        for sql in alterations:
            print(f"  Executing: {sql}")
            session.execute(text(sql))
        
        session.commit()
        print("✓ Successfully added parsed name component columns")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error during migration: {e}")
        raise
    finally:
        close_session()


def downgrade():
    """Remove parsed name component columns"""
    session = get_session()
    
    try:
        print("Removing parsed name component columns...")
        
        alterations = [
            "ALTER TABLE ai_models DROP COLUMN IF EXISTS parsed_company",
            "ALTER TABLE ai_models DROP COLUMN IF EXISTS parsed_model_family",
            "ALTER TABLE ai_models DROP COLUMN IF EXISTS parsed_version",
            "ALTER TABLE ai_models DROP COLUMN IF EXISTS parsed_size",
            "ALTER TABLE ai_models DROP COLUMN IF EXISTS parsed_variants",
            "ALTER TABLE ai_models DROP COLUMN IF EXISTS parsed_modes",
            "ALTER TABLE ai_models DROP COLUMN IF EXISTS parsed_tokens"
        ]
        
        for sql in alterations:
            print(f"  Executing: {sql}")
            session.execute(text(sql))
        
        session.commit()
        print("✓ Successfully removed parsed name component columns")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error during downgrade: {e}")
        raise
    finally:
        close_session()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'down':
        downgrade()
    else:
        upgrade()
