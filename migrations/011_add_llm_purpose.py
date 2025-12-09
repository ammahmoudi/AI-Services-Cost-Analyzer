"""
Migration 011: Add purpose column to llm_configurations
"""
from ai_cost_manager.database import get_session
from sqlalchemy import text

def run_migration():
    """Add purpose column to distinguish extraction vs search LLMs"""
    session = get_session()
    
    try:
        # Add purpose column
        session.execute(text("""
            ALTER TABLE llm_configurations 
            ADD COLUMN IF NOT EXISTS purpose VARCHAR(50) DEFAULT 'extraction'
        """))
        
        session.commit()
        print("✓ Migration 011 completed: Added purpose column to llm_configurations")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Migration 011 failed: {e}")
        raise
    finally:
        session.close()

if __name__ == '__main__':
    run_migration()
