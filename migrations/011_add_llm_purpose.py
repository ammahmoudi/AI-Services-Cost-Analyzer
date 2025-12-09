"""
Migration 011: Add purpose column to llm_configurations
"""
from sqlalchemy import text

def run_migration():
    """Add purpose column to distinguish extraction vs search LLMs"""
    from ai_cost_manager.database import get_session
    
    session = get_session()
    
    try:
        # Add purpose column with a simple approach
        session.execute(text("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name='llm_configurations' AND column_name='purpose'
                ) THEN
                    ALTER TABLE llm_configurations 
                    ADD COLUMN purpose VARCHAR(50) DEFAULT 'extraction';
                END IF;
            END $$;
        """))
        
        session.commit()
        print("✓ Migration 011 completed: Added purpose column to llm_configurations")
        return True
        
    except Exception as e:
        session.rollback()
        print(f"✗ Migration 011 failed: {e}")
        raise
    finally:
        session.close()

if __name__ == '__main__':
    run_migration()
