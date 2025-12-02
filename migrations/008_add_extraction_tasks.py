"""
Migration 008: Add extraction_tasks table

Adds table to track background extraction jobs with status, progress, and cancellation support.
"""
from sqlalchemy import text

def migrate(session):
    """Create extraction_tasks table"""
    
    # Create extraction_tasks table
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS extraction_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            total_models INTEGER DEFAULT 0,
            processed_models INTEGER DEFAULT 0,
            new_models INTEGER DEFAULT 0,
            updated_models INTEGER DEFAULT 0,
            current_model VARCHAR(200),
            error_message TEXT,
            use_llm BOOLEAN DEFAULT 0,
            fetch_schemas BOOLEAN DEFAULT 0,
            force_refresh BOOLEAN DEFAULT 0,
            started_at DATETIME NOT NULL,
            completed_at DATETIME,
            FOREIGN KEY (source_id) REFERENCES api_sources (id) ON DELETE CASCADE
        )
    """))
    
    session.commit()
    print("✅ Created extraction_tasks table")
