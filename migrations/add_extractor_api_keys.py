"""
Migration: Add extractor_api_keys table

Adds a new table to store API keys for different extractors (Together AI, etc.)
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import os

Base = declarative_base()

# Define the new table
class ExtractorAPIKey(Base):
    """API keys for different extractors"""
    __tablename__ = 'extractor_api_keys'
    
    id = Column(Integer, primary_key=True)
    extractor_name = Column(String(50), unique=True, nullable=False)
    api_key = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def run_migration():
    """Run the migration to add extractor_api_keys table"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai_costs.db')
    engine = create_engine(f'sqlite:///{db_path}')
    
    print("Creating extractor_api_keys table...")
    
    try:
        # Create the table
        ExtractorAPIKey.__table__.create(engine, checkfirst=True)
        print("✓ extractor_api_keys table created successfully!")
        
    except Exception as e:
        print(f"Error creating table: {e}")
        raise


if __name__ == "__main__":
    run_migration()
    print("\nMigration completed!")
