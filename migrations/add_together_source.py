"""
Migration: Add Together.ai API Source

Adds Together.ai as an available API source in the database.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ai_cost_manager.models import APISource
from datetime import datetime
import os


def run_migration():
    """Add Together.ai source to the database"""
    # Get database path
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai_costs.db')
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        print("Please run the application first to create the database.")
        return
    
    engine = create_engine(f'sqlite:///{db_path}')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if Together.ai source already exists
        existing = session.query(APISource).filter_by(name="Together AI").first()
        
        if existing:
            print("✓ Together AI source already exists")
            print(f"  ID: {existing.id}")
            print(f"  URL: {existing.url}")
            print(f"  Extractor: {existing.extractor_name}")
            print(f"  Active: {existing.is_active}")
            return
        
        # Create Together.ai source
        together_source = APISource(
            name="Together AI",
            url="https://api.together.xyz/v1/models",
            extractor_name="together",
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(together_source)
        session.commit()
        
        print("✅ Together AI source added successfully!")
        print(f"  ID: {together_source.id}")
        print(f"  Name: {together_source.name}")
        print(f"  URL: {together_source.url}")
        print(f"  Extractor: {together_source.extractor_name}")
        print("\n💡 You can now extract models from Together AI in the web UI!")
        print("   Note: Set up your Together AI API key in Settings for full access.")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error adding Together AI source: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    run_migration()

