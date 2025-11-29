"""
AI Cost Manager - Database

Handles database connections and session management.
Supports SQLite and PostgreSQL backends.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv
from ai_cost_manager.models import Base

# Load environment variables
load_dotenv()

# Get database URL from environment or use default SQLite
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///ai_costs.db')

# Configure engine based on database type
if DATABASE_URL.startswith('postgresql'):
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600    # Recycle connections after 1 hour
    )
elif DATABASE_URL.startswith('sqlite'):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={'check_same_thread': False}  # Allow multi-threading
    )
else:
    # Default fallback
    engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)


def init_db():
    """Initialize the database - create all tables and seed default data"""
    Base.metadata.create_all(engine)
    print("Database initialized successfully!")
    
    # Seed default sources
    _seed_default_sources()


def _seed_default_sources():
    """Add default API sources if they don't exist"""
    from ai_cost_manager.models import APISource
    
    default_sources = [
        {
            'name': 'fal.ai',
            'url': 'https://fal.ai/models',
            'extractor_name': 'fal',
            'is_active': True
        },
        {
            'name': 'Together AI',
            'url': 'https://api.together.xyz/models',
            'extractor_name': 'together',
            'is_active': True
        },
        {
            'name': 'AvalAI',
            'url': 'https://docs.avalai.ir/fa/pricing.md',
            'extractor_name': 'avalai',
            'is_active': True
        },
        {
            'name': 'MetisAI',
            'url': 'https://api.metisai.ir/api/v1/meta/providers/pricing',
            'extractor_name': 'metisai',
            'is_active': True
        },
    ]
    
    session = get_session()
    try:
        for source_data in default_sources:
            # Check if source already exists
            existing = session.query(APISource).filter_by(name=source_data['name']).first()
            if not existing:
                source = APISource(**source_data)
                session.add(source)
                print(f"Added default source: {source_data['name']}")
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error seeding default sources: {e}")
    finally:
        session.close()


def get_session():
    """Get a database session"""
    return Session()


def close_session():
    """Close the scoped session"""
    Session.remove()
