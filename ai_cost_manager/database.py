"""
AI Cost Manager - Database

Handles database connections and session management.
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

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)


def init_db():
    """Initialize the database - create all tables"""
    Base.metadata.create_all(engine)
    print("Database initialized successfully!")


def get_session():
    """Get a database session"""
    return Session()


def close_session():
    """Close the scoped session"""
    Session.remove()
