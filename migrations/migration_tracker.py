"""
Migration tracking system to prevent duplicate migrations
"""
from sqlalchemy import create_engine, text, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///ai_costs.db')

Base = declarative_base()

class MigrationHistory(Base):
    __tablename__ = 'migration_history'
    
    migration_name = Column(String(255), primary_key=True)
    applied_at = Column(DateTime, default=datetime.utcnow)

def ensure_migration_table():
    """Create migration tracking table if it doesn't exist"""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine, checkfirst=True)
    return engine

def is_migration_applied(migration_name: str) -> bool:
    """Check if a migration has already been applied"""
    engine = ensure_migration_table()
    
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM migration_history WHERE migration_name = :name"),
            {"name": migration_name}
        )
        count = result.scalar()
        return count > 0

def mark_migration_applied(migration_name: str):
    """Mark a migration as applied"""
    engine = ensure_migration_table()
    
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO migration_history (migration_name, applied_at) VALUES (:name, :applied_at)"),
            {"name": migration_name, "applied_at": datetime.utcnow()}
        )
        conn.commit()

def get_migration_history():
    """Get all applied migrations"""
    engine = ensure_migration_table()
    
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT migration_name, applied_at FROM migration_history ORDER BY applied_at")
        )
        return [{"name": row[0], "applied_at": row[1]} for row in result]
