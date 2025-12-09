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

# Fix for Heroku/some platforms that use postgres:// instead of postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    print("🔧 Fixed database URL format (postgres:// -> postgresql://)")

# Log database configuration (hide password)
if DATABASE_URL.startswith('postgresql'):
    # Extract and log connection details without password
    try:
        from urllib.parse import urlparse
        parsed = urlparse(DATABASE_URL)
        safe_url = f"{parsed.scheme}://{parsed.username}:***@{parsed.hostname}:{parsed.port}{parsed.path}"
        print(f"✅ Using PostgreSQL: {safe_url}")
    except:
        print("✅ Using PostgreSQL database")
elif DATABASE_URL.startswith('sqlite'):
    print(f"⚠️  Using SQLite: {DATABASE_URL}")
    print("   Note: SQLite data may not persist in containerized environments!")
else:
    print(f"⚠️  Using database: {DATABASE_URL[:50]}...")

# Configure engine based on database type
if DATABASE_URL.startswith('postgresql'):
    # PostgreSQL configuration with robust connection handling
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_size=5,           # Reduced pool size for better resource management
        max_overflow=10,       # Reduced overflow
        pool_pre_ping=True,    # Verify connections before using (catches stale connections)
        pool_recycle=1800,     # Recycle connections after 30 minutes (was 1 hour)
        connect_args={
            'connect_timeout': 10,      # 10 second connection timeout
            'options': '-c statement_timeout=60000'  # 60 second query timeout (increased from 30)
        }
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


def init_db(max_retries=3, retry_delay=2):
    """Initialize the database - create all tables and seed default data
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
    """
    import time
    from sqlalchemy.exc import OperationalError
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # Only create tables, never drop them
            # This is safe to run multiple times - it won't affect existing data
            Base.metadata.create_all(engine, checkfirst=True)
            print("Database initialized successfully!")
            
            # Seed default sources only if the table is empty
            _seed_default_sources()
            return  # Success!
            
        except OperationalError as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"⚠️  Database connection failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                print(f"   Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Dispose of the current engine pool to force reconnection
                engine.dispose()
            else:
                print(f"❌ Failed to initialize database after {max_retries} attempts")
                raise
        except Exception as e:
            print(f"❌ Unexpected error during database initialization: {e}")
            raise
    
    # If we got here, all retries failed
    if last_error:
        raise last_error


def _seed_default_sources():
    """Add default API sources if they don't exist"""
    from ai_cost_manager.models import APISource
    from sqlalchemy.exc import OperationalError
    import time
    
    default_sources = [
        {
            'name': 'fal.ai',
            'url': 'https://fal.ai/api/trpc/models.list',
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
        {
            'name': 'Runware',
            'url': 'https://runware.ai/pricing',
            'extractor_name': 'runware',
            'is_active': True
        },
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
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
            return  # Success!
            
        except OperationalError as e:
            session.rollback()
            if attempt < max_retries - 1:
                print(f"⚠️  Database connection issue while seeding sources (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)
                engine.dispose()
            else:
                print(f"⚠️  Could not seed default sources after {max_retries} attempts: {e}")
        except Exception as e:
            session.rollback()
            print(f"Error seeding default sources: {e}")
            break
        finally:
            session.close()


def get_session():
    """Get a database session"""
    return Session()


def close_session():
    """Close the scoped session"""
    Session.remove()
