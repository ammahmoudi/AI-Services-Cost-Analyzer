#!/usr/bin/env python3
"""
Database Connection Checker

Run this script to verify which database you're connected to
and whether the connection is working.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*60)
print("🔍 Database Connection Checker")
print("="*60)

# Get DATABASE_URL
DATABASE_URL = os.getenv('DATABASE_URL')

print(f"\n📝 Environment:")
print(f"   DATABASE_URL from env: {DATABASE_URL if DATABASE_URL else 'NOT SET'}")

if not DATABASE_URL:
    print("\n⚠️  WARNING: DATABASE_URL is not set!")
    print("   Will use default SQLite: sqlite:///ai_costs.db")
    DATABASE_URL = 'sqlite:///ai_costs.db'

# Parse and display connection info
if DATABASE_URL.startswith('postgresql'):
    from urllib.parse import urlparse
    try:
        parsed = urlparse(DATABASE_URL)
        print(f"\n✅ PostgreSQL Configuration:")
        print(f"   Host: {parsed.hostname}")
        print(f"   Port: {parsed.port or 5432}")
        print(f"   Database: {parsed.path.lstrip('/')}")
        print(f"   Username: {parsed.username}")
    except Exception as e:
        print(f"\n⚠️  Error parsing DATABASE_URL: {e}")
elif DATABASE_URL.startswith('sqlite'):
    print(f"\n⚠️  SQLite Configuration:")
    print(f"   File: {DATABASE_URL.replace('sqlite:///', '')}")
    print(f"   Note: SQLite data will NOT persist in containers!")
else:
    print(f"\n❓ Unknown database type: {DATABASE_URL[:50]}...")

# Try to connect
print(f"\n🔌 Testing Connection...")
try:
    from sqlalchemy import create_engine, text
    
    if DATABASE_URL.startswith('postgresql'):
        engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
    else:
        engine = create_engine(DATABASE_URL, echo=False)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Connection successful!")
        
        # Check if tables exist
        if DATABASE_URL.startswith('postgresql'):
            result = conn.execute(text("""
                SELECT count(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('api_sources', 'ai_models', 'llm_configurations')
            """))
            table_count = result.scalar()
            print(f"   Found {table_count} application tables")
        else:
            result = conn.execute(text("""
                SELECT count(*) 
                FROM sqlite_master 
                WHERE type='table' 
                AND name IN ('api_sources', 'ai_models', 'llm_configurations')
            """))
            table_count = result.scalar()
            print(f"   Found {table_count} application tables")
        
        if table_count == 0:
            print("   ⚠️  No tables found - run migrations!")
        elif table_count < 3:
            print("   ⚠️  Some tables missing - run migrations!")
        else:
            print("   ✅ All core tables present")
            
            # Count models
            result = conn.execute(text("SELECT count(*) FROM ai_models"))
            model_count = result.scalar()
            print(f"   📊 {model_count} models in database")

except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ Database check complete!")
print("="*60)
