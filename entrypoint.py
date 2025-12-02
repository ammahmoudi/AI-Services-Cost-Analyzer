#!/usr/bin/env python3
"""
Entrypoint script for Docker container

Runs migrations before starting the application.
"""
import sys
import subprocess
from pathlib import Path

def run_migrations():
    """Run all database migrations"""
    print("🔧 Running database migrations...")
    
    # Add migrations directory to Python path
    migrations_dir = Path('/app/migrations')
    sys.path.insert(0, str(migrations_dir.parent))
    
    if not migrations_dir.exists():
        print("⚠️  No migrations directory found, skipping...")
        return True
    
    # Import migration tracker
    try:
        from migrations.migration_tracker import is_migration_applied, mark_migration_applied
    except Exception as e:
        print(f"⚠️  Could not import migration tracker: {e}")
        print("  Running migrations without tracking...")
        is_migration_applied = lambda x: False
        mark_migration_applied = lambda x: None
    
    # Get all migration files
    migration_files = sorted([
        f for f in migrations_dir.glob('*.py')
        if not f.name.startswith('__') 
        and not f.name.startswith('.')
        and f.name != 'migration_tracker.py'  # Skip the tracker itself
    ])
    
    if not migration_files:
        print("ℹ️  No migrations to run")
        return True
    
    print(f"Found {len(migration_files)} migration file(s)")
    
    skipped = 0
    applied = 0
    
    for migration_file in migration_files:
        migration_name = migration_file.stem  # filename without .py
        
        # Check if already applied
        if is_migration_applied(migration_name):
            print(f"  {migration_file.name}... ⏭️  (already applied)")
            skipped += 1
            continue
        
        try:
            print(f"  {migration_file.name}...", end=" ")
            result = subprocess.run(
                [sys.executable, str(migration_file)],
                cwd='/app',
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅")
                # Mark as applied
                try:
                    mark_migration_applied(migration_name)
                    applied += 1
                except Exception as e:
                    print(f"    ⚠️  Could not mark migration as applied: {e}")
            else:
                print("⚠️")
                if result.stderr:
                    # Only print if it's not "already exists" error
                    if "already exists" not in result.stderr.lower():
                        print(f"    {result.stderr[:200]}")
                    else:
                        # Still mark as applied if it's just "already exists"
                        try:
                            mark_migration_applied(migration_name)
                        except:
                            pass
        
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            # Don't fail on migration errors - they might be idempotent
    
    print(f"✅ Migrations complete (applied: {applied}, skipped: {skipped})")
    return True

def start_application():
    """Start the Flask application"""
    print("🚀 Starting application...")
    
    # Run with gunicorn for production
    # Increased timeout to 300s to accommodate long extraction jobs with Playwright
    subprocess.run([
        'gunicorn',
        '--bind', '0.0.0.0:5000',
        '--workers', '4',
        '--timeout', '300',
        '--graceful-timeout', '60',
        '--access-logfile', '-',
        '--error-logfile', '-',
        'app:app'
    ])

if __name__ == '__main__':
    print("="*60)
    print("🐳 AI Cost Manager - Container Startup")
    print("="*60)
    
    # Show database configuration
    import os
    db_url = os.getenv('DATABASE_URL', 'Not set')
    if db_url and db_url != 'Not set':
        # Hide password in logs
        if '@' in db_url:
            user_host = db_url.split('@')[1] if '@' in db_url else 'unknown'
            print(f"📊 Database: {user_host.split('/')[0] if '/' in user_host else user_host}")
        else:
            print(f"📊 Database: {db_url[:30]}...")
    else:
        print("⚠️  DATABASE_URL not set - using default SQLite")
    
    # Run migrations
    if not run_migrations():
        print("❌ Migration failed, but continuing anyway...")
    
    print()
    
    # Start application
    start_application()
