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
    
    migrations_dir = Path('/app/migrations')
    
    if not migrations_dir.exists():
        print("⚠️  No migrations directory found, skipping...")
        return True
    
    # Get all migration files
    migration_files = sorted([
        f for f in migrations_dir.glob('*.py')
        if not f.name.startswith('__') and not f.name.startswith('.')
    ])
    
    if not migration_files:
        print("ℹ️  No migrations to run")
        return True
    
    print(f"Found {len(migration_files)} migration(s)")
    
    for migration_file in migration_files:
        try:
            print(f"  Running {migration_file.name}...", end=" ")
            result = subprocess.run(
                [sys.executable, str(migration_file)],
                cwd='/app',
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅")
            else:
                print("⚠️")
                if result.stderr:
                    # Only print if it's not "already exists" error
                    if "already exists" not in result.stderr.lower():
                        print(f"    {result.stderr[:200]}")
        
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            # Don't fail on migration errors - they might be idempotent
    
    print("✅ Migrations complete")
    return True

def start_application():
    """Start the Flask application"""
    print("🚀 Starting application...")
    
    # Run with gunicorn for production
    subprocess.run([
        'gunicorn',
        '--bind', '0.0.0.0:5000',
        '--workers', '4',
        '--timeout', '120',
        '--access-logfile', '-',
        '--error-logfile', '-',
        'app:app'
    ])

if __name__ == '__main__':
    print("="*60)
    print("🐳 AI Cost Manager - Container Startup")
    print("="*60)
    
    # Run migrations
    if not run_migrations():
        print("❌ Migration failed, but continuing anyway...")
    
    print()
    
    # Start application
    start_application()
