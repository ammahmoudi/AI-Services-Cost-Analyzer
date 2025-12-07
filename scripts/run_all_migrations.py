"""
Run all pending migrations

This script runs all migration files in the migrations/ directory.
Safe to run multiple times - migrations will skip if already applied.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def run_migrations():
    """Run all migration files"""
    migrations_dir = Path(__file__).parent / 'migrations'
    
    if not migrations_dir.exists():
        print("❌ Migrations directory not found!")
        return
    
    # Get all migration files
    migration_files = sorted([
        f for f in migrations_dir.glob('*.py')
        if not f.name.startswith('__')
    ])
    
    if not migration_files:
        print("No migration files found")
        return
    
    print(f"Found {len(migration_files)} migration file(s)\n")
    
    for migration_file in migration_files:
        print(f"{'='*60}")
        print(f"Running: {migration_file.name}")
        print(f"{'='*60}")
        
        try:
            # Import and run the migration
            spec = __import__('importlib.util').util.spec_from_file_location(
                migration_file.stem, 
                migration_file
            )
            module = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Run upgrade function if it exists
            if hasattr(module, 'upgrade'):
                module.upgrade()
            else:
                print(f"⚠️  No upgrade() function found in {migration_file.name}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error running {migration_file.name}: {e}")
            print()
    
    print("="*60)
    print("✅ All migrations processed!")
    print("="*60)

if __name__ == '__main__':
    print("\n🔧 AI Cost Manager - Migration Runner\n")
    run_migrations()
