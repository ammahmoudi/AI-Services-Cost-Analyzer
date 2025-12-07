#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from ai_cost_manager.database import get_session, close_session
from sqlalchemy import text
import json

session = get_session()

# Check all tables
print('=== ALL TABLES IN DATABASE ===')
result = session.execute(text("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name
"""))
all_tables = result.fetchall()
for table in all_tables:
    print(f'  - {table[0]}')

# Check cache_entries structure
print('\n=== cache_entries TABLE STRUCTURE ===')
result = session.execute(text("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'cache_entries'
    ORDER BY ordinal_position
"""))
columns = result.fetchall()
for col in columns:
    print(f'  {col[0]}: {col[1]}')

# Check all cache_entries
print('\n=== ALL cache_entries RECORDS ===')
result = session.execute(text('SELECT id, cache_key, source_name, model_identifier, cache_type, cached_at FROM cache_entries'))
all_rows = result.fetchall()
print(f'Total records: {len(all_rows)}')
for i, row in enumerate(all_rows[:20]):  # Show first 20
    print(f'{i+1}. ID={row[0]}, Key={row[1]}, Identifier={row[3]}, Type={row[4]}')

# Search for 316
print('\n=== SEARCHING FOR 316 ===')
result = session.execute(text("SELECT id, cache_key, model_identifier, cache_type FROM cache_entries WHERE cache_key LIKE '%316%' OR model_identifier LIKE '%316%'"))
matches = result.fetchall()
print(f'Found {len(matches)} records with 316:')
for row in matches:
    print(f'  ID={row[0]}, Key={row[1]}, Identifier={row[2]}, Type={row[3]}')

close_session()
