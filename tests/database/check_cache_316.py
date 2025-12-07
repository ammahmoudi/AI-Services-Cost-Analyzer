#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from ai_cost_manager.database import get_session, close_session
from ai_cost_manager.models import CacheEntry, AIModel
import json

session = get_session()

# First get model 316 info
m = session.query(AIModel).filter_by(id=316).first()
if m:
    print('=== MODEL 316 ===')
    print(f'ID: {m.id}')
    print(f'Name: {m.name}')
    print(f'Model ID: {m.model_id}')
    print(f'Source ID: {m.source_id}')
    print(f'Model Type (DB): {m.model_type}')
    print(f'Category: {m.category}')
    print(f'llm_extracted: {m.llm_extracted}')
    print()

# Check cache entries for model 316
caches = session.query(CacheEntry).filter(CacheEntry.model_identifier.contains('316')).all()
print(f'Found {len(caches)} cache entries for model 316:')
for cache in caches:
    print(f'\n--- Cache Entry ---')
    print(f'Cache Key: {cache.cache_key}')
    print(f'Type: {cache.cache_type}')
    print(f'Cached At: {cache.cached_at}')
    print(f'Data (first 1000 chars):')
    if isinstance(cache.data, dict):
        print(json.dumps(cache.data, indent=2)[:1000])
    else:
        print(str(cache.data)[:1000])

close_session()
