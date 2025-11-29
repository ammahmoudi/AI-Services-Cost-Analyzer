"""
Cache manager for storing extracted data.
Supports both database and file backends based on configuration.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Determine cache backend from environment
CACHE_BACKEND = os.getenv('CACHE_BACKEND', 'database').lower()
CACHE_DIR = os.getenv('CACHE_DIR', 'cache')


class FileCacheBackend:
    """File-based cache backend"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_model_cache_dir(self, source_name: str, model_id: str) -> Path:
        safe_model_id = model_id.replace("/", "_").replace(":", "_")
        model_dir = self.cache_dir / source_name / safe_model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def save(self, source_name: str, model_id: str, cache_type: str, data: Dict[str, Any]) -> None:
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / f"{cache_type}.json"
        
        data_with_meta = {
            "cached_at": datetime.utcnow().isoformat(),
            "source": source_name,
            "model_id": model_id,
            "data": data
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data_with_meta, f, indent=2, ensure_ascii=False)
    
    def load(self, source_name: str, model_id: str, cache_type: str) -> Optional[Dict[str, Any]]:
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / f"{cache_type}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                return cached.get("data")
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def has_cache(self, source_name: str, model_id: str, cache_type: str) -> bool:
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / f"{cache_type}.json"
        return cache_file.exists()
    
    def clear(self, source_name: Optional[str] = None, model_id: Optional[str] = None) -> None:
        import shutil
        if source_name and model_id:
            cache_dir = self._get_model_cache_dir(source_name, model_id)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        elif source_name:
            source_dir = self.cache_dir / source_name
            if source_dir.exists():
                shutil.rmtree(source_dir)
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
    
    def clear_by_type(self, cache_type: str) -> None:
        for source_dir in self.cache_dir.iterdir():
            if source_dir.is_dir():
                for model_dir in source_dir.iterdir():
                    if model_dir.is_dir():
                        cache_file = model_dir / f"{cache_type}.json"
                        if cache_file.exists():
                            cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_cached_models': 0,
            'raw_count': 0,
            'schemas_count': 0,
            'playground_count': 0,
            'llm_extractions_count': 0,
            'total_entries': 0,
            'by_source': {}
        }
        
        for source_dir in self.cache_dir.iterdir():
            if not source_dir.is_dir():
                continue
            
            source_name = source_dir.name
            source_stats = {'models': 0, 'raw': 0, 'schemas': 0, 'playground': 0, 'llm_extractions': 0}
            
            for model_dir in source_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                source_stats['models'] += 1
                stats['total_cached_models'] += 1
                
                if (model_dir / "raw.json").exists():
                    source_stats['raw'] += 1
                    stats['raw_count'] += 1
                    stats['total_entries'] += 1
                
                if (model_dir / "schema.json").exists():
                    source_stats['schemas'] += 1
                    stats['schemas_count'] += 1
                    stats['total_entries'] += 1
                
                if (model_dir / "playground.json").exists():
                    source_stats['playground'] += 1
                    stats['playground_count'] += 1
                    stats['total_entries'] += 1
                
                if (model_dir / "llm.json").exists():
                    source_stats['llm_extractions'] += 1
                    stats['llm_extractions_count'] += 1
                    stats['total_entries'] += 1
            
            stats['by_source'][source_name] = source_stats
        
        return stats


class DatabaseCacheBackend:
    """Database-based cache backend"""
    
    def __init__(self):
        pass
    
    def _get_model_by_identifier(self, source_name: str, model_id: str):
        from .database import get_session
        from .models import AIModel, APISource
        
        session = get_session()
        try:
            model = (
                session.query(AIModel)
                .join(APISource)
                .filter(APISource.name == source_name, AIModel.model_id == model_id)
                .first()
            )
            return model
        finally:
            session.close()
    
    def save(self, source_name: str, model_id: str, cache_type: str, data: Dict[str, Any]) -> None:
        from .database import get_session
        from .models import CacheEntry
        
        model = self._get_model_by_identifier(source_name, model_id)
        if not model:
            return
        
        session = get_session()
        try:
            session.query(CacheEntry).filter_by(
                model_id=model.id, cache_type=cache_type
            ).delete()
            
            cache_entry = CacheEntry(
                model_id=model.id,
                cache_type=cache_type,
                data=data,
                cached_at=datetime.utcnow()
            )
            session.add(cache_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving cache: {e}")
        finally:
            session.close()
    
    def load(self, source_name: str, model_id: str, cache_type: str) -> Optional[Dict[str, Any]]:
        from .database import get_session
        from .models import CacheEntry
        
        model = self._get_model_by_identifier(source_name, model_id)
        if not model:
            return None
        
        session = get_session()
        try:
            cache_entry = session.query(CacheEntry).filter_by(
                model_id=model.id, cache_type=cache_type
            ).first()
            return cache_entry.data if cache_entry else None
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
        finally:
            session.close()
    
    def has_cache(self, source_name: str, model_id: str, cache_type: str) -> bool:
        from .database import get_session
        from .models import CacheEntry
        
        model = self._get_model_by_identifier(source_name, model_id)
        if not model:
            return False
        
        session = get_session()
        try:
            exists = session.query(CacheEntry).filter_by(
                model_id=model.id, cache_type=cache_type
            ).first() is not None
            return exists
        finally:
            session.close()
    
    def clear(self, source_name: Optional[str] = None, model_id: Optional[str] = None) -> None:
        from .database import get_session
        from .models import CacheEntry, APISource
        
        session = get_session()
        try:
            if source_name and model_id:
                model = self._get_model_by_identifier(source_name, model_id)
                if model:
                    session.query(CacheEntry).filter_by(model_id=model.id).delete()
            elif source_name:
                source = session.query(APISource).filter_by(name=source_name).first()
                if source:
                    model_ids = [m.id for m in source.models]
                    session.query(CacheEntry).filter(CacheEntry.model_id.in_(model_ids)).delete(synchronize_session=False)
            else:
                session.query(CacheEntry).delete()
            
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error clearing cache: {e}")
        finally:
            session.close()
    
    def clear_by_type(self, cache_type: str) -> None:
        from .database import get_session
        from .models import CacheEntry
        
        session = get_session()
        try:
            session.query(CacheEntry).filter_by(cache_type=cache_type).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error clearing cache by type: {e}")
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        from .database import get_session
        from .models import CacheEntry, APISource
        from sqlalchemy import func
        
        session = get_session()
        try:
            stats = {
                'total_cached_models': 0,
                'raw_count': 0,
                'schemas_count': 0,
                'playground_count': 0,
                'llm_extractions_count': 0,
                'total_entries': 0,
                'by_source': {}
            }
            
            # Count by type
            for cache_type in ['raw', 'schema', 'playground', 'llm']:
                count = session.query(func.count(CacheEntry.id)).filter_by(cache_type=cache_type).scalar()
                if cache_type == 'schema':
                    stats['schemas_count'] = count
                elif cache_type == 'llm':
                    stats['llm_extractions_count'] = count
                elif cache_type == 'playground':
                    stats['playground_count'] = count
                elif cache_type == 'raw':
                    stats['raw_count'] = count
            
            stats['total_cached_models'] = session.query(func.count(func.distinct(CacheEntry.model_id))).scalar()
            stats['total_entries'] = session.query(func.count(CacheEntry.id)).scalar()
            
            # By source
            sources = session.query(APISource).all()
            for source in sources:
                model_ids = [m.id for m in source.models]
                if not model_ids:
                    continue
                
                source_stats = {'models': len(model_ids), 'raw': 0, 'schemas': 0, 'playground': 0, 'llm_extractions': 0}
                
                for cache_type in ['raw', 'schema', 'playground', 'llm']:
                    count = (
                        session.query(func.count(CacheEntry.id))
                        .filter(CacheEntry.model_id.in_(model_ids), CacheEntry.cache_type == cache_type)
                        .scalar()
                    )
                    if cache_type == 'schema':
                        source_stats['schemas'] = count
                    elif cache_type == 'llm':
                        source_stats['llm_extractions'] = count
                    elif cache_type == 'playground':
                        source_stats['playground'] = count
                    elif cache_type == 'raw':
                        source_stats['raw'] = count
                
                stats['by_source'][source.name] = source_stats
            
            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'error': str(e)}
        finally:
            session.close()


class CacheManager:
    """
    Unified cache manager that delegates to backend based on configuration.
    """
    
    def __init__(self):
        """Initialize cache manager with configured backend"""
        if CACHE_BACKEND == 'file':
            self.backend = FileCacheBackend(CACHE_DIR)
            print(f"Using file-based cache backend: {CACHE_DIR}")
        else:
            self.backend = DatabaseCacheBackend()
            print("Using database cache backend")
    
    def save_raw_data(self, source_name: str, model_id: str, raw_data: Dict[str, Any]) -> None:
        """Save raw model data from API"""
        self.backend.save(source_name, model_id, 'raw', raw_data)
    
    def load_raw_data(self, source_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Load cached raw model data"""
        return self.backend.load(source_name, model_id, 'raw')
    
    def save_schema(self, source_name: str, model_id: str, schema: Dict[str, Any]) -> None:
        """Save OpenAPI schema data"""
        self.backend.save(source_name, model_id, 'schema', schema)
    
    def load_schema(self, source_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Load cached schema data"""
        return self.backend.load(source_name, model_id, 'schema')
    
    def save_playground_data(self, source_name: str, model_id: str, playground_data: Dict[str, Any]) -> None:
        """Save playground endpoint data"""
        self.backend.save(source_name, model_id, 'playground', playground_data)
    
    def load_playground_data(self, source_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Load cached playground endpoint data"""
        return self.backend.load(source_name, model_id, 'playground')
    
    def save_llm_extraction(self, source_name: str, model_id: str, llm_data: Dict[str, Any]) -> None:
        """Save LLM-extracted pricing data"""
        self.backend.save(source_name, model_id, 'llm', llm_data)
    
    def load_llm_extraction(self, source_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Load cached LLM extraction data"""
        return self.backend.load(source_name, model_id, 'llm')
    
    def has_cache(self, source_name: str, model_id: str, cache_type: str = "raw") -> bool:
        """Check if cache exists for a model"""
        return self.backend.has_cache(source_name, model_id, cache_type)
    
    def clear_cache(self, source_name: Optional[str] = None, model_id: Optional[str] = None) -> None:
        """Clear cache data"""
        self.backend.clear(source_name, model_id)
    
    def clear_all_cache(self) -> None:
        """Clear all cache"""
        self.backend.clear()
    
    def clear_model_cache(self, source_name: str, model_id: str) -> None:
        """Clear cache for a specific model"""
        self.backend.clear(source_name, model_id)
    
    def clear_llm_cache(self) -> None:
        """Clear only LLM extraction cache entries"""
        self.backend.clear_by_type('llm')
    
    def clear_schema_cache(self) -> None:
        """Clear only schema cache entries"""
        self.backend.clear_by_type('schema')
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.backend.get_stats()


# Global cache manager instance
cache_manager = CacheManager()
