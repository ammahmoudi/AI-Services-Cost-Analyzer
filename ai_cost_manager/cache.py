"""
Cache manager for storing extracted data to avoid re-fetching.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CacheManager:
    """
    Manages caching of extracted model data, schemas, and LLM results.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Base directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_model_cache_dir(self, source_name: str, model_id: str) -> Path:
        """
        Get cache directory for a specific model.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            
        Returns:
            Path to model cache directory
        """
        # Sanitize model_id for filesystem
        safe_model_id = model_id.replace("/", "_").replace(":", "_")
        model_dir = self.cache_dir / source_name / safe_model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def save_raw_data(self, source_name: str, model_id: str, raw_data: Dict[str, Any]) -> None:
        """
        Save raw model data from API.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            raw_data: Raw data from API
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / "raw.json"
        
        data_with_meta = {
            "cached_at": datetime.utcnow().isoformat(),
            "source": source_name,
            "model_id": model_id,
            "data": raw_data
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data_with_meta, f, indent=2, ensure_ascii=False)
    
    def load_raw_data(self, source_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load cached raw model data.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            
        Returns:
            Raw data or None if not cached
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / "raw.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                return cached.get("data")
        except Exception as e:
            print(f"Error loading cached raw data for {model_id}: {e}")
            return None
    
    def save_schema(self, source_name: str, model_id: str, schema: Dict[str, Any]) -> None:
        """
        Save OpenAPI schema data.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            schema: OpenAPI schema data
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / "schema.json"
        
        schema_with_meta = {
            "cached_at": datetime.utcnow().isoformat(),
            "source": source_name,
            "model_id": model_id,
            "schema": schema
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(schema_with_meta, f, indent=2, ensure_ascii=False)
    
    def load_schema(self, source_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load cached schema data.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            
        Returns:
            Schema data or None if not cached
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / "schema.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                return cached.get("schema")
        except Exception as e:
            print(f"Error loading cached schema for {model_id}: {e}")
            return None
    
    def save_playground_data(self, source_name: str, model_id: str, playground_data: Dict[str, Any]) -> None:
        """
        Save playground endpoint data.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            playground_data: Playground data from API
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / "playground.json"
        
        data_with_meta = {
            "cached_at": datetime.utcnow().isoformat(),
            "source": source_name,
            "model_id": model_id,
            "playground_data": playground_data
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data_with_meta, f, indent=2, ensure_ascii=False)
    
    def load_playground_data(self, source_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load cached playground endpoint data.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            
        Returns:
            Playground data or None if not cached
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / "playground.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            return cached.get("playground_data")
        except Exception as e:
            print(f"Error loading cached playground data for {model_id}: {e}")
            return None
    
    def save_llm_extraction(self, source_name: str, model_id: str, llm_data: Dict[str, Any]) -> None:
        """
        Save LLM-extracted pricing data.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            llm_data: LLM extraction results
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / "llm_extraction.json"
        
        llm_with_meta = {
            "cached_at": datetime.utcnow().isoformat(),
            "source": source_name,
            "model_id": model_id,
            "extraction": llm_data
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(llm_with_meta, f, indent=2, ensure_ascii=False)
    
    def load_llm_extraction(self, source_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load cached LLM extraction data.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            
        Returns:
            LLM extraction data or None if not cached
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        cache_file = cache_dir / "llm_extraction.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                return cached.get("extraction")
        except Exception as e:
            print(f"Error loading cached LLM extraction for {model_id}: {e}")
            return None
    
    def has_cache(self, source_name: str, model_id: str, cache_type: str = "raw") -> bool:
        """
        Check if cache exists for a model.
        
        Args:
            source_name: Name of the API source
            model_id: Model identifier
            cache_type: Type of cache ("raw", "schema", "llm")
            
        Returns:
            True if cache exists
        """
        cache_dir = self._get_model_cache_dir(source_name, model_id)
        
        cache_files = {
            "raw": "raw.json",
            "schema": "schema.json",
            "llm": "llm_extraction.json"
        }
        
        cache_file = cache_dir / cache_files.get(cache_type, "raw.json")
        return cache_file.exists()
    
    def clear_cache(self, source_name: Optional[str] = None, model_id: Optional[str] = None) -> None:
        """
        Clear cache data.
        
        Args:
            source_name: If provided, clear only this source. If None, clear all.
            model_id: If provided, clear only this model. Requires source_name.
        """
        if source_name and model_id:
            # Clear specific model
            cache_dir = self._get_model_cache_dir(source_name, model_id)
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
        elif source_name:
            # Clear entire source
            source_dir = self.cache_dir / source_name
            if source_dir.exists():
                import shutil
                shutil.rmtree(source_dir)
        else:
            # Clear all cache
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
    
    def clear_all_cache(self) -> None:
        """Clear all cache."""
        self.clear_cache()
    
    def clear_llm_cache(self) -> None:
        """Clear only LLM extraction cache files."""
        for source_dir in self.cache_dir.iterdir():
            if source_dir.is_dir():
                for model_dir in source_dir.iterdir():
                    if model_dir.is_dir():
                        llm_file = model_dir / "llm_extraction.json"
                        if llm_file.exists():
                            llm_file.unlink()
    
    def clear_schema_cache(self) -> None:
        """Clear only schema cache files."""
        for source_dir in self.cache_dir.iterdir():
            if source_dir.is_dir():
                for model_dir in source_dir.iterdir():
                    if model_dir.is_dir():
                        schema_file = model_dir / "schema.json"
                        if schema_file.exists():
                            schema_file.unlink()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_cached_models': 0,
            'schemas_count': 0,
            'llm_extractions_count': 0,
            'total_size_mb': 0.0,
            'by_source': {}
        }
        
        total_size = 0
        
        for source_dir in self.cache_dir.iterdir():
            if not source_dir.is_dir():
                continue
            
            source_name = source_dir.name
            source_stats = {
                'models': 0,
                'schemas': 0,
                'llm_extractions': 0,
                'size_mb': 0.0
            }
            
            source_size = 0
            
            for model_dir in source_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                source_stats['models'] += 1
                stats['total_cached_models'] += 1
                
                # Check for schema
                if (model_dir / "schema.json").exists():
                    source_stats['schemas'] += 1
                    stats['schemas_count'] += 1
                    source_size += (model_dir / "schema.json").stat().st_size
                
                # Check for LLM extraction
                if (model_dir / "llm_extraction.json").exists():
                    source_stats['llm_extractions'] += 1
                    stats['llm_extractions_count'] += 1
                    source_size += (model_dir / "llm_extraction.json").stat().st_size
                
                # Add raw data size
                if (model_dir / "raw.json").exists():
                    source_size += (model_dir / "raw.json").stat().st_size
            
            source_stats['size_mb'] = round(source_size / (1024 * 1024), 2)
            total_size += source_size
            stats['by_source'][source_name] = source_stats
        
        stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        return stats


# Global cache manager instance
cache_manager = CacheManager()
