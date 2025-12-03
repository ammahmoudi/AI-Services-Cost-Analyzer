"""
Progress Tracker for Extraction Operations

Tracks real-time progress of model extraction operations.
Stores progress state in JSON files that can be queried via API.
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class ProgressTracker:
    """Track progress of extraction operations."""
    
    def __init__(self, source_name: str, source_id: int):
        """
        Initialize progress tracker.
        
        Args:
            source_name: Name of the source being extracted
            source_id: ID of the source being extracted
        """
        self.source_name = source_name
        self.source_id = source_id
        self.progress_dir = Path("cache/progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.progress_dir / f"source_{source_id}.json"
        
        # Initialize progress state
        self.state = {
            'source_id': source_id,
            'source_name': source_name,
            'status': 'initializing',  # initializing, extracting, complete, error
            'total_models': 0,
            'processed_models': 0,
            'current_model': None,
            'current_model_name': None,
            'new_models': 0,
            'updated_models': 0,
            'cached_count': 0,
            'error_count': 0,
            'errors': [],
            'started_at': None,
            'completed_at': None,
            'elapsed_seconds': 0,
            'models_per_second': 0.0,
            'estimated_remaining_seconds': 0,
            'cache_details': {
                'raw': 0,
                'schema': 0,
                'playground': 0,
                'llm': 0,
            },
            'options': {
                'use_llm': False,
                'fetch_schemas': False,
                'force_refresh': False,
            }
        }
    
    def start(self, total_models: int, options: Optional[Dict[str, Any]] = None):
        """
        Start tracking extraction.
        
        Args:
            total_models: Total number of models to extract
            options: Extraction options (use_llm, fetch_schemas, etc.)
        """
        self.state['status'] = 'extracting'
        self.state['total_models'] = total_models
        self.state['started_at'] = datetime.utcnow().isoformat()
        
        if options:
            self.state['options'].update(options)
        
        self._save()
    
    def update(self, processed: int, current_model_id: str = None, 
               current_model_name: str = None, cache_used: list = None,
               has_error: bool = False, error_message: str = None):
        """
        Update progress.
        
        Args:
            processed: Number of models processed so far
            current_model_id: ID of current model being processed
            current_model_name: Name of current model being processed
            cache_used: List of cache types used ['raw', 'schema', 'llm']
            has_error: Whether current model had an error
            error_message: Error message if has_error is True
        """
        self.state['processed_models'] = processed
        self.state['current_model'] = current_model_id
        self.state['current_model_name'] = current_model_name
        
        # Update cache counts
        if cache_used:
            for cache_type in cache_used:
                if cache_type in self.state['cache_details']:
                    self.state['cache_details'][cache_type] += 1
            self.state['cached_count'] = sum(self.state['cache_details'].values())
        
        # Update error tracking
        if has_error:
            self.state['error_count'] += 1
            if error_message and len(self.state['errors']) < 100:  # Keep last 100 errors
                self.state['errors'].append({
                    'model_id': current_model_id,
                    'model_name': current_model_name,
                    'message': error_message,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        # Calculate timing
        if self.state['started_at']:
            start_time = datetime.fromisoformat(self.state['started_at'])
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.state['elapsed_seconds'] = elapsed
            
            if processed > 0:
                self.state['models_per_second'] = processed / elapsed
                remaining = self.state['total_models'] - processed
                if self.state['models_per_second'] > 0:
                    self.state['estimated_remaining_seconds'] = remaining / self.state['models_per_second']
        
        self._save()
    
    def increment_new(self):
        """Increment count of new models."""
        self.state['new_models'] += 1
        self._save()
    
    def increment_updated(self):
        """Increment count of updated models."""
        self.state['updated_models'] += 1
        self._save()
    
    def complete(self):
        """Mark extraction as complete."""
        self.state['status'] = 'complete'
        self.state['completed_at'] = datetime.utcnow().isoformat()
        
        if self.state['started_at']:
            start_time = datetime.fromisoformat(self.state['started_at'])
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.state['elapsed_seconds'] = elapsed
        
        self._save()
    
    def error(self, error_message: str):
        """
        Mark extraction as failed.
        
        Args:
            error_message: Error message
        """
        self.state['status'] = 'error'
        self.state['completed_at'] = datetime.utcnow().isoformat()
        self.state['errors'].append({
            'model_id': None,
            'model_name': None,
            'message': error_message,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if self.state['started_at']:
            start_time = datetime.fromisoformat(self.state['started_at'])
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.state['elapsed_seconds'] = elapsed
        
        self._save()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current progress state.
        
        Returns:
            Progress state dictionary
        """
        return self.state.copy()
    
    def _save(self):
        """Save progress state to file."""
        try:
            # Write to a temp file first, then rename (atomic operation)
            temp_file = self.progress_file.with_suffix('.json.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
            # Atomic rename (avoids partial writes)
            temp_file.replace(self.progress_file)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
    
    @classmethod
    def load(cls, source_id: int) -> Optional[Dict[str, Any]]:
        """
        Load progress state from file.
        
        Args:
            source_id: Source ID to load progress for
            
        Returns:
            Progress state dictionary or None if not found
        """
        progress_file = Path("cache/progress") / f"source_{source_id}.json"
        
        if not progress_file.exists():
            return None
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content or content.strip() == '':
                    print(f"Warning: Progress file is empty for source {source_id}")
                    return None
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse progress JSON: {e}")
            # Try to recover by returning None
            return None
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
            return None
    
    @classmethod
    def clear(cls, source_id: int):
        """
        Clear progress file for a source.
        
        Args:
            source_id: Source ID to clear progress for
        """
        progress_file = Path("cache/progress") / f"source_{source_id}.json"
        
        if progress_file.exists():
            try:
                progress_file.unlink()
            except Exception as e:
                print(f"Warning: Could not clear progress: {e}")
