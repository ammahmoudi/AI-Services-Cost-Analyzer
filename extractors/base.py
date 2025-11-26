"""
Base Extractor Class

Abstract base class for all API extractors.
Each extractor knows how to fetch and parse data from a specific API source.
"""
from abc import ABC, abstractmethod
import requests
from typing import List, Dict, Any


class BaseExtractor(ABC):
    """
    Base class for all extractors.
    
    Subclasses must implement the extract() method to parse API responses
    and return a list of model data dictionaries.
    """
    
    def __init__(self, source_url: str):
        self.source_url = source_url
        
    def fetch_data(self, url: str = None, method: str = 'GET', **kwargs) -> Any:
        """
        Fetch data from the API.
        
        Args:
            url: URL to fetch (defaults to source_url)
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data (usually JSON)
        """
        url = url or self.source_url
        
        # Set a reasonable timeout to avoid freezing
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 10  # 10 seconds default timeout
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, **kwargs)
            elif method.upper() == 'POST':
                response = requests.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Try to parse as JSON
            try:
                return response.json()
            except ValueError:
                return response.text
                
        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            raise
    
    @abstractmethod
    def extract(self) -> List[Dict[str, Any]]:
        """
        Extract model data from the API.
        
        Returns:
            List of dictionaries containing model information.
            Each dictionary should have at minimum:
            - model_id: Unique identifier
            - name: Human-readable name
            - description: Model description
            - cost_per_call: Cost per API call (float)
            - model_type: Type of model (e.g., 'text-to-image')
        """
        pass
    
    def normalize_model_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize raw model data into standard format.
        Override this method to customize normalization.
        
        Args:
            raw_data: Raw model data from API
            
        Returns:
            Normalized model data dictionary
        """
        return {
            'model_id': raw_data.get('id', ''),
            'name': raw_data.get('name', ''),
            'description': raw_data.get('description', ''),
            'cost_per_call': float(raw_data.get('cost', 0.0)),
            'model_type': raw_data.get('type', 'other'),
            'raw_metadata': raw_data
        }
