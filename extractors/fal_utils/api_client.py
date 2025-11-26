"""
Fal.ai API Client

Functions for fetching data from fal.ai APIs.
"""
import json
from typing import List, Dict, Any


def fetch_from_new_api(base_url: str, fetch_data_func) -> List[Dict[str, Any]]:
    """
    Fetch models using the new TRPC API format.
    
    Args:
        base_url: Base API URL
        fetch_data_func: Function to fetch data (from BaseExtractor)
        
    Returns:
        List of raw model data
    """
    all_models = []
    page = 1
    limit = 100
    
    while True:
        params = {
            "batch": "1",
            "input": json.dumps({
                "0": {
                    "json": {
                        "keywords": "",
                        "categories": [],
                        "tags": [],
                        "type": [],
                        "deprecated": False,
                        "pendingEnterprise": False,
                        "sort": "relevant",
                        "page": page,
                        "limit": limit,
                        "favorites": False,
                        "useCache": True
                    }
                }
            })
        }
        
        try:
            data = fetch_data_func(base_url, params=params)
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
        
        # Navigate through the new API structure with careful null checks
        if not data or not isinstance(data, list) or len(data) == 0:
            break
        
        first_item = data[0]
        if not isinstance(first_item, dict):
            break
        
        result = first_item.get("result")
        if not result or not isinstance(result, dict):
            break
            
        data_obj = result.get("data")
        if not data_obj or not isinstance(data_obj, dict):
            break
            
        json_data = data_obj.get("json")
        if not json_data or not isinstance(json_data, dict):
            break
            
        items = json_data.get("items", [])
        
        if not items:
            break
        
        # Filter out deprecated/removed models
        active_items = [
            item for item in items
            if not item.get("deprecated") and not item.get("removed")
        ]
        
        all_models.extend(active_items)
        
        # Check if we have more pages
        pages = json_data.get("pages", 1)
        if page >= pages:
            break
        
        page += 1
    
    return all_models


def fetch_from_old_api(fetch_data_func) -> List[Dict[str, Any]]:
    """
    Fetch models using the old API format (fallback).
    
    Args:
        fetch_data_func: Function to fetch data (from BaseExtractor)
        
    Returns:
        List of raw model data
    """
    url = "https://fal.ai/api/models?categories=&tags=&type=&deprecated=false&pendingEnterprise=false&keywords=&sort=relevant"
    data = fetch_data_func(url)
    
    if isinstance(data, list):
        return [item for item in data if not item.get("deprecated") and not item.get("removed")]
    return []


def fetch_openapi_schema(model_id: str, fetch_data_func) -> Dict[str, Any]:
    """
    Fetch OpenAPI schema for a specific model.
    
    Args:
        model_id: Model identifier (e.g., 'fal-ai/flux-pro')
        fetch_data_func: Function to fetch data (from BaseExtractor)
        
    Returns:
        OpenAPI schema dictionary
    """
    try:
        # Fal.ai provides OpenAPI schemas at this endpoint
        schema_url = f"https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={model_id}"
        schema = fetch_data_func(schema_url, timeout=5)  # Quick timeout for schemas
        return schema if isinstance(schema, dict) else {}
    except Exception:
        # Silently fail for SSL errors and other issues
        return {}
