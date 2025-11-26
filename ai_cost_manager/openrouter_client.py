"""
OpenRouter API client for fetching available models.
"""
import requests
from typing import List, Dict, Any, Optional


def fetch_openrouter_models(api_key: Optional[str] = None, sort_by_free: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch available models from OpenRouter API.
    
    Args:
        api_key: Optional OpenRouter API key (some info available without auth)
        sort_by_free: If True, sort with free models first
        
    Returns:
        List of model dictionaries with id, name, description, pricing
    """
    url = "https://openrouter.ai/api/v1/models"
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = data.get("data", [])
        
        # Format models for easy selection
        formatted_models = []
        for model in models:
            pricing = model.get("pricing", {})
            prompt_price = pricing.get("prompt", 0)
            completion_price = pricing.get("completion", 0)
            
            # Determine if model is free (ends with :free or both prices are 0)
            is_free = (
                model.get("id", "").endswith(":free") or 
                (float(prompt_price) == 0 and float(completion_price) == 0)
            )
            
            formatted_models.append({
                "id": model.get("id"),
                "name": model.get("name"),
                "description": model.get("description", ""),
                "context_length": model.get("context_length"),
                "prompt_price": prompt_price,
                "completion_price": completion_price,
                "architecture": model.get("architecture", {}),
                "is_free": is_free
            })
        
        # Sort by free models first if requested
        if sort_by_free:
            formatted_models.sort(key=lambda m: (not m["is_free"], m["name"]))
        
        return formatted_models
    
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}")
        return []


def get_recommended_models() -> List[str]:
    """
    Get list of recommended model IDs for pricing extraction.
    
    Returns:
        List of recommended model IDs
    """
    return [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.1-8b-instruct",
        "qwen/qwen-2.5-72b-instruct",
    ]


def format_model_for_display(model: Dict[str, Any]) -> str:
    """
    Format model information for display.
    
    Args:
        model: Model dictionary
        
    Returns:
        Formatted string with model info
    """
    name = model.get("name", "Unknown")
    model_id = model.get("id", "")
    context = model.get("context_length")
    prompt_price = model.get("prompt_price", 0)
    completion_price = model.get("completion_price", 0)
    is_free = model.get("is_free", False)
    
    # Format prices (they're in dollars per token)
    prompt_per_1m = float(prompt_price) * 1_000_000 if prompt_price else 0
    completion_per_1m = float(completion_price) * 1_000_000 if completion_price else 0
    
    info = f"{name} ({model_id})"
    
    if context:
        info += f" - {context:,} tokens"
    
    if is_free:
        info += " - 🆓 FREE"
    elif prompt_per_1m or completion_per_1m:
        info += f" - ${prompt_per_1m:.2f}/${completion_per_1m:.2f} per 1M tokens"
    
    return info
