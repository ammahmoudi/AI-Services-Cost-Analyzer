"""
OpenRouter Extractor
Extracts AI model information from OpenRouter API
"""
import requests
from typing import List, Dict, Any, Optional
from extractors.base_extractor import BaseExtractor


class OpenRouterExtractor(BaseExtractor):
    """
    Extractor for OpenRouter AI models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter extractor.
        
        Args:
            api_key: Optional OpenRouter API key for authentication
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
    
    def get_source_info(self) -> Dict[str, str]:
        """Get information about the OpenRouter source."""
        return {
            'name': 'OpenRouter',
            'base_url': 'https://openrouter.ai',
            'api_url': self.base_url,
            'description': 'OpenRouter aggregates AI models from various providers with unified pricing'
        }
    
    def extract(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract AI models from OpenRouter API.
        
        Returns:
            List of normalized model data dictionaries
        """
        print("Extracting models from OpenRouter...")
        
        # Fetch models from API
        models_data = self._fetch_models()
        
        if not models_data:
            print("No models found from OpenRouter")
            return []
        
        print(f"Found {len(models_data)} models from OpenRouter")
        
        # Normalize each model
        normalized_models = []
        for model in models_data:
            try:
                normalized = self._normalize_openrouter_model(model)
                normalized_models.append(normalized)
            except Exception as e:
                print(f"Error normalizing model {model.get('id', 'unknown')}: {e}")
                continue
        
        print(f"Successfully normalized {len(normalized_models)} models")
        return normalized_models
    
    def _fetch_models(self) -> List[Dict[str, Any]]:
        """
        Fetch models from OpenRouter API.
        
        Returns:
            List of raw model data from API
        """
        url = f"{self.base_url}/models"
        
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', [])
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching OpenRouter models: {e}")
            return []
    
    def _normalize_openrouter_model(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize OpenRouter model data to standard format.
        
        Args:
            raw_data: Raw model data from OpenRouter API
            
        Returns:
            Normalized model data dictionary
        """
        model_id = raw_data.get('id', '')
        name = raw_data.get('name', model_id)
        description = raw_data.get('description', '')
        
        # Extract pricing information
        pricing = raw_data.get('pricing', {})
        prompt_price = self._parse_price(pricing.get('prompt'))
        completion_price = self._parse_price(pricing.get('completion'))
        request_price = self._parse_price(pricing.get('request'))
        image_price = self._parse_price(pricing.get('image'))
        
        # Determine model type based on architecture
        model_type = self._determine_model_type(raw_data)
        
        # Extract context length
        context_length = raw_data.get('context_length')
        
        # Extract architecture info
        architecture = raw_data.get('architecture', {})
        input_modalities = architecture.get('input_modalities', [])
        output_modalities = architecture.get('output_modalities', [])
        
        # Build pricing info string
        pricing_parts = []
        if prompt_price:
            pricing_parts.append(f"Input: ${prompt_price}/token")
        if completion_price:
            pricing_parts.append(f"Output: ${completion_price}/token")
        if request_price:
            pricing_parts.append(f"Request: ${request_price}")
        if image_price:
            pricing_parts.append(f"Image: ${image_price}")
        
        pricing_info = " | ".join(pricing_parts) if pricing_parts else "Pricing available"
        
        # Normalize the data
        normalized = {
            'model_id': model_id,
            'name': name,
            'description': description,
            'model_type': model_type,
            'cost_per_call': request_price,
            'credits_required': None,
            'pricing_info': pricing_info,
            'thumbnail_url': '',
            'tags': self._extract_tags(raw_data),
            'category': model_type,
            'input_schema': None,
            'output_schema': None,
            # Pricing details for database
            'pricing_type': self._determine_pricing_type(pricing),
            'pricing_formula': self._build_pricing_formula(pricing),
            'pricing_variables': {
                'prompt_tokens': prompt_price,
                'completion_tokens': completion_price,
                'requests': request_price,
                'images': image_price,
            },
            'input_cost_per_unit': prompt_price,
            'output_cost_per_unit': completion_price,
            'cost_unit': 'token',
            'llm_extracted': False,  # Directly from API, not LLM extracted
            'raw_metadata': {
                **raw_data,
                'context_length': context_length,
                'input_modalities': input_modalities,
                'output_modalities': output_modalities,
            },
        }
        
        return normalized
    
    def _parse_price(self, price_value: Any) -> Optional[float]:
        """
        Parse price value from OpenRouter API.
        
        Args:
            price_value: Price value (can be number, string, or None)
            
        Returns:
            Parsed price as float, or None
        """
        if price_value is None:
            return None
        
        try:
            if isinstance(price_value, (int, float)):
                return float(price_value)
            elif isinstance(price_value, str):
                # Remove any non-numeric characters except decimal point
                cleaned = ''.join(c for c in price_value if c.isdigit() or c == '.')
                return float(cleaned) if cleaned else None
            return None
        except (ValueError, TypeError):
            return None
    
    def _determine_model_type(self, raw_data: Dict[str, Any]) -> str:
        """
        Determine model type from OpenRouter data.
        
        Args:
            raw_data: Raw model data
            
        Returns:
            Model type string
        """
        architecture = raw_data.get('architecture', {})
        output_modalities = architecture.get('output_modalities', [])
        input_modalities = architecture.get('input_modalities', [])
        
        # Check output modalities
        if 'image' in output_modalities:
            if 'text' in input_modalities:
                return 'text-to-image'
            elif 'image' in input_modalities:
                return 'image-to-image'
            return 'image-generation'
        
        if 'embeddings' in output_modalities:
            return 'embeddings'
        
        # Check input modalities for multimodal
        if len(input_modalities) > 1:
            return 'multimodal'
        
        # Default to text generation
        return 'text-generation'
    
    def _determine_pricing_type(self, pricing: Dict[str, Any]) -> str:
        """
        Determine pricing type from pricing data.
        
        Args:
            pricing: Pricing dictionary
            
        Returns:
            Pricing type string
        """
        has_prompt = pricing.get('prompt') is not None
        has_completion = pricing.get('completion') is not None
        has_request = pricing.get('request') is not None
        
        if has_prompt and has_completion:
            return 'per_token'
        elif has_request:
            return 'per_request'
        else:
            return 'usage_based'
    
    def _build_pricing_formula(self, pricing: Dict[str, Any]) -> Optional[str]:
        """
        Build pricing formula from pricing data.
        
        Args:
            pricing: Pricing dictionary
            
        Returns:
            Pricing formula string or None
        """
        prompt_price = pricing.get('prompt')
        completion_price = pricing.get('completion')
        request_price = pricing.get('request')
        
        if prompt_price and completion_price:
            return f"(input_tokens * {prompt_price}) + (output_tokens * {completion_price})"
        elif request_price:
            return f"{request_price} per request"
        
        return None
    
    def _extract_tags(self, raw_data: Dict[str, Any]) -> List[str]:
        """
        Extract tags from model data.
        
        Args:
            raw_data: Raw model data
            
        Returns:
            List of tag strings
        """
        tags = []
        
        # Add architecture tokenizer as tag
        architecture = raw_data.get('architecture', {})
        tokenizer = architecture.get('tokenizer')
        if tokenizer:
            tags.append(tokenizer)
        
        # Add modalities as tags
        input_modalities = architecture.get('input_modalities', [])
        output_modalities = architecture.get('output_modalities', [])
        
        tags.extend([f"input:{m}" for m in input_modalities])
        tags.extend([f"output:{m}" for m in output_modalities])
        
        # Add context length category
        context_length = raw_data.get('context_length')
        if context_length:
            if context_length >= 100000:
                tags.append('long-context')
            elif context_length >= 32000:
                tags.append('extended-context')
        
        return tags
