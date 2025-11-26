"""
Fal.ai Extractor

Extracts AI model pricing and metadata from fal.ai API.
Based on the new TRPC API format.
"""
from typing import List, Dict, Any, Optional
from extractors.base import BaseExtractor
from ai_cost_manager.cache import cache_manager
from extractors.fal_utils import (
    fetch_from_new_api,
    fetch_from_old_api,
    fetch_openapi_schema,
    fetch_playground_data,
    extract_input_schema,
    extract_output_schema,
    simplify_schema
)


class FalAIExtractor(BaseExtractor):
    """
    Extractor for fal.ai models.
    
    Fetches model data from the fal.ai explore API and normalizes it.
    Also fetches OpenAPI schemas for input/output definitions.
    """
    
    def __init__(self, source_url: str = "https://fal.ai/api/trpc/models.list", fetch_schemas: bool = True, use_llm: bool = False):
        super().__init__(source_url)
        self.base_url = source_url
        self.fetch_schemas = fetch_schemas
        self.use_llm = use_llm
        self.force_refresh = False  # Set to True to bypass cache
    
    def extract(self) -> List[Dict[str, Any]]:
        """
        Extract all models from fal.ai API.
        
        Returns:
            List of normalized model data dictionaries
        """
        print("Fetching models from fal.ai...")
        
        try:
            # Try new API format first
            models = fetch_from_new_api(self.base_url, self.fetch_data)
            print(f"✓ Fetched {len(models)} models from fal.ai\n")
            
            # Extract with progress bar
            from tqdm import tqdm
            normalized_models = []
            
            with tqdm(total=len(models), desc="Extracting models", unit=" model") as pbar:
                for i, model in enumerate(models):
                    normalized = self._normalize_fal_model(model, i + 1, len(models))
                    normalized_models.append(normalized)
                    
                    # Build status indicator
                    status_parts = []
                    if normalized.get('_cache_used'):
                        status_parts.append('📦')
                    if normalized.get('_errors'):
                        status_parts.append('⚠️')
                    
                    status = ' '.join(status_parts) if status_parts else ''
                    model_name = normalized.get('name', 'Unknown')
                    display_text = f"{model_name[:45]} {status}" if status else model_name[:50]
                    
                    pbar.set_postfix_str(display_text)
                    pbar.update(1)
            
            print(f"\n✅ Successfully extracted {len(normalized_models)} models")
            return normalized_models
            
        except Exception as e:
            print(f"Error with new API: {e}")
            print("Trying fallback API format...")
            try:
                models = fetch_from_old_api(self.fetch_data)
                print(f"✓ Fetched {len(models)} models from fal.ai (fallback)\n")
                
                from tqdm import tqdm
                normalized_models = []
                
                with tqdm(total=len(models), desc="Extracting models", unit=" model") as pbar:
                    for i, model in enumerate(models):
                        normalized = self._normalize_fal_model(model, i + 1, len(models))
                        normalized_models.append(normalized)
                        
                        # Build status indicator
                        status_parts = []
                        if normalized.get('_cache_used'):
                            status_parts.append('📦')
                        if normalized.get('_errors'):
                            status_parts.append('⚠️')
                        
                        status = ' '.join(status_parts) if status_parts else ''
                        model_name = normalized.get('name', 'Unknown')
                        display_text = f"{model_name[:45]} {status}" if status else model_name[:50]
                        
                        pbar.set_postfix_str(display_text)
                        pbar.update(1)
                
                print(f"\n✅ Successfully extracted {len(normalized_models)} models")
                return normalized_models
            except Exception as e2:
                print(f"Error with fallback API: {e2}")
                return []
    
    def _normalize_fal_model(self, raw_data: Dict[str, Any], current: int = 0, total: int = 0) -> Dict[str, Any]:
        """
        Normalize fal.ai model data to standard format.
        
        Args:
            raw_data: Raw model data from fal.ai API
            current: Current model number (for progress)
            total: Total number of models (for progress)
            
        Returns:
            Normalized model data dictionary
        """
        from datetime import datetime
        
        model_id = raw_data.get("id", "")
        
        # Track cache usage and errors
        cache_used = []
        errors = []
        
        # Save raw data to cache with timestamp
        cache_manager.save_raw_data("fal.ai", model_id, raw_data)
        last_raw_fetched = datetime.utcnow()
        
        # Extract name (handle both formats)
        name = (
            raw_data.get("title") or 
            raw_data.get("name") or 
            model_id.split("/")[-1].replace("-", " ").title()
        )
        
        # Extract description
        description = (
            raw_data.get("shortDescription") or 
            raw_data.get("description") or 
            ""
        )
        
        # Map category to model type
        category = raw_data.get("category", "other")
        model_type = self._map_category_to_type(category)
        
        # Calculate cost
        cost_per_call = self._calculate_cost(raw_data)
        
        # Safely extract credits_required
        credits_required = None
        credits_val = raw_data.get("creditsRequired")
        if credits_val is not None:
            try:
                credits_required = float(credits_val)
            except (ValueError, TypeError):
                # creditsRequired might be a dict or other type, ignore it
                pass
        
        # Extract pricing details with LLM if enabled
        pricing_details = {}
        if self.use_llm:
            # Try loading from cache first (unless force refresh)
            if not self.force_refresh:
                pricing_details = cache_manager.load_llm_extraction("fal.ai", model_id)
            
            if not pricing_details:
                try:
                    from ai_cost_manager.llm_extractor import extract_pricing_with_llm
                    import time
                    
                    # Retry logic for rate limiting
                    max_retries = 3
                    retry_delay = 2  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            pricing_details = extract_pricing_with_llm({
                                'name': name,
                                'pricing_info': raw_data.get("pricingInfoOverride") or raw_data.get("pricingInfo") or "",
                                'creditsRequired': raw_data.get("creditsRequired"),
                                'model_type': model_type,
                            })
                            
                            if pricing_details:
                                # Save to cache
                                cache_manager.save_llm_extraction("fal.ai", model_id, pricing_details)
                                print(f"    ✓ LLM extracted pricing for {name}")
                            break  # Success, exit retry loop
                            
                        except Exception as retry_error:
                            error_msg = str(retry_error)
                            if "Rate limit" in error_msg and attempt < max_retries - 1:
                                print(f"    ⏳ Rate limited, waiting {retry_delay}s before retry...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            elif "server error" in error_msg.lower() and attempt < max_retries - 1:
                                print(f"    ⏳ Server error, retrying in {retry_delay}s...")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                raise  # Re-raise on final attempt or non-retryable error
                                
                except Exception as e:
                    error_msg = str(e)
                    if "Rate limit" in error_msg:
                        print(f"    ⚠ Rate limited, skipping LLM extraction for {name}")
                    elif "server error" in error_msg.lower():
                        print(f"    ⚠ OpenRouter server error, skipping {name}")
                    else:
                        print(f"    ⚠ LLM pricing extraction failed for {name}: {e}")
            else:
                print(f"    📦 Using cached LLM data for {name}")
        
        # Extract group info
        group = raw_data.get("group", {})
        if not isinstance(group, dict):
            group = {}
        
        # Fetch OpenAPI schema if enabled
        openapi_schema = None
        input_schema = None
        output_schema = None
        last_schema_fetched = None
        if self.fetch_schemas:
            # Try loading from cache first (unless force refresh)
            if not self.force_refresh:
                openapi_schema = cache_manager.load_schema("fal.ai", model_id)
                if openapi_schema:
                    cache_used.append('schema')
            
            if not openapi_schema:
                try:
                    openapi_schema = fetch_openapi_schema(model_id, self.fetch_data)
                    if openapi_schema:
                        # Save to cache
                        cache_manager.save_schema("fal.ai", model_id, openapi_schema)
                        last_schema_fetched = datetime.utcnow()
                except Exception as e:
                    # Silently skip schema errors - don't print or freeze
                    errors.append(f'schema: {str(e)[:50]}')
            
            if openapi_schema:
                input_schema = extract_input_schema(openapi_schema)
                output_schema = extract_output_schema(openapi_schema)
                if not last_schema_fetched:
                    last_schema_fetched = datetime.utcnow()

        
        # Fetch playground data for additional pricing info
        playground_data = None
        playground_pricing = None
        last_playground_fetched = None
        if not self.force_refresh:
            playground_data = cache_manager.load_playground_data("fal.ai", model_id)
            if playground_data:
                cache_used.append('playground')
        
        if not playground_data:
            try:
                playground_data = fetch_playground_data(model_id)
                if playground_data:
                    # Save to cache
                    cache_manager.save_playground_data("fal.ai", model_id, playground_data)
                    last_playground_fetched = datetime.utcnow()
            except Exception as e:
                errors.append(f'playground: {str(e)[:50]}')
        else:
            if playground_data:
                last_playground_fetched = datetime.utcnow()
        
        # Extract pricing from playground data
        if playground_data:
            try:
                # playground_data is already the extracted billing dict with keys:
                # 'endpoint', 'billing_unit', 'price', 'pricing_text'
                if 'price' in playground_data and 'billing_unit' in playground_data:
                    playground_pricing = {
                        'price': playground_data['price'],
                        'billing_unit': playground_data['billing_unit'],
                        'endpoint': playground_data.get('endpoint'),
                        'pricing_text': playground_data.get('pricing_text'),  # Full text for LLM
                    }
                    
                    # Use playground pricing if we don't have credits_required
                    if credits_required is None and playground_pricing.get('price'):
                        credits_required = playground_pricing['price']
                        cost_per_call = float(playground_pricing['price'])
            except Exception as e:
                pass
        
        # Build normalized data
        normalized = {
            'model_id': model_id,
            'name': name,
            'description': description,
            'model_type': model_type,
            'cost_per_call': cost_per_call,
            'credits_required': credits_required,
            'pricing_info': raw_data.get("pricingInfoOverride") or raw_data.get("pricingInfo") or "",
            'thumbnail_url': raw_data.get("thumbnailUrl") or "",
            'tags': raw_data.get("tags") or [],
            'category': category,
            'input_schema': input_schema,
            'output_schema': output_schema,
            # Playground pricing (fallback source)
            'playground_pricing': playground_pricing,
            # LLM extracted pricing details
            'pricing_type': pricing_details.get('pricing_type') if pricing_details else None,
            'pricing_formula': pricing_details.get('pricing_formula') if pricing_details else None,
            'pricing_variables': pricing_details.get('pricing_variables') if pricing_details else None,
            'input_cost_per_unit': pricing_details.get('input_cost_per_unit') if pricing_details else None,
            'output_cost_per_unit': pricing_details.get('output_cost_per_unit') if pricing_details else None,
            'cost_unit': pricing_details.get('cost_unit') if pricing_details else None,
            'llm_extracted': pricing_details if pricing_details else None,
            'raw_metadata': {
                **raw_data,
                'openapi_schema': openapi_schema,
                'playground_data': playground_data,
                'pricing_text': playground_pricing.get('pricing_text') if playground_pricing else None,
            },
            # Timestamps for data fetching
            'last_raw_fetched': last_raw_fetched,
            'last_schema_fetched': last_schema_fetched,
            'last_playground_fetched': last_playground_fetched,
            # Internal tracking (not saved to DB)
            '_cache_used': cache_used,
            '_errors': errors,
        }
        
        return normalized
    
    def _map_category_to_type(self, category: str) -> str:
        """
        Map fal.ai category to standard model type.
        
        Args:
            category: fal.ai category string
            
        Returns:
            Standardized model type
        """
        category_map = {
            "text-to-image": "text-to-image",
            "text-to-video": "text-to-video",
            "image-to-image": "image-to-image",
            "image-to-video": "image-to-video",
            "text-generation": "text-generation",
            "image-to-text": "image-to-text",
            "audio-generation": "audio-generation",
        }
        return category_map.get(category, "other")
    
    def _calculate_cost(self, model_data: Dict[str, Any]) -> float:
        """
        Calculate cost per call from various pricing fields.
        
        Args:
            model_data: Raw model data
            
        Returns:
            Cost per call as float
        """
        credits_required = model_data.get("creditsRequired")
        if credits_required is not None:
            try:
                credits = float(credits_required)
                if credits > 0:
                    # Assuming 1 credit = $0.01 (adjust as needed)
                    return credits * 0.01
            except (ValueError, TypeError):
                pass
        
        # Could parse pricingInfoOverride for more accurate pricing
        # For now, return 0.0 if no credits info
        return 0.0

