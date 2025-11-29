"""
Fal.ai Extractor

Extracts AI model pricing and metadata from fal.ai API.
Based on the new TRPC API format.
"""
from typing import List, Dict, Any, Optional
from extractors.base import BaseExtractor
from ai_cost_manager.cache import cache_manager
from ai_cost_manager.progress_tracker import ProgressTracker
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
    
    def extract(self, progress_tracker: ProgressTracker = None) -> List[Dict[str, Any]]:
        """
        Extract all models from fal.ai API.
        
        Args:
            progress_tracker: Optional progress tracker for UI updates
        
        Returns:
            List of normalized model data dictionaries
        """
        # Initialize progress tracking immediately
        if progress_tracker:
            progress_tracker.start(
                total_models=0,  # Will update once we know the count
                options={
                    'use_llm': self.use_llm,
                    'fetch_schemas': self.fetch_schemas,
                    'force_refresh': getattr(self, 'force_refresh', False)
                }
            )
        
        print("Fetching models from fal.ai...")
        
        try:
            # Try new API format first
            models = fetch_from_new_api(self.base_url, self.fetch_data)
            print(f"✓ Fetched {len(models)} models from fal.ai\n")
            
            # Update progress with actual count
            if progress_tracker:
                progress_tracker.state['total_models'] = len(models)
                progress_tracker._save()
            
            # Extract with progress bar
            from tqdm import tqdm
            normalized_models = []
            
            with tqdm(total=len(models), desc="Extracting models", unit=" model") as pbar:
                for i, model in enumerate(models):
                    normalized = self._normalize_fal_model(model, i + 1, len(models))
                    normalized_models.append(normalized)
                    
                    # Update progress tracker
                    if progress_tracker:
                        progress_tracker.update(
                            processed=i + 1,
                            current_model_id=normalized.get('model_id'),
                            current_model_name=normalized.get('name'),
                            cache_used=normalized.get('_cache_used', []),
                            has_error=bool(normalized.get('_errors')),
                            error_message=', '.join(normalized.get('_errors', []))[:200] if normalized.get('_errors') else None
                        )
                    
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
            
            # Complete progress tracking
            if progress_tracker:
                progress_tracker.complete()
            
            print(f"\n✅ Successfully extracted {len(normalized_models)} models")
            return normalized_models
            
        except Exception as e:
            if progress_tracker:
                progress_tracker.error(f"Error fetching models: {str(e)[:200]}")
            print(f"Error with new API: {e}")
            print("Trying fallback API format...")
            try:
                models = fetch_from_old_api(self.fetch_data)
                print(f"✓ Fetched {len(models)} models from fal.ai (fallback)\n")
                
                # Update progress with actual count
                if progress_tracker:
                    progress_tracker.state['total_models'] = len(models)
                    progress_tracker.state['status'] = 'extracting'
                    progress_tracker._save()
                
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
    
    def extract_model(self, model_id: str) -> Dict[str, Any]:
        """
        Extract a single model by ID for re-extraction.
        This is more efficient than extract() because it:
        1. Fetches the model list (lightweight, no schemas)
        2. Finds the target model
        3. Only fetches schema for THAT specific model
        
        Args:
            model_id: The model ID to extract
            
        Returns:
            Normalized model data dictionary or empty dict if not found
        """
        print(f"Re-extracting model: {model_id}")
        
        try:
            # Fetch all models to find the specific one (no schemas yet)
            models = fetch_from_new_api(self.base_url, self.fetch_data)
            
            # Find the model by ID
            model_data = None
            for model in models:
                if model.get('id') == model_id:
                    model_data = model
                    break
            
            if not model_data:
                print(f"Model {model_id} not found in API")
                return {}
            
            # Normalize with schema fetching enabled for this single model
            # The _normalize_fal_model method will fetch the schema from the model's URL
            normalized = self._normalize_fal_model(model_data)
            print(f"✅ Successfully re-extracted {normalized.get('name', model_id)}")
            return normalized
            
        except Exception as e:
            print(f"Error re-extracting model {model_id}: {e}")
            return {}
    
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
        
        # Extract name - use title but append variant suffix if model_id has additional path segments
        base_name = raw_data.get("title") or raw_data.get("name") or ""
        
        # Check if model_id has variant suffix (e.g., /edit, /lora, /pro)
        # Split model_id: "fal-ai/flux-2-pro/edit" -> ["fal-ai", "flux-2-pro", "edit"]
        id_parts = model_id.split("/")
        
        if len(id_parts) > 2 and base_name:
            # Has variant - append it to name
            variant = id_parts[-1].replace("-", " ").title()
            name = f"{base_name} ({variant})"
        elif base_name:
            # No variant - use title as is
            name = base_name
        else:
            # Fallback: use last part of model_id
            name = id_parts[-1].replace("-", " ").title()
        
        # Extract description
        description = (
            raw_data.get("shortDescription") or 
            raw_data.get("description") or 
            ""
        )
        
        # Map category to model type
        category = raw_data.get("category", "other")
        model_type = self._map_category_to_type(category)
        
        # Extract tags - include category as a tag for filtering
        tags = raw_data.get("tags") or []
        if isinstance(tags, list):
            tags = tags.copy()  # Don't modify original
        else:
            tags = []
        
        # Add category as a tag if not already present (for multi-capability filtering)
        if category and category not in tags:
            tags.append(category)
        
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
        
        # Initialize schema and playground variables before LLM call
        input_schema = None
        output_schema = None
        playground_data = None
        
        # Extract group info
        group = raw_data.get("group", {})
        if not isinstance(group, dict):
            group = {}
        
        # Fetch OpenAPI schema if enabled
        openapi_schema = None
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
        
        # Parse playground pricing
        if playground_data and isinstance(playground_data, dict):
            if playground_data.get('price') and playground_data.get('billing_unit'):
                try:
                    playground_pricing = {
                        'price': float(playground_data['price']),
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
                            llm_context = {
                                'name': name,
                                'pricing_info': raw_data.get("pricingInfoOverride") or raw_data.get("pricingInfo") or "",
                                'creditsRequired': raw_data.get("creditsRequired"),
                                'model_type': model_type,
                                'tags': tags,
                                'raw_metadata': raw_data,
                                'input_schema': input_schema,
                                'output_schema': output_schema,
                                'playground_data': playground_data,
                            }
                            print(f"  📤 Sending to LLM: name='{name}', pricing='{llm_context['pricing_info']}', model_type='{model_type}'")
                            pricing_details = extract_pricing_with_llm(llm_context)
                            print(f"  📤 Sending to LLM: name='{name}', pricing='{llm_context['pricing_info']}', model_type='{model_type}'")
                            pricing_details = extract_pricing_with_llm(llm_context)
                            
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
                        print(f"    ⚠️  Rate limited, skipping LLM extraction for {name}")
                    elif "server error" in error_msg.lower():
                        print(f"    ⚠️  OpenRouter server error, skipping {name}")
                    elif "timeout" in error_msg.lower():
                        print(f"    ⚠️  LLM request timed out for {name} (60s timeout)")
                    else:
                        print(f"    ⚠️  LLM pricing extraction failed for {name}: {error_msg[:150]}")
                    errors.append(f'llm: {error_msg[:50]}')
            else:
                print(f"    📦 Using cached LLM data for {name}")
        
        # Use LLM-extracted data to enhance pricing if available
        if pricing_details:
            # Use LLM cost_per_call if we don't have one or it's zero
            if (cost_per_call == 0.0 or cost_per_call is None) and pricing_details.get('cost_per_call'):
                try:
                    llm_cost = float(pricing_details['cost_per_call'])
                    if llm_cost > 0:
                        cost_per_call = llm_cost
                except (ValueError, TypeError):
                    pass
            
            # Use LLM credits_required if we don't have one
            if credits_required is None and pricing_details.get('credits_required'):
                try:
                    llm_credits = float(pricing_details['credits_required'])
                    if llm_credits > 0:
                        credits_required = llm_credits
                except (ValueError, TypeError):
                    pass
            
            # LLM overrides model_type - normalizes types across all sources
            if pricing_details.get('model_type'):
                llm_model_type = pricing_details.get('model_type')
                print(f"  🔍 LLM returned model_type: {llm_model_type} (current: {model_type})")
                
                # Accept only standardized base types from LLM (as defined in prompt)
                valid_types = ['text-generation', 'image-generation', 'video-generation',
                              'audio-generation', 'embeddings', 'code-generation', 'chat',
                              'completion', 'rerank', 'moderation', 'other']
                
                if llm_model_type in valid_types:
                    model_type = llm_model_type
                    print(f"  ✅ Updated model_type to: {model_type}")
                else:
                    print(f"  ⚠️  LLM returned invalid type '{llm_model_type}', ignoring (expected one of: {', '.join(valid_types)})")
            
            # LLM overrides category - better categorization
            if pricing_details.get('category'):
                category = pricing_details.get('category')
                print(f"  🔍 LLM returned category: {category}")
            
            # Enhance tags - merge LLM-suggested tags with existing ones
            if pricing_details.get('tags') and isinstance(pricing_details['tags'], list):
                for llm_tag in pricing_details['tags']:
                    if llm_tag and llm_tag not in tags:
                        tags.append(llm_tag)
                print(f"  🏷️  Final tags after LLM merge: {tags}")
            
            # Use LLM description if we don't have a good one
            if not description and pricing_details.get('description'):
                description = pricing_details['description']
        
        # Build pricing formula based on available data
        # Standard: 1.0 MP = 1,000,000 pixels (1000x1000)
        STANDARD_IMAGE_MP = 1.0
        pricing_formula = None
        pricing_info_text = raw_data.get("pricingInfoOverride") or raw_data.get("pricingInfo") or ""
        
        # LLM formula takes priority if available
        if pricing_details and pricing_details.get('pricing_formula'):
            pricing_formula = pricing_details['pricing_formula']
        elif credits_required and credits_required > 0:
            pricing_formula = f"{credits_required} credits per call ($0.01 per credit)"
        elif pricing_info_text:
            # Parse pricing info to create formula
            if "/MP" in pricing_info_text or "per megapixel" in pricing_info_text.lower():
                import re
                match = re.search(r'\\$(\\d+\\.?\\d*)', pricing_info_text)
                if match:
                    price = float(match.group(1))
                    pricing_formula = f"megapixels * ${price:.4f}/MP (1000x1000 = {STANDARD_IMAGE_MP} MP standard)"
            elif "$" in pricing_info_text:
                pricing_formula = pricing_info_text
        
        # Build normalized data
        # Convert LLM cost values to floats safely
        input_cost_per_unit = None
        output_cost_per_unit = None
        if pricing_details:
            if pricing_details.get('input_cost_per_unit'):
                try:
                    input_cost_per_unit = float(pricing_details['input_cost_per_unit'])
                except (ValueError, TypeError):
                    pass
            if pricing_details.get('output_cost_per_unit'):
                try:
                    output_cost_per_unit = float(pricing_details['output_cost_per_unit'])
                except (ValueError, TypeError):
                    pass
        
        normalized = {
            'model_id': model_id,
            'name': name,
            'description': description,
            'model_type': model_type,
            'cost_per_call': cost_per_call,
            'credits_required': credits_required,
            'pricing_info': raw_data.get("pricingInfoOverride") or raw_data.get("pricingInfo") or "",
            'thumbnail_url': raw_data.get("thumbnailUrl") or "",
            'tags': tags,  # Include category as tag for multi-capability filtering
            'category': category,
            'input_schema': input_schema,
            'output_schema': output_schema,
            # Playground pricing (fallback source)
            'playground_pricing': playground_pricing,
            # LLM extracted pricing details
            'pricing_type': pricing_details.get('pricing_type') if pricing_details else None,
            'pricing_formula': pricing_formula or (pricing_details.get('pricing_formula') if pricing_details else None),
            'pricing_variables': {
                'tags': tags,  # For LLM context and filtering
                'category': category,
                **(pricing_details.get('pricing_variables') or {})
            } if pricing_details or tags else None,
            'input_cost_per_unit': input_cost_per_unit,
            'output_cost_per_unit': output_cost_per_unit,
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
        Standard assumptions:
        - Image: 1.0 MP = 1,000,000 pixels (1000x1000 pixels)
        
        Args:
            model_data: Raw model data
            
        Returns:
            Cost per call as float
        """
        STANDARD_IMAGE_MP = 1.0  # 1 megapixel = 1000x1000 pixels
        
        credits_required = model_data.get("creditsRequired")
        if credits_required is not None:
            try:
                credits = float(credits_required)
                if credits > 0:
                    # Assuming 1 credit = $0.01 (adjust as needed)
                    return credits * 0.01
            except (ValueError, TypeError):
                pass
        
        # Parse pricing info for cost calculation
        pricing_info = model_data.get("pricingInfoOverride") or model_data.get("pricingInfo") or ""
        if pricing_info and "$" in pricing_info:
            # Try to extract dollar amount
            import re
            # Match patterns like "$0.025" or "$0.025/MP"
            match = re.search(r'\$(\d+\.?\d*)', pricing_info)
            if match:
                try:
                    price = float(match.group(1))
                    # For image models with /MP pricing, use 1024x1024 standard (1.05 MP)
                    if "/MP" in pricing_info or "per megapixel" in pricing_info.lower():
                        return price * STANDARD_IMAGE_MP
                    return price
                except (ValueError, TypeError):
                    pass
        
        return 0.0

