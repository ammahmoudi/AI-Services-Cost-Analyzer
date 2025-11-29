"""
Together AI Extractor

Extracts AI model pricing and metadata from Together AI API.
Supports both standard and batch API pricing (50% discount).
Falls back to web scraping when API key is not available.
"""
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from extractors.base import BaseExtractor
from ai_cost_manager.cache import cache_manager
from ai_cost_manager.progress_tracker import ProgressTracker
from extractors.together_utils.pricing_scraper import scrape_pricing_page, merge_scraped_with_api, parse_model_name_to_id, fetch_model_page
from tqdm import tqdm


class TogetherAIExtractor(BaseExtractor):
    """
    Extractor for Together AI models.
    
    Fetches model data from the Together AI API and normalizes it.
    Includes both standard and batch API pricing.
    """
    
    def __init__(self, api_key: Optional[str] = None, source_url: str = "https://api.together.xyz/v1/models", use_llm: bool = False, fetch_schemas: bool = False):
        super().__init__(source_url)
        # Try to get API key from database if not provided
        if not api_key:
            try:
                from ai_cost_manager.database import get_session, close_session
                from ai_cost_manager.models import ExtractorAPIKey
                
                session = get_session()
                try:
                    key_config = session.query(ExtractorAPIKey).filter_by(
                        extractor_name='together',
                        is_active=True
                    ).first()
                    if key_config:
                        api_key = key_config.api_key
                finally:
                    close_session()
            except Exception:
                pass  # No database key found, will use scraping fallback
        
        self.api_key = api_key
        self.base_url = "https://api.together.xyz/v1"
        self.use_llm = use_llm
        self.fetch_schemas = fetch_schemas
        self.force_refresh = False  # Set to True to bypass cache
    
    def get_source_info(self) -> Dict[str, str]:
        """Get information about the Together AI source."""
        return {
            'name': 'Together AI',
            'base_url': 'https://www.together.ai',
            'api_url': self.base_url,
            'description': 'Together AI provides open-source AI models with competitive pricing and batch API support'
        }
    
    def extract(self, progress_tracker: ProgressTracker = None) -> List[Dict[str, Any]]:
        """
        Extract all models from Together AI API.
        
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
        
        print("Fetching models from Together AI...")
        
        try:
            # Fetch models from API
            models = self._fetch_models()
            
            if not models:
                if progress_tracker:
                    progress_tracker.error("No models found from Together AI")
                print("No models found from Together AI")
                return []
            
            print(f"OK: Fetched {len(models)} models from Together AI")
            
            # Filter out invalid models (hardware specs, duplicates, etc.)
            invalid_prefixes = ['1x-', '17b-', '70-', 'per-', 'nvidia-', 'llama-4-scoutllama', 'llama-4-maverickllama', 
                               'deepseek-r1deepseek', 'qwen3-235b-a22bqwen3']
            valid_models = []
            for model in models:
                model_id = model.get('id', '')
                # Skip models with invalid IDs
                if any(model_id.startswith(prefix) for prefix in invalid_prefixes):
                    continue
                # Skip models with malformed compound IDs
                if model_id and '-' in model_id:
                    # Check if it's a hardware spec (e.g., "1x-l40s-48gb")
                    parts = model_id.split('-')
                    if len(parts) >= 3 and parts[0].endswith('x'):
                        continue
                valid_models.append(model)
            
            if len(valid_models) < len(models):
                print(f"   Filtered out {len(models) - len(valid_models)} invalid model entries")
            
            models = valid_models
            print(f"OK: Processing {len(models)} valid models\n")
            
            # Update progress with actual count
            if progress_tracker:
                progress_tracker.state['total_models'] = len(models)
                progress_tracker._save()
            
            # Fetch schemas if enabled
            if self.fetch_schemas:
                self._fetch_all_schemas(models)
            
            # Extract with progress bar
            from tqdm import tqdm
            normalized_models = []
            
            with tqdm(total=len(models), desc="Extracting models", unit=" model") as pbar:
                for i, model in enumerate(models):
                    normalized = self._normalize_together_model(model, i + 1, len(models))
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
                        status_parts.append('[cached]')
                    if normalized.get('_errors'):
                        status_parts.append('[warn]')
                    if normalized.get('batch_pricing') and normalized['batch_pricing'].get('supported'):
                        status_parts.append('50%')
                    
                    status = ' '.join(status_parts) if status_parts else ''
                    model_name = normalized.get('name', 'Unknown')
                    display_text = f"{model_name[:40]} {status}" if status else model_name[:45]
                    
                    pbar.set_postfix_str(display_text)
                    pbar.update(1)
            
            # Complete progress tracking
            if progress_tracker:
                progress_tracker.complete()
            
            print(f"\nOK: Successfully extracted {len(normalized_models)} models")
            return normalized_models
            
        except Exception as e:
            if progress_tracker:
                progress_tracker.error(str(e))
            print(f"Error extracting from Together AI: {e}")
            return []
    
    def extract_model(self, model_id: str) -> Dict[str, Any]:
        """
        Extract a single model by ID for re-extraction.
        This is more efficient than extract() because it:
        1. Fetches the model list (lightweight, no schemas)
        2. Finds the target model
        3. Only fetches schema for THAT specific model (if fetch_schemas=True)
        
        Args:
            model_id: The model ID to extract
            
        Returns:
            Normalized model data dictionary or None if not found
        """
        print(f"Re-extracting model: {model_id}")
        
        try:
            # Fetch all models to find the specific one (no schemas yet)
            models = self._fetch_models()
            
            if not models:
                print("No models found from Together AI")
                return {}
            
            # Find the model by ID
            model_data = None
            for model in models:
                if model.get('id') == model_id:
                    model_data = model
                    break
            
            if not model_data:
                print(f"Model {model_id} not found in API")
                return {}
            
            # Normalize with schema fetching for this single model
            # The _normalize_together_model method will fetch schema if fetch_schemas=True
            normalized = self._normalize_together_model(model_data)
            print(f"✅ Successfully re-extracted {normalized.get('name', model_id)}")
            return normalized
            
        except Exception as e:
            print(f"Error re-extracting model {model_id}: {e}")
            return {}
    
    def _fetch_models(self) -> List[Dict[str, Any]]:
        """
        Fetch models from Together AI API or scrape pricing page as fallback.
        
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
            
            # API returns a list of models directly
            models = response.json()
            
            # Try to enhance with scraped data
            try:
                scraped_data = scrape_pricing_page()
                if scraped_data:
                    total_scraped = sum(len(models) for models in scraped_data.values())
                    print(f"   OK: Scraped {total_scraped} pricing entries from website")
                    models = merge_scraped_with_api(models, scraped_data)
            except Exception as scrape_error:
                print(f"   WARNING: Could not scrape pricing page: {scrape_error}")
            
            return models
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"WARNING: API authentication failed (401). Falling back to web scraping...")
                print("   Note: Scraped data may have less detail than API data.\n")
                return self._fetch_from_scraper()
            else:
                print(f"Error fetching Together AI models: {e}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Together AI models: {e}")
            print("WARNING: Falling back to web scraping...\n")
            return self._fetch_from_scraper()
    
    def _fetch_from_scraper(self) -> List[Dict[str, Any]]:
        """
        Fetch models by scraping the pricing page.
        
        Returns:
            List of model data dictionaries
        """
        try:
            scraped_data = scrape_pricing_page()
            
            if not scraped_data:
                return []
            
            # Convert scraped data to API-like format
            converted_models = []
            
            for category, models in scraped_data.items():
                for scraped in models:
                    model = {
                        'id': parse_model_name_to_id(scraped['name']),
                        'display_name': scraped['name'],
                        'type': self._category_to_type(category),
                        'pricing': {},
                        'link': scraped.get('url', ''),
                    }
                    
                    # Add pricing based on category
                    if scraped.get('input_price_per_million'):
                        model['pricing']['input'] = scraped['input_price_per_million']
                    if scraped.get('output_price_per_million'):
                        model['pricing']['output'] = scraped['output_price_per_million']
                    if scraped.get('price_per_mp'):
                        model['pricing']['per_mp'] = scraped['price_per_mp']
                    if scraped.get('price_per_video'):
                        model['pricing']['per_video'] = scraped['price_per_video']
                    if scraped.get('price_per_million_chars'):
                        model['pricing']['per_million_chars'] = scraped['price_per_million_chars']
                    if scraped.get('price_per_minute'):
                        model['pricing']['per_minute'] = scraped['price_per_minute']
                    if scraped.get('price_per_million_tokens'):
                        model['pricing']['per_million_tokens'] = scraped['price_per_million_tokens']
                    
                    # Add metadata
                    if scraped.get('images_per_dollar'):
                        model['images_per_dollar'] = scraped['images_per_dollar']
                    if scraped.get('default_steps'):
                        model['default_steps'] = scraped['default_steps']
                    if scraped.get('supports_batch') is not None:
                        model['supports_batch'] = scraped['supports_batch']
                    
                    converted_models.append(model)
            
            return converted_models
            
        except Exception as e:
            print(f"Error scraping pricing page: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _category_to_type(self, category: str) -> str:
        """Map category to model type."""
        type_map = {
            'text': 'chat',
            'image': 'image',
            'video': 'video',
            'audio': 'audio',
            'embedding': 'embedding',
            'rerank': 'rerank',
            'moderation': 'moderation',
        }
        return type_map.get(category, 'chat')
    
    def _fetch_all_schemas(self, models: List[Dict[str, Any]]) -> None:
        """
        Fetch schema data for all models from their detail pages.
        
        Args:
            models: List of model data dictionaries
        """
        print("Fetching schemas from model detail pages...")
        
        with tqdm(total=len(models), desc="Fetching schemas", unit=" schema") as pbar:
            for model in models:
                model_id = model.get('id') or model.get('model_id')
                model_url = model.get('link')
                
                # Validate model ID
                if not model_id or model_id.startswith('/') or ' ' in model_id:
                    pbar.set_postfix_str(f"[skip]")
                    pbar.update(1)
                    continue
                
                if not model_url:
                    # Construct URL from model ID
                    model_url = f"https://www.together.ai/models/{model_id}"
                elif not model_url.startswith('http'):
                    # Fix malformed URLs
                    if model_url.startswith('/'):
                        model_url = f"https://www.together.ai{model_url}"
                    else:
                        model_url = f"https://www.together.ai/{model_url}"
                
                # Check cache first
                if not self.force_refresh:
                    cached_schema = cache_manager.load_schema("together.ai", model_id)
                    if cached_schema:
                        pbar.set_postfix_str(f"[cached]")
                        pbar.update(1)
                        continue
                
                # Fetch schema from model page
                try:
                    schema_data = fetch_model_page(model_url)
                    if schema_data:
                        # Save to cache as schema.json
                        cache_manager.save_schema("together.ai", model_id, schema_data)
                        pbar.set_postfix_str(f"OK")
                    else:
                        pbar.set_postfix_str(f"[warn]")
                except Exception as e:
                    pbar.set_postfix_str(f"[error]")
                
                pbar.update(1)
        
        print("OK: Schema fetching complete\n")
    
    def _normalize_together_model(self, raw_data: Dict[str, Any], current: int = 0, total: int = 0) -> Dict[str, Any]:
        """
        Normalize Together AI model data to standard format.
        
        Args:
            raw_data: Raw model data from Together AI API
            current: Current model number (for progress)
            total: Total number of models (for progress)
            
        Returns:
            Normalized model data dictionary
        """
        model_id = raw_data.get('id', '')
        
        # Track cache usage and errors
        cache_used = []
        errors = []
        
        # Save raw data to cache with timestamp
        cache_manager.save_raw_data("together.ai", model_id, raw_data)
        last_raw_fetched = datetime.utcnow()
        
        # Load schema data from cache
        schema_data = None
        last_schema_fetched = None
        if self.fetch_schemas:
            schema_data = cache_manager.load_schema("together.ai", model_id)
            if schema_data:
                cache_used.append('schema')
                last_schema_fetched = datetime.utcnow()
        
        name = raw_data.get('display_name') or model_id
        
        # Extract model type
        model_type = self._map_type_to_category(raw_data.get('type', 'chat'))
        
        # Extract pricing information from API
        pricing = raw_data.get('pricing', {})
        input_price = pricing.get('input', 0)  # Price per million tokens
        output_price = pricing.get('output', 0)  # Price per million tokens
        hourly_price = pricing.get('hourly', 0)
        base_price = pricing.get('base', 0)
        finetune_price = pricing.get('finetune', 0)
        
        # Try to enhance pricing from schema data
        if schema_data:
            schema_pricing = schema_data.get('pricing')
            
            # If schema has clean pricing and API doesn't, use schema pricing
            if schema_pricing and not (input_price or output_price):
                # Schema pricing is already parsed (e.g., "$0.20 Input / $0.60 Output")
                if schema_data.get('input_price'):
                    input_price = schema_data['input_price']
                if schema_data.get('output_price'):
                    output_price = schema_data['output_price']
                if schema_data.get('price_per_mp'):
                    price_per_mp = schema_data['price_per_mp']
                if schema_data.get('price_per_video'):
                    price_per_video = schema_data['price_per_video']
        
        # Image pricing
        price_per_mp = pricing.get('per_mp', 0)
        images_per_dollar = raw_data.get('images_per_dollar', 0)
        default_steps = raw_data.get('default_steps', 0)
        
        # Video pricing
        price_per_video = pricing.get('per_video', 0)
        
        # Audio pricing
        price_per_minute = pricing.get('per_minute', 0)
        price_per_million_chars = pricing.get('per_million_chars', 0)
        
        # Other pricing
        price_per_million_tokens = pricing.get('per_million_tokens', 0)
        
        # Check if model explicitly supports batch API
        # According to Together AI: "Batch Inference API now supports all serverless models" at 50% cost
        # But image, video, and some other models don't support batch
        supports_batch_explicit = raw_data.get('supports_batch')
        if supports_batch_explicit is None:
            # Auto-detect: only text models with input/output pricing support batch
            supports_batch = bool(input_price or output_price) and model_type == 'text-generation'
        else:
            supports_batch = supports_batch_explicit
        
        batch_discount = 0.5  # 50% discount
        
        # Convert from per million to per token
        input_cost_per_token = input_price / 1_000_000 if input_price else None
        output_cost_per_token = output_price / 1_000_000 if output_price else None
        
        # Calculate batch pricing
        batch_input_price = None
        batch_output_price = None
        batch_input_cost_per_token = None
        batch_output_cost_per_token = None
        
        if supports_batch:
            if input_price:
                batch_input_price = input_price * batch_discount
                batch_input_cost_per_token = batch_input_price / 1_000_000
            if output_price:
                batch_output_price = output_price * batch_discount
                batch_output_cost_per_token = batch_output_price / 1_000_000
        
        # Build pricing info string
        pricing_parts = []
        if input_price:
            pricing_parts.append(f"Input: ${input_price}/M")
        if output_price:
            pricing_parts.append(f"Output: ${output_price}/M")
        if price_per_mp:
            pricing_parts.append(f"${price_per_mp}/MP")
            if images_per_dollar:
                pricing_parts.append(f"{images_per_dollar} img/$1")
            if default_steps:
                pricing_parts.append(f"{default_steps} steps")
        if price_per_video:
            pricing_parts.append(f"${price_per_video}/video")
        if price_per_minute:
            pricing_parts.append(f"${price_per_minute}/min")
        if price_per_million_chars:
            pricing_parts.append(f"${price_per_million_chars}/M chars")
        if price_per_million_tokens:
            pricing_parts.append(f"${price_per_million_tokens}/M tokens")
        if supports_batch:
            pricing_parts.append(f"Batch: 50% off")
        if hourly_price:
            pricing_parts.append(f"Hourly: ${hourly_price}")
        
        pricing_info = " | ".join(pricing_parts) if pricing_parts else "Pricing available"
        
        # Build pricing formula
        pricing_formula = None
        if input_cost_per_token and output_cost_per_token:
            pricing_formula = f"(input_tokens * ${input_cost_per_token:.6f}) + (output_tokens * ${output_cost_per_token:.6f})"
        elif price_per_mp:
            # Image models: formula based on megapixels
            pricing_formula = f"megapixels * ${price_per_mp:.4f}/MP"
        elif price_per_video:
            pricing_formula = f"${price_per_video:.4f} per video"
        elif price_per_minute:
            pricing_formula = f"minutes * ${price_per_minute:.4f}/min"
        elif price_per_million_tokens:
            pricing_formula = f"(tokens / 1,000,000) * ${price_per_million_tokens:.4f}"
        elif hourly_price:
            pricing_formula = f"${hourly_price:.2f} per hour"
        
        # Calculate average cost per call based on model type
        # Standard assumptions for realistic usage:
        # - Text: 1000 input + 1000 output tokens (typical conversation)
        # - Image: 1.0 MP = 1,000,000 pixels (1000x1000, matches Together.ai's 40 img/$1 at $0.025/MP)
        # - Video: 1 video generation
        # - Audio: 1 minute
        # - Embeddings: 1000 tokens
        STANDARD_IMAGE_MP = 1.0  # 1 megapixel = 1000x1000 pixels
        
        cost_per_call = 0.0
        if input_cost_per_token and output_cost_per_token:
            # Text models: estimate 1000 input + 1000 output tokens (typical conversation)
            cost_per_call = (input_cost_per_token * 1000) + (output_cost_per_token * 1000)
        elif price_per_video:
            # Video models: cost per video generation
            cost_per_call = price_per_video
        elif price_per_mp:
            # Image models: standard 1024x1024 image (1.05 MP)
            cost_per_call = price_per_mp * STANDARD_IMAGE_MP
        elif price_per_minute:
            # Audio models: estimate 1 minute
            cost_per_call = price_per_minute
        elif price_per_million_tokens:
            # Embedding/rerank models: estimate 1000 tokens
            cost_per_call = (price_per_million_tokens / 1_000_000) * 1000
        elif hourly_price:
            # Hourly models: cost for 1 hour
            cost_per_call = hourly_price
        
        # Build preliminary tags for LLM context
        tags = []
        if raw_data.get('organization'):
            tags.append(raw_data['organization'])
        if raw_data.get('license'):
            tags.append(f"license:{raw_data['license']}")
        if supports_batch:
            tags.append('batch-api')
        if model_type:
            tags.append(model_type)
        
        # Extract LLM pricing if enabled
        llm_extracted = None
        if self.use_llm and pricing_info:
            # Try loading from cache first (unless force refresh)
            if not self.force_refresh:
                llm_extracted = cache_manager.load_llm_extraction("together.ai", model_id)
                if llm_extracted:
                    cache_used.append('llm')
            
            if not llm_extracted:
                try:
                    from ai_cost_manager.llm_extractor import extract_pricing_with_llm
                    import time
                    
                    # Retry logic for rate limiting
                    max_retries = 3
                    retry_delay = 2
                    
                    for attempt in range(max_retries):
                        try:
                            llm_context = {
                                'name': name,
                                'pricing_info': pricing_info,
                                'model_type': model_type,
                                'input_price_per_million': input_price,
                                'output_price_per_million': output_price,
                                'batch_discount': '50%' if supports_batch else None,
                                'tags': tags,
                                'raw_metadata': raw_data,
                            }
                            print(f"  📤 Sending to LLM: name='{name}', pricing='{pricing_info}', model_type='{model_type}'")
                            llm_extracted = extract_pricing_with_llm(llm_context)
                            
                            if llm_extracted:
                                cache_manager.save_llm_extraction("together.ai", model_id, llm_extracted)
                            break
                            
                        except Exception as retry_error:
                            error_msg = str(retry_error)
                            if "Rate limit" in error_msg and attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            elif "server error" in error_msg.lower() and attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                # Log the error but don't fail the entire extraction
                                print(f"  ⚠️  LLM extraction failed: {error_msg[:150]}")
                                errors.append(f'llm: {error_msg[:50]}')
                                break
                                
                except Exception as e:
                    errors.append(f'llm: {str(e)[:50]}')
        
        # Initialize overrides outside conditional
        pricing_type_override = None
        category_override = None
        
        # Use LLM-extracted data to enhance pricing if available
        if llm_extracted:
            # LLM extraction overrides calculated values for better accuracy
            
            # Override model_type - LLM normalizes types across all sources
            if llm_extracted.get('model_type'):
                llm_model_type = llm_extracted.get('model_type')
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
            
            # Override category - LLM can provide better categorization
            if llm_extracted.get('category'):
                category_override = llm_extracted.get('category')
                print(f"  🔍 LLM returned category: {category_override}")
            
            # Override pricing_formula if LLM has a better one
            if llm_extracted.get('pricing_formula'):
                pricing_formula = llm_extracted['pricing_formula']
            
            # Override pricing_type if LLM detected a better one
            if llm_extracted.get('pricing_type'):
                llm_pricing_type = llm_extracted.get('pricing_type')
                # Trust LLM for these specific types
                if llm_pricing_type in ['per_video', 'per_image', 'per_minute', 'per_call', 'per_token', 
                                        'per_megapixel', 'per_second', 'hourly', 'fixed']:
                    # LLM knows better, set a flag to use it later
                    pricing_type_override = llm_pricing_type
            
            # Use LLM-extracted cost_per_call if available and seems reasonable
            if llm_extracted.get('cost_per_call'):
                try:
                    llm_cost = float(llm_extracted['cost_per_call'])
                    # Override if LLM cost is positive (let LLM override even if we have a value)
                    if llm_cost > 0:
                        cost_per_call = llm_cost
                except (ValueError, TypeError):
                    pass
            
            # Use LLM-extracted credits if available
            if llm_extracted.get('credits_required'):
                try:
                    credits = float(llm_extracted['credits_required'])
                    if credits > 0:
                        # Store credits in normalized data
                        pass  # Will be added below
                except (ValueError, TypeError):
                    pass
        
        # Extract tags AFTER LLM overrides so they reflect final model_type
        tags = []
        if raw_data.get('organization'):
            tags.append(raw_data['organization'])
        if raw_data.get('license'):
            tags.append(f"license:{raw_data['license']}")
        if supports_batch:
            tags.append('batch-api')
        
        # Add final model_type as tag for filtering
        if model_type and model_type not in tags:
            tags.append(model_type)
            print(f"  🏷️  Added model_type '{model_type}' to tags")
        
        # Merge LLM-suggested tags with base tags
        if llm_extracted and llm_extracted.get('tags') and isinstance(llm_extracted['tags'], list):
            for llm_tag in llm_extracted['tags']:
                if llm_tag and llm_tag not in tags:
                    tags.append(llm_tag)
            print(f"  🏷️  Final tags after LLM merge: {tags}")
        
        # Normalize the data
        normalized = {
            'model_id': model_id,
            'name': name,
            'description': llm_extracted.get('description', f"Together AI model - {model_type}") if llm_extracted else f"Together AI model - {model_type}",
            'model_type': model_type,
            'cost_per_call': cost_per_call,
            'credits_required': llm_extracted.get('credits_required') if llm_extracted else None,
            'pricing_info': pricing_info,
            'thumbnail_url': raw_data.get('link', ''),
            'tags': tags,
            'category': category_override if category_override else raw_data.get('type', model_type),
            'input_schema': None,
            'output_schema': None,
            # Pricing details for database - determine based on model type
            'pricing_type': (
                pricing_type_override if pricing_type_override else
                'per_token' if (input_cost_per_token or output_cost_per_token) else
                'per_video' if price_per_video else
                'per_image' if price_per_mp else
                'per_minute' if price_per_minute else
                'per_token' if price_per_million_tokens else
                'hourly' if hourly_price else
                'per_call'
            ),
            'pricing_formula': pricing_formula,
            'pricing_variables': {
                'input_tokens': input_cost_per_token,
                'output_tokens': output_cost_per_token,
                'hourly': hourly_price,
                'base': base_price,
                'finetune': finetune_price,
                'price_per_mp': price_per_mp,
                'price_per_video': price_per_video,
                'price_per_minute': price_per_minute,
                'price_per_million_chars': price_per_million_chars,
                'price_per_million_tokens': price_per_million_tokens,
                'images_per_dollar': images_per_dollar,
                'default_steps': default_steps,
            },
            'input_cost_per_unit': input_cost_per_token or (price_per_million_tokens / 1_000_000 if price_per_million_tokens else None),
            'output_cost_per_unit': output_cost_per_token,
            'cost_unit': (
                'token' if (input_cost_per_token or output_cost_per_token or price_per_million_tokens) else
                'video' if price_per_video else
                'megapixel' if price_per_mp else
                'minute' if price_per_minute else
                'hour' if hourly_price else
                'call'
            ),
            'llm_extracted': llm_extracted if llm_extracted else None,
            # Batch API pricing (50% discount for all serverless models)
            'batch_pricing': {
                'supported': supports_batch,
                'discount_percentage': 50 if supports_batch else 0,
                'input_price_per_million': batch_input_price,
                'output_price_per_million': batch_output_price,
                'input_cost_per_token': batch_input_cost_per_token,
                'output_cost_per_token': batch_output_cost_per_token,
            } if supports_batch else None,
            'raw_metadata': {
                **raw_data,
                'batch_supported': supports_batch,
                'context_length': raw_data.get('context_length'),
            },
            # Timestamps for data fetching
            'last_raw_fetched': last_raw_fetched,
            'last_schema_fetched': last_schema_fetched,
            'last_playground_fetched': None,
            # Schema data from model detail page
            'schema_data': schema_data,
            # Internal tracking (not saved to DB)
            '_cache_used': cache_used,
            '_errors': errors,
        }
        
        return normalized
    
    def _map_type_to_category(self, model_type: str) -> str:
        """
        Map Together AI model type to standard category.
        
        Args:
            model_type: Together AI model type
            
        Returns:
            Standardized model type
        """
        type_map = {
            'chat': 'text-generation',
            'language': 'text-generation',
            'code': 'code-generation',
            'image': 'image-generation',
            'embedding': 'embeddings',
            'moderation': 'moderation',
            'rerank': 'reranking',
        }
        return type_map.get(model_type, 'other')
