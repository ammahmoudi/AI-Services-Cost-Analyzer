"""
MetisAI Extractor

Extracts AI model pricing and metadata from MetisAI API.
MetisAI is an Iranian AI service aggregator providing access to multiple AI providers
(OpenAI, Google, Anthropic, Cohere, Meta, Mistral, DeepSeek, Grok, etc.)
with pricing in USD.

API endpoint: https://api.metisai.ir/api/v1/meta/providers/pricing
"""
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from extractors.base import BaseExtractor
from ai_cost_manager.cache import cache_manager
from ai_cost_manager.progress_tracker import ProgressTracker
from tqdm import tqdm
import time


class MetisAIExtractor(BaseExtractor):
    """
    Extractor for MetisAI models.
    
    Fetches model data from MetisAI's pricing API and normalizes it.
    Supports multiple model types: llm, embedding, imaginator, chunking, reranking, audiotor, transcriptor, videonator, searching.
    """
    
    def __init__(
        self, 
        source_url: str = "https://api.metisai.ir/api/v1/meta/providers/pricing",
        use_llm: bool = False, 
        fetch_schemas: bool = False
    ):
        super().__init__(source_url)
        self.api_url = source_url
        self.use_llm = use_llm
        self.fetch_schemas = fetch_schemas
        self.force_refresh = False
    
    def get_source_info(self) -> Dict[str, str]:
        """Get information about the MetisAI source."""
        return {
            'name': 'MetisAI',
            'base_url': 'https://metisai.ir',
            'api_url': 'https://api.metisai.ir',
            'description': 'Iranian AI service aggregator with comprehensive pricing API'
        }
    
    def extract(self, progress_tracker: ProgressTracker = None) -> List[Dict[str, Any]]:
        """
        Extract all models from MetisAI API.
        
        Args:
            progress_tracker: Optional progress tracker for UI updates
            
        Returns:
            List of normalized model dictionaries
        """
        # Initialize progress tracking
        if progress_tracker:
            progress_tracker.start(
                total_models=0,  # Will update once we know the count
                options={
                    'use_llm': self.use_llm,
                    'fetch_schemas': self.fetch_schemas,
                    'force_refresh': self.force_refresh
                }
            )
        
        # Fetch pricing data
        raw_data = self._fetch_pricing_data()
        
        if not raw_data:
            if progress_tracker:
                progress_tracker.error('Failed to fetch pricing data from MetisAI API')
            return []
        
        # Parse all models from API response
        all_models = []
        
        # The API returns data grouped by type (llm, embedding, imaginator, etc.)
        for model_type, models_list in raw_data.items():
            if not isinstance(models_list, list):
                continue
            
            for model_data in models_list:
                all_models.append({
                    **model_data,
                    'api_type': model_type  # Store the API type category
                })
        
        if progress_tracker:
            progress_tracker.start(
                total_models=len(all_models),
                options={
                    'use_llm': self.use_llm,
                    'fetch_schemas': self.fetch_schemas,
                    'force_refresh': self.force_refresh
                }
            )
        
        # Normalize models with progress tracking
        normalized_models = []
        
        with tqdm(total=len(all_models), desc="Extracting models", unit=" model", disable=progress_tracker is not None) as pbar:
            for idx, raw_model in enumerate(all_models):
                try:
                    normalized = self._normalize_metis_model(raw_model)
                    if normalized:
                        normalized_models.append(normalized)
                    
                    if progress_tracker:
                        cache_used_list = []
                        cache_dict = normalized.get('_cache_used', {}) if normalized else {}
                        if isinstance(cache_dict, dict):
                            cache_used_list = [k for k, v in cache_dict.items() if v]
                        
                        progress_tracker.update(
                            processed=idx + 1,
                            current_model_id=normalized.get('model_id', 'Unknown') if normalized else 'Unknown',
                            current_model_name=normalized.get('name', 'Unknown') if normalized else 'Unknown',
                            cache_used=cache_used_list,
                            has_error=bool(normalized.get('_errors')) if normalized else False,
                            error_message=', '.join(normalized.get('_errors', []))[:200] if normalized and normalized.get('_errors') else ''
                        )
                    
                    pbar.update(1)
                    
                except Exception as e:
                    if progress_tracker:
                        progress_tracker.error(f"Error normalizing model: {str(e)}")
                    print(f"Error normalizing model {raw_model.get('model', 'unknown')}: {e}")
        
        if progress_tracker:
            progress_tracker.complete()
        
        return normalized_models
    
    def extract_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract a specific model by ID.
        
        Args:
            model_id: The model identifier to extract
            
        Returns:
            Normalized model dictionary or None if not found
        """
        # Fetch all models and find the specific one
        all_models = self.extract()
        
        for model in all_models:
            if model.get('model_id') == model_id:
                return model
        
        return None
    
    def _fetch_pricing_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch pricing data from MetisAI API.
        
        Returns:
            Dict with pricing data grouped by type or None on error
        """
        try:
            response = requests.get(self.api_url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching MetisAI pricing data: {e}")
            return None
    
    def _normalize_metis_model(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize a MetisAI model to standard format.
        
        Args:
            raw_data: Raw model data from API
            
        Returns:
            Normalized model dictionary
        """
        # Extract basic fields
        provider = raw_data.get('name', 'unknown')
        model_id = raw_data.get('model', '')
        api_type = raw_data.get('api_type', 'unknown')
        currency = raw_data.get('currency', 'USD')
        
        # Create full model identifier
        full_model_id = f"{provider}/{model_id}" if provider != model_id else model_id
        
        # Extract pricing fields
        fixed_call_income = raw_data.get('fixedCallIncome', 0.0)
        input_token_unit_income = raw_data.get('inputTokenUnitIncome', 0.0)
        output_token_unit_income = raw_data.get('outputTokenUnitIncome', 0.0)
        per_second_income = raw_data.get('perSecondIncome', 0.0)
        
        # Map API type to model type
        model_type = self._map_api_type_to_model_type(api_type)
        
        # Build tags
        tags = ['metis', provider, api_type, model_type]
        
        # Determine pricing structure and build formula
        pricing_type = 'per_call'
        pricing_formula = None
        cost_per_call = None
        pricing_info = []
        
        # Pricing in USD per token (need to convert to per million)
        input_price_per_million = input_token_unit_income * 1_000_000 if input_token_unit_income > 0 else None
        output_price_per_million = output_token_unit_income * 1_000_000 if output_token_unit_income > 0 else None
        
        if input_token_unit_income > 0 or output_token_unit_income > 0:
            # Token-based pricing
            pricing_type = 'per_token'
            
            if input_price_per_million and output_price_per_million:
                pricing_formula = f"(input_tokens * ${input_token_unit_income}) + (output_tokens * ${output_token_unit_income})"
                cost_per_call = (input_price_per_million + output_price_per_million) / 1000  # Cost for 1K tokens
                pricing_info.append(f"${input_price_per_million:.6f} per 1M input tokens")
                pricing_info.append(f"${output_price_per_million:.6f} per 1M output tokens")
            elif input_price_per_million:
                pricing_formula = f"(input_tokens * ${input_token_unit_income})"
                cost_per_call = input_price_per_million / 1000
                pricing_info.append(f"${input_price_per_million:.6f} per 1M tokens")
            elif output_price_per_million:
                pricing_formula = f"(output_tokens * ${output_token_unit_income})"
                cost_per_call = output_price_per_million / 1000
                pricing_info.append(f"${output_price_per_million:.6f} per 1M output tokens")
        
        elif per_second_income > 0:
            # Per-second pricing (video, audio generation)
            pricing_type = 'per_second'
            pricing_formula = f"(duration_seconds * ${per_second_income})"
            cost_per_call = per_second_income  # Cost for 1 second
            pricing_info.append(f"${per_second_income:.6f} per second")
        
        elif fixed_call_income > 0:
            # Fixed per-call pricing (images, searches, etc.)
            pricing_type = 'per_call'
            pricing_formula = f"${fixed_call_income} per call"
            cost_per_call = fixed_call_income
            pricing_info.append(f"${fixed_call_income:.4f} per call")
        
        # LLM extraction if enabled
        llm_extracted = None
        cache_used = {'raw': False, 'schema': False, 'playground': False, 'llm': False}
        errors = []
        
        if self.use_llm:
            # Try to load from cache first
            cached_llm = cache_manager.load_llm_extraction("metis", full_model_id)
            
            if cached_llm:
                llm_extracted = cached_llm
                cache_used['llm'] = True
            else:
                # Perform LLM extraction with retry logic
                from ai_cost_manager.llm_extractor import extract_pricing_with_llm
                
                llm_context = {
                    'provider': provider,
                    'model_id': model_id,
                    'api_type': api_type,
                    'fixed_call_income': fixed_call_income,
                    'input_token_unit_income': input_token_unit_income,
                    'output_token_unit_income': output_token_unit_income,
                    'per_second_income': per_second_income,
                    'currency': currency,
                    'pricing_info': ', '.join(pricing_info),
                    'model_type': model_type,
                    'tags': tags
                }
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        llm_result = extract_pricing_with_llm(llm_context)
                        if llm_result:
                            llm_extracted = llm_result
                            # Save to cache
                            cache_manager.save_llm_extraction("metis", full_model_id, llm_result)
                            break
                    except Exception as e:
                        error_msg = str(e)
                        if 'rate limit' in error_msg.lower() and attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            time.sleep(wait_time)
                            continue
                        errors.append(f"LLM extraction failed: {error_msg}")
                        break
        
        # Apply LLM overrides if available
        model_type_override = None
        category_override = None
        pricing_type_override = None
        
        if llm_extracted:
            # Override model_type if LLM provides a valid one
            if llm_extracted.get('model_type'):
                suggested_type = llm_extracted['model_type'].lower()
                standard_types = ['chat', 'image', 'video', 'audio', 'embedding', 'rerank', 'search', 'other']
                if suggested_type in standard_types:
                    model_type_override = suggested_type
                    model_type = model_type_override
            
            # Override category if provided
            if llm_extracted.get('category'):
                category_override = llm_extracted['category']
            
            # Override pricing_type if provided
            if llm_extracted.get('pricing_type'):
                pricing_type_override = llm_extracted['pricing_type']
            
            # Override pricing_formula if provided
            if llm_extracted.get('pricing_formula'):
                pricing_formula = llm_extracted['pricing_formula']
            
            # Override cost_per_call if provided
            if llm_extracted.get('cost_per_call'):
                try:
                    cost_per_call = float(llm_extracted['cost_per_call'])
                except (ValueError, TypeError):
                    pass
        
        # Rebuild tags after overrides
        tags = ['metis', provider, api_type, model_type]
        if category_override:
            tags.append(category_override)
        
        # Merge LLM-suggested tags
        if llm_extracted and llm_extracted.get('tags') and isinstance(llm_extracted['tags'], list):
            for llm_tag in llm_extracted['tags']:
                if llm_tag and llm_tag not in tags:
                    tags.append(llm_tag)
        
        # Build description
        description = f"MetisAI {provider} model - {model_type}"
        if llm_extracted and llm_extracted.get('description'):
            description = llm_extracted['description']
        
        # Convert LLM cost values to floats safely
        input_cost_per_unit = None
        output_cost_per_unit = None
        if llm_extracted:
            if llm_extracted.get('input_cost_per_unit'):
                try:
                    input_cost_per_unit = float(llm_extracted['input_cost_per_unit'])
                except (ValueError, TypeError):
                    pass
            if llm_extracted.get('output_cost_per_unit'):
                try:
                    output_cost_per_unit = float(llm_extracted['output_cost_per_unit'])
                except (ValueError, TypeError):
                    pass
        
        # Use parsed pricing if LLM didn't provide
        if not input_cost_per_unit and input_token_unit_income:
            input_cost_per_unit = input_token_unit_income
        if not output_cost_per_unit and output_token_unit_income:
            output_cost_per_unit = output_token_unit_income
        
        # Determine cost_unit
        if pricing_type == 'per_token':
            cost_unit = 'token'
        elif pricing_type == 'per_second':
            cost_unit = 'second'
        elif pricing_type == 'per_call':
            cost_unit = 'call'
        else:
            cost_unit = 'unit'
        
        # Build pricing_variables
        pricing_variables = {
            'fixed_call_income': fixed_call_income,
            'input_token_unit_income': input_token_unit_income,
            'output_token_unit_income': output_token_unit_income,
            'per_second_income': per_second_income,
            'input_price_per_million': input_price_per_million,
            'output_price_per_million': output_price_per_million,
            'currency': currency,
            'provider': provider,
            'api_type': api_type,
            'tags': tags,
        }
        
        # Merge LLM pricing variables if available
        if llm_extracted and llm_extracted.get('pricing_variables'):
            pricing_variables.update(llm_extracted['pricing_variables'])
        
        # Build normalized data
        normalized = {
            'model_id': full_model_id,
            'name': f"{provider} {model_id}",
            'description': description,
            'model_type': model_type,
            'cost_per_call': cost_per_call,
            'credits_required': llm_extracted.get('credits_required') if llm_extracted else None,
            'pricing_info': ' | '.join(pricing_info) if pricing_info else 'Contact provider',
            'thumbnail_url': '',
            'tags': tags,
            'category': category_override if category_override else api_type,
            'input_schema': None,
            'output_schema': None,
            'pricing_type': (
                pricing_type_override if pricing_type_override else pricing_type
            ),
            'pricing_formula': pricing_formula,
            'pricing_variables': pricing_variables,
            'input_cost_per_unit': input_cost_per_unit,
            'output_cost_per_unit': output_cost_per_unit,
            'cost_unit': llm_extracted.get('cost_unit') if llm_extracted else cost_unit,
            'llm_extracted': llm_extracted if llm_extracted else None,
            'raw_metadata': raw_data,
            'last_raw_fetched': datetime.utcnow(),
            'last_schema_fetched': None,
            'last_playground_fetched': None,
            '_cache_used': cache_used,
            '_errors': errors,
        }
        
        return normalized
    
    def _map_api_type_to_model_type(self, api_type: str) -> str:
        """
        Map MetisAI API type to standard model type.
        
        Args:
            api_type: The type from API (llm, embedding, imaginator, etc.)
            
        Returns:
            Standard model type
        """
        type_mapping = {
            'llm': 'chat',
            'embedding': 'embedding',
            'imaginator': 'image',
            'videonator': 'video',
            'audiotor': 'audio',
            'transcriptor': 'audio',
            'chunking': 'other',
            'reranking': 'rerank',
            'searching': 'search',
        }
        
        return type_mapping.get(api_type.lower(), 'other')
