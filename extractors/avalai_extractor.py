"""
AvalAI Extractor

Extracts AI model pricing and metadata from AvalAI documentation.
AvalAI is an Iranian AI service aggregator providing access to multiple AI providers
(OpenAI, Google, Anthropic, XAI, Meta, Mistral, Alibaba, DeepSeek, etc.)
with pricing in USD per million tokens.

Pricing sources:
- Model details: https://docs.avalai.ir/fa/models/model-details.md
- Pricing page: https://docs.avalai.ir/fa/pricing.md
"""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from extractors.base import BaseExtractor
from ai_cost_manager.cache import cache_manager
from ai_cost_manager.progress_tracker import ProgressTracker
from playwright.sync_api import sync_playwright
from tqdm import tqdm


class AvalAIExtractor(BaseExtractor):
    """
    Extractor for AvalAI models.
    
    Fetches model data from AvalAI's pricing documentation pages and normalizes it.
    Supports multiple model types: chat, image, video, audio, embedding, rerank, etc.
    """
    
    def __init__(
        self, 
        source_url: str = "https://docs.avalai.ir/fa/pricing.md",
        model_details_url: str = "https://docs.avalai.ir/fa/models/model-details.md",
        use_llm: bool = False, 
        fetch_schemas: bool = False
    ):
        super().__init__(source_url)
        self.pricing_url = source_url
        self.model_details_url = model_details_url
        self.use_llm = use_llm
        self.fetch_schemas = fetch_schemas
        self.force_refresh = False
    
    def get_source_info(self) -> Dict[str, str]:
        """Get information about the AvalAI source."""
        return {
            'name': 'AvalAI',
            'base_url': 'https://avalai.ir',
            'api_url': 'https://api.avalai.ir',
            'description': 'Iranian AI service aggregator with transparent pricing (no markup) and 20,000 IRR free credit'
        }
    
    def extract(self, progress_tracker: ProgressTracker = None) -> List[Dict[str, Any]]:
        """
        Extract all models from AvalAI pricing pages.
        
        Args:
            progress_tracker: Optional progress tracker for UI updates
        
        Returns:
            List of normalized model data dictionaries
        """
        # Initialize progress tracking
        if progress_tracker:
            progress_tracker.start(
                total_models=0,
                options={
                    'use_llm': self.use_llm,
                    'fetch_schemas': self.fetch_schemas,
                    'force_refresh': self.force_refresh
                }
            )
        
        print("Fetching models from AvalAI pricing pages...")
        
        try:
            # Fetch pricing data from both URLs
            pricing_html = self._fetch_page_html(self.pricing_url)
            # model_details_html = self._fetch_page_html(self.model_details_url)  # Reserved for future use
            
            # Parse pricing tables
            models = self._parse_pricing_tables(pricing_html)
            
            print(f"✓ Parsed {len(models)} models from AvalAI pricing pages\n")
            
            # Update progress with actual count
            if progress_tracker:
                progress_tracker.state['total_models'] = len(models)
                progress_tracker._save()
            
            # Normalize with progress bar
            normalized_models = []
            
            with tqdm(total=len(models), desc="Extracting models", unit=" model") as pbar:
                for i, model in enumerate(models):
                    normalized = self._normalize_avalai_model(model, i + 1, len(models))
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
            print(f"Error extracting from AvalAI: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_model(self, model_id: str) -> Dict[str, Any]:
        """
        Extract a single model by ID for re-extraction.
        
        Args:
            model_id: The model ID to extract
            
        Returns:
            Normalized model data dictionary or empty dict if not found
        """
        print(f"Re-extracting model: {model_id}")
        
        try:
            # Fetch pricing data
            pricing_html = self._fetch_page_html(self.pricing_url)
            models = self._parse_pricing_tables(pricing_html)
            
            # Find the model by ID
            model_data = None
            for model in models:
                if model.get('model_id') == model_id:
                    model_data = model
                    break
            
            if not model_data:
                print(f"Model {model_id} not found in AvalAI pricing")
                return {}
            
            # Normalize
            normalized = self._normalize_avalai_model(model_data)
            print(f"✅ Successfully re-extracted {normalized.get('name', model_id)}")
            return normalized
            
        except Exception as e:
            print(f"Error re-extracting model {model_id}: {e}")
            return {}
    
    def _fetch_page_html(self, url: str) -> str:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Content as string (may be markdown or HTML)
        """
        # For .md files, fetch the raw content directly
        if url.endswith('.md'):
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        else:
            # For other URLs, use Playwright to handle JavaScript rendering
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until='networkidle', timeout=30000)
                html = page.content()
                browser.close()
                return html
    
    def _parse_pricing_tables(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse pricing tables from markdown content.
        
        The pricing page contains multiple tables for different model categories.
        
        Args:
            content: Markdown content from pricing page
            
        Returns:
            List of raw model data dictionaries
        """
        models = []
        
        # Split content into lines
        lines = content.split('\n')
        
        current_category = 'other'
        in_table = False
        headers = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect category from headings
            if line.startswith('###'):
                heading_text = line.replace('#', '').strip()
                current_category = self._heading_to_category(heading_text)
                in_table = False
                continue
            
            # Detect table start (header row with |)
            if '|' in line and not in_table:
                # This is a header row
                headers = [h.strip() for h in line.split('|')[1:-1]]  # Remove empty first/last elements
                in_table = True
                continue
            
            # Skip separator row (|---|---|)
            if in_table and '---' in line:
                continue
            
            # Parse table data rows
            if in_table and '|' in line and '---' not in line:
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if len(cells) >= 2:
                    # Extract model data from this row
                    model_data = self._parse_markdown_row(cells, headers, current_category)
                    if model_data:
                        models.extend(model_data)
            
            # End of table
            if in_table and '|' not in line:
                in_table = False
                headers = []
        
        return models
    
    def _heading_to_category(self, heading_text: str) -> str:
        """
        Determine the category from a heading text.
        
        Args:
            heading_text: Heading text
            
        Returns:
            Category string
        """
        heading_lower = heading_text.lower()
        
        if 'chat' in heading_lower or 'completion' in heading_lower or 'چت' in heading_lower:
            return 'chat'
        elif 'cloudflare' in heading_lower:
            return 'cloudflare'
        elif 'image' in heading_lower or 'تصویر' in heading_lower:
            return 'image'
        elif 'video' in heading_lower or 'ویدیو' in heading_lower:
            return 'video'
        elif 'embedding' in heading_lower or 'تعبیه' in heading_lower:
            return 'embedding'
        elif 'rerank' in heading_lower or 'مرتب' in heading_lower:
            return 'rerank'
        elif 'audio' in heading_lower or 'speech' in heading_lower or 'صوتی' in heading_lower:
            return 'audio'
        elif 'moderation' in heading_lower or 'نظارت' in heading_lower:
            return 'moderation'
        elif 'fine-tun' in heading_lower or 'تنظیم دقیق' in heading_lower:
            return 'fine-tuning'
        elif 'search' in heading_lower or 'جستجو' in heading_lower:
            return 'search'
        elif 'ocr' in heading_lower:
            return 'ocr'
        
        return 'other'
    
    def _parse_markdown_row(self, cells: List[str], headers: List[str], category: str) -> List[Dict[str, Any]]:
        """
        Parse a markdown table row into model data.
        
        Args:
            cells: List of cell strings
            headers: List of header names
            category: Model category
            
        Returns:
            List of model data dictionaries (may be multiple if row has variants)
        """
        if len(cells) < 2:
            return []
        
        # First cell contains model ID(s) - may have multiple variants separated by <br>
        model_ids_cell = cells[0]
        
        # Skip rows that are actually table headers or descriptions in Persian/English
        # These contain pricing unit descriptions, not model IDs
        skip_keywords = [
            'هزینه',  # Cost/Price in Persian
            'unit',
            'irt',
            'usd/usdt',
            'محاسبه',  # Calculation in Persian
            'نرخ تبدیل',  # Exchange rate in Persian
            'تومان',  # Toman (Iranian currency)
            'دلار',  # Dollar in Persian
        ]
        
        model_ids_lower = model_ids_cell.lower()
        if any(keyword in model_ids_lower for keyword in skip_keywords):
            return []
        
        model_ids = self._extract_model_ids_from_markdown(model_ids_cell)
        
        if not model_ids:
            return []
        
        # Extract provider from second cell
        provider = cells[1] if len(cells) > 1 else 'Unknown'
        
        # Extract pricing based on category
        pricing_data = self._extract_pricing_by_category_markdown(cells, headers, category)
        
        # Create model entry for each variant
        models = []
        for model_id in model_ids:
            model_data = {
                'model_id': model_id,
                'provider': provider,
                'category': category,
                **pricing_data,
                'raw_row': cells
            }
            models.append(model_data)
        
        return models
    
    def _extract_model_ids_from_markdown(self, cell_text: str) -> List[str]:
        """
        Extract model IDs from a markdown cell.
        Model IDs are in backticks, may be multiple per cell separated by <br>.
        
        Args:
            cell_text: Cell text
            
        Returns:
            List of model ID strings
        """
        model_ids = []
        
        # Split by <br> and process each part
        parts = cell_text.split('<br>')
        
        for part in parts:
            # Extract text within backticks
            if '`' in part:
                # Find all text within backticks
                import re
                matches = re.findall(r'`([^`]+)`', part)
                for match in matches:
                    model_id = match.strip()
                    # Skip deprecated models (wrapped in ~~)
                    if model_id and not model_id.startswith('~~'):
                        # Clean up strikethrough
                        model_id = model_id.replace('~~', '')
                        if model_id:
                            model_ids.append(model_id)
        
        return model_ids
    
    def _extract_pricing_by_category_markdown(self, cells: List[str], headers: List[str], category: str) -> Dict[str, Any]:
        """
        Extract pricing data from markdown cells based on model category.
        
        Args:
            cells: List of cell strings
            headers: List of header names
            category: Model category
            
        Returns:
            Dictionary of pricing data
        """
        pricing = {}
        
        # Common fields across all categories
        if category in ['chat', 'cloudflare', 'fine-tuning']:
            # Text models: input/output pricing per million tokens
            pricing['input_price_per_million'] = self._parse_price(cells[2]) if len(cells) > 2 else None
            pricing['cached_input_price_per_million'] = self._parse_price(cells[3]) if len(cells) > 3 else None
            pricing['output_price_per_million'] = self._parse_price(cells[4]) if len(cells) > 4 else None
            pricing['notes'] = cells[5] if len(cells) > 5 else ''
            
        elif category == 'image':
            # Image models: per image or per megapixel
            pricing['price_per_image'] = self._parse_price(cells[2]) if len(cells) > 2 else None
            pricing['notes'] = cells[3] if len(cells) > 3 else ''
            
        elif category == 'video':
            # Video models: per second
            pricing['resolution'] = cells[2] if len(cells) > 2 else ''
            pricing['price_per_second'] = self._parse_price(cells[3]) if len(cells) > 3 else None
            pricing['notes'] = cells[4] if len(cells) > 4 else ''
            
        elif category == 'embedding':
            # Embedding models: per million tokens
            pricing['input_price_per_million'] = self._parse_price(cells[2]) if len(cells) > 2 else None
            pricing['cached_input_price_per_million'] = self._parse_price(cells[3]) if len(cells) > 3 else None
            
        elif category == 'rerank':
            # Reranking models: per query
            pricing['price_per_query'] = self._parse_price(cells[2]) if len(cells) > 2 else None
            
        elif category == 'audio':
            # Audio models: complex pricing
            pricing['input_price_per_million'] = self._parse_price(cells[2]) if len(cells) > 2 else None
            pricing['cached_input_price_per_million'] = self._parse_price(cells[3]) if len(cells) > 3 else None
            pricing['output_price_per_million'] = self._parse_price(cells[4]) if len(cells) > 4 else None
            pricing['special_cost'] = cells[5] if len(cells) > 5 else ''
            pricing['notes'] = cells[6] if len(cells) > 6 else ''
            
        elif category == 'search':
            # Search models: per query
            pricing['price_per_query'] = self._parse_price(cells[2]) if len(cells) > 2 else None
            pricing['notes'] = cells[3] if len(cells) > 3 else ''
            
        elif category == 'ocr':
            # OCR models: per page
            pricing['price_per_page'] = self._parse_price(cells[2]) if len(cells) > 2 else None
            pricing['notes'] = cells[3] if len(cells) > 3 else ''
        
        return pricing
    
    def _parse_price(self, price_str: str) -> Optional[float]:
        """
        Parse a price string to float.
        Handles formats like: "$1.25", "1.25", "$0.075", "-", "رایگان"
        
        Args:
            price_str: Price string
            
        Returns:
            Price as float or None
        """
        if not price_str or price_str in ['-', 'رایگان', 'free', '']:
            return None
        
        # Remove currency symbols and whitespace
        price_str = price_str.replace('$', '').replace('،', '').strip()
        
        # Extract first number (may have multiple prices in notes)
        match = re.search(r'\d+\.?\d*', price_str)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        
        return None
    
    def _normalize_avalai_model(self, raw_data: Dict[str, Any], current: int = 0, total: int = 0) -> Dict[str, Any]:
        """
        Normalize AvalAI model data to standard format.
        
        Args:
            raw_data: Raw model data from parsing
            current: Current model number (for progress)
            total: Total number of models (for progress)
            
        Returns:
            Normalized model data dictionary
        """
        model_id = raw_data.get('model_id', '')
        provider = raw_data.get('provider', '')
        category = raw_data.get('category', 'other')
        
        # Track cache usage and errors
        cache_used = []
        errors = []
        
        # Save raw data to cache
        cache_manager.save_raw_data("avalai", model_id, raw_data)
        last_raw_fetched = datetime.utcnow()
        
        # Build name from model_id and provider
        name = f"{provider} {model_id}" if provider != 'Unknown' else model_id
        
        # Map category to model type
        model_type = self._map_category_to_type(category)
        
        # Extract pricing
        input_price = raw_data.get('input_price_per_million')
        cached_input_price = raw_data.get('cached_input_price_per_million')
        output_price = raw_data.get('output_price_per_million')
        price_per_image = raw_data.get('price_per_image')
        price_per_second = raw_data.get('price_per_second')
        price_per_query = raw_data.get('price_per_query')
        price_per_page = raw_data.get('price_per_page')
        
        # Calculate cost per call based on model type
        cost_per_call = 0.0
        pricing_formula = None
        cost_unit = 'token'
        
        if input_price and output_price:
            # Text models: estimate 1000 input + 1000 output tokens
            input_cost_per_token = input_price / 1_000_000
            output_cost_per_token = output_price / 1_000_000
            cost_per_call = (input_cost_per_token * 1000) + (output_cost_per_token * 1000)
            pricing_formula = f"(input_tokens * ${input_cost_per_token:.6f}) + (output_tokens * ${output_cost_per_token:.6f})"
            
        elif price_per_image:
            # Image models
            cost_per_call = price_per_image
            pricing_formula = f"${price_per_image:.4f} per image"
            cost_unit = 'image'
            
        elif price_per_second:
            # Video models: estimate 5 seconds
            cost_per_call = price_per_second * 5
            pricing_formula = f"seconds * ${price_per_second:.4f}/sec"
            cost_unit = 'second'
            
        elif price_per_query:
            # Search/rerank models
            cost_per_call = price_per_query
            pricing_formula = f"${price_per_query:.4f} per query"
            cost_unit = 'query'
            
        elif price_per_page:
            # OCR models
            cost_per_call = price_per_page
            pricing_formula = f"${price_per_page:.4f} per page"
            cost_unit = 'page'
        
        # Build pricing info string
        pricing_parts = []
        if input_price:
            pricing_parts.append(f"Input: ${input_price}/M tokens")
        if cached_input_price:
            pricing_parts.append(f"Cached: ${cached_input_price}/M tokens")
        if output_price:
            pricing_parts.append(f"Output: ${output_price}/M tokens")
        if price_per_image:
            pricing_parts.append(f"${price_per_image}/image")
        if price_per_second:
            pricing_parts.append(f"${price_per_second}/second")
        if price_per_query:
            pricing_parts.append(f"${price_per_query}/query")
        if price_per_page:
            pricing_parts.append(f"${price_per_page}/page")
        
        pricing_info = " | ".join(pricing_parts) if pricing_parts else "Contact provider"
        
        # Build preliminary tags for LLM context
        tags = [provider, category]
        if model_type and model_type not in tags:
            tags.append(model_type)
        if raw_data.get('notes'):
            notes_lower = raw_data['notes'].lower()
            if 'deprecated' in notes_lower or 'منسوخ' in notes_lower:
                tags.append('deprecated')
            if 'experimental' in notes_lower or 'آزمایشی' in notes_lower:
                tags.append('experimental')
            if 'preview' in notes_lower:
                tags.append('preview')
        
        # Extract pricing details with LLM if enabled
        llm_extracted = None
        if self.use_llm and pricing_info:
            # Try loading from cache first (unless force refresh)
            if not self.force_refresh:
                llm_extracted = cache_manager.load_llm_extraction("avalai", model_id)
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
                                'cached_input_price_per_million': cached_input_price,
                                'price_per_image': price_per_image,
                                'price_per_second': price_per_second,
                                'price_per_query': price_per_query,
                                'price_per_page': price_per_page,
                                'tags': tags,
                                'raw_metadata': raw_data,
                            }
                            llm_extracted = extract_pricing_with_llm(llm_context)
                            
                            if llm_extracted:
                                cache_manager.save_llm_extraction("avalai", model_id, llm_extracted)
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
                                errors.append(f'llm: {error_msg[:50]}')
                                break
                                
                except Exception as e:
                    errors.append(f'llm: {str(e)[:50]}')
            else:
                cache_used.append('llm')
        
        # Initialize overrides
        pricing_type_override = None
        category_override = None
        
        # Use LLM-extracted data to enhance pricing if available
        if llm_extracted:
            # LLM overrides model_type - normalizes types across all sources
            if llm_extracted.get('model_type'):
                llm_model_type = llm_extracted.get('model_type')
                
                # Accept only standardized base types from LLM
                valid_types = ['text-generation', 'image-generation', 'video-generation',
                              'audio-generation', 'embeddings', 'code-generation', 'chat',
                              'completion', 'rerank', 'moderation', 'other']
                
                if llm_model_type in valid_types:
                    model_type = llm_model_type
            
            # LLM overrides category
            if llm_extracted.get('category'):
                category_override = llm_extracted.get('category')
            
            # Override pricing_formula if LLM has a better one
            if llm_extracted.get('pricing_formula'):
                pricing_formula = llm_extracted['pricing_formula']
            
            # Override pricing_type if LLM detected a better one
            if llm_extracted.get('pricing_type'):
                llm_pricing_type = llm_extracted.get('pricing_type')
                if llm_pricing_type in ['per_video', 'per_image', 'per_minute', 'per_call', 'per_token', 
                                        'per_megapixel', 'per_second', 'hourly', 'fixed', 'per_query', 'per_page']:
                    pricing_type_override = llm_pricing_type
            
            # Use LLM-extracted cost_per_call if available and reasonable
            if llm_extracted.get('cost_per_call'):
                try:
                    llm_cost = float(llm_extracted['cost_per_call'])
                    if llm_cost > 0:
                        cost_per_call = llm_cost
                except (ValueError, TypeError):
                    pass
            
            # Use LLM-extracted credits if available
            if llm_extracted.get('credits_required'):
                try:
                    credits = float(llm_extracted['credits_required'])
                    if credits > 0 and cost_per_call == 0:
                        # Assuming 1 credit = $0.01
                        cost_per_call = credits * 0.01
                except (ValueError, TypeError):
                    pass
        
        # Rebuild tags AFTER LLM overrides to reflect final model_type
        tags = [provider, category_override if category_override else category]
        if model_type and model_type not in tags:
            tags.append(model_type)
        if raw_data.get('notes'):
            notes_lower = raw_data['notes'].lower()
            if 'deprecated' in notes_lower or 'منسوخ' in notes_lower:
                tags.append('deprecated')
            if 'experimental' in notes_lower or 'آزمایشی' in notes_lower:
                tags.append('experimental')
            if 'preview' in notes_lower:
                tags.append('preview')
        
        # Merge LLM-suggested tags with base tags
        if llm_extracted and llm_extracted.get('tags') and isinstance(llm_extracted['tags'], list):
            for llm_tag in llm_extracted['tags']:
                if llm_tag and llm_tag not in tags:
                    tags.append(llm_tag)
        
        # Merge LLM-suggested tags with base tags
        if llm_extracted and llm_extracted.get('tags') and isinstance(llm_extracted['tags'], list):
            for llm_tag in llm_extracted['tags']:
                if llm_tag and llm_tag not in tags:
                    tags.append(llm_tag)
        
        # Build description
        description = f"AvalAI {provider} model - {model_type}"
        if llm_extracted and llm_extracted.get('description'):
            description = llm_extracted['description']
        elif raw_data.get('notes'):
            description += f" - {raw_data.get('notes', '')[:100]}"
        
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
        if not input_cost_per_unit and input_price:
            input_cost_per_unit = input_price / 1_000_000
        if not output_cost_per_unit and output_price:
            output_cost_per_unit = output_price / 1_000_000
        
        # Build pricing_variables
        pricing_variables = {
            'input_price_per_million': input_price,
            'cached_input_price_per_million': cached_input_price,
            'output_price_per_million': output_price,
            'price_per_image': price_per_image,
            'price_per_second': price_per_second,
            'price_per_query': price_per_query,
            'price_per_page': price_per_page,
            'resolution': raw_data.get('resolution'),
            'special_cost': raw_data.get('special_cost'),
            'tags': tags,  # For LLM context and filtering
        }
        
        # Merge LLM pricing variables if available
        if llm_extracted and llm_extracted.get('pricing_variables'):
            pricing_variables.update(llm_extracted['pricing_variables'])
        
        # Build normalized data
        normalized = {
            'model_id': model_id,
            'name': name,
            'description': description,
            'model_type': model_type,
            'cost_per_call': cost_per_call,
            'credits_required': llm_extracted.get('credits_required') if llm_extracted else None,
            'pricing_info': pricing_info,
            'thumbnail_url': '',
            'tags': tags,
            'category': category_override if category_override else category,
            'input_schema': None,
            'output_schema': None,
            'pricing_type': (
                pricing_type_override if pricing_type_override else
                'per_token' if (input_price or output_price) else
                'per_image' if price_per_image else
                'per_second' if price_per_second else
                'per_query' if price_per_query else
                'per_page' if price_per_page else
                'per_call'
            ),
            'pricing_formula': pricing_formula,
            'pricing_variables': pricing_variables,
            'input_cost_per_unit': input_cost_per_unit,
            'output_cost_per_unit': output_cost_per_unit,
            'cost_unit': llm_extracted.get('cost_unit') if llm_extracted else cost_unit,
            'llm_extracted': llm_extracted if llm_extracted else None,
            'raw_metadata': raw_data,
            'last_raw_fetched': last_raw_fetched,
            'last_schema_fetched': None,
            'last_playground_fetched': None,
            '_cache_used': cache_used,
            '_errors': errors,
        }
        
        return normalized
    
    def _map_category_to_type(self, category: str) -> str:
        """
        Map AvalAI category to standard model type.
        
        Args:
            category: AvalAI category string
            
        Returns:
            Standardized model type
        """
        type_map = {
            'chat': 'text-generation',
            'cloudflare': 'text-generation',
            'image': 'image-generation',
            'video': 'video-generation',
            'audio': 'audio-generation',
            'embedding': 'embeddings',
            'rerank': 'reranking',
            'moderation': 'moderation',
            'fine-tuning': 'fine-tuning',
            'search': 'search',
            'ocr': 'ocr',
        }
        return type_map.get(category, 'other')
