"""
Runware API Pricing Extractor

Extracts pricing information from Runware by:
1. Fetching all models from authenticated models page (my.runware.ai/models/all)
2. For each model, visiting playground to extract approximate cost
3. Falling back to public pricing page if authentication fails
"""

import re
import requests
import json
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

from ai_cost_manager.progress_tracker import ProgressTracker
from ai_cost_manager.llm_extractor import extract_pricing_with_llm
from ai_cost_manager.model_types import VALID_MODEL_TYPES, get_valid_types_string


class RunwareExtractor:
    """
    Extractor for Runware API pricing information.
    
    Scrapes the public pricing page HTML to extract:
    - Image generation models (FLUX, Midjourney, Stable Diffusion, etc.)
    - Video generation models (Sora, Kling, Google Veo, etc.)
    - AI tools (background removal, upscaling, etc.)
    """

    def __init__(
        self,
        source_url: str = "https://runware.ai/pricing",
        fetch_schemas: bool = False,
        use_llm: bool = False
    ):
        """
        Initialize the Runware extractor.

        Args:
            source_url: URL to Runware pricing page
            fetch_schemas: Not used for Runware (no schema support)
            use_llm: Not used for Runware (no LLM models)
        """
        self.source_url = source_url
        self.fetch_schemas = fetch_schemas
        self.use_llm = use_llm
        
        # Provider mapping for better categorization
        self.provider_map = {
            'flux': 'Black Forest Labs',
            'midjourney': 'Midjourney',
            'sora': 'OpenAI',
            'kling': 'KlingAI',
            'google': 'Google',
            'veo': 'Google',
            'imagen': 'Google',
            'minimax': 'MiniMax',
            'hailuo': 'MiniMax',
            'vidu': 'Vidu',
            'ltx': 'Lightricks',
            'seedance': 'ByteDance',
            'wan': 'ByteDance',
            'pixverse': 'PixVerse',
            'bria': 'Bria',
            'qwen': 'Alibaba',
            'seedream': 'Tencent',
            'hidream': 'Alibaba',
            'riverflow': 'ImagineArt',
            'nano banana': 'Google',
            'gemini': 'Google',
            'sd': 'Stability AI',
            'sdxl': 'Stability AI',
            'runway': 'Runway',
        }

    def extract(self, progress_tracker: Optional[ProgressTracker] = None) -> List[Dict]:
        """
        Extract all pricing information from Runware.
        
        Tries authenticated extraction first, falls back to public pricing page.

        Args:
            progress_tracker: Optional progress tracker for UI updates

        Returns:
            List of model dictionaries
        """
        all_models = []
        
        # Try authenticated extraction first
        try:
            from ai_cost_manager.database import get_session
            from ai_cost_manager.models import AuthSettings
            
            session_db = get_session()
            auth = session_db.query(AuthSettings).filter_by(
                source_name='runware',
                is_active=True
            ).first()
            
            if auth:
                # Check for username/password auth
                if hasattr(auth, 'username') and hasattr(auth, 'password'):
                    username_str = str(auth.username) if auth.username is not None else None
                    password_str = str(auth.password) if auth.password is not None else None
                    
                    if username_str and password_str:
                        print(f"Found Runware credentials (user: {username_str}), attempting authenticated extraction...")
                        all_models = self._extract_with_auth(
                            cookies_json=None,
                            progress_tracker=progress_tracker,
                            username=username_str,
                            password=password_str
                        )
                        
                        if all_models:
                            print(f"✅ Successfully extracted {len(all_models)} models via authenticated method")
                            return all_models
                        else:
                            print("⚠️ Authenticated extraction returned no models")
                # Legacy cookie-based auth (deprecated)
                elif hasattr(auth, 'cookies') and auth.cookies is not None:
                    cookies_str = str(auth.cookies)
                    print("Found Runware cookies (deprecated), attempting authenticated extraction...")
                    all_models = self._extract_with_auth(cookies_str, progress_tracker)
                    
                    if all_models:
                        print(f"✅ Successfully extracted {len(all_models)} models via authenticated method")
                        return all_models
                else:
                    print("Runware auth found but no credentials/cookies configured")
            else:
                print("No Runware authentication found, using default credentials")
                
        except Exception as e:
            print(f"Error checking authentication: {e}")
            print("Falling back to public pricing page")

        # Fallback to public pricing page
        try:
            print(f"Fetching Runware pricing from {self.source_url}...")
            response = requests.get(self.source_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            image_models = self._parse_image_table(soup)
            video_models = self._parse_video_table(soup)
            tool_models = self._parse_tool_table(soup)
            
            raw_models = image_models + video_models + tool_models
            
            print(f"Parsed {len(raw_models)} models from Runware pricing page")
            
            if progress_tracker:
                progress_tracker.start(len(all_models), {
                    'use_llm': self.use_llm,
                    'supports_llm': False
                })

            processed_count = 0
            for model_data in raw_models:
                try:
                    model_dict = self._normalize_model(model_data)
                    if model_dict:
                        all_models.append(model_dict)
                        processed_count += 1
                        
                        if progress_tracker:
                            progress_tracker.update(
                                processed_count,
                                model_dict['model_id'],
                                model_dict['model_name'],
                                [],
                                False,
                                ""
                            )
                except Exception as e:
                    print(f"Error processing model {model_data.get('model_name', 'unknown')}: {e}")
                    continue

            if progress_tracker:
                progress_tracker.complete()

        except Exception as e:
            print(f"Error extracting from Runware: {e}")
            if progress_tracker:
                progress_tracker.error(str(e))

        return all_models

    def _extract_with_auth(self, cookies_json: Optional[str] = None, progress_tracker: Optional[ProgressTracker] = None, 
                          username: Optional[str] = None, password: Optional[str] = None) -> List[Dict]:
        """
        Extract models using authentication via login
        
        Args:
            cookies_json: JSON string of cookies (not used, we'll login directly)
            progress_tracker: Optional progress tracker
            username: Email for login (if not provided, uses default)
            password: Password for login (if not provided, uses default)
            
        Returns:
            List of model dictionaries
        """
        from playwright.sync_api import sync_playwright
        
        all_models = []
        
        # Start progress tracker early with estimated count
        if progress_tracker:
            progress_tracker.start(200, {
                'use_llm': self.use_llm,
                'supports_llm': False
            })
        
        # Login credentials (use provided or fallback to defaults)
        email = username if username else "rzrd2024@gmail.com"
        pwd = password if password else "REZVANI@rzrd2024"
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            try:
                # Login
                print("Logging in to Runware...")
                page.goto('https://my.runware.ai/login', timeout=60000)
                page.wait_for_load_state('domcontentloaded', timeout=30000)
                time.sleep(2)
                
                page.fill('input[type="email"]', email)
                page.fill('input[type="password"]', pwd)
                page.click('button[type="submit"]')
                time.sleep(8)  # Wait for login
                
                # Check if login succeeded - navigate to models page to verify
                print("Navigating to models page...")
                page.goto('https://my.runware.ai/models/all', timeout=60000)
                page.wait_for_load_state('domcontentloaded', timeout=30000)
                time.sleep(3)
                
                # Check page title to verify we're logged in
                page_title = page.title()
                if page_title != "Models | Runware":
                    print(f"❌ Login failed - page title: {page_title}")
                    return []
                
                print("✅ Login successful, on models page")
                
                # Click on "Certified Models" tab to get the curated list
                print("Switching to Certified Models tab...")
                try:
                    # Wait for tabs to be visible and click "Certified Models"
                    page.click('button[role="tab"]:has-text("Certified Models")')
                    time.sleep(2)
                    print("✅ Switched to Certified Models tab")
                except Exception as e:
                    print(f"⚠️ Could not switch to Certified Models tab: {e}")
                
                # Wait for network to be idle
                try:
                    page.wait_for_load_state('networkidle', timeout=30000)
                    print("Network idle state reached")
                except Exception as e:
                    print(f"Network idle timeout (continuing anyway): {e}")
                
                time.sleep(3)  # Extra wait for dynamic content
                
                # Scroll down multiple times to load more models (page uses virtual scrolling)
                print("Scrolling to load more models...")
                all_airs = set()
                no_new_models_count = 0
                max_scrolls = 50  # Increased scroll attempts
                
                for scroll_iteration in range(max_scrolls):
                    # Extract AIRs from current view
                    visible_text = page.evaluate("() => document.body.innerText")
                    airs = re.findall(r'[a-z]+:\d+@[a-z0-9-]+', visible_text, re.IGNORECASE)
                    previous_count = len(all_airs)
                    all_airs.update(airs)
                    new_count = len(all_airs)
                    
                    # Stop if no new models found for 3 consecutive scrolls
                    if new_count == previous_count:
                        no_new_models_count += 1
                        if no_new_models_count >= 5:
                            print(f"No new models found after {scroll_iteration + 1} scrolls, stopping...")
                            break
                    else:
                        no_new_models_count = 0
                    
                    # Scroll down to load more
                    page.evaluate("window.scrollBy(0, 2000)")  # Scroll by 2000px
                    time.sleep(0.5)  # Wait for new content
                    
                    if scroll_iteration % 5 == 0:
                        print(f"Scroll {scroll_iteration + 1}/{max_scrolls}: Found {len(all_airs)} unique AIRs...")
                
                unique_airs = list(all_airs)
                print(f"✅ Found {len(unique_airs)} total model AIRs from authenticated page")

                
                if not unique_airs:
                    print("No models found, checking HTML for JSON data...")
                    content = page.content()
                    json_matches = re.findall(r'"air"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
                    if json_matches:
                        unique_airs = list(set(json_matches))
                        print(f"Found {len(unique_airs)} AIRs in JSON data")
                
                if not unique_airs:
                    print("Still no AIRs found. Saving debug files...")
                    page.screenshot(path='debug_models_page.png')
                    with open('debug_models_page.html', 'w', encoding='utf-8') as f:
                        f.write(page.content())
                    print("Saved debug_models_page.png and debug_models_page.html")
                
                if not unique_airs:
                    print("No models found")
                    return []
                
                # For each AIR, visit playground to get pricing
                processed_count = 0
                for air in unique_airs:
                    try:
                        model_dict = self._extract_model_pricing_from_playground(page, air)
                        if model_dict:
                            all_models.append(model_dict)
                            processed_count += 1
                            print(f"Extracted: {model_dict['model_name']} - ${model_dict.get('output_price_per_image') or model_dict.get('output_price_per_request', 0)}")
                            
                            if progress_tracker:
                                progress_tracker.update(
                                    processed_count,
                                    model_dict['model_id'],
                                    model_dict['model_name'],
                                    [],
                                    False,
                                    ""
                                )
                        
                        time.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        print(f"Error extracting {air}: {e}")
                        continue
                
                if progress_tracker:
                    progress_tracker.complete()
                
            except Exception as e:
                print(f"Authentication error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                page.close()
                context.close()
                browser.close()
        
        return all_models

    def _extract_model_airs_from_page(self, page) -> List[str]:
        """
        Extract all model AIR identifiers from the models page.
        
        Returns:
            List of AIR identifiers (e.g., 'vidu:1@1', 'flux:1@dev')
        """
        airs = []
        
        # Wait for model cards to load
        try:
            page.wait_for_selector('[data-testid="model-card"], .model-card, [class*="model"]', timeout=5000)
        except Exception:
            pass
        
        # Try multiple strategies to find model links/data
        # Strategy 1: Look for links to playground with modelAIR parameter
        links = page.query_selector_all('a[href*="playground"]')
        for link in links:
            href = link.get_attribute('href')
            if href and 'modelAIR=' in href:
                match = re.search(r'modelAIR=([^&]+)', href)
                if match:
                    air = match.group(1)
                    if air not in airs:
                        airs.append(air)
        
        # Strategy 2: Look for data attributes with AIR
        elements = page.query_selector_all('[data-air], [data-model-air]')
        for elem in elements:
            air = elem.get_attribute('data-air') or elem.get_attribute('data-model-air')
            if air and air not in airs:
                airs.append(air)
        
        # Strategy 3: Extract from page content/JavaScript
        content = page.content()
        air_matches = re.findall(r'"air":\s*"([^"]+)"', content)
        for air in air_matches:
            if air not in airs and ':' in air and '@' in air:
                airs.append(air)
        
        return airs

    def _extract_model_pricing_from_playground(self, page, air: str) -> Optional[Dict]:
        """
        Visit playground for a specific model AIR and extract pricing.
        
        Args:
            page: Playwright page object
            air: Model AIR identifier (e.g., 'vidu:1@1')
            
        Returns:
            Model dictionary with pricing or None
        """
        try:
            # Determine category (needed for pricing logic below)
            category = self._determine_category_from_air(air)
            
            # Try category-specific playground URL first (shows approximate pricing for some models)
            playground_url = f'https://my.runware.ai/playground/{category}?modelAIR={air}'
            
            print(f"Visiting playground for {air}...")
            try:
                page.goto(playground_url, wait_until='domcontentloaded', timeout=45000)
                time.sleep(1)
                
                # Check if we got a 404 or error page
                page_content = page.content().lower()
                page_title = page.title().lower()
                if 'not found' in page_title or '404' in page_title or 'oops' in page_content or 'page not found' in page_content:
                    # Fallback to generic playground URL
                    print(f"  Category URL failed, trying generic URL...")
                    playground_url = f'https://my.runware.ai/playground?modelAIR={air}'
                    page.goto(playground_url, wait_until='domcontentloaded', timeout=45000)
            except Exception:
                # If category-specific URL fails, try generic
                print("  Category URL error, trying generic URL...")
                playground_url = f'https://my.runware.ai/playground?modelAIR={air}'
                page.goto(playground_url, wait_until='domcontentloaded', timeout=45000)
            time.sleep(2)  # Wait for page to fully load and pricing to display
            
            # Extract model name and try to get description/details
            model_name = self._extract_model_name_from_playground(page, air)
            
            # Try to extract category tags and provider info
            provider_from_air = air.split(':')[0] if ':' in air else None
            
            # Extract pricing information
            # Look for cost display elements - pricing is calculated after setting parameters
            price_text = None
            price = None
            
            # Try multiple selectors for pricing (look for displayed cost)
            selectors = [
                '[data-testid="cost"]',
                '[data-testid*="cost"]',
                '[data-testid*="price"]',
                '[class*="cost"]',
                '[class*="price"]',
                '[class*="total"]',
                '[class*="approx"]',
                'text=/approx/i',
                'text=/\\$[\\d.]+/',
            ]
            
            # Try to find all price elements and pick the most likely one
            potential_prices = []
            for selector in selectors:
                try:
                    elems = page.query_selector_all(selector)
                    for elem in elems:
                        text = elem.text_content()
                        if text:
                            # Look for dollar amounts in the text
                            matches = re.findall(r'\$?([\d.]+)', text)
                            for match in matches:
                                try:
                                    price_val = float(match)
                                    # Filter out common non-price values
                                    if price_val > 0 and price_val < 100:  # Reasonable range
                                        # Check context - prefer "cost" or "price" mentions
                                        context_score = 0
                                        text_lower = text.lower()
                                        if 'cost' in text_lower or 'price' in text_lower or 'approx' in text_lower:
                                            context_score = 1
                                        potential_prices.append((price_val, text, elem, context_score))
                                except ValueError:
                                    pass
                except Exception:
                    continue
            
            # Debug: print all found prices
            if potential_prices:
                print(f"  Found prices: {[(p[0], p[1].strip()) for p in potential_prices]}")
            
            # If we found multiple prices, analyze them
            if potential_prices:
                # Sort by context score (prefer cost/price mentions) then by price value
                potential_prices.sort(key=lambda x: (x[3], x[0]), reverse=True)
                
                # Remove duplicates by price value
                unique_prices = {}
                for price_val, price_text, elem, context_score in potential_prices:
                    if price_val not in unique_prices:
                        unique_prices[price_val] = (price_text, elem, context_score)
                
                # Take the best price (highest context score, then highest value)
                best_price = max(unique_prices.items(), key=lambda x: (x[1][2], x[0]))
                price = best_price[0]
                price_text = best_price[1][0]
            else:
                # Fallback: search entire page content
                content = page.content()
                price_matches = re.findall(r'\$[\d.]+', content)
                if price_matches:
                    # Parse all prices and pick the highest reasonable one
                    parsed_prices = []
                    for match in price_matches:
                        try:
                            val = float(match.replace('$', ''))
                            if 0 < val < 100:
                                parsed_prices.append(val)
                        except Exception:
                            pass
                    if parsed_prices:
                        price = max(parsed_prices)
                        price_text = f'${price}'
            
            # If still no valid price found, check if it's $0.05 placeholder
            # If so, set price to 0 as user requested (needs config to determine actual price)
            if not price_text or price is None:
                # Check if page has the "Choose your settings" message indicating config needed
                content = page.content()
                if 'choose your settings' in content.lower() or 'click generate' in content.lower():
                    # Model needs configuration, set to 0 for now
                    price = 0.0
                    price_text = '$0.00'
                    print(f"  No price found, needs configuration - setting to $0")
                else:
                    return None
            elif price == 0.05 and not potential_prices:
                # Only found $0.05 from fallback search, likely a placeholder
                content = page.content()
                if 'choose your settings' in content.lower() or 'click generate' in content.lower():
                    price = 0.0
                    price_text = '$0.00'
                    print(f"  Only $0.05 found (placeholder) - setting to $0")
            
            # Extract resolution/duration if available
            resolution = self._extract_resolution_from_playground(page)
            duration = self._extract_duration_from_playground(page)
            
            # Determine unit type and pricing fields
            unit_type, pricing_field = self._determine_unit_type(category, price_text)
            
            # Generate model ID
            model_id = self._generate_model_id_from_air(air, resolution, duration)
            
            # Determine provider (prefer from model name, fallback to AIR)
            provider = self._determine_provider(model_name or air)
            if not provider and provider_from_air:
                provider = provider_from_air.capitalize()
            
            # Map category to model type (use broad types)
            # Runware categories: imageInference, videoInference, audioInference
            type_map = {
                'imageInference': 'image-generation',
                'videoInference': 'video-generation',
                'audioInference': 'audio-generation',
            }
            model_type = type_map.get(category, 'other')
            
            # Determine specific category based on model name/AIR
            air_lower = air.lower()
            model_name_lower = (model_name or '').lower()
            combined_text = f"{air_lower} {model_name_lower}"
            
            # Default categories by type
            specific_category = None
            
            # Image types
            if category == 'imageInference':
                if 'upscale' in combined_text or 'enhance' in combined_text or 'super' in combined_text:
                    specific_category = 'image-to-image'
                elif 'remove-background' in air_lower or 'background' in combined_text:
                    specific_category = 'image-to-image'
                elif 'control' in combined_text or 'canny' in combined_text or 'depth' in combined_text:
                    specific_category = 'image-to-image'
                else:
                    specific_category = 'text-to-image'  # Default for image generation
            
            # Video types
            elif category == 'videoInference':
                if 'image-to-video' in air_lower or 'img2vid' in air_lower:
                    specific_category = 'image-to-video'
                else:
                    specific_category = 'text-to-video'  # Default for video generation
            
            # Build pricing dict
            pricing: Dict[str, Optional[float]] = {
                'input_price_per_token': None,
                'output_price_per_token': None,
                'input_price_per_image': None,
                'output_price_per_image': None,
                'input_price_per_second': None,
                'output_price_per_second': None,
                'input_price_per_request': None,
                'output_price_per_request': None,
            }
            pricing[pricing_field] = price
            
            # Build description
            desc_parts = [model_name or air]
            if resolution:
                desc_parts.append(f"Resolution: {resolution}")
            if duration:
                desc_parts.append(f"Duration: {duration}")
            
            description = " | ".join(desc_parts)
            
            # Build pricing_formula based on category
            pricing_formula = None
            cost_unit = None
            pricing_type = None
            
            if category == 'imageInference':
                pricing_formula = f"${price} per image"
                cost_unit = "image"
                pricing_type = "per_image"
            elif category == 'videoInference':
                if 'per sec' in price_text.lower() or 'per second' in price_text.lower():
                    pricing_formula = f"${price} per second"
                    cost_unit = "second"
                    pricing_type = "per_second"
                else:
                    pricing_formula = f"${price} per video"
                    cost_unit = "video"
                    pricing_type = "per_request"
            elif category == 'audioInference':
                pricing_formula = f"${price} per audio"
                cost_unit = "audio"
                pricing_type = "per_request"
            else:
                pricing_formula = f"${price}"
                cost_unit = "request"
                pricing_type = "per_request"
            
            # Build pricing_variables
            pricing_variables = {
                'air_identifier': air,
                'category': category,
                'resolution': resolution,
                'duration': duration,
                'base_price': price,
                'price_text': price_text,
            }
            
            # LLM extraction if enabled
            llm_extracted = None
            if self.use_llm:
                try:
                    from ai_cost_manager.llm_extractor import extract_pricing_with_llm
                    from ai_cost_manager.cache import CacheManager
                    
                    cache_manager = CacheManager()
                    
                    # Try loading from cache first
                    llm_extracted = cache_manager.load_llm_extraction("runware", model_id)
                    
                    if not llm_extracted:
                        # Prepare context for LLM
                        llm_context = {
                            'name': model_name or air,
                            'provider': provider,
                            'model_type': model_type,
                            'category': category,
                            'pricing_info': price_text,
                            'pricing_formula': pricing_formula,
                            'description': description,
                            'tags': [category, air.split(':')[0], model_type],
                            'raw_metadata': {
                                'air': air,
                                'category': category,
                                'resolution': resolution,
                                'duration': duration,
                                'price': price,
                            },
                        }
                        
                        llm_extracted = extract_pricing_with_llm(llm_context)
                        
                        if llm_extracted:
                            cache_manager.save_llm_extraction("runware", model_id, llm_extracted)
                            
                            # LLM can override model_type with broad type
                            if llm_extracted.get('model_type'):
                                llm_model_type = llm_extracted.get('model_type')
                                
                                # Accept only standardized broad types from LLM
                                if llm_model_type in VALID_MODEL_TYPES:
                                    model_type = llm_model_type
                                    print(f"  ✅ LLM updated model_type to: {model_type}")
                                else:
                                    print(f"  ⚠️  LLM returned invalid type '{llm_model_type}', ignoring (expected one of: {get_valid_types_string()})")
                            
                            # LLM provides specific category
                            if llm_extracted.get('category'):
                                specific_category = llm_extracted.get('category')
                                print(f"  ✅ LLM updated category to: {specific_category}")
                            
                            # Enhance description with LLM data
                            if llm_extracted.get('description'):
                                description = llm_extracted['description']
                            
                            # Update pricing formula if LLM provided better one
                            if llm_extracted.get('pricing_formula'):
                                pricing_formula = llm_extracted['pricing_formula']
                                
                except Exception as e:
                    print(f"  LLM extraction failed: {e}")
            
            return {
                'model_id': model_id,
                'model_name': model_name or air,
                'name': model_name or air,  # Add 'name' field for compatibility
                'provider': provider,
                'model_type': model_type,
                'description': description,
                'cost_per_call': price,
                'thumbnail_url': None,
                'image_url': None,
                'pricing_type': pricing_type,
                'pricing_formula': pricing_formula,
                'pricing_variables': pricing_variables,
                'input_cost_per_unit': None,
                'output_cost_per_unit': price if pricing_field.startswith('output_') else None,
                'cost_unit': cost_unit,
                **pricing,
                'tags': [category, air.split(':')[0], specific_category] if specific_category else [category, air.split(':')[0]],
                'category': specific_category or model_type,  # Specific category (text-to-image, image-to-video, etc.)
                'llm_extracted': llm_extracted if llm_extracted else None,
                'raw_metadata': {
                    'air': air,
                    'category': category,
                    'resolution': resolution,
                    'duration': duration,
                    'price_text': price_text,
                },
                'last_raw_fetched': None,
                'last_schema_fetched': None,
                'last_playground_fetched': None,
            }
            
        except Exception as e:
            print(f"Error extracting playground pricing for {air}: {e}")
            return None

    def _determine_category_from_air(self, air: str) -> str:
        """Determine playground category from AIR identifier."""
        # Common patterns: vidu:1@1, flux:1@dev, sora:2@standard
        provider = air.split(':')[0].lower()
        
        video_providers = ['vidu', 'sora', 'kling', 'minimax', 'hailuo', 'pixverse', 'veo', 'ltx', 'seedance', 'wan']
        audio_providers = ['audio', 'speech', 'tts']
        
        if any(p in provider for p in video_providers):
            return 'videoInference'
        elif any(p in provider for p in audio_providers):
            return 'audioInference'
        else:
            return 'imageInference'

    def _extract_model_name_from_playground(self, page, air: str) -> str:
        """Extract model name from playground page."""
        # Try to find model name in title or header
        try:
            title = page.query_selector('h1, h2, [class*="title"], [class*="model-name"]')
            if title:
                return title.text_content().strip()
        except Exception:
            pass
        
        # Fallback: use AIR as name
        return air.replace(':', ' ').replace('@', ' v')

    def _extract_resolution_from_playground(self, page) -> Optional[str]:
        """Extract resolution from playground controls."""
        try:
            # Look for resolution selector or display
            res_elem = page.query_selector('[name="resolution"], select:has-text("1024"), input[placeholder*="resolution"]')
            if res_elem:
                return res_elem.get_attribute('value') or res_elem.text_content()
        except Exception:
            pass
        return None

    def _extract_duration_from_playground(self, page) -> Optional[str]:
        """Extract duration from playground controls."""
        try:
            # Look for duration selector or display
            dur_elem = page.query_selector('[name="duration"], select:has-text("5s"), input[placeholder*="duration"]')
            if dur_elem:
                return dur_elem.get_attribute('value') or dur_elem.text_content()
        except Exception:
            pass
        return None

    def _determine_unit_type(self, category: str, price_text: str) -> tuple:
        """
        Determine unit type and corresponding pricing field.
        
        Returns:
            (unit_type, pricing_field_name)
        """
        if category == 'imageInference':
            return ('per_image', 'output_price_per_image')
        elif category == 'videoInference':
            if '/ sec' in price_text or 'per second' in price_text.lower():
                return ('per_second', 'output_price_per_second')
            else:
                return ('per_video', 'output_price_per_request')
        elif category == 'audioInference':
            if '/ sec' in price_text or 'per second' in price_text.lower():
                return ('per_second', 'output_price_per_second')
            else:
                return ('per_request', 'output_price_per_request')
        else:
            return ('per_request', 'output_price_per_request')

    def _generate_model_id_from_air(self, air: str, resolution: Optional[str], duration: Optional[str]) -> str:
        """Generate model ID from AIR identifier."""
        base_id = f"runware-{air.replace(':', '-').replace('@', '-').lower()}"
        
        if resolution:
            base_id += f"-{resolution.lower().replace('x', '')}"
        if duration:
            base_id += f"-{duration.lower()}"
        
        # Clean up
        base_id = re.sub(r'[^a-z0-9-]', '-', base_id)
        base_id = re.sub(r'-+', '-', base_id)
        
        return base_id.strip('-')

    def _parse_image_table(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse image pricing table from HTML."""
        models = []
        
        # Find all tables in the page
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if this is the image table by looking at headers
            headers = table.find_all('th')
            header_text = ' '.join([h.get_text(strip=True) for h in headers]).lower()
            
            # Image table has: Model, Size, Steps, Price/Image
            if 'steps' in header_text and 'image' in header_text:
                rows = table.find_all('tr')[1:]  # Skip header row
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        model_name = cells[0].get_text(strip=True)
                        resolution = cells[1].get_text(strip=True)
                        steps_text = cells[2].get_text(strip=True)
                        price_text = cells[3].get_text(strip=True)
                        
                        # Skip empty or invalid rows
                        if not model_name or model_name == '-' or 'view pricing' in price_text.lower():
                            continue
                        
                        # Parse price
                        price_match = re.search(r'\$?([\d.]+)', price_text)
                        if not price_match:
                            continue
                        
                        price = float(price_match.group(1))
                        
                        # Parse steps
                        steps = None
                        if steps_text and steps_text != '-':
                            steps_match = re.search(r'(\d+)', steps_text)
                            if steps_match:
                                steps = int(steps_match.group(1))
                        
                        models.append({
                            'model_name': model_name,
                            'resolution': resolution if resolution != '-' else None,
                            'steps': steps,
                            'price': price,
                            'unit_type': 'per_image',
                            'category': 'image'
                        })
        
        return models

    def _parse_video_table(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse video pricing table from HTML."""
        models = []
        
        tables = soup.find_all('table')
        
        for table in tables:
            headers = table.find_all('th')
            header_text = ' '.join([h.get_text(strip=True) for h in headers]).lower()
            
            # Video table has: Model, Resolution, Duration, Price/Video
            if 'duration' in header_text and 'video' in header_text:
                rows = table.find_all('tr')[1:]  # Skip header row
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        model_name = cells[0].get_text(strip=True)
                        resolution = cells[1].get_text(strip=True)
                        duration = cells[2].get_text(strip=True)
                        price_text = cells[3].get_text(strip=True)
                        
                        # Skip empty, invalid, or promo rows
                        if not model_name or model_name == '-' or '50%' in model_name or '15%' in model_name:
                            continue
                        
                        # Parse price
                        price_match = re.search(r'\$?([\d.]+)', price_text)
                        if not price_match:
                            continue
                        
                        price = float(price_match.group(1))
                        
                        # Determine unit type (per video or per second)
                        unit_type = 'per_video'
                        if '/ sec' in price_text or 'per sec' in price_text.lower():
                            unit_type = 'per_second'
                        
                        models.append({
                            'model_name': model_name,
                            'resolution': resolution if resolution != '-' else None,
                            'duration': duration if duration != '-' else None,
                            'price': price,
                            'unit_type': unit_type,
                            'category': 'video'
                        })
        
        return models

    def _parse_tool_table(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse tools pricing table from HTML."""
        models = []
        
        tables = soup.find_all('table')
        
        for table in tables:
            headers = table.find_all('th')
            header_text = ' '.join([h.get_text(strip=True) for h in headers]).lower()
            
            # Tools table has: Model, Size, Price (no steps/duration)
            if 'model' in header_text and 'price' in header_text and 'steps' not in header_text and 'duration' not in header_text:
                rows = table.find_all('tr')[1:]  # Skip header row
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        model_name = cells[0].get_text(strip=True)
                        resolution = cells[1].get_text(strip=True)
                        price_text = cells[2].get_text(strip=True)
                        
                        # Skip empty rows
                        if not model_name or model_name == '-':
                            continue
                        
                        # Parse price (might have range like $0.0006 / $0.0032, take first)
                        price_match = re.search(r'\$?([\d.]+)', price_text)
                        if not price_match:
                            continue
                        
                        price = float(price_match.group(1))
                        
                        # Determine unit type
                        unit_type = 'per_call'
                        if '/ sec' in price_text or 'per sec' in price_text.lower():
                            unit_type = 'per_second'
                        
                        models.append({
                            'model_name': model_name,
                            'resolution': resolution if resolution != '-' else None,
                            'duration': None,
                            'price': price,
                            'unit_type': unit_type,
                            'category': 'tool'
                        })
        
        return models

    def _determine_provider(self, model_name: str) -> str:
        """Determine the provider from model name."""
        model_lower = model_name.lower()
        
        for key, provider in self.provider_map.items():
            if key in model_lower:
                return provider
        
        # Default to Runware if no specific provider found
        return 'Runware'

    def _generate_model_id(self, model_name: str, resolution: Optional[str], duration: Optional[str], steps: Optional[int]) -> str:
        """Generate a unique model ID."""
        base_id = f"runware-{model_name.lower()}"
        base_id = re.sub(r'[^a-z0-9-]', '-', base_id)
        base_id = re.sub(r'-+', '-', base_id)
        
        if resolution:
            base_id += f"-{resolution.lower()}"
        if duration:
            base_id += f"-{duration.lower()}"
        if steps:
            base_id += f"-{steps}steps"
        
        return base_id.strip('-')

    def _normalize_model(self, model_data: Dict) -> Optional[Dict]:
        """
        Normalize a Runware model to standard dictionary format.

        Args:
            model_data: Raw model data from parsing

        Returns:
            Model dictionary or None if normalization fails
        """
        try:
            model_name = model_data['model_name']
            resolution = model_data.get('resolution')
            duration = model_data.get('duration')
            steps = model_data.get('steps')
            price = model_data['price']
            unit_type = model_data['unit_type']
            category = model_data['category']
            
            # Generate model ID
            model_id = self._generate_model_id(model_name, resolution, duration, steps)
            
            # Determine provider
            provider = self._determine_provider(model_name)
            
            # Map category to broad model type
            type_map = {
                'image': 'image-generation',
                'video': 'video-generation',
                'tool': 'other'
            }
            model_type = type_map.get(category, 'other')
            
            # Determine specific category based on model name
            model_name_lower = model_name.lower()
            specific_category = None
            
            if category == 'image':
                if 'upscale' in model_name_lower or 'enhance' in model_name_lower:
                    specific_category = 'image-to-image'
                elif 'background' in model_name_lower or 'remove' in model_name_lower:
                    specific_category = 'image-to-image'
                elif 'control' in model_name_lower:
                    specific_category = 'image-to-image'
                else:
                    specific_category = 'text-to-image'
            elif category == 'video':
                if 'image-to-video' in model_name_lower or 'img2vid' in model_name_lower:
                    specific_category = 'image-to-video'
                else:
                    specific_category = 'text-to-video'

            # Initialize pricing fields
            input_price_per_token = None
            output_price_per_token = None
            input_price_per_image = None
            output_price_per_image = None
            input_price_per_second = None
            output_price_per_second = None
            input_price_per_request = None
            output_price_per_request = None

            # Set pricing based on unit type
            if unit_type == 'per_image':
                output_price_per_image = price
            elif unit_type == 'per_video':
                output_price_per_request = price
            elif unit_type == 'per_second':
                output_price_per_second = price
            elif unit_type == 'per_call':
                output_price_per_request = price

            # Build description
            description_parts = [model_name]
            if resolution:
                description_parts.append(f"Resolution: {resolution}")
            if duration:
                description_parts.append(f"Duration: {duration}")
            if steps:
                description_parts.append(f"Steps: {steps}")
            
            description = " | ".join(description_parts)

            return {
                'model_id': model_id,
                'model_name': model_name,
                'provider': provider,
                'model_type': model_type,
                'input_price_per_token': input_price_per_token,
                'output_price_per_token': output_price_per_token,
                'input_price_per_image': input_price_per_image,
                'output_price_per_image': output_price_per_image,
                'input_price_per_second': input_price_per_second,
                'output_price_per_second': output_price_per_second,
                'input_price_per_request': input_price_per_request,
                'output_price_per_request': output_price_per_request,
                'tags': [specific_category] if specific_category else [],
                'category': specific_category or model_type,  # Specific category
                'description': description
            }

        except Exception as e:
            print(f"Error normalizing model {model_data.get('model_name', 'unknown')}: {e}")
            return None
