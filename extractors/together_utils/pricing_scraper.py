"""
Scraper for Together AI pricing page

Fallback method when API key is not available.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import re


def scrape_pricing_page(url: str = "https://www.together.ai/pricing") -> Dict[str, List[Dict[str, Any]]]:
    """
    Scrape pricing information from Together AI pricing page.
    
    Args:
        url: URL of the pricing page
        
    Returns:
        Dictionary with model categories and their pricing data
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find sections
        models_by_category = {
            'text': [],
            'image': [],
            'video': [],
            'audio': [],
            'embedding': [],
            'rerank': [],
            'moderation': [],
        }
        
        # Find all tables in the pricing section
        tables = soup.find_all('table')
        
        current_category = None
        
        # Look for section headers to identify category
        for element in soup.find_all(['h3', 'h4', 'div', 'table']):
            text = element.get_text(strip=True).lower()
            
            # Identify category from headers
            if 'text' in text and 'vision' in text:
                current_category = 'text'
            elif 'image model' in text:
                current_category = 'image'
            elif 'video model' in text:
                current_category = 'video'
            elif 'audio model' in text or 'transcription' in text:
                current_category = 'audio'
            elif 'embedding' in text:
                current_category = 'embedding'
            elif 'rerank' in text:
                current_category = 'rerank'
            elif 'moderation' in text:
                current_category = 'moderation'
            
            # Process tables
            if element.name == 'table' and current_category:
                rows = element.find_all('tr')
                
                # Parse header to understand columns
                header_row = rows[0] if rows else None
                headers = []
                if header_row:
                    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
                
                for row in rows[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) < 2:
                        continue
                    
                    # Extract model name
                    model_name_elem = cols[0]
                    model_name = model_name_elem.get_text(strip=True)
                    if not model_name:
                        continue
                    
                    # Try to extract model URL
                    model_url = None
                    link = model_name_elem.find('a')
                    if link and link.get('href'):
                        href = link.get('href')
                        # Handle relative URLs
                        if href.startswith('http'):
                            model_url = href
                        elif href.startswith('/'):
                            model_url = 'https://www.together.ai' + href
                        else:
                            # Relative path without leading slash
                            model_url = 'https://www.together.ai/' + href
                    
                    pricing_data = {
                        'name': model_name,
                        'url': model_url,
                        'category': current_category,
                    }
                    
                    # Parse based on category
                    if current_category == 'text':
                        # Text models: Input price | Output price | Batch price (optional)
                        if len(cols) >= 2:
                            input_text = cols[1].get_text(strip=True)
                            input_match = re.search(r'\$?([\d.]+)', input_text)
                            if input_match:
                                pricing_data['input_price_per_million'] = float(input_match.group(1))
                        
                        if len(cols) >= 3:
                            output_text = cols[2].get_text(strip=True)
                            output_match = re.search(r'\$?([\d.]+)', output_text)
                            if output_match:
                                pricing_data['output_price_per_million'] = float(output_match.group(1))
                        
                        # Batch price is optional (not shown in table anymore based on your feedback)
                        # All serverless models get 50% discount for batch API
                        if pricing_data.get('input_price_per_million'):
                            pricing_data['supports_batch'] = True
                    
                    elif current_category == 'image':
                        # Image models: Price per MP | Images per $1 | Default steps
                        if len(cols) >= 2:
                            price_text = cols[1].get_text(strip=True)
                            price_match = re.search(r'\$?([\d.]+)', price_text)
                            if price_match:
                                pricing_data['price_per_mp'] = float(price_match.group(1))
                        
                        if len(cols) >= 3:
                            images_text = cols[2].get_text(strip=True)
                            images_match = re.search(r'([\d.,\s]+)', images_text)
                            if images_match:
                                # Remove spaces and parse
                                images_str = images_match.group(1).replace(' ', '').replace(',', '')
                                try:
                                    pricing_data['images_per_dollar'] = float(images_str)
                                except ValueError:
                                    pass
                        
                        if len(cols) >= 4:
                            steps_text = cols[3].get_text(strip=True)
                            steps_match = re.search(r'(\d+)', steps_text)
                            if steps_match:
                                pricing_data['default_steps'] = int(steps_match.group(1))
                        
                        pricing_data['supports_batch'] = False  # Images don't support batch
                    
                    elif current_category == 'video':
                        # Video models: Price per video
                        if len(cols) >= 2:
                            price_text = cols[1].get_text(strip=True)
                            price_match = re.search(r'\$?([\d.]+)', price_text)
                            if price_match:
                                pricing_data['price_per_video'] = float(price_match.group(1))
                        
                        pricing_data['supports_batch'] = False
                    
                    elif current_category == 'audio':
                        # Audio models: Price per audio minute
                        if len(cols) >= 2:
                            price_text = cols[1].get_text(strip=True)
                            price_match = re.search(r'\$?([\d.]+)', price_text)
                            if price_match:
                                price = float(price_match.group(1))
                                # Always use price per minute for audio models
                                pricing_data['price_per_minute'] = price
                        
                        # Check for batch support (Whisper supports it)
                        pricing_data['supports_batch'] = 'whisper' in model_name.lower()
                    
                    elif current_category in ['embedding', 'rerank', 'moderation']:
                        # These models: Price per 1M tokens
                        if len(cols) >= 2:
                            price_text = cols[1].get_text(strip=True)
                            price_match = re.search(r'\$?([\d.]+)', price_text)
                            if price_match:
                                pricing_data['price_per_million_tokens'] = float(price_match.group(1))
                        
                        pricing_data['supports_batch'] = False
                    
                    models_by_category[current_category].append(pricing_data)
        
        return models_by_category
        
    except Exception as e:
        print(f"Error scraping pricing page: {e}")
        import traceback
        traceback.print_exc()
        return {}


def fetch_model_page(model_url: str) -> Optional[Dict[str, Any]]:
    """
    Fetch additional details from a model's page.
    
    Args:
        model_url: URL to the model page
        
    Returns:
        Dictionary with model details including pricing field
    """
    try:
        # Validate URL format
        if not model_url or not model_url.startswith('http'):
            print(f"Invalid model URL: {model_url}")
            return None
        
        response = requests.get(model_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        details = {
            'url': model_url,
        }
        
        # Extract model ID from URL
        model_id_match = re.search(r'/models/([^/?]+)', model_url)
        if model_id_match:
            details['model_id'] = model_id_match.group(1)
        
        # Try to find JSON data embedded in the page (often in script tags)
        scripts = soup.find_all('script', type='application/json')
        for script in scripts:
            try:
                data = script.string
                if data:
                    import json
                    json_data = json.loads(data) if data.startswith('{') or data.startswith('[') else None
                    if json_data and isinstance(json_data, dict):
                        details['embedded_data'] = json_data
                        break
            except:
                pass
        
        # Look for pricing section in the page text
        page_text = soup.get_text(separator=' ', strip=True)
        
        # Pattern 1: "Pricing: $X.XX Input / $Y.YY Output"
        pricing_pattern = r'Pricing:\s*\$?([\d.]+)\s*(?:Input|input)\s*/\s*\$?([\d.]+)\s*(?:Output|output)'
        match = re.search(pricing_pattern, page_text, re.IGNORECASE)
        if match:
            input_price = match.group(1)
            output_price = match.group(2)
            details['pricing'] = f"${input_price} Input / ${output_price} Output"
            details['input_price'] = float(input_price)
            details['output_price'] = float(output_price)
        
        # Pattern 2: Look for "$/MP" or image pricing
        elif '$/MP' in page_text or '/MP' in page_text:
            image_pricing_pattern = r'\$?([\d.]+)\s*/\s*MP'
            match = re.search(image_pricing_pattern, page_text, re.IGNORECASE)
            if match:
                price_per_mp = match.group(1)
                details['pricing'] = f"${price_per_mp}/MP"
                details['price_per_mp'] = float(price_per_mp)
        
        # Pattern 3: Video pricing
        elif '/video' in page_text.lower():
            video_pricing_pattern = r'\$?([\d.]+)\s*/\s*(?:video|Video)'
            match = re.search(video_pricing_pattern, page_text, re.IGNORECASE)
            if match:
                price_per_video = match.group(1)
                details['pricing'] = f"${price_per_video}/video"
                details['price_per_video'] = float(price_per_video)
        
        # Pattern 4: Audio pricing (per audio minute)
        elif 'audio minute' in page_text.lower() or '/min' in page_text:
            audio_pricing_pattern = r'\$?([\d.]+)\s*(?:per|/)\s*(?:audio\s+)?min(?:ute)?'
            match = re.search(audio_pricing_pattern, page_text, re.IGNORECASE)
            if match:
                price_per_minute = match.group(1)
                details['pricing'] = f"${price_per_minute}/audio minute"
                details['price_per_minute'] = float(price_per_minute)
        
        # Extract short meta description only
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            if meta.get('property') in ['og:description', 'twitter:description']:
                content = meta.get('content', '')
                if content and len(content) < 300:  # Keep it short
                    details['description'] = content
                    break
        
        return details
        
    except Exception as e:
        print(f"Error fetching model page {model_url}: {e}")
        return None




def parse_model_name_to_id(name: str) -> str:
    """
    Convert display name to model ID format.
    
    Args:
        name: Display name from pricing page
        
    Returns:
        Model ID string
    """
    # Generate ID from name (basic conversion)
    model_id = name.lower()
    # Remove extra text after model name
    model_id = re.sub(r'\s+(llama|qwen|deepseek|mistral|flux|imagen|veo|sora).*$', '', model_id, flags=re.I)
    model_id = model_id.replace(' ', '-').replace('(', '').replace(')', '').replace('.', '-')
    return model_id


def merge_scraped_with_api(api_models: List[Dict[str, Any]], scraped_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge scraped pricing data with API model data.
    
    Args:
        api_models: Models from API
        scraped_data: Models scraped from pricing page (by category)
        
    Returns:
        Merged list with enhanced pricing info
    """
    # Flatten scraped data
    all_scraped = []
    for category_models in scraped_data.values():
        all_scraped.extend(category_models)
    
    # Create lookup by model name
    scraped_lookup = {m['name'].lower(): m for m in all_scraped}
    
    for api_model in api_models:
        name = api_model.get('display_name', api_model.get('id', '')).lower()
        
        # Try to find matching scraped data
        matching_scraped = None
        for scraped_name, scraped in scraped_lookup.items():
            # Fuzzy match
            if scraped_name in name or name in scraped_name:
                matching_scraped = scraped
                break
        
        if matching_scraped:
            # Update pricing with scraped data
            if 'pricing' not in api_model:
                api_model['pricing'] = {}
            
            # Update based on category
            if matching_scraped.get('input_price_per_million'):
                api_model['pricing']['input'] = matching_scraped['input_price_per_million']
            if matching_scraped.get('output_price_per_million'):
                api_model['pricing']['output'] = matching_scraped['output_price_per_million']
            if matching_scraped.get('price_per_mp'):
                api_model['pricing']['per_mp'] = matching_scraped['price_per_mp']
            if matching_scraped.get('price_per_video'):
                api_model['pricing']['per_video'] = matching_scraped['price_per_video']
            if matching_scraped.get('price_per_million_chars'):
                api_model['pricing']['per_million_chars'] = matching_scraped['price_per_million_chars']
            if matching_scraped.get('price_per_minute'):
                api_model['pricing']['per_minute'] = matching_scraped['price_per_minute']
            
            # Add metadata
            if matching_scraped.get('images_per_dollar'):
                api_model['images_per_dollar'] = matching_scraped['images_per_dollar']
            if matching_scraped.get('default_steps'):
                api_model['default_steps'] = matching_scraped['default_steps']
            if matching_scraped.get('url'):
                api_model['link'] = matching_scraped['url']
            if matching_scraped.get('supports_batch') is not None:
                api_model['supports_batch'] = matching_scraped['supports_batch']
    
    return api_models

