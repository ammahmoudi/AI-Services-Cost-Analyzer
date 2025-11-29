"""
LLM Pricing Extractor

Uses LLM (via OpenRouter) to intelligently extract and parse pricing information.
"""
import json
import requests
from typing import Dict, Any, Optional
from ai_cost_manager.database import get_session
from ai_cost_manager.models import LLMConfiguration


class LLMPricingExtractor:
    """
    Uses LLM to extract structured pricing information from text.
    """
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Optional[LLMConfiguration]:
        """Load active LLM configuration from database"""
        session = get_session()
        try:
            config = session.query(LLMConfiguration).filter_by(is_active=True).first()
            return config
        finally:
            session.close()
    
    def extract_pricing(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pricing information using LLM.
        
        Args:
            model_data: Raw model data including pricing_info, credits, etc.
            
        Returns:
            Structured pricing information
        """
        if not self.config:
            return self._fallback_extraction(model_data)
        
        # Build prompt
        prompt = self._build_prompt(model_data)
        
        try:
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse response
            result = self._parse_llm_response(response)
            
            return result
            
        except Exception as e:
            print(f"  LLM extraction failed: {e}")
            return self._fallback_extraction(model_data)
    
    def _build_prompt(self, model_data: Dict[str, Any]) -> str:
        """Build prompt for LLM"""
        model_name = model_data.get('name', 'Unknown')
        pricing_info = model_data.get('pricing_info', '')
        credits = model_data.get('creditsRequired', 0)
        model_type = model_data.get('model_type', 'other')
        
        # Extract additional context from all available sources
        tags = model_data.get('tags', [])
        raw_metadata = model_data.get('raw_metadata', {})
        input_schema = model_data.get('input_schema', {})
        output_schema = model_data.get('output_schema', {})
        playground_data = model_data.get('playground_data', {})
        
        # Build comprehensive context
        context_parts = []
        
        # Add tags/categories
        if tags:
            context_parts.append(f"Categories/Tags: {', '.join(tags) if isinstance(tags, list) else tags}")
        
        # Add schema information (parameter hints)
        schema_hints = []
        if input_schema and isinstance(input_schema, dict):
            properties = input_schema.get('properties', {})
            if 'resolution' in properties:
                schema_hints.append("supports resolution parameter")
            if 'duration' in properties or 'num_frames' in properties:
                schema_hints.append("supports duration/frames parameter")
            if 'steps' in properties or 'num_inference_steps' in properties:
                schema_hints.append("supports inference steps parameter")
        
        if schema_hints:
            context_parts.append(f"Model capabilities: {', '.join(schema_hints)}")
        
        # Add playground pricing if available
        if playground_data and isinstance(playground_data, dict):
            playground_price = playground_data.get('price')
            playground_pricing_text = playground_data.get('pricing_text')
            if playground_price:
                context_parts.append(f"Playground price: {playground_price} credits")
            if playground_pricing_text:
                context_parts.append(f"Playground pricing: {playground_pricing_text}")
        
        # Add raw metadata pricing details
        if raw_metadata and isinstance(raw_metadata, dict):
            # Check for pricing override
            pricing_override = raw_metadata.get('pricingInfoOverride')
            if pricing_override:
                context_parts.append(f"Pricing override: {pricing_override}")
            
            # Check for billing message
            billing_msg = raw_metadata.get('billingMessage')
            if billing_msg:
                context_parts.append(f"Billing info: {billing_msg}")
            
            # Check for machine type (cost indicator)
            machine_type = raw_metadata.get('machineType')
            if machine_type:
                context_parts.append(f"Machine type: {machine_type}")
        
        additional_context = "\n".join(context_parts) if context_parts else "No additional context available"
        
        prompt = f"""Extract pricing information from this AI model data and return a JSON object.

Model Name: {model_name}
Model Type: {model_type}
Credits Required: {credits}
Pricing Info Text: "{pricing_info}"

ADDITIONAL CONTEXT:
{additional_context}

MODEL TYPE DETECTION HINTS:
- If model name contains "video", "seedance", "mochi", "runway", "sora", "pika", or pricing mentions "video"/"frame"/"fps", use "video-generation"
- If pricing is per video (e.g., "$0.14/video"), use "video-generation" and pricing_type="per_video"
- If model name contains "image", "flux", "dall-e", "midjourney", "stable-diffusion", or pricing mentions "image"/"megapixel"/"MP", use "image-generation"
- If pricing mentions "token" or "per million", use "text-generation" or "chat"
- For text-to-video, image-to-video models: use "video-generation" and add the specific type (e.g., "text-to-video") to tags array
- For text-to-image, image-to-image models: use "image-generation" and add the specific type (e.g., "text-to-image") to tags array
- For image-to-text models: use "text-generation" and add "image-to-text" to tags array

IMPORTANT: Analyze the pricing carefully and return a JSON object with these exact fields:

REQUIRED FIELDS:
- model_type: MUST be EXACTLY one of these values: "text-generation", "image-generation", "video-generation", "audio-generation", "embeddings", "code-generation", "chat", "completion", "rerank", "moderation", "other"
  DO NOT use values like "text-to-video", "image-to-video", "text-to-image" - use the base generation type and add specifics to tags
- pricing_type: MUST be one of: "per_token", "per_image", "per_video", "per_minute", "per_second", "per_call", "hourly", "fixed", "tiered", "per_megapixel"
- cost_unit: MUST match pricing_type - "token", "image", "video", "minute", "second", "call", "hour", "megapixel"
- cost_per_call: numeric value in USD for ONE typical call (required, use 0 if unknown)

OPTIONAL FIELDS:
- pricing_formula: human-readable description
- input_cost_per_unit: numeric cost per input unit (e.g., per 1K tokens)
- output_cost_per_unit: numeric cost per output unit
- credits_required: numeric credits needed per call
- pricing_variables: object with pricing factors like resolution, duration, steps
- description: clear explanation of the pricing model
- tags: array of relevant tags (e.g., ["fast", "high-quality", "enterprise", "open-source"])
- category: original source category if available

PRICING FORMAT EXAMPLES:

1. Text models (tokens):
   "$0.15 / 1M input tokens, $0.60 / 1M output tokens"
   → {{"model_type": "text-generation", "pricing_type": "per_token", "cost_unit": "token", "input_cost_per_unit": 0.00015, "output_cost_per_unit": 0.0006, "cost_per_call": 0.00075}}

2. Image models (megapixels):
   "$0.025/MP | 40.0 img/$1 | 28 steps"
   → {{"model_type": "image-generation", "pricing_type": "per_image", "cost_unit": "megapixel", "cost_per_call": 0.025, "pricing_variables": {{"price_per_mp": 0.025, "images_per_dollar": 40.0, "default_steps": 28}}, "tags": ["fast", "affordable"]}}

3. Video models:
   "$0.19 per 5s 720p video"
   → {{"model_type": "video-generation", "pricing_type": "per_video", "cost_unit": "video", "cost_per_call": 0.19, "pricing_variables": {{"duration_seconds": 5, "resolution": "720p"}}, "tags": ["short-form", "hd"]}}

3b. Text-to-video models:
   "AnimateDiff - text to video"
   → {{"model_type": "video-generation", "pricing_type": "per_video", "cost_unit": "video", "cost_per_call": 0.25, "tags": ["text-to-video", "animation", "ai-generated"]}}

4. Audio models:
   "$0.006 per minute"
   → {{"model_type": "audio-generation", "pricing_type": "per_minute", "cost_unit": "minute", "cost_per_call": 0.006}}

5. Credits-based:
   "2.5 credits" (assuming 1 credit = $0.01)
   → {{"model_type": "image-generation", "pricing_type": "per_call", "cost_unit": "call", "cost_per_call": 0.025, "credits_required": 2.5}}

6. Hourly:
   "$2.40/hour"
   → {{"model_type": "other", "pricing_type": "hourly", "cost_unit": "hour", "cost_per_call": 2.40}}

CRITICAL RULES:
- If pricing is in $/MP (dollars per megapixel), use pricing_type="per_image" and cost_unit="megapixel"
- If pricing shows "img/$" (images per dollar), calculate: cost_per_call = 1 / images_per_dollar
- Always extract numeric values from pricing_variables (e.g., "28 steps" → 28, not "28")
- cost_per_call should be a realistic USD amount (typically 0.0001 to 10.0)
- Return ONLY valid JSON with no markdown, no explanations, no code blocks"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call OpenRouter API"""
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/ai-cost-manager',
        }
        
        data = {
            'model': self.config.model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.1,  # Low temperature for consistent extraction
            'max_tokens': 500,
        }
        
        response = requests.post(
            f'{self.config.base_url}/chat/completions',
            headers=headers,
            json=data,
            timeout=60  # Increased from 30s to prevent timeouts with complex prompts
        )
        
        # Handle rate limiting and server errors
        if response.status_code == 429:
            raise Exception("Rate limit exceeded. Please wait before making more requests.")
        elif response.status_code == 500:
            raise Exception("OpenRouter server error. The API may be experiencing issues.")
        elif response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
        
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        return content
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        # Clean response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        if response.startswith('```json'):
            response = response[7:]
        
        # Parse JSON
        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise
    
    def _fallback_extraction(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback extraction without LLM"""
        credits = model_data.get('creditsRequired')
        pricing_info = model_data.get('pricing_info', '')
        
        # Safely handle credits
        credits_value = 0
        if credits is not None:
            try:
                credits_value = float(credits)
            except (ValueError, TypeError):
                credits_value = 0
        
        result = {
            'pricing_type': 'fixed' if credits_value > 0 else 'unknown',
            'pricing_formula': f'{credits_value} credits per call' if credits_value > 0 else 'Pricing not specified',
            'input_cost_per_unit': None,
            'output_cost_per_unit': None,
            'cost_unit': 'calls',
            'pricing_variables': {},
            'estimated_cost_per_call': credits_value * 0.01 if credits_value > 0 else 0.0,
            'notes': pricing_info if pricing_info else 'No pricing information available',
        }
        
        return result


def extract_pricing_with_llm(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to extract pricing using LLM.
    
    Args:
        model_data: Raw model data
        
    Returns:
        Structured pricing information
    """
    extractor = LLMPricingExtractor()
    return extractor.extract_pricing(model_data)
