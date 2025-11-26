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
        
        prompt = f"""Extract pricing information from this AI model data and return a JSON object.

Model Name: {model_name}
Model Type: {model_type}
Credits Required: {credits}
Pricing Info Text: "{pricing_info}"

Analyze the pricing and return a JSON object with these fields:
- pricing_type: one of ["fixed", "per_token", "per_second", "per_image", "per_video", "per_request", "tiered", "variable", "unknown"]
- pricing_formula: human-readable description of how cost is calculated
- input_cost_per_unit: numeric cost per input unit (or null)
- output_cost_per_unit: numeric cost per output unit (or null)
- cost_unit: unit of measurement ("tokens", "seconds", "images", "calls", "minutes", etc.)
- pricing_variables: object with any variables that affect pricing (e.g., {{"resolution": "affects cost", "duration": "affects cost"}})
- estimated_cost_per_call: estimated cost in USD for a typical call
- notes: any additional pricing notes

Examples:
- "$0.045/sec" → pricing_type: "per_second", cost_unit: "seconds", input_cost_per_unit: 0.045
- "100 credits" → pricing_type: "fixed", cost_unit: "calls", estimated_cost_per_call: 1.00 (if 1 credit = $0.01)
- "$0.03/1K input tokens, $0.06/1K output tokens" → pricing_type: "per_token", input_cost_per_unit: 0.00003, output_cost_per_unit: 0.00006

Return ONLY valid JSON, no markdown or explanation:"""
        
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
            timeout=30
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
