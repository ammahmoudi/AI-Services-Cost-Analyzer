"""
LLM Client Utility

Provides unified chat/completion endpoint logic for OpenRouter and OpenAI-compatible APIs.
"""
import requests
from typing import Dict, Any
from ai_cost_manager.models import LLMConfiguration

class LLMClient:
    def __init__(self, config: LLMConfiguration):
        self.config = config

    def chat(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500, timeout: int = 30, retries: int = 2) -> str:
        """Send a chat/completion request to the configured LLM endpoint.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            timeout: Request timeout in seconds (default: 30)
            retries: Number of retry attempts on failure (default: 2)
            
        Raises:
            Exception: On API errors, timeouts, or network issues
        """
        if not self.config:
            raise Exception("No active LLM configuration.")

        # OpenRouter-compatible endpoint
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/ai-cost-manager',
        }
        data = {
            'model': self.config.model_name,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    f'{self.config.base_url}/chat/completions',
                    headers=headers,
                    json=data,
                    timeout=timeout
                )
                
                if response.status_code == 429:
                    raise Exception("Rate limit exceeded. Please wait before making more requests.")
                elif response.status_code == 500:
                    raise Exception("LLM server error. The API may be experiencing issues.")
                elif response.status_code >= 400:
                    raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
                
                response.raise_for_status()
                
                try:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError) as e:
                    raise Exception(f"Unexpected LLM API response format: {str(e)}")
                except ValueError:
                    raise Exception(f"Invalid JSON in LLM API response: {response.text[:200]}")
                    
            except requests.Timeout:
                last_error = Exception(f"LLM API request timed out after {timeout} seconds. The API may be slow or unresponsive.")
                if attempt < retries:
                    print(f"  Retry {attempt + 1}/{retries} after timeout...")
                    continue
            except requests.ConnectionError as e:
                last_error = Exception(f"Failed to connect to LLM API: {str(e)}")
                if attempt < retries:
                    print(f"  Retry {attempt + 1}/{retries} after connection error...")
                    continue
            except requests.RequestException as e:
                last_error = Exception(f"LLM API request failed: {str(e)}")
                if attempt < retries:
                    print(f"  Retry {attempt + 1}/{retries} after request error...")
                    continue
            except Exception as e:
                # Don't retry on other exceptions (like parsing errors)
                raise
        
        # If we exhausted retries, raise the last error
        if last_error:
            raise last_error
        raise Exception("LLM API request failed after all retries")

    @staticmethod
    def parse_response(response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data (JSON) with normalization."""
        import json
        from ai_cost_manager.model_types import normalize_model_type
        
        response = response.strip()
        
        # Remove markdown code blocks
        if '```' in response:
            # Extract content between code fences
            import re
            code_block = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
            if code_block:
                response = code_block.group(1).strip()
        
        response = response.strip()
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            # Try to extract JSON from text - be more aggressive
            import re
            
            # First, try to find a complete JSON array
            # Look for [ ... ] with proper nesting
            array_matches = list(re.finditer(r'\[', response))
            for match in array_matches:
                start = match.start()
                # Try to parse from this position
                try:
                    # Find the matching closing bracket
                    depth = 0
                    for i in range(start, len(response)):
                        if response[i] == '[':
                            depth += 1
                        elif response[i] == ']':
                            depth -= 1
                            if depth == 0:
                                # Found matching bracket
                                json_str = response[start:i+1]
                                data = json.loads(json_str)
                                return data
                except:
                    continue
            
            # Fall back to finding a JSON object
            object_matches = list(re.finditer(r'\{', response))
            for match in object_matches:
                start = match.start()
                try:
                    depth = 0
                    for i in range(start, len(response)):
                        if response[i] == '{':
                            depth += 1
                        elif response[i] == '}':
                            depth -= 1
                            if depth == 0:
                                json_str = response[start:i+1]
                                data = json.loads(json_str)
                                # Normalize and return
                                if isinstance(data, dict) and 'model_type' in data and data['model_type']:
                                    original_type = data['model_type']
                                    normalized = normalize_model_type(original_type)
                                    if normalized != original_type:
                                        print(f"      [parse_response] Normalized model_type: '{original_type}' → '{normalized}'")
                                    data['model_type'] = normalized
                                return data
                except:
                    continue
            
            # Nothing worked
            raise ValueError(f"Could not extract valid JSON from response. First 200 chars: {response[:200]}")
        
        # Normalize model_type if present to canonical value (for dict responses)
        if isinstance(data, dict):
            if 'model_type' in data and data['model_type']:
                original_type = data['model_type']
                normalized = normalize_model_type(original_type)
                if normalized != original_type:
                    print(f"      [parse_response] Normalized model_type: '{original_type}' → '{normalized}'")
                data['model_type'] = normalized
        
        return data
