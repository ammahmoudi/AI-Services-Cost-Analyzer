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
        """Parse LLM response into structured data (JSON)."""
        import json
        response = response.strip()
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        if response.startswith('```json'):
            response = response[7:]
        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise
