"""
LLM Client Utility

Provides unified chat/completion endpoint logic for OpenRouter and OpenAI-compatible APIs.
"""
import requests
from typing import Optional, Dict, Any
from ai_cost_manager.models import LLMConfiguration

class LLMClient:
    def __init__(self, config: LLMConfiguration):
        self.config = config

    def chat(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """Send a chat/completion request to the configured LLM endpoint."""
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
        response = requests.post(
            f'{self.config.base_url}/chat/completions',
            headers=headers,
            json=data,
            timeout=60
        )
        if response.status_code == 429:
            raise Exception("Rate limit exceeded. Please wait before making more requests.")
        elif response.status_code == 500:
            raise Exception("LLM server error. The API may be experiencing issues.")
        elif response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']

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
