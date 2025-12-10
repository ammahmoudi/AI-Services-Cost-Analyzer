"""
Model Name Parser - Extracts structured components from AI model names.

Parses model names into structured components:
- Company/Provider (e.g., "Meta", "OpenAI", "BFL")
- Model Family (e.g., "Llama", "GPT", "Flux")
- Version (e.g., "3.1", "4", "1.1")
- Size/Scale (e.g., "8B", "70B", "405B")
- Variant/Type (e.g., "Instruct", "Chat", "Pro", "Ultra")
- Mode (e.g., "Fill", "Redux", "Edit")
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict


# Known companies/providers
KNOWN_COMPANIES = {
    'meta', 'openai', 'anthropic', 'google', 'microsoft', 'mistral', 'cohere',
    'bfl', 'black-forest-labs', 'black forest labs', 'stability', 'stabilityai',
    'midjourney', 'runway', 'pika', 'together', 'replicate', 'huggingface',
    'nvidia', 'aws', 'azure', 'databricks', '01.ai', 'deepseek', 'alibaba',
    'baidu', 'bytedance', 'imagination', 'rundiffusion', 'juggernaut', 'qwen',
    'klingai', 'pruna', 'prunaai', 'hidream', 'fal', 'inference', 'tensor', 
    'cerebras', 'groq', 'tencent', 'jina', 'eleutheral', 'avail', 'avalai', 
    'metisai', 'runware', 'luma', 'kling', 'kuaishou', 'minimax', 'moonshot',
    'xai', 'x.ai', 'ideogram', 'recraft', 'bria', 'salesforce', 'arcee', 
    'cartesia', 'elevenlabs', '11labs', 'suno', 'udio', 'beatoven', 'cassette',
    'veed', 'easel', 'lightricks', 'tripo3d', 'meshy', 'pixverse', 'vidu',
    'togethercomputer', 'together.ai', 'perplexity', 'tavily', 'exa', 'firecrawl',
    'decart', 'wan', 'wan-ai', 'krea', 'cognition', 'cogito', 'deepcogito',
    'arize', 'marin', 'virtue', 'refuel', 'essentialai', 'canopy', 'hexgrad',
    'mercor', 'higgsfield', 'hvision', 'lucataco', 'nightmareai', 'philz1337x',
    'fofr', 'pseudoram', 'smoretalk', 'clarityai', 'simalabs', 'perceptron',
    'isaac', 'zai', 'z.ai', 'sourceful', 'imagineart', 'leonardo', 'dreamshaper',
    'lykon', 'servicenow', 'apriel', 'resemble', 'argil', 'mirelo', 'moonvalley',
    'marey', 'creatify', 'groq', 'llama', 'typhoon', 'qwq', 'qvq', 'kimi'
}

# Known model families
KNOWN_MODEL_FAMILIES = {
    'gpt', 'llama', 'claude', 'gemini', 'mistral', 'command', 'flux', 'stable-diffusion',
    'sd', 'sdxl', 'dalle', 'midjourney', 'seedance', 'seedream', 'seededit', 'seedvr',
    'pulid', 'qwen', 'deepseek', 'yi', 'wizard', 'vicuna', 'orca', 'falcon', 'mpt', 
    'phi', 'mixtral', 'ministral', 'juggernaut', 'imagen', 'pixtral', 'grok', 'kling', 
    'klingai', 'hidream', 'pixelcraft', 'moondream', 'molmo', 'llava', 'idefics', 
    'lava', 'kosmos', 'blip', 'dali', 'sora', 'veo', 'hailuo', 'minimax', 'wan',
    'cogvideo', 'cogview', 'hunyuan', 'mochi', 'luma', 'ray', 'photon', 'dreamachine',
    'pika', 'runway', 'gen', 'video', 'kolors', 'auraflow', 'playground', 'sana',
    'kandinsky', 'pixart', 'omnigen', 'recraft', 'ideogram', 'bria', 'fibo',
    'gemma', 'nemotron', 'cogito', 'apriel', 'marin', 'orpheus', 'kokoro', 'whisper',
    'chirp', 'bark', 'musicgen', 'lyria', 'sonauto', 'yue', 'diffrhythm', 'udio',
    'thinksound', 'mmaudio', 'stable-audio', 'elevenlabs', 'chatterbox', 'f5-tts',
    'dia', 'vibevoice', 'index', 'kokoro', 'cartesia', 'sonic', 'playai', 'resemble',
    'florence', 'sam', 'segment-anything', 'clip', 'dinov2', 'dino', 'siglip',
    'eva', 'bge', 'gte', 'e5', 'jina', 'nomic', 'mxbai', 'nv-embed', 'text-embedding',
    'rerank', 'cohere', 'mixedbread', 'voyage', 'snowflake', 'arctic', 'embed',
    'triposr', 'trellis', 'rodin', 'era3d', 'wonder3d', 'zero123', 'stable-zero123',
    'hunyuan3d', 'meshy', 'csm', 'shap-e', 'point-e', 'dreamfusion', 'magic3d',
    'ltx', 'magi', 'vidu', 'pixverse', 'marey', 'fabric', 'infinity-star', 'ovi',
    'skyreels', 'nextstep', 'decart', 'lucy', 'framepack', 'chronoedit', 'editto',
    'reve', 'omnihuman', 'echomimic', 'sadtalker', 'musetalk', 'live-portrait',
    'infinitalk', 'lipsync', 'sync', 'vace', 'dreamo', 'dreamomni', 'bagel',
    'fashn', 'leffa', 'cat-vton', 'oot-diffusion', 'outfit', 'tryon', 'omnizero',
    'pulid', 'ip-adapter', 'photomaker', 'instant-id', 'face-to-sticker', 'ghiblify',
    'plushify', 'transpixar', 'cartoonify', 'pixelate', 'image2pixel', 'starvector'
}

# Mapping of model family to likely company/provider
FAMILY_TO_COMPANY = {
    'qwen': 'Alibaba',
    'qwq': 'Alibaba',
    'qvq': 'Alibaba',
    'tongyi': 'Alibaba',
    'imagen': 'Google',
    'gemini': 'Google',
    'gemma': 'Google',
    'veo': 'Google',
    'deepseek': 'DeepSeek',
    'mistral': 'Mistral',
    'mixtral': 'Mistral',
    'ministral': 'Mistral',
    'pixtral': 'Mistral',
    'codestral': 'Mistral',
    'falcon': 'Technology Innovation Institute',
    'mpt': 'MosaicML',
    'phi': 'Microsoft',
    'nemotron': 'Nvidia',
    'yi': '01.ai',
    'wizard': 'NousResearch',
    'grok': 'xAI',
    'kling': 'Klingai',
    'klingai': 'Klingai',
    'kolors': 'Klingai',
    'hidream': 'HiDream',
    'llava': 'LLaVA',
    'moondream': 'Moondream',
    'molmo': 'AllenAI',
    'claude': 'Anthropic',
    'llama': 'Meta',
    'gpt': 'OpenAI',
    'dalle': 'OpenAI',
    'sora': 'OpenAI',
    'whisper': 'OpenAI',
    'flux': 'BFL',
    'stable-diffusion': 'Stability',
    'sd': 'Stability',
    'sdxl': 'Stability',
    'stable-audio': 'Stability',
    'command': 'Cohere',
    'seedance': 'Bytedance',
    'seedream': 'Bytedance',
    'seededit': 'Bytedance',
    'lynx': 'Bytedance',
    'hunyuan': 'Tencent',
    'hunyuan3d': 'Tencent',
    'cogvideo': 'Tencent',
    'cogview': 'Tencent',
    'minimax': 'MiniMax',
    'hailuo': 'MiniMax',
    'kimi': 'Moonshot',
    'moonshot': 'Moonshot',
    'wan': 'Alibaba',
    'midjourney': 'Midjourney',
    'runway': 'Runway',
    'gen': 'Runway',
    'pika': 'Pika',
    'luma': 'Luma',
    'ray': 'Luma',
    'photon': 'Luma',
    'dreamachine': 'Luma',
    'ideogram': 'Ideogram',
    'recraft': 'Recraft',
    'playground': 'Playground',
    'leonardo': 'Leonardo',
    'juggernaut': 'Juggernaut',
    'mochi': 'Genmo',
    'ltx': 'Lightricks',
    'fabric': 'Veed',
    'pixverse': 'Pixverse',
    'vidu': 'Vidu',
    'ovi': 'Kuaishou',
    'decart': 'Decart',
    'lucy': 'Decart',
    'sam': 'Meta',
    'segment-anything': 'Meta',
    'florence': 'Microsoft',
    'dinov2': 'Meta',
    'clip': 'OpenAI',
    'bge': 'BAAI',
    'gte': 'Alibaba',
    'e5': 'Microsoft',
    'jina': 'Jina',
    'nomic': 'Nomic',
    'nv-embed': 'Nvidia',
    'voyage': 'Voyage',
    'cohere': 'Cohere',
    'arctic': 'Snowflake',
    'musicgen': 'Meta',
    'elevenlabs': 'ElevenLabs',
    'cartesia': 'Cartesia',
    'sonic': 'Cartesia',
    'suno': 'Suno',
    'udio': 'Udio',
    'kokoro': 'Hexgrad',
    'orpheus': 'Canopy',
    'triposr': 'Tripo3D',
    'trellis': 'Microsoft',
    'meshy': 'Meshy',
    'rodin': 'Hyper3D',
    'era3d': 'ERA3D',
    'wonder3d': 'Wonder3D',
    'cogito': 'DeepCogito',
    'apriel': 'ServiceNow',
    'marin': 'Marin'
}

# Known size indicators - exclude 'ultra' and similar variant-like words
SIZE_PATTERNS = [
    r'\b(\d+(?:\.\d+)?[BM])\b',  # 8B, 70B, 1.5B, 13M
    r'\b(\d+x\d+[BM])\b',         # 8x7B (MoE models)
    r'\b(tiny|small|base|medium|large|xl|xxl)\b'
]

# Known variant/type keywords
VARIANT_KEYWORDS = {
    'instruct', 'chat', 'code', 'coder', 'vision', 'turbo', 'preview', 'pro', 'ultra',
    'lite', 'mini', 'max', 'plus', 'premium', 'standard', 'basic', 'free', 'nano',
    'dev', 'alpha', 'beta', 'experimental', 'finetuned', 'fine-tuned', 'flash',
    'quantized', 'gguf', 'awq', 'gptq', 'ggml', 'uncensored', 'censored', 'fp8',
    'multilingual', 'english', 'it', 'de', 'fr', 'es', 'zh', 'ja', 'ko', 'ar',
    'base', 'v1', 'v2', 'v3', 'v4', 'v5', 'latest', 'thinking', 'reasoning',
    'preview', 'fast', 'distill', 'distilled', 'maverick', 'scout', 'guard',
    'master', 'director', 'live', 'reference', 'subject', 'hd', 'hq', 'reference',
    'tput', 'throughput', 'audio', 'realtime', 'transcribe', 'diarize', 'tts',
    'embedding', 'embed', 'rerank', 'retriever', 'qa', 'search', 'ocr'
}

# Known mode keywords (for image generation, video, and multimodal models)
MODE_KEYWORDS = {
    'fill', 'redux', 'edit', 'inpaint', 'outpaint', 'upscale', 'enhance', 'remix',
    'text-to-image', 't2i', 'image-to-image', 'i2i', 'controlnet', 'lora', 'kontext',
    'krea', 'srpo', 'schnell', 'stream', 'multi', 'portrait', 'realism', 'anime',
    'text-to-video', 't2v', 'image-to-video', 'i2v', 'video-to-video', 'v2v',
    'text-to-audio', 't2a', 'audio-to-audio', 'a2a', 'video-to-audio', 'v2a',
    'speech-to-speech', 's2s', 'speech-to-text', 's2t', 'text-to-speech', 'tts',
    'image-to-3d', 'i23d', 'text-to-3d', 't23d', 'multiview', '3d', 'depth',
    'reframe', 'depth', 'pose', 'canny', 'effects', 'elements', 'transition',
    'lipsync', 'avatar', 'animate', 'stylize', 'transfer', 'removal', 'replace',
    'background', 'segment', 'detect', 'caption', 'grounding', 'generation',
    'trainer', 'training', 'turbo', 'fast', 'pro', 'standard', 'lite'
}


@dataclass
class ParsedModelName:
    """Structured representation of a parsed model name."""
    
    # Original strings
    original_name: str
    original_model_id: str
    
    # Parsed components
    company: Optional[str] = None
    model_family: Optional[str] = None
    version: Optional[str] = None  # e.g., "3.1", "4", "1.1"
    size: Optional[str] = None      # e.g., "8B", "70B"
    variants: List[str] = None      # e.g., ["Instruct", "Chat"]
    modes: List[str] = None         # e.g., ["Fill", "Redux"]
    
    # Additional metadata
    full_version: Optional[str] = None  # e.g., "v1.1", "V3.5"
    tokens: Set[str] = None             # All significant tokens
    
    def __post_init__(self):
        if self.variants is None:
            self.variants = []
        if self.modes is None:
            self.modes = []
        if self.tokens is None:
            self.tokens = set()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, handling sets."""
        data = asdict(self)
        data['tokens'] = list(self.tokens)
        return data
    
    def get_search_key(self) -> str:
        """Generate a normalized search key for matching."""
        parts = []
        if self.company:
            parts.append(self.company.lower())
        if self.model_family:
            parts.append(self.model_family.lower())
        if self.version:
            parts.append(self.version)
        if self.size:
            parts.append(self.size.lower())
        for variant in self.variants:
            parts.append(variant.lower())
        return ' '.join(parts)


class ModelNameParser:
    """Parser for extracting structured components from model names."""
    
    def __init__(self):
        self.company_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(c) for c in KNOWN_COMPANIES) + r')\b',
            re.IGNORECASE
        )
        self.model_family_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(m) for m in KNOWN_MODEL_FAMILIES) + r')\b',
            re.IGNORECASE
        )
    
    def parse(self, name: str, model_id: Optional[str] = None) -> ParsedModelName:
        """
        Parse a model name into structured components.
        
        Args:
            name: The display name of the model
            model_id: Optional model ID for additional context
            
        Returns:
            ParsedModelName with extracted components
        """
        result = ParsedModelName(
            original_name=name,
            original_model_id=model_id or ''
        )
        
        # Combine name and model_id for analysis
        combined = f"{name} {model_id or ''}"
        
        # Extract version numbers (prioritize decimal versions like 1.1, 3.5)
        result.version = self._extract_version(combined)
        result.full_version = self._extract_full_version(combined)
        
        # Extract company
        result.company = self._extract_company(combined)
        
        # Extract model family
        result.model_family = self._extract_model_family(combined)
        
        # Infer company from family if not found
        if not result.company and result.model_family:
            family_lower = result.model_family.lower()
            if family_lower in FAMILY_TO_COMPANY:
                result.company = FAMILY_TO_COMPANY[family_lower]
        
        # Extract size
        result.size = self._extract_size(combined)
        
        # Extract variants and modes
        result.variants = self._extract_variants(combined)
        result.modes = self._extract_modes(combined)
        
        # Extract all significant tokens
        result.tokens = self._extract_tokens(combined)
        
        return result
    
    def _extract_version(self, text: str) -> Optional[str]:
        """Extract version number, prioritizing decimal versions."""
        # Look for decimal versions first (1.1, 3.5, 2.0) even when stuck to words (e.g., FLUX1.1)
        decimal_match = re.search(r'(?<!\d)(\d+\.\d+)', text)
        if decimal_match:
            return decimal_match.group(1)
        
        # Flux-specific fused patterns like "flux1.1", "flux-1", "flux.1"
        flux_match = re.search(r'flux[\s\._-]*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
        if flux_match:
            return flux_match.group(1)

        # Look for single digit versions (but be careful with sizes like "8B")
        # Only match if followed by whitespace or end, not "B" or "M"
        single_match = re.search(r'(?<!\d)(\d{1,2})(?!\.\d)(?![BMbm])', text)
        if single_match:
            version = single_match.group(1)
            # Filter out obvious non-versions (like years, large numbers)
            if len(version) <= 2 and int(version) < 30:
                return version
        
        return None
    
    def _extract_full_version(self, text: str) -> Optional[str]:
        """Extract full version string including 'v' prefix."""
        match = re.search(r'\b(v\d+(?:\.\d+)?)\b', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _extract_company(self, text: str) -> Optional[str]:
        """Extract company/provider name - tries exact match first, then partial."""
        text_lower = text.lower()
        
        # Exact word match first
        match = self.company_pattern.search(text)
        if match:
            company = match.group(1).lower()
            # Normalize variations
            if company in ['black-forest-labs', 'black forest labs']:
                return 'BFL'
            if company == 'stabilityai':
                return 'Stability'
            if company == 'qwen':
                return 'Alibaba'
            return company.title()
        
        # Try partial matching for known companies (e.g., "klingai" matches "klingai")
        for known_company in sorted(KNOWN_COMPANIES, key=len, reverse=True):  # Longest first
            if known_company in text_lower:
                # Avoid matching parts of other words (e.g., "openai" in "openai-gpt")
                if re.search(r'\b' + re.escape(known_company) + r'\b', text_lower, re.IGNORECASE):
                    company = known_company.lower()
                    if company == 'qwen':
                        return 'Alibaba'
                    if company in ['klingai', 'kling']:
                        return 'Klingai'
                    if company in ['prunaai', 'pruna']:
                        return 'Pruna'
                    if company == 'hidream':
                        return 'HiDream'
                    return company.title()
        
        return None
    
    def _extract_model_family(self, text: str) -> Optional[str]:
        """Extract model family name - tries exact match first, then partial."""
        text_lower = text.lower()
        
        # Exact word match first
        match = self.model_family_pattern.search(text)
        if match:
            family = match.group(1).lower()
            # Normalize variations
            if family == 'sd':
                return 'Stable-Diffusion'
            if family in ['kling', 'klingai']:
                return 'Kling'
            if family == 'flux':
                return 'Flux'
            return family.title()
        
        # Try partial matching for known families (e.g., "flux" in "FLUX1.1")
        for known_family in sorted(KNOWN_MODEL_FAMILIES, key=len, reverse=True):  # Longest first
            if known_family in text_lower:
                # Avoid matching parts of other words
                if re.search(r'\b' + re.escape(known_family) + r'\b', text_lower, re.IGNORECASE):
                    family = known_family.lower()
                    if family == 'sd':
                        return 'Stable-Diffusion'
                    if family in ['kling', 'klingai']:
                        return 'Kling'
                    if family == 'flux':
                        return 'Flux'
                    return family.title()
        
        return None
    
    def _extract_size(self, text: str) -> Optional[str]:
        """Extract model size (e.g., 8B, 70B, Large)."""
        for pattern in SIZE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    
    def _extract_variants(self, text: str) -> List[str]:
        """Extract variant keywords."""
        variants = []
        text_lower = text.lower()
        for keyword in VARIANT_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                variants.append(keyword.title())
        return variants
    
    def _extract_modes(self, text: str) -> List[str]:
        """Extract mode keywords (for image generation)."""
        modes = []
        text_lower = text.lower()
        for keyword in MODE_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                modes.append(keyword.title())
        return modes
    
    def _extract_tokens(self, text: str) -> Set[str]:
        """Extract all significant tokens for searching."""
        # Remove special characters and split
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out noise words and very short tokens
        noise_words = {'by', 'the', 'a', 'an', 'and', 'or', 'with', 'for', 'from', 'to', 'of'}
        significant_tokens = {
            t for t in tokens 
            if len(t) > 1 and t not in noise_words and not t.isdigit()
        }
        
        return significant_tokens
    
    def match_score(self, parsed_search: ParsedModelName, parsed_model: ParsedModelName) -> float:
        """
        Calculate match score between search query and model (0-100).
        
        Uses structured components for intelligent matching:
        - Version must match exactly (if present in search)
        - Company, family, size, variants contribute to score
        - Token overlap provides baseline matching
        """
        score = 0.0
        
        # VERSION MATCH - Critical (40 points)
        if parsed_search.version:
            if parsed_model.version == parsed_search.version:
                score += 40
            elif parsed_model.version:
                # Different version = automatic fail
                return 0.0
        
        # MODEL FAMILY - Very important (25 points)
        if parsed_search.model_family and parsed_model.model_family:
            if parsed_search.model_family.lower() == parsed_model.model_family.lower():
                score += 25
        
        # COMPANY - Important (10 points)
        if parsed_search.company and parsed_model.company:
            if parsed_search.company.lower() == parsed_model.company.lower():
                score += 10
        
        # SIZE - Important for LLMs (10 points)
        if parsed_search.size and parsed_model.size:
            if parsed_search.size.lower() == parsed_model.size.lower():
                score += 10
        
        # VARIANTS - Important (10 points)
        if parsed_search.variants and parsed_model.variants:
            search_variants = {v.lower() for v in parsed_search.variants}
            model_variants = {v.lower() for v in parsed_model.variants}
            overlap = search_variants & model_variants
            if overlap:
                score += 10 * (len(overlap) / len(search_variants))
        
        # TOKEN OVERLAP - Baseline (5 points)
        if parsed_search.tokens and parsed_model.tokens:
            overlap = parsed_search.tokens & parsed_model.tokens
            if overlap:
                score += 5 * (len(overlap) / len(parsed_search.tokens))
        
        return min(score, 100.0)


# Singleton instance
_parser = None

def get_parser() -> ModelNameParser:
    """Get the global parser instance."""
    global _parser
    if _parser is None:
        _parser = ModelNameParser()
    return _parser


def parse_model_name(name: str, model_id: Optional[str] = None) -> ParsedModelName:
    """Convenience function to parse a model name."""
    return get_parser().parse(name, model_id)
