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
# NOTE: "fal" is intentionally excluded from this list to avoid false positives
# since it's primarily an API provider prefix (fal-ai/), not a model creator
KNOWN_COMPANIES = {
    'meta', 'openai', 'anthropic', 'google', 'microsoft', 'mistral', 'cohere',
    'bfl', 'black-forest-labs', 'black forest labs', 'stability', 'stabilityai',
    'midjourney', 'runway', 'pika', 'together', 'replicate', 'huggingface',
    'nvidia', 'aws', 'azure', 'databricks', '01.ai', 'deepseek', 'alibaba',
    'baidu', 'bytedance', 'imagination', 'rundiffusion', 'juggernaut', 'qwen',
    'klingai', 'pruna', 'prunaai', 'hidream', 'inference', 'tensor', 
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
    'marey', 'creatify', 'groq', 'llama', 'typhoon', 'qwq', 'qvq', 'kimi',
    'civitai', 'dataforseo', 'baai', 'byteplus', 'imagination', 'yue', 'zonos'
}

# Company name capitalization mapping for consistency
# Maps lowercase company names to their proper capitalization
COMPANY_CAPITALIZATION = {
    'bfl': 'BFL',
    'black-forest-labs': 'BFL',
    'black forest labs': 'BFL',
    'openai': 'OpenAI',
    'deepseek': 'DeepSeek',
    'stabilityai': 'Stability',
    'midjourney': 'Midjourney',
    'huggingface': 'HuggingFace',
    'bytedance': 'Bytedance',
    'alibaba': 'Alibaba',
    'anthropic': 'Anthropic',
    'cohere': 'Cohere',
    'klingai': 'Klingai',
    'kling': 'Klingai',
    'prunaai': 'Pruna',
    'pruna': 'Pruna',
    'hidream': 'HiDream',
    'avalai': 'AvalAI',
    'avail': 'AvalAI',
    'metisai': 'MetisAI',
    'runware': 'Runware',
    'rundiffusion': 'RunDiffusion',
    'elevenlabs': 'ElevenLabs',
    '11labs': 'ElevenLabs',
    'xai': 'xAI',
    'x.ai': 'xAI',
    'ideogram': 'Ideogram',
    'recraft': 'Recraft',
    'bria': 'Bria',
    'salesforce': 'Salesforce',
    'arcee': 'Arcee',
    'cartesia': 'Cartesia',
    'suno': 'Suno',
    'udio': 'Udio',
    'beatoven': 'Beatoven',
    'cassette': 'Cassette',
    'veed': 'Veed',
    'easel': 'Easel',
    'lightricks': 'Lightricks',
    'tripo3d': 'Tripo3D',
    'meshy': 'Meshy',
    'pixverse': 'Pixverse',
    'vidu': 'Vidu',
    'togethercomputer': 'Together',
    'together': 'Together',
    'together.ai': 'Together',
    'perplexity': 'Perplexity',
    'tavily': 'Tavily',
    'exa': 'Exa',
    'firecrawl': 'Firecrawl',
    'decart': 'Decart',
    'wan': 'Alibaba',
    'wan-ai': 'Alibaba',
    'krea': 'Krea',
    'cognition': 'Cognition',
    'cogito': 'DeepCogito',
    'deepcogito': 'DeepCogito',
    'arize': 'Arize',
    'marin': 'Marin',
    'virtue': 'Virtue',
    'refuel': 'Refuel',
    'essentialai': 'EssentialAI',
    'canopy': 'Canopy',
    'hexgrad': 'Hexgrad',
    'mercor': 'Mercor',
    'higgsfield': 'Higgsfield',
    'hvision': 'HVision',
    'lucataco': 'Lucataco',
    'nightmareai': 'NightmareAI',
    'philz1337x': 'Philz1337x',
    'fofr': 'Fofr',
    'pseudoram': 'Pseudoram',
    'smoretalk': 'SmoreTalk',
    'clarityai': 'ClarityAI',
    'simalabs': 'SimaLabs',
    'perceptron': 'Perceptron',
    'isaac': 'Isaac',
    'zai': 'Z.AI',
    'z.ai': 'Z.AI',
    'sourceful': 'Sourceful',
    'imagineart': 'ImagineArt',
    'leonardo': 'Leonardo',
    'dreamshaper': 'DreamShaper',
    'lykon': 'Lykon',
    'servicenow': 'ServiceNow',
    'apriel': 'Apriel',
    'resemble': 'Resemble',
    'argil': 'Argil',
    'mirelo': 'Mirelo',
    'moonvalley': 'MoonValley',
    'marey': 'Marey',
    'creatify': 'Creatify',
    'groq': 'Groq',
    'typhoon': 'Typhoon',
    'civitai': 'Civitai',
    'dataforseo': 'DataForSEO',
    'baai': 'BAAI',
    'byteplus': 'BytePlus',
    'imagination': 'Imagination',
    'yue': 'YuE',
    'zonos': 'Zonos',
    'qwen': 'Alibaba',
    'meta': 'Meta',
    'google': 'Google',
    'microsoft': 'Microsoft',
    'mistral': 'Mistral',
    'nvidia': 'Nvidia',
    'aws': 'AWS',
    'azure': 'Azure',
    'databricks': 'Databricks',
    '01.ai': '01.AI',
    'baidu': 'Baidu',
    'tencent': 'Tencent',
    'jina': 'Jina',
    'eleutheral': 'EleutherAL',
    'luma': 'Luma',
    'minimax': 'Minimax',
    'moonshot': 'Moonshot',
    'juggernaut': 'Juggernaut',
    'kuaishou': 'Kuaishou'
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
    'omnihuman': 'Bytedance',
    'hunyuan': 'Tencent',
    'hunyuan3d': 'Tencent',
    'cogvideo': 'Tencent',
    'cogview': 'Tencent',
    'minimax': 'Minimax',
    'hailuo': 'Minimax',
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
    'magi': 'Google',
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
    'marin': 'Marin',
    'chatterbox': 'Resemble',
    'csm': 'Stability',
    'yue': 'YuE',
    'zonos': 'Zonos'
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
    'embedding', 'embed', 'rerank', 'retriever', 'qa', 'search', 'ocr', 'kontext'
}

# Known mode keywords (for image generation, video, and multimodal models)
MODE_KEYWORDS = {
    'fill', 'redux', 'edit', 'inpaint', 'outpaint', 'upscale', 'enhance', 'remix',
    'text-to-image', 't2i', 'image-to-image', 'i2i', 'controlnet', 'lora',
    'krea', 'srpo', 'schnell', 'stream', 'multi', 'portrait', 'realism', 'anime',
    'text-to-video', 't2v', 'image-to-video', 'i2v', 'video-to-video', 'v2v',
    'text-to-audio', 't2a', 'audio-to-audio', 'a2a', 'video-to-audio', 'v2a',
    'speech-to-speech', 's2s', 'speech-to-text', 's2t', 'text-to-speech', 'tts',
    'image-to-3d', 'i23d', 'text-to-3d', 't23d', 'multiview', '3d', 'depth',
    'reframe', 'depth', 'pose', 'canny', 'effects', 'elements', 'transition',
    'lipsync', 'avatar', 'animate', 'stylize', 'transfer', 'removal', 'replace',
    'background', 'segment', 'detect', 'caption', 'grounding', 'generation',
    'trainer', 'training'
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
        
        # Model ID prefix patterns (company/model or company-model)
        self.model_id_company_patterns = [
            (r'^([a-zA-Z][a-zA-Z0-9._-]*)/.*', 'slash'),  # anthropic/claude, black-forest-labs/flux
            (r'^runware-([a-zA-Z][a-zA-Z0-9]*)-.*', 'runware'),  # runware-bfl-1-1
            (r'^([a-zA-Z][a-zA-Z0-9]*)-([a-zA-Z][a-zA-Z0-9]*)-.*', 'dash_prefix'),  # bytedance-seedance
        ]
    
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
        
        # Extract company (pass model_id for prefix detection)
        result.company = self._extract_company(combined, model_id or '')
        
        # Extract model family
        result.model_family = self._extract_model_family(combined, model_id or '')
        
        # Infer company from family if not found
        if not result.company and result.model_family:
            family_lower = result.model_family.lower()
            if family_lower in FAMILY_TO_COMPANY:
                result.company = FAMILY_TO_COMPANY[family_lower]
        
        # Extract size
        result.size = self._extract_size(combined)
        
        # Extract variants and modes (pass model_id for better mode detection)
        result.variants = self._extract_variants(combined)
        result.modes = self._extract_modes(combined, model_id or '')
        
        # Extract all significant tokens
        result.tokens = self._extract_tokens(combined)
        
        return result
    
    def _extract_version(self, text: str) -> Optional[str]:
        """Extract version number, prioritizing decimal versions and filtering false positives."""
        # Look for decimal versions first (1.1, 3.5, 2.0)
        decimal_match = re.search(r'(?<!\d)(\d+\.\d+)(?!\d)', text)
        if decimal_match:
            version = decimal_match.group(1)
            # Filter out years (2024.0, 2025.5, etc.)
            if float(version) < 100:  # Reasonable version numbers are < 100
                return version
        
        # Flux-specific fused patterns like "flux1.1", "flux-1", "flux.1"
        flux_match = re.search(r'flux[\s\._-]*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
        if flux_match:
            return flux_match.group(1)

        # Look for v-prefixed versions (v1, v2, v3.5) - most reliable
        v_match = re.search(r'\bv(\d{1,2}(?:\.\d+)?)\b', text, re.IGNORECASE)
        if v_match:
            return v_match.group(1)

        # Look for single/double digit versions, but avoid:
        # - Years (2024, 2025, 20240620)
        # - Dates (01-25, 240620)
        # - Model IDs (civitai-101055, runware-bfl-1-1)
        # - Long numbers (101055, 128078)
        # Only match if:
        # 1. Not in runware-{company}-{num} pattern
        # 2. Not followed by more digits (years/dates)
        # 3. Not part of civitai ID pattern
        # 4. Reasonable range (1-20)
        
        # Skip if it's a runware model ID pattern
        if re.search(r'runware-[a-z]+-\d', text, re.IGNORECASE):
            return None
        
        # Skip if it's a civitai pattern
        if re.search(r'civitai-\d', text, re.IGNORECASE):
            return None
        
        # Look for isolated single/double digit (space/dash separated, reasonable range)
        single_match = re.search(r'(?<![\d-])(\d{1,2})(?![\d-])(?![BMbm])', text)
        if single_match:
            version = single_match.group(1)
            version_num = int(version)
            # Only accept 1-20 as version (filters out years, dates, large IDs)
            if 1 <= version_num <= 20:
                # Additional check: not part of a date pattern (YYYY-MM-DD, MM-DD)
                if not re.search(r'\d{2,4}[-/]' + version + r'[-/]\d{2,4}', text):
                    return version
        
        return None
    
    def _extract_full_version(self, text: str) -> Optional[str]:
        """Extract full version string including 'v' prefix."""
        match = re.search(r'\b(v\d+(?:\.\d+)?)\b', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _extract_company(self, text: str, model_id: str = '') -> Optional[str]:
        """Extract company/provider name from text and model_id, ignoring API provider prefixes."""
        model_id_lower = model_id.lower()
        
        # IMPORTANT: Provider prefixes like 'fal-ai/', 'anthropic/', 'cohere/' are API providers,
        # NOT the model company. We need to look at the model name itself or after the prefix.
        
        # Clean model_id: Remove common API provider prefixes to get actual model identifier
        api_providers = ['fal-ai/', 'together/', 'replicate/', 'openrouter/', 'huggingface/']
        cleaned_model_id = model_id_lower
        for provider in api_providers:
            if cleaned_model_id.startswith(provider):
                cleaned_model_id = cleaned_model_id[len(provider):]
                break
        
        # Special case: runware-{company}-* pattern (runware is provider, but pattern indicates company)
        if model_id_lower.startswith('runware-'):
            runware_match = re.match(r'^runware-([a-zA-Z][a-zA-Z0-9]*)-', model_id_lower)
            if runware_match:
                company = runware_match.group(1)
                if company in ['bfl', 'black-forest-labs']:
                    return 'BFL'
                if company == 'bytedance':
                    return 'Bytedance'
                if company == 'bria':
                    return 'Bria'
                if company == 'civitai':
                    return 'Civitai'
                # Check if it's a known company
                if company in [c.lower() for c in KNOWN_COMPANIES]:
                    return company.title()
        
        # For model_ids like 'company/model' or 'company-model' (AFTER removing provider prefix)
        # This handles cases like anthropic/claude, black-forest-labs/flux in OpenRouter
        if cleaned_model_id != model_id_lower:  # Had a provider prefix removed
            # Check for company/model pattern
            slash_match = re.match(r'^([a-zA-Z][a-zA-Z0-9._-]+)/', cleaned_model_id)
            if slash_match:
                company = slash_match.group(1).replace('-', ' ').replace('_', ' ').lower()
                # Only use if it's a known company
                if company in [c.lower() for c in KNOWN_COMPANIES]:
                    # Use capitalization mapping
                    return COMPANY_CAPITALIZATION.get(company, company.title())
        
        # Check cleaned model_id for company-model pattern (e.g., bytedance-seedance, alibaba-qwen)
        dash_match = re.match(r'^([a-zA-Z][a-zA-Z0-9]+)-([a-zA-Z][a-zA-Z0-9]+)', cleaned_model_id)
        if dash_match:
            potential_company = dash_match.group(1).lower()
            # Check if first part is a known company
            if potential_company in [c.lower() for c in KNOWN_COMPANIES]:
                # Use capitalization mapping
                return COMPANY_CAPITALIZATION.get(potential_company, potential_company.title())
        
        # Exact word match in combined text (name + cleaned model_id)
        combined_text = f"{text} {cleaned_model_id}"
        match = self.company_pattern.search(combined_text)
        
        # Check if model NAME starts with a company name (e.g., "Alibaba qwen3-32b", "Anthropic claude-4")
        # This handles cases where the display name includes the company - check this FIRST for priority
        name_parts = text.split()
        if name_parts:
            first_word = name_parts[0].lower()
            if first_word in [c.lower() for c in KNOWN_COMPANIES]:
                # Use capitalization mapping for consistency
                return COMPANY_CAPITALIZATION.get(first_word, first_word.title())
        
        # Pattern match from combined text
        if match:
            company = match.group(1).lower()
            # Use capitalization mapping for consistency
            return COMPANY_CAPITALIZATION.get(company, company.title())
        
        # Try partial matching for known companies (use cleaned text)
        combined_text_lower = f"{text} {cleaned_model_id}".lower()
        for known_company in sorted(KNOWN_COMPANIES, key=len, reverse=True):
            if known_company in combined_text_lower:
                if re.search(r'\b' + re.escape(known_company) + r'\b', combined_text_lower, re.IGNORECASE):
                    company = known_company.lower()
                    # Use capitalization mapping for consistency
                    return COMPANY_CAPITALIZATION.get(company, company.title())
        
        return None
    
    def _extract_model_family(self, text: str, model_id: str = '') -> Optional[str]:
        """Extract model family name - tries exact match first, then partial."""
        text_lower = text.lower()
        model_id_lower = model_id.lower()
        
        # Clean model_id: Remove API provider prefixes to get actual model identifier
        api_providers = ['fal-ai/', 'together/', 'replicate/', 'openrouter/', 'huggingface/']
        cleaned_model_id = model_id_lower
        for provider in api_providers:
            if cleaned_model_id.startswith(provider):
                cleaned_model_id = cleaned_model_id[len(provider):]
                break
        
        # Also remove runware prefix when looking for family
        if cleaned_model_id.startswith('runware-'):
            cleaned_model_id = cleaned_model_id.replace('runware-', '', 1)
        
        # Exact word match first (check both text and cleaned model_id)
        combined_for_family = f"{text} {cleaned_model_id}"
        match = self.model_family_pattern.search(combined_for_family)
        if match:
            family = match.group(1).lower()
            # Normalize variations
            if family == 'sd':
                return 'Stable-Diffusion'
            if family in ['kling', 'klingai']:
                return 'Kling'
            if family == 'flux':
                return 'Flux'
            if family == 'claude':
                return 'Claude'
            if family in ['seedance', 'seedream', 'seededit']:
                return family.title()
            return family.title()
        
        # Try partial matching for known families
        for known_family in sorted(KNOWN_MODEL_FAMILIES, key=len, reverse=True):
            if known_family in text_lower or known_family in cleaned_model_id:
                # Check in combined text
                if re.search(r'\b' + re.escape(known_family) + r'\b', combined_for_family, re.IGNORECASE):
                    family = known_family.lower()
                    if family == 'sd':
                        return 'Stable-Diffusion'
                    if family in ['kling', 'klingai']:
                        return 'Kling'
                    if family == 'flux':
                        return 'Flux'
                    if family == 'claude':
                        return 'Claude'
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
    
    def _extract_modes(self, text: str, model_id: str = '') -> List[str]:
        """Extract mode keywords (for image generation, video, etc.)."""
        modes = []
        text_lower = text.lower()
        
        # First, extract from model_id path (e.g., fal-ai/wan/v2.2-5b/text-to-video)
        if model_id:
            model_id_lower = model_id.lower()
            
            # Remove API provider prefixes
            api_providers = ['fal-ai/', 'together/', 'replicate/', 'openrouter/', 'huggingface/']
            cleaned_model_id = model_id_lower
            for provider in api_providers:
                if cleaned_model_id.startswith(provider):
                    cleaned_model_id = cleaned_model_id[len(provider):]
                    break
            
            # Look for modes in the path (e.g., /text-to-video, /image-to-image)
            for keyword in MODE_KEYWORDS:
                if keyword in cleaned_model_id:
                    # Check if it's part of a path segment (after a slash)
                    if '/' + keyword in cleaned_model_id or cleaned_model_id.startswith(keyword):
                        mode_title = keyword.title()
                        # Special handling for hyphenated modes
                        if '-to-' in keyword:
                            parts = keyword.split('-')
                            mode_title = '-'.join(p.capitalize() for p in parts)
                        if mode_title not in modes:
                            modes.append(mode_title)
        
        # Then extract from display name/text
        for keyword in MODE_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                mode_title = keyword.title()
                # Special handling for hyphenated modes
                if '-to-' in keyword:
                    parts = keyword.split('-')
                    mode_title = '-'.join(p.capitalize() for p in parts)
                if mode_title not in modes:
                    modes.append(mode_title)
        
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
