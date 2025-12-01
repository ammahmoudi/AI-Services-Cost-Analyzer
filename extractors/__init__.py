"""
Extractor Registry

Central registry for all available extractors.
"""
from extractors.fal_extractor import FalAIExtractor
from extractors.together_extractor import TogetherAIExtractor
from extractors.avalai_extractor import AvalAIExtractor
from extractors.metisai_extractor import MetisAIExtractor
from extractors.runware_extractor import RunwareExtractor

# Registry of available extractors (with aliases)
EXTRACTORS = {
    'fal': FalAIExtractor,
    'falai': FalAIExtractor,  # Alias
    'fal.ai': FalAIExtractor,  # Alias
    'together': TogetherAIExtractor,
    'togetherai': TogetherAIExtractor,  # Alias
    'together.ai': TogetherAIExtractor,  # Alias
    'avalai': AvalAIExtractor,
    'aval': AvalAIExtractor,  # Alias
    'aval.ai': AvalAIExtractor,  # Alias
    'metisai': MetisAIExtractor,
    'metis': MetisAIExtractor,  # Alias
    'metis.ai': MetisAIExtractor,  # Alias
    'runware': RunwareExtractor,
    'runware.ai': RunwareExtractor,  # Alias
}

# Primary extractor names (for display)
PRIMARY_EXTRACTORS = {
    'fal': 'Fal.ai (aliases: falai, fal.ai)',
    'together': 'Together AI (aliases: togetherai, together.ai)',
    'avalai': 'AvalAI (aliases: aval, aval.ai)',
    'metisai': 'MetisAI (aliases: metis, metis.ai)',
    'runware': 'Runware (aliases: runware.ai)',
}

# Extractor feature support matrix
EXTRACTOR_FEATURES = {
    'fal': {
        'supports_schemas': True,
        'supports_llm': True,
        'supports_playground': True,
    },
    'together': {
        'supports_schemas': False,
        'supports_llm': True,
        'supports_playground': False,
    },
    'avalai': {
        'supports_schemas': False,
        'supports_llm': True,
        'supports_playground': False,
    },
    'metisai': {
        'supports_schemas': False,
        'supports_llm': True,
        'supports_playground': False,
    },
    'runware': {
        'supports_schemas': False,
        'supports_llm': True,
        'supports_playground': False,
    },
}


def get_extractor(extractor_name: str):
    """
    Get an extractor class by name.
    
    Args:
        extractor_name: Name of the extractor (e.g., 'fal')
        
    Returns:
        Extractor class (not instance)
        
    Raises:
        ValueError: If extractor not found
    """
    extractor_class = EXTRACTORS.get(extractor_name.lower())
    
    if not extractor_class:
        available = ', '.join(EXTRACTORS.keys())
        raise ValueError(
            f"Unknown extractor: '{extractor_name}'. "
            f"Available extractors: {available}"
        )
    
    return extractor_class


def list_extractors(include_aliases=False):
    """
    List all available extractors.
    
    Args:
        include_aliases: If True, include all aliases. If False, only primary names.
    
    Returns:
        List of extractor names or dict of extractors with descriptions
    """
    if include_aliases:
        return list(EXTRACTORS.keys())
    else:
        return list(PRIMARY_EXTRACTORS.keys())


def get_extractor_features(extractor_name: str) -> dict:
    """
    Get feature support information for an extractor.
    
    Args:
        extractor_name: Name of the extractor (e.g., 'fal')
        
    Returns:
        Dict with feature flags (supports_schemas, supports_llm, etc.)
    """
    # Normalize to primary name
    extractor_class = EXTRACTORS.get(extractor_name.lower())
    if not extractor_class:
        return {
            'supports_schemas': False,
            'supports_llm': False,
            'supports_playground': False,
        }
    
    # Find primary name
    for primary_name, cls in EXTRACTORS.items():
        if cls == extractor_class and primary_name in EXTRACTOR_FEATURES:
            return EXTRACTOR_FEATURES[primary_name]
    
    # Default to no features
    return {
        'supports_schemas': False,
        'supports_llm': False,
        'supports_playground': False,
    }
