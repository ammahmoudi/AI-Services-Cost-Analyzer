"""
Extractor Registry

Central registry for all available extractors.
"""
from extractors.fal_extractor import FalAIExtractor

# Registry of available extractors (with aliases)
EXTRACTORS = {
    'fal': FalAIExtractor,
    'falai': FalAIExtractor,  # Alias
    'fal.ai': FalAIExtractor,  # Alias
}

# Primary extractor names (for display)
PRIMARY_EXTRACTORS = {
    'fal': 'Fal.ai (aliases: falai, fal.ai)',
}


def get_extractor(extractor_name: str, source_url: str, **kwargs):
    """
    Get an extractor instance by name.
    
    Args:
        extractor_name: Name of the extractor (e.g., 'fal')
        source_url: URL for the API source
        **kwargs: Additional arguments for extractor
        
    Returns:
        Extractor instance
        
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
    
    return extractor_class(source_url)


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
