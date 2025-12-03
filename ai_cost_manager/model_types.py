"""
Centralized model type definitions for AI Cost Manager.

This module defines the standardized model types used across all extractors.
Model types are strictly validated to ensure consistency across the system.

Categories/tags are intentionally kept flexible and are NOT validated,
allowing LLMs to provide nuanced categorization (e.g., text-to-image, 
image-to-video, audio-to-text, etc.) without strict constraints.
"""

# Standardized broad model types - these are STRICTLY validated
# across all extractors and LLM extraction
VALID_MODEL_TYPES = [
    'text-generation',      # Text completion, chat, language models
    'image-generation',     # Image creation and manipulation
    'video-generation',     # Video creation and manipulation
    'audio-generation',     # Audio/speech creation and manipulation
    '3d-generation',       # 3D model creation and manipulation
    'embeddings',          # Vector embeddings models
    'code-generation',     # Code completion and generation
    'reranking',           # Search result reranking
    'moderation',          # Content moderation
    'search',              # Search and retrieval
    'training',            # Model training, fine-tuning, LoRA training
    'detection',           # Object detection, segmentation (SAM, YOLO, etc.)
    'other',               # Fallback for unclassified models
]

# Common category examples (NOT exhaustive, NOT validated)
# These are provided as guidance for LLM extraction and human reference
# LLMs are free to return any category that makes sense for the model
COMMON_CATEGORIES = {
    'image-generation': [
        'text-to-image',
        'image-to-image',
        'image-upscaling',
        'image-editing',
        'background-removal',
        'inpainting',
        'outpainting',
    ],
    'video-generation': [
        'text-to-video',
        'image-to-video',
        'video-editing',
        'video-upscaling',
    ],
    'audio-generation': [
        'text-to-speech',
        'speech-to-text',
        'audio-to-text',
        'music-generation',
        'voice-cloning',
    ],
    'text-generation': [
        'chat',
        'completion',
        'translation',
        'summarization',
        'question-answering',
    ],
    '3d-generation': [
        'text-to-3d',
        'image-to-3d',
        '3d-modeling',
        '3d-reconstruction',
        'mesh-generation',
    ],
    'training': [
        'fine-tuning',
        'lora-training',
        'dreambooth',
        'model-training',
        'adapter-training',
        'custom-model-training',
    ],
    'detection': [
        'object-detection',
        'segmentation',
        'instance-segmentation',
        'semantic-segmentation',
        'pose-detection',
        'face-detection',
        'sam',  # Segment Anything Model
        'yolo',
    ],
}


def is_valid_model_type(model_type: str) -> bool:
    """
    Check if a model type is valid.
    
    Args:
        model_type: The model type to validate
        
    Returns:
        True if the model type is in VALID_MODEL_TYPES, False otherwise
    """
    return model_type in VALID_MODEL_TYPES


def get_valid_types_string() -> str:
    """
    Get a comma-separated string of valid model types for error messages.
    
    Returns:
        Comma-separated string of valid types
    """
    return ', '.join(VALID_MODEL_TYPES)
