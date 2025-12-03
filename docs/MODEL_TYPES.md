# Model Type Configuration

This document explains the centralized model type system used in the AI Cost Manager.

## Overview

The model type system uses a **two-tier structure**:
- **`model_type`**: Broad, strictly validated categories
- **`category`**: Specific, flexible tags (NOT validated)

This design ensures consistency across extractors while preserving flexibility for nuanced categorization.

## Configuration File

All valid model types are defined in: **`ai_cost_manager/model_types.py`**

### Valid Model Types (Strict)

These are the only allowed values for `model_type` field:

```python
VALID_MODEL_TYPES = [
    'text-generation',      # Text completion, chat, language models
    'image-generation',     # Image creation and manipulation
    'video-generation',     # Video creation and manipulation
    'audio-generation',     # Audio/speech creation and manipulation
    'embeddings',          # Vector embeddings models
    'code-generation',     # Code completion and generation
    'reranking',           # Search result reranking
    'moderation',          # Content moderation
    'search',              # Search and retrieval
    'other',               # Fallback for unclassified models
]
```

### Categories/Tags (Flexible)

The `category` field is **NOT validated** and can contain any value that makes sense for the model. Common examples include:

**Image Generation:**
- `text-to-image`
- `image-to-image`
- `image-upscaling`
- `image-editing`
- `background-removal`
- `inpainting`
- `outpainting`

**Video Generation:**
- `text-to-video`
- `image-to-video`
- `video-editing`
- `video-upscaling`

**Audio Generation:**
- `text-to-speech`
- `speech-to-text`
- `audio-to-text`
- `music-generation`
- `voice-cloning`

**Text Generation:**
- `chat`
- `completion`
- `translation`
- `summarization`
- `question-answering`

## Usage in Extractors

All extractors follow this pattern:

1. **Import the centralized types:**
   ```python
   from ai_cost_manager.model_types import VALID_MODEL_TYPES, get_valid_types_string
   ```

2. **Map native types to broad categories:**
   ```python
   # Map provider-specific types to broad types
   if native_type == "image":
       model_type = "image-generation"
   elif native_type == "video":
       model_type = "video-generation"
   ```

3. **Determine specific category:**
   ```python
   # Set specific category based on model characteristics
   if "upscale" in model_name:
       category = "image-upscaling"
   elif "background" in model_name:
       category = "background-removal"
   else:
       category = "text-to-image"
   ```

4. **Validate LLM-provided types:**
   ```python
   if llm_extracted.get('model_type'):
       llm_model_type = llm_extracted['model_type']
       
       # Strict validation
       if llm_model_type in VALID_MODEL_TYPES:
           model_type = llm_model_type
       else:
           print(f"Invalid type '{llm_model_type}', expected: {get_valid_types_string()}")
   
   # Category is accepted without validation
   if llm_extracted.get('category'):
       category = llm_extracted['category']
   ```

## LLM Extraction

When using LLM extraction (`use_llm=True`), the LLM is instructed to:

1. Return a **broad `model_type`** from the strict list
2. Optionally provide a **specific `category`** with full flexibility

The LLM prompt includes:
- The complete list of valid model types
- Examples showing the two-tier structure
- Guidance that categories should be specific and descriptive

## Adding New Model Types

To add a new valid model type:

1. Edit `ai_cost_manager/model_types.py`
2. Add the new type to `VALID_MODEL_TYPES` list
3. Add documentation comment explaining when to use it
4. Optionally add common categories to `COMMON_CATEGORIES` dict
5. Update LLM prompt in `ai_cost_manager/llm_extractor.py` if needed
6. All extractors will automatically use the new type

## Rationale

**Why strict validation for `model_type`?**
- Ensures consistency across all providers
- Enables reliable filtering and grouping in UI
- Prevents type proliferation (e.g., "chat" vs "completion" vs "text-generation")
- Makes database queries predictable

**Why flexible `category` field?**
- Allows nuanced categorization without constraints
- LLMs can provide detailed, provider-specific tags
- New model capabilities don't require code changes
- Better for display and detailed filtering

## Extractors Updated

All extractors use centralized model types:
- ✅ FAL Extractor
- ✅ Runware Extractor
- ✅ AvalAI Extractor
- ✅ Together Extractor
- ✅ MetisAI Extractor
- ✅ OpenRouter Extractor
