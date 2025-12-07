"""
LLM Pricing Extractor

Uses LLM (via OpenRouter) to intelligently extract and parse pricing information.
"""
import json
from typing import Dict, Any, Optional
from ai_cost_manager.database import get_session
from ai_cost_manager.models import LLMConfiguration
from ai_cost_manager.llm_client import LLMClient


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
            llm_client = LLMClient(self.config)
            response = llm_client.chat(prompt, temperature=0.1, max_tokens=500)
            result = llm_client.parse_response(response)
            return result
        except Exception as e:
            print(f"  LLM extraction failed: {e}")
            return self._fallback_extraction(model_data)
    
    def _build_prompt(self, model_data: Dict[str, Any]) -> str:
        """Build prompt for LLM"""
        model_name = model_data.get('name', 'Unknown')
        model_description = model_data.get('description', '')
        pricing_info = model_data.get('pricing_info', '')
        credits = model_data.get('creditsRequired', 0)
        model_type = model_data.get('model_type', 'other')
        
        # Extract additional context from all available sources
        tags = model_data.get('tags', [])
        raw_metadata = model_data.get('raw_metadata', {})
        input_schema = model_data.get('input_schema', {})
        output_schema = model_data.get('output_schema', {})
        playground_data = model_data.get('playground_data', {})
        
        # Build comprehensive context
        context_parts = []
        
        # Add tags/categories
        if tags:
            context_parts.append(f"Categories/Tags: {', '.join(tags) if isinstance(tags, list) else tags}")
        
        # Add schema information (parameter hints)
        schema_hints = []
        if input_schema and isinstance(input_schema, dict):
            properties = input_schema.get('properties', {})
            if 'resolution' in properties:
                schema_hints.append("supports resolution parameter")
            if 'duration' in properties or 'num_frames' in properties:
                schema_hints.append("supports duration/frames parameter")
            if 'steps' in properties or 'num_inference_steps' in properties:
                schema_hints.append("supports inference steps parameter")
        
        if schema_hints:
            context_parts.append(f"Model capabilities: {', '.join(schema_hints)}")
        
        # Add playground pricing if available
        if playground_data and isinstance(playground_data, dict):
            playground_price = playground_data.get('price')
            playground_pricing_text = playground_data.get('pricing_text')
            if playground_price:
                context_parts.append(f"Playground price: {playground_price} credits")
            if playground_pricing_text:
                context_parts.append(f"Playground pricing: {playground_pricing_text}")
        
        # Add raw metadata pricing details
        if raw_metadata and isinstance(raw_metadata, dict):
            # Check for pricing override
            pricing_override = raw_metadata.get('pricingInfoOverride')
            if pricing_override:
                context_parts.append(f"Pricing override: {pricing_override}")
            
            # Check for billing message
            billing_msg = raw_metadata.get('billingMessage')
            if billing_msg:
                context_parts.append(f"Billing info: {billing_msg}")
            
            # Check for machine type (cost indicator)
            machine_type = raw_metadata.get('machineType')
            if machine_type:
                context_parts.append(f"Machine type: {machine_type}")
        
        additional_context = "\n".join(context_parts) if context_parts else "No additional context available"
        
        prompt = f"""Extract pricing information from this AI model data and return a JSON object.

Model Name: {model_name}
Model Description: {model_description}
Current Type: {model_type}
Credits Required: {credits}
Pricing Info Text: "{pricing_info}"

ADDITIONAL CONTEXT:
{additional_context}

CRITICAL TYPE DETECTION RULES (check ALL signals, not just name):

IMPORTANT: Check the ADDITIONAL CONTEXT section above for Categories/Tags! Use them as PRIMARY indicators!

1. **3D MODELS** → "model-3d"
   ✓ Name contains: 3D, Hyper3D, Meshy, TripoSR, Shap-E, Point-E, InstantMesh, Part, Hunyuan
   ✓ Description mentions: 3d, mesh, point cloud, nerf, gaussian splatting, 3D files
   ✓ Categories/Tags contain: 3D, 3D-to-3D, text-to-3d, image-to-3d, mesh-generation, point-cloud, point cloud
   ⚠️ PRIORITY: If Categories/Tags mention "3D" anywhere, return "model-3d" (NOT "other")!

2. **TRAINING MODELS** → "training"
   ✓ Name contains: LoRA, DreamBooth, Fine-tune, Training, Trainer, Adapter, Qwen, Custom
   ✓ Description mentions: train, fine-tuning, custom model, adapter, lora
   ✓ Category: fine-tuning, lora-training, dreambooth, custom-training, image-trainer
   ✗ NOT if name contains: generate, generation (those are for image/video generation)

3. **TEXT-TO-VIDEO & VIDEO CAPTIONING** - Be careful with distinction!
   
   a) **VIDEO GENERATION** → "video-generation"
      ✓ Name contains: Runway, Pika, Sora, AnimateDiff, SVD, Video Generation, GenVideo
      ✓ Description mentions: "generate video", "create video", "text-to-video", "video synthesis"
      ✓ Category: text-to-video, image-to-video, video-generation
   
   b) **VIDEO CAPTIONING/DESCRIPTION** → "text-generation" (it's an LLM analyzing video!)
      ✓ Name contains: Video Captioner, Video Caption, Video Description, Video-LLM, VideoLLaMA
      ✓ Description mentions: "caption video", "describe video", "video understanding", "video Q&A", "analyze video"
      ✓ Category: video-captioning, video-to-text, video-description, video-qa
      ⚠️ CRITICAL: If model takes VIDEO as INPUT and generates TEXT, it's "text-generation" NOT "video-generation"!

4. **VISION/MULTIMODAL MODELS** → "text-generation"
   ✓ Name contains: Florence, CLIP, BLIP, LLaVA, GPT-4V, Gemini Vision, Qwen-VL, InternVL
   ✓ Description mentions: vision-language, multimodal, image understanding, visual question answering, OCR
   ✓ Category: vision-language, vqa, multimodal, image-to-text, ocr
   ⚠️ These are LLMs with vision capabilities, return "text-generation"!

5. **MODERATION** → "moderation"
   ✓ Name contains: NSFW, Moderation, Safety, Filter, Content Filter
   ✓ Description mentions: moderation, safety, inappropriate, filter content
   ✓ Category: content-moderation, nsfw-detection, safety-filter
   ⚠️ PRIORITY: Check this BEFORE audio/video types

6. **IMAGE GENERATION** → "image-generation"
   ✓ Name contains: FLUX, Stable Diffusion, DALL-E, Midjourney, Imagen, Kandinsky
   ✓ Description mentions: image generation, create images, photo, art
   ✓ Category: text-to-image, image-to-image, inpainting, upscaling
   ✗ NOT if name is about detection/moderation

7. **AUDIO GENERATION** → "audio-generation"
   ✓ Name contains: ElevenLabs, Whisper, Bark, MusicGen, TTS, Speech
   ✓ Description mentions: audio, speech, voice, music, sound generation
   ✓ Category: text-to-speech, speech-to-text, music-generation
   ✗ NOT if name just contains random words

8. **CODE GENERATION** → "code-generation"
   ✓ Name contains: Code, CodeLlama, StarCoder, Codex, Replit, CodeGen
   ✓ Description mentions: code, programming, developer
   ✓ Category: code-completion, code-generation

9. **TEXT/LLM** → "text-generation"
   ✓ Name contains: GPT, Claude, Llama, Mistral, Gemini, Chat, LLM
   ✓ Description mentions: language model, chat, text generation
   ✓ Category: chat, completion, vision (if accepts images)
   ⚠️ INCLUDES vision-language models (Florence, BLIP, etc.) and video captioning models!

10. **EMBEDDINGS** → "embeddings"
    ✓ Name contains: embedding, CLIP, E5, BGE, embed
    ✓ Description mentions: vector, embedding, semantic search

11. **RERANKING** → "reranking"
    ✓ Name contains: rerank, cross-encoder
    ✓ Description mentions: reranking, relevance scoring

12. **SEARCH** → "search"
    ✓ Name contains: search, retrieval
    ✓ Description mentions: search engine, information retrieval

13. **DETECTION** → "detection"
    ✓ Name contains: SAM, YOLO, Detect, Segment, Mask
    ✓ Description mentions: object detection, segmentation, instance segmentation
    ✓ Category: object-detection, segmentation, instance-segmentation

DETECTION PRIORITY (check in this order):
1. First: Is it MODERATION/NSFW filter? → "moderation"
2. Second: Is it TRAINING model? → "training" (includes fine-tuning, LoRA, custom trainers)
3. Third: Is it 3D model? → "model-3d"
4. Fourth: Is it DETECTION/SEGMENTATION? → "detection"
5. Fifth: Is it VIDEO CAPTIONING (video→text)? → "text-generation" (NOT video-generation!)
6. Sixth: Is it VISION/MULTIMODAL (image→text)? → "text-generation" (Florence, BLIP, etc.)
7. Then check: Other specific types (code, image generation, video generation, audio, etc.)
8. Last resort: → "other"

IMPORTANT: Analyze the pricing carefully and return a JSON object with these exact fields:

⚠️ CRITICAL CONSTRAINT - model_type MUST BE ONE OF THESE EXACT VALUES ONLY:
   'text-generation', 'image-generation', 'video-generation', 'audio-generation', 'model-3d',
   'embeddings', 'code-generation', 'reranking', 'moderation', 'search', 'training', 'detection', 'other'
   
   NO OTHER VALUES ARE ACCEPTABLE. If unsure, use 'other'.
   Do NOT return: "chat", "chat-completion", "completion", "image-gen", "video-gen", "audio-gen", "tts", "embed", "code", or any variant.
   ALWAYS return the EXACT canonical value from the list above.

REQUIRED FIELDS:
- model_type: MUST be ONE of the 13 canonical types listed above. Use PRIORITY order!
  * Check for moderation/NSFW first → ONLY return "moderation" (not "filter", not "safety")
  * Then check for 3D/detection/training → ONLY "model-3d", "detection", "training"
  * VIDEO CAPTIONING (video→text) → "text-generation" (NOT "video-generation")
  * VISION MODELS (Florence, BLIP) → "text-generation" (NOT "other")
  * Be careful not to misclassify based on partial word matches
  * "NSFW Filter" → "moderation" (NOT "audio" or anything else)
  * "SAM" → "detection" (NOT "training")
  * "Hyper3D" → "model-3d" (NOT "other")
  * "GPT-4" → "text-generation" (NOT "chat" or "completion")

- category: SPECIFIC type (optional but recommended), one of:
  * "text-to-image", "image-to-image", "image-upscaling" (for image models)
  * "text-to-video", "image-to-video" (for video models)
  * "image-to-text", "audio-to-text", "text-to-speech" (for transcription/generation)
  * "fine-tuning", "lora-training", "dreambooth", "custom-training", "image-trainer" (for training)
  * "object-detection", "segmentation", "sam", "yolo" (for detection models)
  * Or leave empty/null if no specific variant applies
  * NEVER return "none" - either return a valid category or leave empty
- pricing_type: MUST be one of: "per_token", "per_image", "per_video", "per_minute", "per_second", "per_call", "hourly", "fixed", "tiered", "per_megapixel"
- cost_unit: MUST match pricing_type - "token", "image", "video", "minute", "second", "call", "hour", "megapixel"
- cost_per_call: numeric value in USD for ONE typical call (required, use 0 if unknown)

OPTIONAL FIELDS:
- pricing_formula: human-readable description
- input_cost_per_unit: numeric cost per input unit (e.g., per 1K tokens)
- output_cost_per_unit: numeric cost per output unit
- credits_required: numeric credits needed per call
- pricing_variables: object with pricing factors like resolution, duration, steps
- description: clear explanation of the pricing model
- tags: array of relevant tags (e.g., ["fast", "high-quality", "enterprise", "open-source"])
- category: original source category if available

PRICING FORMAT EXAMPLES:

1. Text models (tokens):
   "$0.15 / 1M input tokens, $0.60 / 1M output tokens"
   → {{"model_type": "text-generation", "pricing_type": "per_token", "cost_unit": "token", "input_cost_per_unit": 0.00015, "output_cost_per_unit": 0.0006, "cost_per_call": 0.00075}}

2. Image models (megapixels):
   "$0.025/MP | 40.0 img/$1 | 28 steps"
   → {{"model_type": "image-generation", "category": "text-to-image", "pricing_type": "per_image", "cost_unit": "megapixel", "cost_per_call": 0.025, "pricing_variables": {{"price_per_mp": 0.025, "images_per_dollar": 40.0, "default_steps": 28}}, "tags": ["fast", "affordable"]}}

3. Video models:
   "$0.19 per 5s 720p video"
   → {{"model_type": "video-generation", "category": "text-to-video", "pricing_type": "per_video", "cost_unit": "video", "cost_per_call": 0.19, "pricing_variables": {{"duration_seconds": 5, "resolution": "720p"}}, "tags": ["short-form", "hd"]}}

3b. Image-to-video models:
   "AnimateDiff - image to video"
   → {{"model_type": "video-generation", "category": "image-to-video", "pricing_type": "per_video", "cost_unit": "video", "cost_per_call": 0.25, "tags": ["animation", "ai-generated"]}}

3c. Image upscaling models:
   "Upscale 4x - $0.01 per image"
   → {{"model_type": "image-generation", "category": "image-to-image", "pricing_type": "per_image", "cost_unit": "image", "cost_per_call": 0.01, "tags": ["upscaling", "enhancement"]}}

4. Audio models:
   "$0.006 per minute"
   → {{"model_type": "audio-generation", "pricing_type": "per_minute", "cost_unit": "minute", "cost_per_call": 0.006}}

5. Credits-based:
   "2.5 credits" (assuming 1 credit = $0.01)
   → {{"model_type": "image-generation", "pricing_type": "per_call", "cost_unit": "call", "cost_per_call": 0.025, "credits_required": 2.5}}

6. Hourly:
   "$2.40/hour"
   → {{"model_type": "other", "pricing_type": "hourly", "cost_unit": "hour", "cost_per_call": 2.40}}

CRITICAL RULES:
- If pricing is in $/MP (dollars per megapixel), use pricing_type="per_image" and cost_unit="megapixel"
- If pricing shows "img/$" (images per dollar), calculate: cost_per_call = 1 / images_per_dollar
- Always extract numeric values from pricing_variables (e.g., "28 steps" → 28, not "28")
- cost_per_call should be a realistic USD amount (typically 0.0001 to 10.0)
- Return ONLY valid JSON with no markdown, no explanations, no code blocks"""
        
        return prompt
    
    # _call_llm and _parse_llm_response are now handled by LLMClient
    
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
