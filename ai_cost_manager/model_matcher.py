"""
Model Matcher - Identifies same models across different providers using LLM

This module helps connect models that are the same underlying model but offered
through different providers (e.g., Flux Dev on Replicate vs FAL vs Runware)
"""

from typing import List, Dict, Optional, Any
import json
from dataclasses import dataclass
from ai_cost_manager.llm_client import LLMClient
from ai_cost_manager.database import get_session
from ai_cost_manager.models import LLMConfiguration
from ai_cost_manager.model_types import VALID_MODEL_TYPES


@dataclass
class ModelMatch:
    """Represents a matched model across providers"""
    canonical_name: str  # Standard model name (e.g., "flux-dev-1.0")
    confidence: float  # Match confidence 0-1
    models: List[Dict]  # List of model records that match
    
    def get_best_price(self) -> Optional[Dict]:
        """Get the provider offering the best price"""
        valid_models = [m for m in self.models if m.get('cost_per_call') and m['cost_per_call'] > 0]
        if not valid_models:
            return None
        return min(valid_models, key=lambda x: x['cost_per_call'])
    
    def get_all_providers(self) -> List[Dict]:
        """Get all providers offering this model with their prices"""
        return sorted(
            self.models,
            key=lambda x: x.get('cost_per_call', float('inf'))
        )




class ModelMatcher:
    """Matches models across different providers using LLM (OpenRouter or OpenAI via unified config)"""

    def __init__(self):
        # Load active LLM configuration directly
        session = get_session()
        try:
            self.config = session.query(LLMConfiguration).filter_by(is_active=True).first()
        finally:
            session.close()
        self.match_cache = {}  # Cache for model matches
        
    def normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for comparison"""
        if not model_name:
            return ""
        
        # Convert to lowercase and remove common variations
        normalized = model_name.lower().strip()
        
        # Remove version suffixes that are just formatting differences
        replacements = {
            '-': ' ',
            '_': ' ',
            '.': ' ',
            '  ': ' ',
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized.strip()
    
    def extract_model_features(self, model: Dict) -> Dict[str, Any]:
        """Extract key features from model for matching"""
        return {
            'name': model.get('name', ''),
            'model_name': model.get('model_name', ''),
            'model_id': model.get('model_id', ''),
            'description': model.get('description', ''),
            'model_type': model.get('model_type', ''),
            'provider': model.get('provider', ''),
            'tags': model.get('tags', []),
            'normalized_name': self.normalize_model_name(model.get('name', '')),
        }
    
    def match_models_with_llm(self, models: List[Dict]) -> List[ModelMatch]:
        """
        Use LLM to match models across providers using unified LLM config (OpenRouter or OpenAI)
        """
        if not self.config:
            print("Warning: No active LLM configuration, using basic matching")
            return self._match_models_basic(models)

        # Group by model type first for efficiency
        # Model types are validated against VALID_MODEL_TYPES (from model_types.py)
        by_type = {}
        for model in models:
            model_type = model.get('model_type', 'other')
            
            # Validate model_type against centralized list
            if model_type not in VALID_MODEL_TYPES:
                print(f"⚠️  Invalid model_type '{model_type}' found during matching, treating as 'other'")
                model_type = 'other'
            
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append(model)

        all_matches = []

        # Process each type separately
        for model_type, type_models in by_type.items():
            if len(type_models) < 2:
                # Single model, no matching needed
                for model in type_models:
                    all_matches.append(ModelMatch(
                        canonical_name=model.get('name', model.get('model_id', 'unknown')),
                        confidence=1.0,
                        models=[model]
                    ))
                continue

            # Build context for LLM
            model_summaries = []
            for i, model in enumerate(type_models):
                features = self.extract_model_features(model)
                # Include model_name and more description for better matching
                summary = {
                    'index': i,
                    'name': features['name'],
                    'model_name': features['model_name'] if features['model_name'] else features['name'],
                    'provider': features['provider'],
                    'model_id': features['model_id'],
                    'description': features['description'][:300] if features['description'] else '',
                    'tags': features['tags'][:8] if features['tags'] else [],
                }
                model_summaries.append(summary)

            # Debug: Print sample of models being matched
            if len(model_summaries) > 0:
                print(f"\n🔍 Matching {len(model_summaries)} {model_type} models...")
                if len(model_summaries) <= 5:
                    for summary in model_summaries:
                        print(f"  [{summary['index']}] {summary['name']} ({summary['provider']})")
                else:
                    print(f"  Sample: {model_summaries[0]['name']}, {model_summaries[1]['name']}, ...")

            # Process in batches if too many models (max 50 per batch to avoid context overflow)
            BATCH_SIZE = 50
            all_batch_matches = []
            
            if len(model_summaries) > BATCH_SIZE:
                num_batches = (len(model_summaries) + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"  ⚡ Processing in {num_batches} batches...")
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min((batch_idx + 1) * BATCH_SIZE, len(model_summaries))
                    batch = model_summaries[start_idx:end_idx]
                    
                    print(f"  📦 Batch {batch_idx + 1}/{num_batches}: models {start_idx}-{end_idx-1}")
                    
                    try:
                        batch_matches = self._llm_match_request_unified(model_type, batch)
                        
                        # Adjust indices to account for batch offset
                        for match_group in batch_matches:
                            if isinstance(match_group, dict) and 'indices' in match_group:
                                original_indices = match_group['indices']
                                if isinstance(original_indices, list):
                                    match_group['indices'] = [idx + start_idx for idx in original_indices]
                        
                        if batch_matches:
                            print(f"    ✅ Found {len(batch_matches)} groups in this batch")
                        all_batch_matches.extend(batch_matches)
                    except Exception as e:
                        print(f"    ⚠️  Batch failed: {e}")
                        continue
                
                matches = all_batch_matches
                
                # Summary
                if matches:
                    print(f"✅ Total: Found {len(matches)} match groups across all batches")
                else:
                    print(f"ℹ️  No matches found (all models will be individual)")
            else:
                # Small enough to process in one go
                matches = self._llm_match_request_unified(model_type, model_summaries)
                
                # Debug: Print matches found
                if matches:
                    print(f"✅ Found {len(matches)} match groups for {model_type}")
                    for match in matches:
                        if isinstance(match, dict):
                            print(f"  → {match.get('canonical_name', '?')}: indices {match.get('indices', [])} (confidence: {match.get('confidence', 0)})")
                else:
                    print(f"ℹ️  No matches found for {model_type} (all models will be individual)")

            # Process matches and build ModelMatch objects
            try:
                # Track which models have been matched
                matched_indices = set()
                
                # Build ModelMatch objects from LLM groups
                for match_group in matches:
                    # Skip if match_group is not a dict (LLM returned malformed data)
                    if not isinstance(match_group, dict):
                        continue
                    
                    # Check confidence threshold - accept reasonably confident matches
                    confidence = match_group.get('confidence', 0.0)
                    if confidence < 0.85:
                        print(f"⚠️  Skipping low-confidence match ({confidence:.2f}): {match_group.get('canonical_name', 'unknown')}")
                        continue
                    
                    # Ensure indices is a list
                    indices = match_group.get('indices', [])
                    if not isinstance(indices, list):
                        # If single integer, wrap it in a list
                        indices = [indices] if isinstance(indices, int) else []
                    
                    # Validate indices are within range
                    valid_indices = [i for i in indices if isinstance(i, int) and 0 <= i < len(type_models)]
                    
                    if not valid_indices:
                        print(f"⚠️  Skipping match group with invalid indices: {match_group}")
                        continue
                    
                    # Only match if we have at least 2 models (no point matching single model)
                    if len(valid_indices) < 2:
                        print(f"⚠️  Skipping single-model match group: {match_group.get('canonical_name', 'unknown')}")
                        continue
                    
                    # Mark these indices as matched
                    matched_indices.update(valid_indices)
                    
                    matched_models = [type_models[i] for i in valid_indices]
                    all_matches.append(ModelMatch(
                        canonical_name=match_group.get('canonical_name', 'unknown'),
                        confidence=confidence,
                        models=matched_models
                    ))
                
                # Add unmatched models as individual entries (preserve their model_type)
                for i, model in enumerate(type_models):
                    if i not in matched_indices:
                        all_matches.append(ModelMatch(
                            canonical_name=model.get('name', model.get('model_id', 'unknown')),
                            confidence=1.0,
                            models=[model]
                        ))
                        
            except Exception as e:
                print(f"LLM matching failed for {model_type}: {e}")
                # Fallback to basic matching
                all_matches.extend(self._match_models_basic(type_models))

        return all_matches

    def _llm_match_request_unified(self, model_type: str, model_summaries: List[Dict]) -> List[Dict]:
        """Make LLM request to match models using the shared LLMClient utility"""
        prompt = f"""You are a model matching expert. Your task is to identify models that are the SAME underlying AI model offered by different providers.

Model Type: {model_type}

Models to analyze:
{json.dumps(model_summaries, indent=2)}

MATCHING RULES:
1. **Ignore formatting variations**: "flux-1.1-pro", "FLUX1.1 [pro]", "Flux 1.1 Pro", "flux_1_1_pro" are ALL the same
2. **Version MUST match exactly**: "FLUX 1.1" ≠ "FLUX 1.0", "GPT-4o" ≠ "GPT-4"
3. **Edition/variant MUST match**: "Pro" ≠ "Dev" ≠ "Schnell", "Turbo" ≠ "Standard"
4. **Base model name must be the same**: "FLUX" = "Flux", "Claude" = "claude"
5. **Provider names don't matter**: Same model from Replicate, FAL, Runware, etc. should be matched

Positive Examples:
✅ "BFL flux-1.1-pro" + "FLUX1.1 [pro]" + "Flux 1.1 Pro (Replicate)" → MATCH (all are Flux 1.1 Pro)
✅ "Kling 1.6 Standard" + "kling-1-6-standard" + "Kling 1.6 Standard (Runware)" → MATCH
✅ "Claude 3.5 Sonnet" + "anthropic/claude-3.5-sonnet" + "claude-3-5-sonnet-20241022" → MATCH
✅ "Stable Diffusion XL" + "SDXL" + "stable-diffusion-xl-1024-v1-0" → MATCH

Negative Examples:
❌ "FLUX 1.1 Pro" + "FLUX 1.1 Dev" → NO MATCH (different editions)
❌ "FLUX 1.1" + "FLUX 1.0" → NO MATCH (different versions)
❌ "GPT-4o" + "GPT-4o-mini" → NO MATCH (mini is a different model)
❌ "Claude 3.5 Sonnet" + "Claude 3 Opus" → NO MATCH (different models)

OUTPUT FORMAT:
Return a JSON array of match groups. Each group represents models that are identical.
If NO matches found, return empty array: []

[{{"canonical_name": "flux-1-1-pro", "indices": [0, 3, 7], "confidence": 0.95, "reasoning": "All three are Flux 1.1 Pro with formatting variations"}}]

REQUIREMENTS:
- Only group if at least 2 models match
- Use confidence 0.95 for exact matches, 0.90-0.94 for likely matches
- canonical_name should be a clean, standardized version of the model name
- Include clear reasoning for each match

Analyze the models above and identify ALL matching groups:
"""
        try:
            llm_client = LLMClient(self.config)
            response = llm_client.chat(prompt, temperature=0.1, max_tokens=4000)
            
            # Debug: Log raw LLM response for troubleshooting
            response_preview = response[:500] if len(response) > 500 else response
            print(f"\n📝 LLM Response preview: {response_preview}...")
            
            parsed = llm_client.parse_response(response)
            
            # Additional debug info
            if isinstance(parsed, list):
                print(f"   Parsed {len(parsed)} potential match groups")
            elif isinstance(parsed, dict):
                print(f"   Parsed dict with keys: {list(parsed.keys())}")
            
            # Handle different response formats
            if isinstance(parsed, dict) and 'matches' in parsed:
                return parsed['matches']
            elif isinstance(parsed, list):
                # Validate that list contains dicts with proper structure
                valid_groups = []
                for item in parsed:
                    if isinstance(item, dict) and 'indices' in item and 'canonical_name' in item:
                        valid_groups.append(item)
                return valid_groups
            elif isinstance(parsed, dict) and 'groups' in parsed:
                return parsed['groups']
            else:
                print(f"⚠️  Unexpected LLM response format: {type(parsed)}")
                return []
        except Exception as e:
            print(f"LLM request failed: {e}")
            raise
    
    def _match_models_basic(self, models: List[Dict]) -> List[ModelMatch]:
        """Basic matching without LLM - uses string similarity"""
        matches = []
        used_indices = set()
        
        for i, model in enumerate(models):
            if i in used_indices:
                continue
            
            # Find similar models
            group = [model]
            used_indices.add(i)
            
            norm_name = self.normalize_model_name(model.get('name', ''))
            
            for j, other in enumerate(models[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                other_norm = self.normalize_model_name(other.get('name', ''))
                
                # Simple similarity check
                if self._are_similar(norm_name, other_norm):
                    group.append(other)
                    used_indices.add(j)
            
            matches.append(ModelMatch(
                canonical_name=model.get('name', model.get('model_id', 'unknown')),
                confidence=0.7,  # Lower confidence for basic matching
                models=group
            ))
        
        return matches
    
    def _are_similar(self, name1: str, name2: str) -> bool:
        """Check if two normalized names are similar (STRICT matching)"""
        if not name1 or not name2:
            return False
        
        # Exact match
        if name1 == name2:
            return True
        
        # Check for version/variant keywords that indicate different models
        variant_keywords = ['pro', 'dev', 'turbo', 'mini', 'ultra', 'lite', 'plus', 'kontex', 'schnell']
        name1_parts = set(name1.split())
        name2_parts = set(name2.split())
        
        # If one has a variant keyword the other doesn't, they're different
        for keyword in variant_keywords:
            if (keyword in name1_parts) != (keyword in name2_parts):
                return False
        
        # One contains the other - but must be at least 80% overlap
        if name1 in name2 or name2 in name1:
            shorter = min(len(name1), len(name2))
            longer = max(len(name1), len(name2))
            # Require 80% overlap for meaningful match
            if shorter / longer >= 0.8 and shorter > 5:
                return True
            else:
                return False
        
        # Check for common tokens
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())
        
        common = tokens1 & tokens2
        total = tokens1 | tokens2
        
        if len(total) == 0:
            return False
        
        # Jaccard similarity
        similarity = len(common) / len(total)
        return similarity > 0.6
    
    def find_alternatives(self, model_id: str, all_models: List[Dict]) -> List[Dict]:
        """
        Find alternative providers for a specific model
        
        Args:
            model_id: ID of the model to find alternatives for
            all_models: All available models
            
        Returns:
            List of alternative models sorted by price
        """
        # Find the target model
        target = None
        for model in all_models:
            if model.get('model_id') == model_id:
                target = model
                break
        
        if not target:
            return []
        
        # Match all models
        matches = self.match_models_with_llm(all_models)
        
        # Find the match group containing our target
        for match in matches:
            model_ids = [m.get('model_id') for m in match.models]
            if model_id in model_ids:
                # Return all alternatives (excluding the original)
                alternatives = [m for m in match.models if m.get('model_id') != model_id]
                return sorted(
                    alternatives,
                    key=lambda x: x.get('cost_per_call', float('inf'))
                )
        
        return []


def match_and_cache_models(models: List[Dict], cache_file: str = 'model_matches.json') -> List[ModelMatch]:
    """
    Match models and cache results
    
    Args:
        models: List of model dictionaries
        cache_file: Path to cache file
        
    Returns:
        List of ModelMatch objects
    """
    matcher = ModelMatcher()
    matches = matcher.match_models_with_llm(models)
    
    # Save to cache
    cache_data = []
    for match in matches:
        cache_data.append({
            'canonical_name': match.canonical_name,
            'confidence': match.confidence,
            'model_ids': [m.get('model_id') for m in match.models],
            'providers': [m.get('provider') for m in match.models],
        })
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved {len(matches)} model matches to {cache_file}")
    except Exception as e:
        print(f"Failed to save cache: {e}")
    
    return matches
