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
                summary = {
                    'index': i,
                    'name': features['name'],
                    'provider': features['provider'],
                    'model_id': features['model_id'],
                    'description': features['description'][:200] if features['description'] else '',
                    'tags': features['tags'][:5] if features['tags'] else [],
                }
                model_summaries.append(summary)

            # Call LLM to identify matches using the unified LLM client
            try:
                matches = self._llm_match_request_unified(model_type, model_summaries)

                # Track which models have been matched
                matched_indices = set()
                
                # Build ModelMatch objects from LLM groups
                for match_group in matches:
                    # Skip if match_group is not a dict (LLM returned malformed data)
                    if not isinstance(match_group, dict):
                        continue
                    
                    # Check confidence threshold - only accept high-confidence matches
                    confidence = match_group.get('confidence', 0.0)
                    if confidence < 0.9:
                        print(f"⚠️  Skipping low-confidence match ({confidence}): {match_group.get('canonical_name', 'unknown')}")
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
        prompt = f"""You are an AI model expert tasked with identifying IDENTICAL models from different providers.

⚠️ TASK: Match models that are the EXACT SAME model, accounting for formatting differences.

Models to analyze:
{json.dumps(model_summaries, indent=2)}

🔴 MATCHING RULES:

1. **Ignore Formatting Differences** (these are the SAME model):
   ✅ "flux-1.1-pro" = "FLUX1.1 [pro]" = "Flux 1.1 Pro" = "FLUX 1.1 (Pro)" (same model, different formatting)
   ✅ "sdxl-1.0" = "SDXL 1.0" = "Stable Diffusion XL 1.0" (abbreviation vs full name)
   ✅ "gpt-4-turbo" = "GPT-4 Turbo" = "GPT 4 (Turbo)" (same model)
   
   Formatting to ignore:
   - Hyphens vs spaces vs underscores: "flux-1.1" = "flux 1.1" = "flux_1_1"
   - Brackets/parentheses: "[pro]" = "(pro)" = "pro"
   - Capitalization: "FLUX" = "flux" = "Flux"
   - Extra version labels: "V1.1" = "1.1" = "version 1.1"

2. **Version Numbers MUST Match Exactly**:
   ❌ "FLUX 1.1" ≠ "FLUX 1.0" (different versions)
   ❌ "FLUX 1" ≠ "FLUX 2" (different major versions)
   ❌ "GPT-4" ≠ "GPT-4.5" (different versions)
   ❌ "SDXL 1.0" ≠ "SDXL 2.1" (different versions)

3. **Edition/Variant Keywords MUST Match**:
   ❌ "FLUX 1.1 Pro" ≠ "FLUX 1.1 Dev" (Pro ≠ Dev)
   ❌ "FLUX 1.1 Pro" ≠ "FLUX 1.1" (one has Pro, one doesn't)
   ❌ "Schnell" ≠ "Dev" ≠ "Pro" (different editions)
   ❌ "Turbo" ≠ "Standard" (different variants)
   ❌ "Ultra" ≠ "Pro" (different tiers)

4. **Matching Algorithm**:
   a) Extract base name (e.g., "FLUX", "GPT-4", "SDXL")
   b) Extract version (e.g., "1.1", "2.0", "4.5")
   c) Extract edition/variant (e.g., "Pro", "Dev", "Turbo", "Schnell")
   d) MATCH if: base name matches AND version matches AND edition matches
   e) Ignore all formatting (spaces, hyphens, brackets, case)

5. **Examples**:
   ✅ MATCH: "BFL flux-1.1-pro" + "FLUX1.1 [pro]" + "Flux 1.1 Pro"
      → Same: base="FLUX", version="1.1", edition="pro"
   
   ✅ MATCH: "SDXL-1.0" + "Stable Diffusion XL 1.0" + "sdxl_1_0"
      → Same: base="SDXL", version="1.0", edition=none
   
   ❌ DON'T MATCH: "FLUX 1.1 Pro" + "FLUX 1.1 Dev"
      → Different editions: "Pro" ≠ "Dev"
   
   ❌ DON'T MATCH: "FLUX 1.1 Pro" + "FLUX 1.0 Pro"
      → Different versions: "1.1" ≠ "1.0"

6. **When Uncertain**: If confidence < 90% → DON'T MATCH

✅ Return ONLY a JSON array of match groups:
[
    {{
        "canonical_name": "flux-1-1-pro",
        "indices": [2, 5, 9],
        "confidence": 0.95,
        "reasoning": "All are FLUX version 1.1 Pro edition - same base, version, and edition despite formatting"
    }}
]

⚠️ Only include groups with confidence > 0.9. Empty array [] if no confident matches.
"""
        try:
            llm_client = LLMClient(self.config)
            response = llm_client.chat(prompt, temperature=0.3, max_tokens=800)
            parsed = llm_client.parse_response(response)
            
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
