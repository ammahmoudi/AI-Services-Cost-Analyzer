"""
Model Matching Service

Handles matching models across providers and finding alternatives
"""

from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
from ai_cost_manager.models import AIModel, CanonicalModel, ModelMatch, APISource
from ai_cost_manager.model_matcher import ModelMatcher
from ai_cost_manager.model_types import VALID_MODEL_TYPES
from datetime import datetime


class ModelMatchingService:
    """Service for managing model matches across providers"""
    
    def __init__(self, session: Session):
        self.session = session
        self.matcher = ModelMatcher()
    
    def match_all_models(self, force_refresh: bool = False) -> Dict[str, any]:
        """
        Match all models in the database across providers
        
        Args:
            force_refresh: If True, re-match even if matches exist
            
        Returns:
            Summary of matching results
        """
        # Check if we already have matches
        existing_matches = self.session.query(ModelMatch).count()
        if existing_matches > 0 and not force_refresh:
            return {
                'status': 'already_matched',
                'existing_matches': existing_matches,
                'message': 'Models already matched. Use force_refresh=True to re-match.'
            }
        
        # Get all active models
        models = self.session.query(AIModel).filter(AIModel.is_active == True).all()
        
        if not models:
            return {
                'status': 'no_models',
                'message': 'No models found to match'
            }
        
        # Convert to dictionaries
        model_dicts = []
        for model in models:
            model_dicts.append({
                'id': model.id,
                'model_id': model.model_id,
                'name': model.name,
                'model_name': model.name,
                'description': model.description,
                'model_type': model.model_type,
                'provider': model.source.name if model.source else 'unknown',
                'tags': model.tags or [],
                'cost_per_call': model.cost_per_call,
            })
        
        # Match using LLM
        print(f"Matching {len(model_dicts)} models...")
        matches = self.matcher.match_models_with_llm(model_dicts)
        
        # Clear existing matches if force refresh
        if force_refresh:
            self.session.query(ModelMatch).delete()
            self.session.query(CanonicalModel).delete()
            self.session.commit()
        
        # Save matches to database
        created_canonical = 0
        created_matches = 0
        
        for match in matches:
            # Create or get canonical model
            canonical = self.session.query(CanonicalModel).filter(
                CanonicalModel.canonical_name == match.canonical_name
            ).first()
            
            if not canonical:
                # Get representative model for description
                best_model = match.get_best_price()
                if not best_model:
                    best_model = match.models[0]
                
                # Validate and use model_type from VALID_MODEL_TYPES
                model_type = best_model.get('model_type')
                if model_type not in VALID_MODEL_TYPES:
                    print(f"⚠️  Invalid model_type '{model_type}' for canonical model, defaulting to 'other'")
                    model_type = 'other'
                
                canonical = CanonicalModel(
                    canonical_name=match.canonical_name,
                    display_name=match.canonical_name.replace('-', ' ').title(),
                    description=best_model.get('description', ''),
                    model_type=model_type,
                    tags=list(set([tag for m in match.models for tag in (m.get('tags') or [])]))
                )
                self.session.add(canonical)
                self.session.flush()  # Get the ID
                created_canonical += 1
            
            # Create model matches
            for model_dict in match.models:
                ai_model = self.session.query(AIModel).filter(
                    AIModel.id == model_dict['id']
                ).first()
                
                if ai_model:
                    model_match = ModelMatch(
                        canonical_model_id=canonical.id,
                        ai_model_id=ai_model.id,
                        confidence=match.confidence,
                        matched_by='llm',
                        matched_at=datetime.utcnow()
                    )
                    self.session.add(model_match)
                    created_matches += 1
        
        self.session.commit()
        
        return {
            'status': 'success',
            'canonical_models_created': created_canonical,
            'model_matches_created': created_matches,
            'total_models_processed': len(model_dicts),
            'match_groups': len(matches)
        }
    
    def get_alternatives(self, model_id: int) -> List[Dict]:
        """
        Get alternative providers for a specific model
        
        Args:
            model_id: Database ID of the model
            
        Returns:
            List of alternative models with provider and pricing info
        """
        # Find the model
        model = self.session.query(AIModel).filter(AIModel.id == model_id).first()
        if not model:
            return []
        
        # Find its canonical model
        match = self.session.query(ModelMatch).filter(
            ModelMatch.ai_model_id == model_id
        ).first()
        
        if not match:
            # No matches found, return empty
            return []
        
        # Get all models linked to the same canonical model
        all_matches = self.session.query(ModelMatch).filter(
            ModelMatch.canonical_model_id == match.canonical_model_id
        ).all()
        
        alternatives = []
        for alt_match in all_matches:
            if alt_match.ai_model_id == model_id:
                continue  # Skip the original model
            
            alt_model = alt_match.ai_model
            if alt_model and alt_model.is_active:
                alternatives.append({
                    'model_id': alt_model.id,
                    'name': alt_model.name,
                    'provider': alt_model.source.name if alt_model.source else 'unknown',
                    'cost_per_call': alt_model.cost_per_call,
                    'pricing_formula': alt_model.pricing_formula,
                    'confidence': alt_match.confidence,
                    'model_type': alt_model.model_type,
                    'description': alt_model.description,
                })
        
        # Sort by price (lowest first)
        alternatives.sort(key=lambda x: x['cost_per_call'] if x['cost_per_call'] else float('inf'))
        
        return alternatives
    
    def get_best_price_for_model(self, model_id: int) -> Optional[Dict]:
        """Get the provider offering the best price for this model"""
        alternatives = self.get_alternatives(model_id)
        if not alternatives:
            return None
        return alternatives[0]  # Already sorted by price
    
    def get_canonical_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """
        Get all canonical models with their providers and pricing
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List of canonical models with provider details
        """
        query = self.session.query(CanonicalModel)
        
        if model_type:
            query = query.filter(CanonicalModel.model_type == model_type)
        
        canonical_models = query.all()
        
        result = []
        for canonical in canonical_models:
            # Get all providers for this model
            matches = self.session.query(ModelMatch).filter(
                ModelMatch.canonical_model_id == canonical.id
            ).all()
            
            providers = []
            best_price = None
            
            for match in matches:
                if match.ai_model and match.ai_model.is_active:
                    provider_data = {
                        'model_id': match.ai_model.id,
                        'provider': match.ai_model.source.name if match.ai_model.source else 'unknown',
                        'cost_per_call': match.ai_model.cost_per_call,
                        'pricing_formula': match.ai_model.pricing_formula,
                        'confidence': match.confidence,
                    }
                    providers.append(provider_data)
                    
                    # Track best price
                    if match.ai_model.cost_per_call and match.ai_model.cost_per_call > 0:
                        if best_price is None or match.ai_model.cost_per_call < best_price['cost_per_call']:
                            best_price = provider_data
            
            # Sort providers by price
            providers.sort(key=lambda x: x['cost_per_call'] if x['cost_per_call'] else float('inf'))
            
            result.append({
                'canonical_id': canonical.id,
                'canonical_name': canonical.canonical_name,
                'display_name': canonical.display_name,
                'description': canonical.description,
                'model_type': canonical.model_type,
                'tags': canonical.tags,
                'provider_count': len(providers),
                'providers': providers,
                'best_price': best_price,
            })
        
        # Sort by best price (lowest first)
        result.sort(key=lambda x: x['best_price']['cost_per_call'] if x.get('best_price') else float('inf'))
        
        return result
    
    def get_model_with_alternatives(self, model_id: int) -> Optional[Dict]:
        """Get a model with all its alternatives"""
        model = self.session.query(AIModel).filter(AIModel.id == model_id).first()
        if not model:
            return None
        
        alternatives = self.get_alternatives(model_id)
        best_alternative = alternatives[0] if alternatives else None
        
        return {
            'model': {
                'id': model.id,
                'name': model.name,
                'provider': model.source.name if model.source else 'unknown',
                'cost_per_call': model.cost_per_call,
                'pricing_formula': model.pricing_formula,
                'model_type': model.model_type,
                'description': model.description,
            },
            'alternatives': alternatives,
            'best_alternative': best_alternative,
            'potential_savings': (
                model.cost_per_call - best_alternative['cost_per_call']
                if best_alternative and model.cost_per_call and best_alternative['cost_per_call']
                else 0
            )
        }
