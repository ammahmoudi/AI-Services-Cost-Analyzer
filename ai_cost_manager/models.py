"""
AI Cost Manager - Database Models

Defines the core data structures for tracking AI model costs.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class LLMConfiguration(Base):
    """Configuration for LLM API (OpenRouter)"""
    __tablename__ = 'llm_configurations'
    
    id = Column(Integer, primary_key=True)
    provider = Column(String(50), default='openrouter')
    api_key = Column(Text, nullable=False)
    model_name = Column(String(200), default='openai/gpt-4o-mini')
    base_url = Column(String(500), default='https://openrouter.ai/api/v1')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<LLMConfiguration(model='{self.model_name}', active={self.is_active})>"


class APISource(Base):
    """Represents an API source (e.g., fal.ai, OpenAI, etc.)"""
    __tablename__ = 'api_sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    url = Column(Text, nullable=False)
    extractor_name = Column(String(50), nullable=False)  # e.g., 'fal', 'openai'
    is_active = Column(Boolean, default=True)
    last_extracted = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    models = relationship('AIModel', back_populates='source', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<APISource(name='{self.name}', extractor='{self.extractor_name}')>"


class AIModel(Base):
    """Represents an AI model with its pricing and metadata"""
    __tablename__ = 'ai_models'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('api_sources.id'), nullable=False)
    
    # Basic info
    model_id = Column(String(200), nullable=False)  # e.g., 'fal-ai/flux-pro'
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=True)  # e.g., 'text-to-image', 'text-generation'
    
    # Pricing
    cost_per_call = Column(Float, default=0.0)
    cost_per_1k_tokens = Column(Float, nullable=True)
    credits_required = Column(Float, nullable=True)
    pricing_info = Column(Text, nullable=True)  # Raw pricing text
    
    # Additional metadata
    thumbnail_url = Column(Text, nullable=True)
    tags = Column(JSON, default=list)
    category = Column(String(100), nullable=True)
    
    # Schema information
    input_schema = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)
    
    # Full metadata from source
    raw_metadata = Column(JSON, nullable=True)
    
    # Data fetch timestamps
    last_raw_fetched = Column(DateTime, nullable=True)  # When raw data was last fetched
    last_schema_fetched = Column(DateTime, nullable=True)  # When schema was last fetched
    last_playground_fetched = Column(DateTime, nullable=True)  # When playground data was last fetched
    last_llm_fetched = Column(DateTime, nullable=True)  # When LLM extraction was last performed
    
    # Pricing calculation details (LLM extracted)
    pricing_type = Column(String(50), nullable=True)  # 'fixed', 'per_token', 'per_second', etc.
    pricing_formula = Column(Text, nullable=True)  # Human-readable formula
    pricing_variables = Column(JSON, nullable=True)  # Variables used in calculation
    input_cost_per_unit = Column(Float, nullable=True)  # Cost per input unit
    output_cost_per_unit = Column(Float, nullable=True)  # Cost per output unit
    cost_unit = Column(String(50), nullable=True)  # 'tokens', 'seconds', 'images', etc.
    llm_extracted = Column(JSON, nullable=True)  # Full LLM extraction result
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    source = relationship('APISource', back_populates='models')
    
    def __repr__(self):
        return f"<AIModel(name='{self.name}', cost=${self.cost_per_call:.4f})>"
