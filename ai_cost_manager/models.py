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


class ExtractorAPIKey(Base):
    """API keys for different extractors"""
    __tablename__ = 'extractor_api_keys'
    
    id = Column(Integer, primary_key=True)
    extractor_name = Column(String(50), unique=True, nullable=False)  # e.g., 'together', 'openai'
    api_key = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ExtractorAPIKey(extractor='{self.extractor_name}', active={self.is_active})>"


class AuthSettings(Base):
    """Authentication settings for API sources that require login"""
    __tablename__ = 'auth_settings'
    
    id = Column(Integer, primary_key=True)
    source_name = Column(String(100), unique=True, nullable=False)  # e.g., 'fal.ai'
    cookies = Column(Text, nullable=True)  # JSON string of cookies
    headers = Column(Text, nullable=True)  # JSON string of custom headers
    session_data = Column(Text, nullable=True)  # Additional session data
    username = Column(String(255), nullable=True)  # For credential-based auth (e.g., Runware)
    password = Column(Text, nullable=True)  # For credential-based auth (encrypted recommended)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text, nullable=True)  # User notes about the auth
    
    def __repr__(self):
        return f"<AuthSettings(source='{self.source_name}', active={self.is_active})>"


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
    cache_entries = relationship('CacheEntry', back_populates='model', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<AIModel(name='{self.name}', cost=${self.cost_per_call:.4f})>"


class CacheEntry(Base):
    """Cache storage for model data (raw, schema, playground, llm)"""
    __tablename__ = 'cache_entries'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ai_models.id'), nullable=False)
    cache_type = Column(String(50), nullable=False)  # 'raw', 'schema', 'playground', 'llm'
    data = Column(JSON, nullable=False)  # The actual cached data
    cached_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    model = relationship('AIModel', back_populates='cache_entries')
    
    def __repr__(self):
        return f"<CacheEntry(model_id={self.model_id}, type='{self.cache_type}')>"


class CanonicalModel(Base):
    """Represents a canonical/unified model identity across providers"""
    __tablename__ = 'canonical_models'
    
    id = Column(Integer, primary_key=True)
    canonical_name = Column(String(200), unique=True, nullable=False)  # e.g., 'flux-dev-1.0'
    display_name = Column(String(200), nullable=False)  # User-friendly name
    description = Column(Text, nullable=True)  # Description of the base model
    model_type = Column(String(50), nullable=True)  # 'text-to-image', etc.
    tags = Column(JSON, default=list)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    model_matches = relationship('ModelMatch', back_populates='canonical_model', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<CanonicalModel(name='{self.canonical_name}')>"


class ModelMatch(Base):
    """Links AIModel instances to their canonical model"""
    __tablename__ = 'model_matches'
    
    id = Column(Integer, primary_key=True)
    canonical_model_id = Column(Integer, ForeignKey('canonical_models.id'), nullable=False)
    ai_model_id = Column(Integer, ForeignKey('ai_models.id'), nullable=False)
    confidence = Column(Float, default=1.0)  # Match confidence 0-1
    matched_by = Column(String(50), default='manual')  # 'llm', 'manual', 'rule'
    matched_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    canonical_model = relationship('CanonicalModel', back_populates='model_matches')
    ai_model = relationship('AIModel')
    
    def __repr__(self):
        return f"<ModelMatch(canonical_id={self.canonical_model_id}, model_id={self.ai_model_id}, confidence={self.confidence})>"


class ExtractionTask(Base):
    """Tracks background extraction jobs"""
    __tablename__ = 'extraction_tasks'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('api_sources.id'), nullable=False)
    status = Column(String(20), default='pending')  # pending, running, completed, failed, cancelled
    progress = Column(Integer, default=0)  # 0-100
    total_models = Column(Integer, default=0)
    processed_models = Column(Integer, default=0)
    new_models = Column(Integer, default=0)
    updated_models = Column(Integer, default=0)
    current_model = Column(String(200), nullable=True)
    error_message = Column(Text, nullable=True)
    use_llm = Column(Boolean, default=False)
    fetch_schemas = Column(Boolean, default=False)
    force_refresh = Column(Boolean, default=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationship
    source = relationship('APISource')
    
    def __repr__(self):
        return f"<ExtractionTask(id={self.id}, source_id={self.source_id}, status='{self.status}', progress={self.progress}%)>"
