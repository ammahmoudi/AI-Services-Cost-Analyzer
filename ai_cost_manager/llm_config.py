"""
LLM Configuration Model

Stores LLM API settings for intelligent data extraction.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from ai_cost_manager.models import Base


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
