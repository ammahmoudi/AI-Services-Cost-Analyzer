"""
Pricing Calculation Type Enum

Defines how model pricing is calculated.
"""
from sqlalchemy import Column, String, Text, Float, JSON
from ai_cost_manager.models import AIModel


# Extend AIModel with new pricing fields
def add_pricing_fields():
    """Add pricing calculation fields to AIModel"""
    # These will be added via migration or dynamically
    return {
        'pricing_type': String(50),  # 'fixed', 'per_token', 'per_second', 'per_image', 'per_video', 'variable'
        'pricing_formula': Text,  # Human-readable formula
        'pricing_variables': JSON,  # Variables used in calculation
        'input_cost_per_unit': Float,  # Cost per input unit
        'output_cost_per_unit': Float,  # Cost per output unit
        'cost_unit': String(50),  # 'tokens', 'seconds', 'images', 'calls'
        'llm_extracted': JSON,  # Full LLM extraction result
    }


# Pricing types
PRICING_TYPES = {
    'fixed': 'Fixed cost per call',
    'per_token': 'Cost per token (input/output may differ)',
    'per_second': 'Cost per second of processing',
    'per_image': 'Cost per image generated',
    'per_video': 'Cost per video second',
    'per_request': 'Cost per API request',
    'tiered': 'Tiered pricing based on usage',
    'variable': 'Variable based on parameters',
    'unknown': 'Pricing structure unknown',
}
