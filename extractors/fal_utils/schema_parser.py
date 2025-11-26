"""
Fal.ai OpenAPI Schema Parser

Functions for extracting and simplifying OpenAPI schemas.
"""
from typing import Dict, Any


def extract_input_schema(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract clean input schema from OpenAPI schema.
    
    Args:
        openapi_schema: Full OpenAPI schema
        
    Returns:
        Simplified input schema with fields
    """
    if not openapi_schema:
        return {}
    
    try:
        # Navigate OpenAPI structure to find input schema
        paths = openapi_schema.get("paths", {})
        
        # Usually the POST endpoint at root or /
        for path_key in ["/", "", "/api"]:
            if path_key in paths:
                path_data = paths[path_key]
                post_data = path_data.get("post", {})
                request_body = post_data.get("requestBody", {})
                content = request_body.get("content", {})
                json_content = content.get("application/json", {})
                schema = json_content.get("schema", {})
                
                if schema:
                    # Extract properties and convert to cleaner format
                    return simplify_schema(schema)
        
        # Alternative: Check components/schemas
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})
        
        # Look for Input or Request schema
        for key in ["Input", "Request", "RequestBody"]:
            if key in schemas:
                return simplify_schema(schemas[key])
        
        return {}
        
    except Exception as e:
        print(f"    Error extracting input schema: {e}")
        return {}


def extract_output_schema(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract clean output schema from OpenAPI schema.
    
    Args:
        openapi_schema: Full OpenAPI schema
        
    Returns:
        Simplified output schema with fields
    """
    if not openapi_schema:
        return {}
    
    try:
        # Navigate OpenAPI structure to find output schema
        paths = openapi_schema.get("paths", {})
        
        # Usually the POST endpoint at root or /
        for path_key in ["/", "", "/api"]:
            if path_key in paths:
                path_data = paths[path_key]
                post_data = path_data.get("post", {})
                responses = post_data.get("responses", {})
                success_response = responses.get("200", {}) or responses.get("201", {})
                content = success_response.get("content", {})
                json_content = content.get("application/json", {})
                schema = json_content.get("schema", {})
                
                if schema:
                    return simplify_schema(schema)
        
        # Alternative: Check components/schemas
        components = openapi_schema.get("components", {})
        schemas = components.get("schemas", {})
        
        # Look for Output or Response schema
        for key in ["Output", "Response", "ResponseBody"]:
            if key in schemas:
                return simplify_schema(schemas[key])
        
        return {}
        
    except Exception as e:
        print(f"    Error extracting output schema: {e}")
        return {}


def simplify_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAPI schema to simplified format.
    
    Args:
        schema: OpenAPI schema object
        
    Returns:
        Simplified schema with 'inputs' or 'properties' list
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    inputs = []
    for field_name, field_spec in properties.items():
        field_info = {
            "name": field_name,
            "type": field_spec.get("type", "string"),
            "description": field_spec.get("description", ""),
            "required": field_name in required,
        }
        
        # Add additional useful metadata
        if "default" in field_spec:
            field_info["default"] = field_spec["default"]
        if "enum" in field_spec:
            field_info["enum"] = field_spec["enum"]
        if "minimum" in field_spec:
            field_info["minimum"] = field_spec["minimum"]
        if "maximum" in field_spec:
            field_info["maximum"] = field_spec["maximum"]
        if "format" in field_spec:
            field_info["format"] = field_spec["format"]
        
        inputs.append(field_info)
    
    return {
        "inputs": inputs,
        "properties": properties,
        "required": required
    }
