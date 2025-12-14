from app import app
from ai_cost_manager.database import get_session
from ai_cost_manager.models import AIModel
import re

session = get_session()

# Search for Hunyuan models
models = session.query(AIModel).filter(AIModel.name.ilike('%hunyuan%')).all()

print("Testing version extraction:")
print("="*80)

search = "Hunyuan 3D V2.0"
search_lower = search.lower()

# Extract version from search
search_versions = re.findall(r'(?<!\d)(\d+\.\d+)(?!\d)', search_lower)
print(f"\nSearch: '{search}'")
print(f"Extracted versions from search: {search_versions}")

print("\n" + "="*80)
print("Checking models with '3d' in name:")

for m in models:
    if '3d' in m.name.lower() or '3d' in m.model_id.lower():
        model_name = m.name.lower()
        model_id = m.model_id.lower()
        
        # Extract versions from model
        name_versions = re.findall(r'(?<!\d)(\d+\.\d+)(?!\d)', model_name)
        id_versions = re.findall(r'(?<!\d)(\d+\.\d+)(?!\d)', model_id)
        
        # If no decimal versions, try single digits
        if not name_versions and not id_versions:
            name_versions = re.findall(r'(?<!\d)(\d+)(?!\d)', model_name)
            id_versions = re.findall(r'(?<!\d)(\d+)(?!\d)', model_id)
        
        all_versions = set(name_versions + id_versions)
        
        print(f"\nModel: {m.name}")
        print(f"  ID: {m.model_id}")
        print(f"  Extracted versions: {all_versions}")
        
        if search_versions:
            search_version_set = set(search_versions)
            version_match = bool(search_version_set & all_versions)
            print(f"  Version match with '{search_versions[0]}': {version_match}")
            if not version_match:
                print(f"  ❌ FILTERED OUT - version mismatch")
