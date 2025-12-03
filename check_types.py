#!/usr/bin/env python3
"""Check current model type distribution"""

from ai_cost_manager.database import get_session
from ai_cost_manager.models import AIModel
from sqlalchemy import func

session = get_session()

print("\n📊 Current Model Type Distribution:\n")

results = session.query(
    AIModel.model_type, 
    func.count(AIModel.id)
).group_by(
    AIModel.model_type
).order_by(
    AIModel.model_type
).all()

total = 0
for mtype, count in results:
    label = mtype if mtype else '(null)'
    print(f"  {label:25} {count:4} models")
    total += count

print(f"\n  {'TOTAL':25} {total:4} models\n")

session.close()
