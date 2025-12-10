"""
Test cases for improved parsed component search:

1. Search "BFL" -> should find all BFL (Black Forest Labs) models
2. Search "flux" -> should find all Flux family models
3. Search "anthropic" -> should find all Anthropic models
4. Search "claude" -> should find all Claude family models
5. Search "pro" -> should find models with Pro variant
6. Search "text-to-video" -> should find models with that mode
7. Search "alibaba wan" -> should find Alibaba's Wan models
8. Search "bytedance" -> should find all Bytedance models
"""

# The improvements:
# 1. Better token extraction handles "flux1.1" -> "flux"
# 2. Parsed component bonuses:
#    - Company match: +30 points
#    - Family match: +35 points  
#    - Size match: +15 points
#    - Version match: +10 points
# 3. Search text includes all parsed components

print(__doc__)
