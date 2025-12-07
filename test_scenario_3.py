"""
Test script to verify Scenario 3 has multiple viable paths
"""
from tamer_astar_baseline import create_scenario_3, AStarPlanner, TAMERRewardModel
import numpy as np

print("=" * 70)
print("Testing Scenario 3 Configuration")
print("=" * 70)

world, start, goal, title = create_scenario_3()

print(f"\nScenario: {title}")
print(f"Start: {start}")
print(f"Goal: {goal}")
print(f"Grid size: {world.width}x{world.height}")
print(f"Obstacles: {len(world.obstacles)} cells")
print(f"Hazard zones: {len(world.hazard_zones)} cells")

# Test baseline A*
print("\n" + "-" * 70)
print("Testing Baseline A* (no TAMER)")
print("-" * 70)

planner = AStarPlanner(world)
path = planner.plan(start, goal)

if path:
    print(f"✓ Path found: {len(path)} cells")
    
    # Analyze path
    avg_y = sum(p[1] for p in path) / len(path)
    min_y = min(p[1] for p in path)
    max_y = max(p[1] for p in path)
    hazards_crossed = sum(1 for p in path if p in world.hazard_zones)
    
    print(f"  Average Y: {avg_y:.1f}")
    print(f"  Y range: [{min_y}, {max_y}]")
    print(f"  Route: {'UPPER' if avg_y < 20 else 'LOWER' if avg_y > 30 else 'MIDDLE'}")
    print(f"  Hazards crossed: {hazards_crossed}")
    
    # Show first, middle, last points
    print(f"  Path sample: {path[0]} → {path[len(path)//2]} → {path[-1]}")
else:
    print("✗ No path found!")

# Test with strong upper bias
print("\n" + "-" * 70)
print("Testing A* with STRONG UPPER BIAS")
print("-" * 70)

model = TAMERRewardModel(feature_dim=21)
model.alpha = 0.3

# Manually set weights to prefer upper routes
# Feature 14 is "above_hazard" - make it very positive
model.weights[14] = 20.0  # Strong positive for being above center
model.weights[15] = -20.0  # Strong negative for being below center
model.weights[16] = 10.0  # Positive for moving up
model.weights[17] = -10.0  # Negative for moving down

# Need to wrap predict function
def enhanced_predict(pos, action):
    x, y = pos
    ax, ay = action
    
    hazard_center_y = 25
    above_hazard = 1.0 if y < hazard_center_y else 0.0
    below_hazard = 1.0 if y > hazard_center_y else 0.0
    moving_up = 1.0 if ay < 0 else 0.0
    moving_down = 1.0 if ay > 0 else 0.0
    
    # Simple approximation for testing
    reward = (above_hazard * 20.0 + 
              below_hazard * (-20.0) + 
              moving_up * 10.0 + 
              moving_down * (-10.0))
    return reward

model.predict = enhanced_predict

planner_upper = AStarPlanner(world, model, lambda_weight=10.0)
path_upper = planner_upper.plan(start, goal)

if path_upper:
    avg_y_upper = sum(p[1] for p in path_upper) / len(path_upper)
    print(f"✓ Path found: {len(path_upper)} cells")
    print(f"  Average Y: {avg_y_upper:.1f}")
    print(f"  Route: {'UPPER' if avg_y_upper < 20 else 'LOWER' if avg_y_upper > 30 else 'MIDDLE'}")
    print(f"  Path sample: {path_upper[0]} → {path_upper[len(path_upper)//2]} → {path_upper[-1]}")
    
    if abs(avg_y_upper - avg_y) < 2:
        print("\n⚠️  WARNING: Path didn't change much despite strong bias!")
        print("  This suggests:")
        print("  1. Obstacles might force a specific route")
        print("  2. Lambda weight might need to be even higher")
        print("  3. Feature engineering might need adjustment")
    else:
        print(f"\n✓ SUCCESS: Path shifted by {avg_y - avg_y_upper:.1f} units upward!")
else:
    print("✗ No path found with upper bias!")

# Test with strong lower bias
print("\n" + "-" * 70)
print("Testing A* with STRONG LOWER BIAS")
print("-" * 70)

model.weights[14] = -20.0  # Negative for being above
model.weights[15] = 20.0  # Positive for being below
model.weights[16] = -10.0  # Negative for moving up
model.weights[17] = 10.0  # Positive for moving down

planner_lower = AStarPlanner(world, model, lambda_weight=10.0)
path_lower = planner_lower.plan(start, goal)

if path_lower:
    avg_y_lower = sum(p[1] for p in path_lower) / len(path_lower)
    print(f"✓ Path found: {len(path_lower)} cells")
    print(f"  Average Y: {avg_y_lower:.1f}")
    print(f"  Route: {'UPPER' if avg_y_lower < 20 else 'LOWER' if avg_y_lower > 30 else 'MIDDLE'}")
    print(f"  Path sample: {path_lower[0]} → {path_lower[len(path_lower)//2]} → {path_lower[-1]}")
    
    if abs(avg_y_lower - avg_y) < 2:
        print("\n⚠️  WARNING: Path didn't change much despite strong bias!")
    else:
        print(f"\n✓ SUCCESS: Path shifted by {avg_y_lower - avg_y:.1f} units downward!")
else:
    print("✗ No path found with lower bias!")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if path and path_upper and path_lower:
    print(f"Baseline path:    Y = {avg_y:.1f} ({'UPPER' if avg_y < 20 else 'LOWER' if avg_y > 30 else 'MIDDLE'})")
    print(f"Upper-biased path: Y = {avg_y_upper:.1f} ({'UPPER' if avg_y_upper < 20 else 'LOWER' if avg_y_upper > 30 else 'MIDDLE'})")
    print(f"Lower-biased path: Y = {avg_y_lower:.1f} ({'UPPER' if avg_y_lower < 20 else 'LOWER' if avg_y_lower > 30 else 'MIDDLE'})")
    
    if abs(avg_y_upper - avg_y_lower) > 5:
        print("\n✓ Scenario 3 is WORKING: Paths respond to preferences!")
        print("  Interactive training should work with enough feedback.")
    else:
        print("\n⚠️  Scenario 3 might be CONSTRAINED:")
        print("  Obstacles may limit route options.")
        print("  Consider adjusting obstacle placement.")
else:
    print("\n✗ Some paths failed - check scenario configuration!")

print("=" * 70)
