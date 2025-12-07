import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
from collections import defaultdict
import pickle
import os

class GridWorld:
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.obstacles = set()
        self.hazard_zones = set()  # Implicit hazards for TAMER learning
        
    def add_obstacle(self, x, y, w=1, h=1):
        """Add rectangular obstacle"""
        for i in range(x, x + w):
            for j in range(y, y + h):
                if 0 <= i < self.width and 0 <= j < self.height:
                    self.obstacles.add((i, j))
    
    def add_hazard_zone(self, x, y, w=1, h=1):
        """Add hazard zone (not visible to baseline A*, only for TAMER)"""
        for i in range(x, x + w):
            for j in range(y, y + h):
                if 0 <= i < self.width and 0 <= j < self.height:
                    self.hazard_zones.add((i, j))
    
    def is_valid(self, pos):
        """Check if position is valid (within bounds and not obstacle)"""
        x, y = pos
        return (0 <= x < self.width and 0 <= y < self.height and 
                pos not in self.obstacles)
    
    def get_neighbors(self, pos):
        """Get valid 8-connected neighbors"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            new_pos = (x + dx, y + dy)
            if self.is_valid(new_pos):
                # Cost: 1.0 for cardinal, 1.414 for diagonal
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                neighbors.append((new_pos, cost))
        return neighbors

class AStarPlanner:
    def __init__(self, world, reward_model=None, lambda_weight=0.5):
        self.world = world
        self.reward_model = reward_model
        self.lambda_weight = lambda_weight
        
    def heuristic(self, pos, goal):
        """Euclidean distance heuristic"""
        return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    def plan(self, start, goal):
        """A* planning with optional TAMER reward shaping"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            for neighbor, edge_cost in self.world.get_neighbors(current):
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    h = self.heuristic(neighbor, goal)
                    
                    # TAMER augmentation: subtract learned reward
                    reward_bonus = 0
                    if self.reward_model is not None:
                        action = (neighbor[0] - current[0], neighbor[1] - current[1])
                        reward_bonus = self.lambda_weight * self.reward_model.predict(current, action)
                    
                    f = tentative_g + h - reward_bonus
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        
        return None  # No path found
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dict"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

class TAMERRewardModel:
    """Simple linear reward model learned from human feedback"""
    def __init__(self, feature_dim=10):
        self.weights = np.zeros(feature_dim)
        self.feedback_history = []
        self.alpha = 0.3  # Increased learning rate for faster convergence
        
    def extract_features(self, pos, action):
        """Extract features for position and action"""
        x, y = pos
        ax, ay = action
        
        # Normalize by grid size
        norm_x = x / 50.0
        norm_y = y / 50.0
        
        features = np.array([
            norm_x,  # Normalized x position
            norm_y,  # Normalized y position
            ax / 2.0,  # Normalized action x
            ay / 2.0,  # Normalized action y
            (x + ax) / 50.0,  # Next x position
            (y + ay) / 50.0,  # Next y position
            np.sqrt(ax**2 + ay**2) / 2.0,  # Action magnitude
            norm_x * norm_y,  # Position interaction
            (norm_x - 0.5) ** 2,  # Distance from center
            1.0  # Bias term
        ])
        return features
    
    def predict(self, pos, action):
        """Predict reward for state-action pair"""
        features = self.extract_features(pos, action)
        # Clip to prevent numerical issues
        reward = np.dot(self.weights[:len(features)], features)
        return np.clip(reward, -100, 100)
    
    def update(self, pos, action, feedback):
        """Update model based on human feedback (+1 or -1)"""
        features = self.extract_features(pos, action)
        self.feedback_history.append((pos, action, feedback))
        
        # Gradient descent update
        prediction = self.predict(pos, action)
        error = feedback - prediction
        self.weights += self.alpha * error * features
    
    def save(self, filename):
        """Save model to file"""
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'history': self.feedback_history}, f)
    
    def load(self, filename):
        """Load model from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.weights = data['weights']
                self.feedback_history = data['history']
            return True
        return False

class Visualizer:
    def __init__(self, world):
        self.world = world
        
    def plot_scenario(self, start, goal, baseline_path=None, tamer_path=None, 
                     title="Navigation Scenario", show_hazards=False):
        """Create visualization comparing paths"""
        fig, axes = plt.subplots(1, 2 if tamer_path else 1, figsize=(12, 6))
        if not tamer_path:
            axes = [axes]
        
        for idx, (ax, path, label) in enumerate([
            (axes[0], baseline_path, "Baseline A*"),
            (axes[1], tamer_path, "TAMER-Augmented A*") if tamer_path else (None, None, None)
        ]):
            if ax is None:
                continue
                
            ax.set_xlim(-1, self.world.width)
            ax.set_ylim(-1, self.world.height)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(label)
            
            # Draw obstacles
            for obs in self.world.obstacles:
                rect = Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, 
                               facecolor='black', alpha=0.8)
                ax.add_patch(rect)
            
            # Draw hazard zones (if enabled)
            if show_hazards:
                for hz in self.world.hazard_zones:
                    rect = Rectangle((hz[0]-0.5, hz[1]-0.5), 1, 1, 
                                   facecolor='red', alpha=0.2)
                    ax.add_patch(rect)
            
            # Draw path
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Path')
                
                # Calculate metrics
                path_length = sum(np.sqrt((path[i+1][0]-path[i][0])**2 + 
                                         (path[i+1][1]-path[i][1])**2) 
                                for i in range(len(path)-1))
                hazard_cells = sum(1 for p in path if p in self.world.hazard_zones)
                
                # Add metrics text
                metrics_text = f"Length: {path_length:.1f}\nHazard cells: {hazard_cells}"
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.8), fontsize=9)
            
            # Draw start and goal
            ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
            ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
            ax.legend(loc='upper right')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

def create_scenario_1():
    """Scenario 1: Complex Hazard Avoidance with Multiple Zones"""
    world = GridWorld(50, 50)
    
    # Create a more complex corridor with multiple passages
    world.add_obstacle(10, 0, 2, 15)
    world.add_obstacle(10, 20, 2, 10)
    world.add_obstacle(10, 35, 2, 15)
    
    world.add_obstacle(38, 0, 2, 15)
    world.add_obstacle(38, 20, 2, 10)
    world.add_obstacle(38, 35, 2, 15)
    
    # Add scattered obstacles for complexity
    world.add_obstacle(20, 8, 2, 2)
    world.add_obstacle(28, 8, 2, 2)
    world.add_obstacle(20, 40, 2, 2)
    world.add_obstacle(28, 40, 2, 2)
    
    # Multiple hazard zones - represents crowded areas at different times
    world.add_hazard_zone(18, 18, 6, 6)  # Primary hazard (center-left)
    world.add_hazard_zone(26, 18, 6, 6)  # Primary hazard (center-right)
    world.add_hazard_zone(22, 26, 4, 4)  # Secondary hazard (below)
    
    start = (5, 25)
    goal = (45, 25)
    
    return world, start, goal, "Scenario 1: Multi-Hazard Navigation"

def create_scenario_2():
    """Scenario 2: Narrow Passage with Safety Margin Challenges"""
    world = GridWorld(50, 50)
    
    # Create a complex maze-like structure
    world.add_obstacle(15, 0, 2, 15)
    world.add_obstacle(15, 20, 2, 10)
    world.add_obstacle(15, 35, 2, 15)
    
    world.add_obstacle(25, 5, 2, 10)
    world.add_obstacle(25, 20, 2, 10)
    world.add_obstacle(25, 35, 2, 10)
    
    world.add_obstacle(35, 0, 2, 15)
    world.add_obstacle(35, 20, 2, 10)
    world.add_obstacle(35, 35, 2, 15)
    
    # Add narrow obstacles near passages - requires careful navigation
    world.add_obstacle(18, 16, 3, 2)
    world.add_obstacle(18, 32, 3, 2)
    world.add_obstacle(28, 16, 3, 2)
    world.add_obstacle(28, 32, 3, 2)
    
    # Small hazard zones near obstacles (dangerous tight spaces)
    world.add_hazard_zone(17, 18, 2, 2)
    world.add_hazard_zone(26, 18, 2, 2)
    
    start = (5, 25)
    goal = (45, 25)
    
    return world, start, goal, "Scenario 2: Maze Navigation with Safety Constraints"

def create_scenario_3():
    """Scenario 3: Multi-Route Preference with Environmental Factors"""
    world = GridWorld(50, 50)
    
    # Create three distinct route options with varying obstacles
    # Upper route obstacles
    world.add_obstacle(15, 8, 15, 2)
    world.add_obstacle(18, 10, 2, 3)
    world.add_obstacle(26, 10, 2, 3)
    
    # Lower route obstacles  
    world.add_obstacle(15, 40, 15, 2)
    world.add_obstacle(18, 37, 2, 3)
    world.add_obstacle(26, 37, 2, 3)
    
    # Middle route obstacles (force choice between upper/lower)
    world.add_obstacle(22, 23, 6, 4)
    
    # Complex hazard pattern - simulates different environmental concerns
    # Lower hazards (e.g., rough terrain, high traffic)
    world.add_hazard_zone(12, 32, 8, 6)
    world.add_hazard_zone(20, 30, 10, 4)
    world.add_hazard_zone(30, 32, 8, 6)
    
    # Middle hazards (obstacles force detour)
    world.add_hazard_zone(22, 20, 6, 2)
    
    start = (5, 25)
    goal = (45, 25)
    
    return world, start, goal, "Scenario 3: Multi-Route Preference with Complex Hazards"

def create_scenario_4():
    """Scenario 4: Dense Urban Environment (NEW!)"""
    world = GridWorld(50, 50)
    
    # Simulate city blocks with buildings
    buildings = [
        (10, 10, 5, 5),
        (10, 25, 5, 5),
        (10, 40, 5, 5),
        (20, 10, 5, 5),
        (20, 25, 5, 5),
        (20, 40, 5, 5),
        (30, 10, 5, 5),
        (30, 25, 5, 5),
        (30, 40, 5, 5),
    ]
    
    for bx, by, bw, bh in buildings:
        world.add_obstacle(bx, by, bw, bh)
    
    # Hazard zones represent busy intersections and crowded streets
    world.add_hazard_zone(8, 15, 2, 10)    # Busy street
    world.add_hazard_zone(15, 8, 10, 2)    # Busy intersection
    world.add_hazard_zone(25, 15, 2, 10)   # Another busy street
    world.add_hazard_zone(18, 38, 10, 2)   # Crowded area
    world.add_hazard_zone(35, 20, 2, 15)   # High traffic zone
    
    start = (2, 25)
    goal = (48, 25)
    
    return world, start, goal, "Scenario 4: Dense Urban Navigation"

def create_scenario_5():
    """Scenario 5: Dynamic Obstacle Field (NEW!)"""
    world = GridWorld(50, 50)
    
    # Create a scattered obstacle field
    import random
    random.seed(42)  # Reproducible
    
    # Random obstacles simulating dynamic environment
    for _ in range(25):
        x = random.randint(8, 42)
        y = random.randint(5, 45)
        world.add_obstacle(x, y, 2, 2)
    
    # Hazard zones in strategic locations
    world.add_hazard_zone(15, 15, 8, 8)
    world.add_hazard_zone(25, 25, 10, 10)
    world.add_hazard_zone(15, 35, 8, 8)
    
    start = (2, 25)
    goal = (48, 25)
    
    return world, start, goal, "Scenario 5: Dynamic Obstacle Field"

# Demo functions
def run_baseline_demo():
    """Run baseline A* on all scenarios"""
    scenarios = [
        create_scenario_1(), 
        create_scenario_2(), 
        create_scenario_3(),
        create_scenario_4(),
        create_scenario_5()
    ]
    
    for world, start, goal, title in scenarios:
        planner = AStarPlanner(world)
        path = planner.plan(start, goal)
        
        viz = Visualizer(world)
        fig = viz.plot_scenario(start, goal, baseline_path=path, 
                               title=title, show_hazards=True)
        plt.savefig(f"{title.replace(' ', '_').replace(':', '')}.png", dpi=150)
        plt.show()
        
        print(f"\n{title}")
        print(f"Path length: {len(path)} cells")
        if path:
            hazard_count = sum(1 for p in path if p in world.hazard_zones)
            print(f"Hazard cells crossed: {hazard_count}")

if __name__ == "__main__":
    print("=" * 60)
    print("TAMER-Augmented A* Navigation System")
    print("=" * 60)
    print("\nRunning baseline A* demonstration...")
    print("This will generate visualizations for 3 scenarios.\n")
    
    run_baseline_demo()
    
    print("\n" + "=" * 60)
    print("Next step: Run interactive TAMER training!")
    print("Use the interactive training script to teach the agent.")
    print("=" * 60)