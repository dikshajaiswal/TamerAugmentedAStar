import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import sys

# Import from the baseline implementation
try:
    from tamer_astar_baseline import (GridWorld, AStarPlanner, TAMERRewardModel, 
                                      Visualizer, create_scenario_1, 
                                      create_scenario_2, create_scenario_3)
except:
    print("ERROR: Please save the baseline implementation as 'tamer_astar_baseline.py'")
    print("Then run this interactive training script.")
    sys.exit(1)

class InteractiveTAMERTrainer:
    def __init__(self, world, start, goal, scenario_name):
        self.world = world
        self.start = start
        self.goal = goal
        self.scenario_name = scenario_name
        # Use 15-dimensional model with enhanced features (same as evaluation)
        self.reward_model = TAMERRewardModel(feature_dim=15)
        self.reward_model.alpha = 0.3  # Slightly higher for interactive learning
        self.current_path = None
        self.iteration = 0
        self.feedback_count = 0
        
        # Try to load existing model
        model_file = f"tamer_model_{scenario_name}.pkl"
        if self.reward_model.load(model_file):
            print(f"Loaded existing model from {model_file}")
            self.iteration = len(self.reward_model.feedback_history)
        
        self.setup_ui()
    
    def extract_enhanced_features(self, pos, action):
        """Extract enhanced features that include hazard zone info (same as evaluation)"""
        x, y = pos
        ax, ay = action
        next_x, next_y = x + ax, y + ay
        
        # Check if in hazard zone
        in_hazard = 1.0 if pos in self.world.hazard_zones else 0.0
        next_in_hazard = 1.0 if (next_x, next_y) in self.world.hazard_zones else 0.0
        
        # Distance to nearest hazard
        hazard_dist = self.distance_to_nearest_hazard(pos) if self.world.hazard_zones else 10.0
        
        # Distance to nearest obstacle
        obs_dist = self.distance_to_nearest_obstacle(pos)
        
        features = np.array([
            x / 50.0,  # Normalized position
            y / 50.0,
            ax / 2.0,  # Normalized action
            ay / 2.0,
            in_hazard,  # CRITICAL: directly encode hazard
            next_in_hazard,  # Next step hazard
            min(hazard_dist / 10.0, 1.0),  # Normalized hazard distance
            min(obs_dist / 10.0, 1.0),  # Normalized obstacle distance
            (next_x) / 50.0,  # Next position
            (next_y) / 50.0,
            np.sqrt(ax**2 + ay**2) / 2.0,  # Action magnitude
            (x / 50.0) * (y / 50.0),  # Position interaction
            in_hazard * (ax / 2.0),  # Hazard-action interaction
            (1.0 - in_hazard) * min(obs_dist / 5.0, 1.0),  # Safe area indicator
            1.0  # Bias
        ])
        return features
    
    def distance_to_nearest_hazard(self, pos):
        """Calculate distance to nearest hazard zone"""
        if not self.world.hazard_zones:
            return float('inf')
        
        x, y = pos
        min_dist = float('inf')
        
        for hz in self.world.hazard_zones:
            dist = np.sqrt((hz[0] - x)**2 + (hz[1] - y)**2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def distance_to_nearest_obstacle(self, pos):
        """Calculate distance to nearest obstacle"""
        if not self.world.obstacles:
            return float('inf')
        
        x, y = pos
        min_dist = float('inf')
        
        for obs in self.world.obstacles:
            dist = np.sqrt((obs[0] - x)**2 + (obs[1] - y)**2)
            min_dist = min(min_dist, dist)
        
        return min_dist
        
    def setup_ui(self):
        """Setup interactive matplotlib UI"""
        self.fig = plt.figure(figsize=(14, 8))
        
        # Main plot
        self.ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.2)
        
        # Buttons
        ax_replan = plt.axes([0.2, 0.05, 0.15, 0.05])
        ax_save = plt.axes([0.4, 0.05, 0.15, 0.05])
        ax_done = plt.axes([0.6, 0.05, 0.15, 0.05])
        
        self.btn_replan = Button(ax_replan, 'Replan with Feedback')
        self.btn_save = Button(ax_save, 'Save Model')
        self.btn_done = Button(ax_done, 'Done & Compare')
        
        self.btn_replan.on_clicked(self.replan)
        self.btn_save.on_clicked(self.save_model)
        self.btn_done.on_clicked(self.finish_training)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial planning
        self.replan(None)
        
    def replan(self, event):
        """Replan with current reward model"""
        self.iteration += 1
        
        # Use enhanced predict function
        original_predict = self.reward_model.predict
        def enhanced_predict(pos, action):
            features = self.extract_enhanced_features(pos, action)
            reward = np.dot(self.reward_model.weights, features)
            return np.clip(reward, -50, 50)
        
        self.reward_model.predict = enhanced_predict
        
        # Plan with TAMER - increase lambda progressively
        lambda_weight = min(5.0, 1.0 * self.iteration) if self.iteration > 1 else 0.0
        planner = AStarPlanner(self.world, self.reward_model, lambda_weight)
        self.current_path = planner.plan(self.start, self.goal)
        
        # Restore original predict
        self.reward_model.predict = original_predict
        
        # Print debug info
        if self.iteration > 1:
            sample_hazard_pos = list(self.world.hazard_zones)[0] if self.world.hazard_zones else (25, 25)
            sample_action = (1, 0)
            features = self.extract_enhanced_features(sample_hazard_pos, sample_action)
            reward = np.dot(self.reward_model.weights, features)
            print(f"Iteration {self.iteration}: Lambda={lambda_weight:.1f}, "
                  f"Sample hazard reward={reward:.2f}, "
                  f"Weights norm={np.linalg.norm(self.reward_model.weights):.2f}")
        
        self.draw()
        
    def draw(self):
        """Draw current state"""
        self.ax.clear()
        self.ax.set_xlim(-1, self.world.width)
        self.ax.set_ylim(-1, self.world.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        title = f"{self.scenario_name} - Iteration {self.iteration}\n"
        title += f"Feedback given: {self.feedback_count} | "
        title += "LEFT CLICK near path: Good (+1) | RIGHT CLICK: Bad (-1)"
        self.ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Draw obstacles
        for obs in self.world.obstacles:
            rect = Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, 
                           facecolor='black', alpha=0.8)
            self.ax.add_patch(rect)
        
        # Draw hazard zones with transparency
        for hz in self.world.hazard_zones:
            rect = Rectangle((hz[0]-0.5, hz[1]-0.5), 1, 1, 
                           facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
            self.ax.add_patch(rect)
        
        # Add hazard zone label
        if self.world.hazard_zones:
            hz_center_x = sum(hz[0] for hz in self.world.hazard_zones) / len(self.world.hazard_zones)
            hz_center_y = sum(hz[1] for hz in self.world.hazard_zones) / len(self.world.hazard_zones)
            self.ax.text(hz_center_x, hz_center_y, 'HAZARD\nZONE', 
                        ha='center', va='center', fontsize=12, 
                        fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Draw path with segments
        if self.current_path:
            for i in range(len(self.current_path) - 1):
                p1 = self.current_path[i]
                p2 = self.current_path[i+1]
                
                # Color based on hazard
                color = 'red' if p1 in self.world.hazard_zones else 'blue'
                alpha = 0.8 if p1 in self.world.hazard_zones else 0.7
                linewidth = 4 if p1 in self.world.hazard_zones else 3
                
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=color, linewidth=linewidth, alpha=alpha)
                
                # Draw small circles at path nodes for clicking
                self.ax.plot(p1[0], p1[1], 'o', color=color, 
                           markersize=8, alpha=0.6)
            
            # Metrics
            path_length = sum(np.sqrt((self.current_path[i+1][0]-self.current_path[i][0])**2 + 
                                     (self.current_path[i+1][1]-self.current_path[i][1])**2) 
                            for i in range(len(self.current_path)-1))
            hazard_cells = sum(1 for p in self.current_path if p in self.world.hazard_zones)
            
            box_color = 'lightcoral' if hazard_cells > 0 else 'lightgreen'
            metrics = f"Path Length: {path_length:.1f}\nHazard Cells: {hazard_cells}"
            self.ax.text(0.02, 0.98, metrics, transform=self.ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor=box_color, alpha=0.9), fontsize=11, fontweight='bold')
        
        # Draw start and goal
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=15, label='Start', zorder=10)
        self.ax.plot(self.goal[0], self.goal[1], 'r*', markersize=20, label='Goal', zorder=10)
        
        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        
    def on_click(self, event):
        """Handle mouse clicks for feedback"""
        if event.inaxes != self.ax or self.current_path is None:
            return
        
        click_pos = (event.xdata, event.ydata)
        
        # Find nearest path segment
        min_dist = float('inf')
        nearest_segment = None
        
        for i in range(len(self.current_path) - 1):
            p1 = self.current_path[i]
            p2 = self.current_path[i+1]
            
            # Distance to segment
            dist = self.point_to_segment_distance(click_pos, p1, p2)
            if dist < min_dist:
                min_dist = dist
                nearest_segment = (p1, p2, i)
        
        if min_dist < 3.0:  # Within 3 units
            p1, p2, idx = nearest_segment
            action = (p2[0] - p1[0], p2[1] - p1[1])
            
            # Left click = positive, Right click = negative
            # Make feedback stronger for better learning
            feedback = 3.0 if event.button == 1 else -3.0
            
            # Update using enhanced features
            features = self.extract_enhanced_features(p1, action)
            prediction = np.dot(self.reward_model.weights, features)
            error = feedback - prediction
            self.reward_model.weights += self.reward_model.alpha * error * features
            self.reward_model.feedback_history.append((p1, action, feedback))
            
            self.feedback_count += 1
            
            # Visual feedback
            color = 'green' if feedback > 0 else 'red'
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color=color, linewidth=6, alpha=0.9, zorder=5)
            self.ax.text(p1[0], p1[1], f"{'+3' if feedback > 0 else '-3'}", 
                       fontsize=14, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.fig.canvas.draw()
            
            in_hazard = "IN HAZARD" if p1 in self.world.hazard_zones else "safe"
            print(f"Feedback {self.feedback_count}: {'+3' if feedback > 0 else '-3'} at {p1} ({in_hazard}), "
                  f"prediction was {prediction:.2f}")
    
    def point_to_segment_distance(self, point, seg_start, seg_end):
        """Calculate distance from point to line segment"""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        
        return np.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
    
    def save_model(self, event):
        """Save current model"""
        filename = f"tamer_model_{self.scenario_name}.pkl"
        self.reward_model.save(filename)
        print(f"\nModel saved to {filename}")
        print(f"Total feedback provided: {self.feedback_count}")
        print(f"Weights norm: {np.linalg.norm(self.reward_model.weights):.2f}")
    
    def finish_training(self, event):
        """Finish training and show comparison"""
        self.save_model(None)
        plt.close(self.fig)
        
        # Generate comparison
        print("\nGenerating comparison visualization...")
        
        # Baseline A*
        baseline_planner = AStarPlanner(self.world)
        baseline_path = baseline_planner.plan(self.start, self.goal)
        
        # TAMER-augmented A* with enhanced features
        def make_enhanced_predict():
            def enhanced_predict(pos, action):
                features = self.extract_enhanced_features(pos, action)
                reward = np.dot(self.reward_model.weights, features)
                return np.clip(reward, -50, 50)
            return enhanced_predict
        
        self.reward_model.predict = make_enhanced_predict()
        tamer_planner = AStarPlanner(self.world, self.reward_model, lambda_weight=5.0)
        tamer_path = tamer_planner.plan(self.start, self.goal)
        
        # Visualize
        viz = Visualizer(self.world)
        fig = viz.plot_scenario(self.start, self.goal, 
                               baseline_path=baseline_path,
                               tamer_path=tamer_path,
                               title=f"{self.scenario_name} - Comparison",
                               show_hazards=True)
        
        output_file = f"comparison_{self.scenario_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {output_file}")
        
        plt.show()

def train_scenario(scenario_num):
    """Train on a specific scenario"""
    scenarios = [create_scenario_1(), create_scenario_2(), create_scenario_3()]
    
    if scenario_num < 1 or scenario_num > 3:
        print("Invalid scenario number. Choose 1, 2, or 3.")
        return
    
    world, start, goal, title = scenarios[scenario_num - 1]
    scenario_name = f"scenario_{scenario_num}"
    
    print("=" * 70)
    print(f"TAMER Interactive Training - {title}")
    print("=" * 70)
    print("\nInstructions:")
    print("1. LEFT CLICK near path segments you LIKE (good behavior)")
    print("2. RIGHT CLICK near path segments you DON'T LIKE (bad behavior)")
    print("3. Click 'Replan with Feedback' to see updated path")
    print("4. Repeat steps 1-3 until satisfied")
    print("5. Click 'Save Model' to save progress")
    print("6. Click 'Done & Compare' to see final results")
    print("\nTips:")
    print("- RIGHT CLICK on RED path segments (in hazard zones)")
    print("- LEFT CLICK on BLUE path segments (safe areas)")
    print("- Give 5-10 feedback per iteration, then replan")
    print("- Watch the path change to avoid hazards!")
    print("=" * 70)
    print()
    
    trainer = InteractiveTAMERTrainer(world, start, goal, scenario_name)
    plt.show()

if __name__ == "__main__":
    print("\nSelect scenario to train:")
    print("1. Scenario 1: Implicit Hazard Avoidance")
    print("2. Scenario 2: Safety Margin Preference")
    print("3. Scenario 3: Subjective Route Preference")
    
    choice = input("\nEnter scenario number (1-3): ")
    
    try:
        scenario_num = int(choice)
        train_scenario(scenario_num)
    except ValueError:
        print("Invalid input. Please enter a number 1, 2, or 3.")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup interactive matplotlib UI"""
        self.fig = plt.figure(figsize=(14, 8))
        
        # Main plot
        self.ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.2)
        
        # Buttons
        ax_replan = plt.axes([0.2, 0.05, 0.15, 0.05])
        ax_save = plt.axes([0.4, 0.05, 0.15, 0.05])
        ax_done = plt.axes([0.6, 0.05, 0.15, 0.05])
        
        self.btn_replan = Button(ax_replan, 'Replan with Feedback')
        self.btn_save = Button(ax_save, 'Save Model')
        self.btn_done = Button(ax_done, 'Done & Compare')
        
        self.btn_replan.on_clicked(self.replan)
        self.btn_save.on_clicked(self.save_model)
        self.btn_done.on_clicked(self.finish_training)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial planning
        self.replan(None)
        
    def replan(self, event):
        """Replan with current reward model"""
        self.iteration += 1
        
        # Plan with TAMER
        lambda_weight = 1.0 if self.iteration > 1 else 0.0
        planner = AStarPlanner(self.world, self.reward_model, lambda_weight)
        self.current_path = planner.plan(self.start, self.goal)
        
        self.draw()
        
    def draw(self):
        """Draw current state"""
        self.ax.clear()
        self.ax.set_xlim(-1, self.world.width)
        self.ax.set_ylim(-1, self.world.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        title = f"{self.scenario_name} - Iteration {self.iteration}\n"
        title += f"Feedback given: {self.feedback_count} | "
        title += "LEFT CLICK near path: Good (+1) | RIGHT CLICK: Bad (-1)"
        self.ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Draw obstacles
        for obs in self.world.obstacles:
            rect = Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, 
                           facecolor='black', alpha=0.8)
            self.ax.add_patch(rect)
        
        # Draw hazard zones
        for hz in self.world.hazard_zones:
            rect = Rectangle((hz[0]-0.5, hz[1]-0.5), 1, 1, 
                           facecolor='red', alpha=0.2)
            self.ax.add_patch(rect)
        
        # Draw path with segments
        if self.current_path:
            for i in range(len(self.current_path) - 1):
                p1 = self.current_path[i]
                p2 = self.current_path[i+1]
                
                # Color based on hazard
                color = 'red' if p1 in self.world.hazard_zones else 'blue'
                alpha = 0.5 if p1 in self.world.hazard_zones else 0.7
                
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=color, linewidth=3, alpha=alpha)
                
                # Draw small circles at path nodes for clicking
                self.ax.plot(p1[0], p1[1], 'o', color=color, 
                           markersize=6, alpha=0.5)
            
            # Metrics
            path_length = sum(np.sqrt((self.current_path[i+1][0]-self.current_path[i][0])**2 + 
                                     (self.current_path[i+1][1]-self.current_path[i][1])**2) 
                            for i in range(len(self.current_path)-1))
            hazard_cells = sum(1 for p in self.current_path if p in self.world.hazard_zones)
            
            metrics = f"Path Length: {path_length:.1f}\nHazard Cells: {hazard_cells}"
            self.ax.text(0.02, 0.98, metrics, transform=self.ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='lightgreen', alpha=0.8), fontsize=10)
        
        # Draw start and goal
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=15, label='Start')
        self.ax.plot(self.goal[0], self.goal[1], 'r*', markersize=20, label='Goal')
        
        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        
    def on_click(self, event):
        """Handle mouse clicks for feedback"""
        if event.inaxes != self.ax or self.current_path is None:
            return
        
        click_pos = (event.xdata, event.ydata)
        
        # Find nearest path segment
        min_dist = float('inf')
        nearest_segment = None
        
        for i in range(len(self.current_path) - 1):
            p1 = self.current_path[i]
            p2 = self.current_path[i+1]
            
            # Distance to segment
            dist = self.point_to_segment_distance(click_pos, p1, p2)
            if dist < min_dist:
                min_dist = dist
                nearest_segment = (p1, p2, i)
        
        if min_dist < 3.0:  # Within 3 units
            p1, p2, idx = nearest_segment
            action = (p2[0] - p1[0], p2[1] - p1[1])
            
            # Left click = positive, Right click = negative
            feedback = 1.0 if event.button == 1 else -1.0
            
            self.reward_model.update(p1, action, feedback)
            self.feedback_count += 1
            
            # Visual feedback
            color = 'green' if feedback > 0 else 'red'
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color=color, linewidth=5, alpha=0.8)
            self.ax.text(p1[0], p1[1], f"{'+1' if feedback > 0 else '-1'}", 
                       fontsize=12, color=color, fontweight='bold')
            
            self.fig.canvas.draw()
            
            print(f"Feedback {self.feedback_count}: {'+' if feedback > 0 else '-'}1 at position {p1}")
    
    def point_to_segment_distance(self, point, seg_start, seg_end):
        """Calculate distance from point to line segment"""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        
        return np.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
    
    def save_model(self, event):
        """Save current model"""
        filename = f"tamer_model_{self.scenario_name}.pkl"
        self.reward_model.save(filename)
        print(f"\nModel saved to {filename}")
        print(f"Total feedback provided: {self.feedback_count}")
    
    def finish_training(self, event):
        """Finish training and show comparison"""
        self.save_model(None)
        plt.close(self.fig)
        
        # Generate comparison
        print("\nGenerating comparison visualization...")
        
        # Baseline A*
        baseline_planner = AStarPlanner(self.world)
        baseline_path = baseline_planner.plan(self.start, self.goal)
        
        # TAMER-augmented A*
        tamer_planner = AStarPlanner(self.world, self.reward_model, lambda_weight=1.0)
        tamer_path = tamer_planner.plan(self.start, self.goal)
        
        # Visualize
        viz = Visualizer(self.world)
        fig = viz.plot_scenario(self.start, self.goal, 
                               baseline_path=baseline_path,
                               tamer_path=tamer_path,
                               title=f"{self.scenario_name} - Comparison",
                               show_hazards=True)
        
        output_file = f"comparison_{self.scenario_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {output_file}")
        
        plt.show()

def train_scenario(scenario_num):
    """Train on a specific scenario"""
    scenarios = [create_scenario_1(), create_scenario_2(), create_scenario_3()]
    
    if scenario_num < 1 or scenario_num > 3:
        print("Invalid scenario number. Choose 1, 2, or 3.")
        return
    
    world, start, goal, title = scenarios[scenario_num - 1]
    scenario_name = f"scenario_{scenario_num}"
    
    print("=" * 70)
    print(f"TAMER Interactive Training - {title}")
    print("=" * 70)
    print("\nInstructions:")
    print("1. LEFT CLICK near path segments you LIKE (good behavior)")
    print("2. RIGHT CLICK near path segments you DON'T LIKE (bad behavior)")
    print("3. Click 'Replan with Feedback' to see updated path")
    print("4. Repeat steps 1-3 until satisfied")
    print("5. Click 'Save Model' to save progress")
    print("6. Click 'Done & Compare' to see final results")
    print("\nTips:")
    print("- Give feedback on 5-10 segments per iteration")
    print("- Focus on hazard zones and areas too close to obstacles")
    print("- The red zones are hazards you want the agent to avoid")
    print("=" * 70)
    print()
    
    trainer = InteractiveTAMERTrainer(world, start, goal, scenario_name)
    plt.show()

if __name__ == "__main__":
    print("\nSelect scenario to train:")
    print("1. Scenario 1: Implicit Hazard Avoidance")
    print("2. Scenario 2: Safety Margin Preference")
    print("3. Scenario 3: Subjective Route Preference")
    
    choice = input("\nEnter scenario number (1-3): ")
    
    try:
        scenario_num = int(choice)
        train_scenario(scenario_num)
    except ValueError:
        print("Invalid input. Please enter a number 1, 2, or 3.")