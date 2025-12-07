import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import Arrow
import matplotlib.patches as mpatches

# Use a nice style
plt.style.use('seaborn-v0_8-darkgrid')

def create_title_slide():
    """Create an aesthetic title slide graphic"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background gradient effect
    for i in range(100):
        alpha = 0.3 * (1 - i/100)
        circle = Circle((5, 5), 0.05 * i, color='#2196F3', alpha=alpha, zorder=0)
        ax.add_patch(circle)
    
    # Robot icon (simple representation)
    robot_body = FancyBboxPatch((3.5, 5.5), 1.5, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#4CAF50', edgecolor='#2E7D32', 
                                linewidth=3, zorder=5)
    ax.add_patch(robot_body)
    
    # Robot eyes
    eye1 = Circle((4, 6.3), 0.15, color='white', zorder=6)
    eye2 = Circle((4.5, 6.3), 0.15, color='white', zorder=6)
    ax.add_patch(eye1)
    ax.add_patch(eye2)
    
    # Pupil
    pupil1 = Circle((4, 6.3), 0.07, color='black', zorder=7)
    pupil2 = Circle((4.5, 6.3), 0.07, color='black', zorder=7)
    ax.add_patch(pupil1)
    ax.add_patch(pupil2)
    
    # Antenna
    ax.plot([4.25, 4.25], [6.7, 7.2], 'k-', linewidth=3, zorder=5)
    antenna_top = Circle((4.25, 7.2), 0.1, color='#FF5722', zorder=6)
    ax.add_patch(antenna_top)
    
    # Human icon
    human_head = Circle((6.5, 6.5), 0.3, color='#FF9800', zorder=5)
    ax.add_patch(human_head)
    human_body = FancyBboxPatch((6.2, 5.5), 0.6, 1.0,
                                boxstyle="round,pad=0.05",
                                facecolor='#FF9800', edgecolor='#E65100',
                                linewidth=2, zorder=5)
    ax.add_patch(human_body)
    
    # Feedback arrows
    arrow1 = FancyArrowPatch((5.2, 6.2), (6, 6.2),
                            arrowstyle='->', mutation_scale=30,
                            color='#9C27B0', linewidth=3, zorder=4)
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((6, 5.8), (5.2, 5.8),
                            arrowstyle='->', mutation_scale=30,
                            color='#9C27B0', linewidth=3, zorder=4)
    ax.add_patch(arrow2)
    
    # Text labels
    ax.text(4.25, 4.8, 'Robot', fontsize=16, ha='center', fontweight='bold', color='#2E7D32')
    ax.text(6.5, 4.8, 'Human', fontsize=16, ha='center', fontweight='bold', color='#E65100')
    ax.text(5.6, 6.6, 'Feedback', fontsize=12, ha='center', 
            fontweight='bold', color='#9C27B0', style='italic')
    
    # Title
    ax.text(5, 8.5, 'Learning to Navigate', fontsize=32, ha='center', 
            fontweight='bold', color='#1565C0')
    ax.text(5, 8.0, 'TAMER-Augmented A*', fontsize=28, ha='center',
            fontweight='bold', color='#0D47A1')
    
    # Subtitle
    ax.text(5, 1.5, 'Teaching Robots Human Preferences Through Feedback', 
            fontsize=16, ha='center', style='italic', color='#424242')
    
    plt.tight_layout()
    plt.savefig('slide_title.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created: slide_title.png")
    plt.close()

def create_problem_illustration():
    """Create a beautiful problem statement illustration"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LEFT: Baseline A* (bad path)
    ax = axes[0]
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 21)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_title('❌ Baseline A*: Shortest ≠ Best', fontsize=16, fontweight='bold', color='#D32F2F')
    
    # Obstacles
    for obs in [(5, 5, 2, 10), (15, 5, 2, 10)]:
        rect = Rectangle((obs[0], obs[1]), obs[2], obs[3], 
                        facecolor='#424242', edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
    
    # Hazard zone (red)
    hazard = Rectangle((8, 8), 6, 6, facecolor='#FF5252', alpha=0.4, 
                       edgecolor='#D32F2F', linewidth=3, linestyle='--')
    ax.add_patch(hazard)
    ax.text(11, 11, '⚠\nCrowded\nArea', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#B71C1C')
    
    # Baseline path (goes through hazard)
    path = [(2, 10), (5, 10), (8, 10), (11, 11), (14, 10), (17, 10), (19, 10)]
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, 'b-', linewidth=4, alpha=0.7, label='Shortest Path')
    
    # Add arrow heads
    for i in range(len(path)-1):
        if i % 2 == 0:
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            ax.arrow(path[i][0], path[i][1], dx*0.3, dy*0.3,
                    head_width=0.5, head_length=0.3, fc='blue', ec='blue', alpha=0.7)
    
    # Start and goal
    start = Circle((2, 10), 0.6, color='#4CAF50', edgecolor='#2E7D32', linewidth=3, zorder=10)
    goal = Circle((19, 10), 0.7, color='#F44336', edgecolor='#C62828', linewidth=3, zorder=10)
    ax.add_patch(start)
    ax.add_patch(goal)
    ax.text(2, 10, 'S', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(19, 10, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Metrics box
    metrics = "Path Length: 17.0\nThrough Hazard: Yes ❌\nUser Satisfaction: Low"
    ax.text(0.05, 0.95, metrics, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.9),
            fontweight='bold')
    
    # RIGHT: Desired behavior (good path)
    ax = axes[1]
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 21)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_title('✓ Desired: Human-Preferred Path', fontsize=16, fontweight='bold', color='#388E3C')
    
    # Same obstacles
    for obs in [(5, 5, 2, 10), (15, 5, 2, 10)]:
        rect = Rectangle((obs[0], obs[1]), obs[2], obs[3], 
                        facecolor='#424242', edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
    
    # Hazard zone
    hazard = Rectangle((8, 8), 6, 6, facecolor='#FF5252', alpha=0.4,
                       edgecolor='#D32F2F', linewidth=3, linestyle='--')
    ax.add_patch(hazard)
    ax.text(11, 11, '⚠\nCrowded\nArea', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#B71C1C')
    
    # Better path (goes around)
    path2 = [(2, 10), (4, 10), (5, 8), (6, 6), (8, 5), (11, 4), (14, 5), (16, 6), (17, 8), (18, 10), (19, 10)]
    path_x = [p[0] for p in path2]
    path_y = [p[1] for p in path2]
    ax.plot(path_x, path_y, color='#4CAF50', linewidth=4, alpha=0.8, label='Preferred Path')
    
    # Add arrow heads
    for i in range(0, len(path2)-1, 3):
        dx = path2[i+1][0] - path2[i][0]
        dy = path2[i+1][1] - path2[i][1]
        ax.arrow(path2[i][0], path2[i][1], dx*0.3, dy*0.3,
                head_width=0.5, head_length=0.3, fc='#4CAF50', ec='#4CAF50', alpha=0.8)
    
    # Start and goal
    start = Circle((2, 10), 0.6, color='#4CAF50', edgecolor='#2E7D32', linewidth=3, zorder=10)
    goal = Circle((19, 10), 0.7, color='#F44336', edgecolor='#C62828', linewidth=3, zorder=10)
    ax.add_patch(start)
    ax.add_patch(goal)
    ax.text(2, 10, 'S', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(19, 10, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Metrics box
    metrics = "Path Length: 19.5\nThrough Hazard: No ✓\nUser Satisfaction: High"
    ax.text(0.05, 0.95, metrics, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.9),
            fontweight='bold')
    
    plt.suptitle('The Challenge: Encoding Human Preferences', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('slide_problem.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created: slide_problem.png")
    plt.close()

def create_approach_diagram():
    """Create the TAMER approach flowchart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define positions
    boxes = {
        'astar': (1.5, 7, 1.5, 1),
        'feedback': (5, 7, 2, 1),
        'tamer': (1.5, 4, 1.5, 1),
        'replan': (5, 4, 2, 1),
    }
    
    # Box 1: A* Plans
    box1 = FancyBboxPatch((boxes['astar'][0], boxes['astar'][1]), 
                         boxes['astar'][2], boxes['astar'][3],
                         boxstyle="round,pad=0.1", 
                         facecolor='#2196F3', edgecolor='#1565C0', 
                         linewidth=3)
    ax.add_patch(box1)
    ax.text(boxes['astar'][0] + boxes['astar'][2]/2, 
            boxes['astar'][1] + boxes['astar'][3]/2,
            'A* Plans\nPath', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Box 2: Human Feedback
    box2 = FancyBboxPatch((boxes['feedback'][0], boxes['feedback'][1]),
                         boxes['feedback'][2], boxes['feedback'][3],
                         boxstyle="round,pad=0.1",
                         facecolor='#FF9800', edgecolor='#E65100',
                         linewidth=3)
    ax.add_patch(box2)
    ax.text(boxes['feedback'][0] + boxes['feedback'][2]/2,
            boxes['feedback'][1] + boxes['feedback'][3]/2,
            'Human\nFeedback\n(+1/-1)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Box 3: TAMER Learns
    box3 = FancyBboxPatch((boxes['tamer'][0], boxes['tamer'][1]),
                         boxes['tamer'][2], boxes['tamer'][3],
                         boxstyle="round,pad=0.1",
                         facecolor='#9C27B0', edgecolor='#6A1B9A',
                         linewidth=3)
    ax.add_patch(box3)
    ax.text(boxes['tamer'][0] + boxes['tamer'][2]/2,
            boxes['tamer'][1] + boxes['tamer'][3]/2,
            'TAMER\nLearns Ĥ(p,a)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Box 4: Replan
    box4 = FancyBboxPatch((boxes['replan'][0], boxes['replan'][1]),
                         boxes['replan'][2], boxes['replan'][3],
                         boxstyle="round,pad=0.1",
                         facecolor='#4CAF50', edgecolor='#2E7D32',
                         linewidth=3)
    ax.add_patch(box4)
    ax.text(boxes['replan'][0] + boxes['replan'][2]/2,
            boxes['replan'][1] + boxes['replan'][3]/2,
            'Replan with\nPreferences', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Arrows
    # 1 -> 2
    arrow1 = FancyArrowPatch((2.25, 7.5), (5, 7.5),
                            arrowstyle='->', mutation_scale=30,
                            color='#424242', linewidth=3)
    ax.add_patch(arrow1)
    
    # 2 -> 3
    arrow2 = FancyArrowPatch((6, 7), (2.25, 5),
                            arrowstyle='->', mutation_scale=30,
                            color='#424242', linewidth=3)
    ax.add_patch(arrow2)
    
    # 3 -> 4
    arrow3 = FancyArrowPatch((3, 4.5), (5, 4.5),
                            arrowstyle='->', mutation_scale=30,
                            color='#424242', linewidth=3)
    ax.add_patch(arrow3)
    
    # 4 -> 1 (loop back)
    arrow4 = FancyArrowPatch((6, 5), (2.25, 7),
                            arrowstyle='->', mutation_scale=30,
                            color='#D32F2F', linewidth=3, linestyle='--')
    ax.add_patch(arrow4)
    ax.text(4, 6.2, 'Iterate', fontsize=12, fontweight='bold', 
            color='#D32F2F', style='italic')
    
    # Title
    ax.text(5, 9, 'TAMER-Augmented A* Approach', fontsize=20, 
            ha='center', fontweight='bold', color='#1565C0')
    
    # Formula box
    formula_box = FancyBboxPatch((1, 1.5), 8, 1.2,
                                boxstyle="round,pad=0.15",
                                facecolor='#E3F2FD', edgecolor='#1565C0',
                                linewidth=3)
    ax.add_patch(formula_box)
    ax.text(5, 2.4, 'Modified Cost Function:', fontsize=13, ha='center', fontweight='bold')
    ax.text(5, 1.9, 'f(n) = g(n) + h(n) - λ·Ĥ(p, aₙ)', fontsize=16, 
            ha='center', fontweight='bold', family='monospace', color='#0D47A1')
    
    # Legend
    ax.text(1, 0.8, 'g(n): cost from start', fontsize=10, style='italic')
    ax.text(1, 0.5, 'h(n): heuristic to goal', fontsize=10, style='italic')
    ax.text(5, 0.8, 'λ: preference weight', fontsize=10, style='italic')
    ax.text(5, 0.5, 'Ĥ(p,aₙ): learned reward', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('slide_approach.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created: slide_approach.png")
    plt.close()

def create_features_diagram():
    """Create feature engineering explanation"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # LEFT: Without hazard features
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('❌ Original Features (Failed)', fontsize=14, fontweight='bold', color='#D32F2F')
    
    features_bad = [
        'x position',
        'y position',
        'action x',
        'action y',
        'next x',
        'next y',
        'action magnitude',
        'position interaction',
        'bias'
    ]
    
    y_pos = 8
    for i, feat in enumerate(features_bad):
        box = FancyBboxPatch((1, y_pos - i*0.7), 8, 0.5,
                            boxstyle="round,pad=0.05",
                            facecolor='#FFCDD2', edgecolor='#D32F2F',
                            linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(5, y_pos - i*0.7 + 0.25, feat, ha='center', va='center',
                fontsize=11, fontweight='bold')
    
    # Problem text
    ax.text(5, 1.5, '⚠ Problem: No direct hazard info!', ha='center',
            fontsize=12, fontweight='bold', color='#B71C1C',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
    ax.text(5, 0.8, 'Model can\'t "see" hazards', ha='center',
            fontsize=10, style='italic', color='#D32F2F')
    
    # RIGHT: With hazard features
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('✓ Enhanced Features (Success)', fontsize=14, fontweight='bold', color='#388E3C')
    
    features_good = [
        ('x position', False),
        ('y position', False),
        ('IN HAZARD? ⭐', True),
        ('NEXT IN HAZARD? ⭐', True),
        ('DISTANCE TO HAZARD ⭐', True),
        ('DISTANCE TO OBSTACLE ⭐', True),
        ('action x, y', False),
        ('hazard-action interaction ⭐', True),
        ('safe area indicator ⭐', True),
    ]
    
    y_pos = 8.5
    for i, (feat, is_new) in enumerate(features_good):
        color = '#C8E6C9' if is_new else '#E0E0E0'
        edge_color = '#388E3C' if is_new else '#9E9E9E'
        box = FancyBboxPatch((1, y_pos - i*0.7), 8, 0.5,
                            boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor=edge_color,
                            linewidth=2 if is_new else 1, alpha=0.8)
        ax.add_patch(box)
        ax.text(5, y_pos - i*0.7 + 0.25, feat, ha='center', va='center',
                fontsize=11, fontweight='bold' if is_new else 'normal')
    
    # Success text
    ax.text(5, 1.2, '✓ Solution: Explicit hazard awareness!', ha='center',
            fontsize=12, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))
    ax.text(5, 0.5, 'Model can learn general preferences', ha='center',
            fontsize=10, style='italic', color='#388E3C')
    
    plt.suptitle('The Critical Fix: Feature Engineering', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('slide_features.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created: slide_features.png")
    plt.close()

def create_results_highlight():
    """Create a results highlight graphic"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.2, 'Key Results', fontsize=28, ha='center', 
            fontweight='bold', color='#1565C0')
    
    # Three result boxes
    boxes_info = [
        (1, 5.5, '80-100%', 'Hazard\nReduction', '#4CAF50'),
        (4, 5.5, '5-15%', 'Path Length\nIncrease', '#FF9800'),
        (7, 5.5, '3-5', 'Iterations to\nConverge', '#9C27B0'),
    ]
    
    for x, y, value, label, color in boxes_info:
        # Box
        box = FancyBboxPatch((x, y), 2, 2.5,
                            boxstyle="round,pad=0.15",
                            facecolor=color, edgecolor='white',
                            linewidth=4, alpha=0.9)
        ax.add_patch(box)
        
        # Value
        ax.text(x + 1, y + 1.7, value, ha='center', va='center',
                fontsize=32, fontweight='bold', color='white')
        
        # Label
        ax.text(x + 1, y + 0.7, label, ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
    
    # Bottom conclusion
    conclusion_box = FancyBboxPatch((1, 2), 8, 2,
                                   boxstyle="round,pad=0.15",
                                   facecolor='#E3F2FD', edgecolor='#1565C0',
                                   linewidth=3)
    ax.add_patch(conclusion_box)
    
    ax.text(5, 3.3, '✓ Successfully learned human preferences', 
            ha='center', fontsize=15, fontweight='bold', color='#1565C0')
    ax.text(5, 2.8, '✓ Balanced safety vs. efficiency trade-off',
            ha='center', fontsize=15, fontweight='bold', color='#1565C0')
    ax.text(5, 2.3, '✓ Fast convergence (minutes, not hours)',
            ha='center', fontsize=15, fontweight='bold', color='#1565C0')
    
    # Check marks
    for y_val in [3.3, 2.8, 2.3]:
        check = Circle((1.5, y_val), 0.15, color='#4CAF50', zorder=10)
        ax.add_patch(check)
    
    plt.tight_layout()
    plt.savefig('slide_results_highlight.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created: slide_results_highlight.png")
    plt.close()

def create_future_work():
    """Create future work slide"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.2, 'Future Work', fontsize=28, ha='center',
            fontweight='bold', color='#1565C0')
    
    # Timeline boxes
    timeline_items = [
        (1, 7, '📊 User Study', '10+ participants\nReal feedback collection', '#4CAF50'),
        (1, 5, '🤖 RRT* Extension', 'Continuous spaces\nComplex environments', '#2196F3'),
        (1, 3, '🧠 Neural Networks', 'Deep learning rewards\nRicher representations', '#9C27B0'),
        (6, 7, '📈 Statistical Analysis', 'Significance testing\nPerformance metrics', '#FF9800'),
        (6, 5, '🎮 Interactive Demo', 'Real-time GUI\nLive demonstrations', '#F44336'),
        (6, 3, '🦾 Robot Deployment', 'Physical robot\nReal-world testing', '#00BCD4'),
    ]
    
    for x, y, title, desc, color in timeline_items:
        box = FancyBboxPatch((x, y), 3.5, 1.3,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='white',
                            linewidth=3, alpha=0.85)
        ax.add_patch(box)
        
        ax.text(x + 1.75, y + 0.9, title, ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
        ax.text(x + 1.75, y + 0.4, desc, ha='center', va='center',
                fontsize=9, color='white')
    
    # Bottom note
    ax.text(5, 1.2, 'Goal: Demonstrate robots can learn individual preferences', 
            ha='center', fontsize=14, fontweight='bold', color='#1565C0',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9, pad=0.5))
    ax.text(5, 0.6, 'in just minutes of interaction',
            ha='center', fontsize=14, fontweight='bold', style='italic', color='#1565C0')
    
    plt.tight_layout()
    plt.savefig('slide_future_work.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Created: slide_future_work.png")
    plt.close()

# Generate all slides
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating Aesthetic Presentation Images")
    print("="*60 + "\n")
    
    create_title_slide()
    create_problem_illustration()
    create_approach_diagram()
    create_features_diagram()
    create_results_highlight()
    create_future_work()
    
    print("\n" + "="*60)
    print("✨ All presentation images generated!")
    print("="*60)
    print("\nFiles created:")
    print("  1. slide_title.png - Title slide with robot/human graphic")
    print("  2. slide_problem.png - Problem illustration (bad vs good path)")
    print("  3. slide_approach.png - TAMER flowchart diagram")
    print