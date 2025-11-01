# TAMER-Augmented A* Navigation System

**Authors:** Diksha Jaiswal & Rakshit Kadam  
**Course:** Imitation Learning for Robotics, Purdue University

## Quick Start (For Lightning Talk)

### Step 1: Install Dependencies
```bash
pip install numpy matplotlib pandas
```

### Step 2: Save the Code Files

Save these three files in the same directory:
1. `tamer_astar_baseline.py` - Core A* and TAMER implementation
2. `tamer_interactive_training.py` - Interactive training interface
3. `tamer_evaluation_suite.py` - Automated evaluation and visualization

### Step 3: Generate All Results (FASTEST - 1 minute)

```bash
python tamer_evaluation_suite.py
```

This will automatically:
- Run baseline A* on 3 scenarios
- Train TAMER models with simulated feedback
- Generate all comparison visualizations
- Create training progress plots
- Generate summary report

**Output files** (ready for presentation):
- `scenario_1_comparison.png`
- `scenario_1_training.png`
- `scenario_2_comparison.png`
- `scenario_2_training.png`
- `scenario_3_comparison.png`
- `scenario_3_training.png`
- `summary_report.png`

---

## Alternative: Interactive Training (Optional)

If you want to provide real human feedback:

```bash
python tamer_interactive_training.py
```

Then select a scenario (1, 2, or 3) and:
1. LEFT CLICK on path segments you like (positive feedback)
2. RIGHT CLICK on path segments you don't like (negative feedback)
3. Click "Replan with Feedback" to see updated path
4. Repeat until satisfied
5. Click "Done & Compare" to generate final visualization

---

## Project Overview

### Motivation
Classic motion planners like A* find mathematically optimal paths but ignore human preferences like safety margins, comfort, or subjective route choices. This project uses **TAMER** (Training an Agent via Evaluative Reinforcement) to learn these preferences from human feedback.

### Technical Approach

**Baseline:** Standard A* with cost function:
```
f(n) = g(n) + h(n)
```

**TAMER-Augmented:** Modified cost function:
```
f(n) = g(n) + h(n) - λ · Ĥ(p, a_n)
```

Where:
- `g(n)` = cost from start to node n
- `h(n)` = heuristic (Euclidean distance to goal)
- `Ĥ(p, a_n)` = learned human reward for action a_n from parent p
- `λ` = weight controlling influence of human preferences

### Three Evaluation Scenarios

1. **Implicit Hazard Avoidance**
   - Shortest path crosses a "hazard zone"
   - TAMER learns to route around it despite longer path

2. **Safety Margin Preference**
   - Shortest path passes too close to obstacles
   - TAMER learns to maintain comfortable distance

3. **Subjective Route Preference**
   - Two equivalent paths exist
   - TAMER learns user's preferred route

---

## Implementation Details

### TAMER Reward Model
- **Architecture:** Linear model with hand-crafted features
- **Features:** Position, action, obstacle proximity, hazard proximity
- **Learning:** Gradient descent with learning rate α = 0.1
- **Feedback:** Binary (+1 for good, -1 for bad)

### A* Integration
- **Search:** Standard A* with modified cost function
- **Heuristic:** Euclidean distance (admissible)
- **Graph:** 8-connected grid (cardinal + diagonal moves)
- **Cost Adjustment:** Learned reward subtracted from f-score

### Automated Feedback Simulation
For quick evaluation, we simulate human feedback using rules:
- Penalize paths through hazard zones (-1.0)
- Penalize proximity to obstacles (-0.8 if dist < 2)
- Reward safe corridors (+0.5)

---

## Results Summary

### Expected Outcomes
- **Hazard Avoidance:** 80-100% reduction in hazard cells crossed
- **Path Length:** 5-15% increase (acceptable trade-off)
- **Safety Margin:** Improved minimum obstacle distance
- **Convergence:** 3-5 iterations typically sufficient

### Key Insights
1. TAMER successfully captures human preferences without explicit cost engineering
2. Trade-off between optimality and preference alignment is controllable via λ
3. Learned models generalize to similar scenarios
4. Framework extends naturally to continuous spaces (RRT*, etc.)

---

## File Structure

```
tamer-augmented-astar/
├── tamer_astar_baseline.py          # Core implementation
├── tamer_interactive_training.py     # Interactive GUI
├── tamer_evaluation_suite.py         # Automated evaluation
├── README.md                          # This file
├── scenario_*.png                     # Generated visualizations
└── tamer_model_*.pkl                  # Saved reward models
```

---

## For Your Lightning Talk

### Recommended Slides (5 minutes)

1. **Title Slide** (15 sec)
   - Project name, team, motivation

2. **Problem Statement** (45 sec)
   - A* finds shortest path, not "best" path
   - Show example: hazard zone visual
   - Challenge: encoding human preferences

3. **Approach** (60 sec)
   - TAMER overview (human feedback → reward function)
   - Modified A* cost function equation
   - Training loop diagram

4. **Results - Scenario 1** (45 sec)
   - Side-by-side comparison: baseline vs TAMER
   - Metrics: hazard reduction
   - Training progress plot

5. **Results - All Scenarios** (45 sec)
   - Summary table
   - Key findings

6. **Conclusion** (30 sec)
   - Successfully demonstrated preference learning
   - Extensions: RRT*, real robots
   - Q&A

### Demo Tips
- Use pre-generated PNGs (don't run live code)
- Emphasize visual comparison (red hazard zones)
- Highlight metric improvements
- Keep technical details minimal

---

## Extensions (Future Work)

1. **Advanced Planners:** Integrate with RRT*, PRM
2. **Continuous Spaces:** 2D/3D continuous environments
3. **Multi-Objective:** Balance speed, safety, comfort
4. **Deep Learning:** Neural network reward models
5. **Real Robot:** Deploy on mobile robot platform
6. **Active Learning:** Strategic feedback queries

---

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Pickle (standard library)

---

## Troubleshooting

**Import Error:**
```bash
# Make sure all files are in the same directory
ls tamer_*.py
```

**Matplotlib Backend Issues:**
```python
# Add to top of file if GUI doesn't work
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

**No Visualizations Generated:**
```bash
# Check if files were created
ls *.png
```

---

## Contact

For questions or issues, please reach out to:
- Diksha Jaiswal: [email]
- Rakshit Kadam: [email]

---

## Acknowledgments

- TAMER framework: Knox & Stone (2009)
- A* algorithm: Hart, Nilsson & Raphael (1968)
- Course: Imitation Learning for Robotics, Purdue University

---

## License

Educational project for academic purposes.
