# TrackLearnerAI

A neuroevolution-based self-driving car simulation where AI agents learn to drive on custom race tracks from scratch using the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm. No training data, no labels — just evolution.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [The Evolution Loop](#the-evolution-loop)
  - [Agent Perception (Radar Sensors)](#agent-perception-radar-sensors)
  - [Neural Network Architecture](#neural-network-architecture)
  - [Fitness Function](#fitness-function)
- [Track System](#track-system)
  - [Color-Coded Design](#color-coded-design)
  - [Track Processing Pipeline](#track-processing-pipeline)
  - [Checkpoint Ordering](#checkpoint-ordering)
- [Real-Time Neural Network Visualization](#real-time-neural-network-visualization)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [NEAT Configuration](#neat-configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Creating Custom Tracks](#creating-custom-tracks)

## Overview

TrackLearnerAI drops a population of 60 car agents onto a race track. None of them know how to drive. Through evolutionary pressure — selection, crossover, and mutation — they evolve over up to 150 generations into agents that can navigate complex tracks smoothly and efficiently.

The entire learning process is visualized in real-time using Pygame, including a live neural network topology overlay showing the best agent's brain activity as it drives.

### Key Features

- **NEAT-based neuroevolution** — both network weights and topology evolve simultaneously
- **5-direction radar sensor system** for environment perception
- **Multi-layered fitness function** with checkpoint rewards, speed bonuses, proximity shaping, and penalties
- **Real-time neural network visualization** of the best-performing agent
- **Color-coded PNG track system** — create new tracks by simply drawing them in any image editor
- **Track selector UI** — choose from multiple tracks before training begins
- **Manual driving mode** — test tracks yourself with keyboard controls
- **Automatic checkpoint detection and ordering** using OpenCV contour analysis

## How It Works

### The Evolution Loop

```
Generation 1: 60 random agents → most crash immediately
Generation 10: agents learn to go straight, some navigate turns
Generation 30: agents consistently pass multiple checkpoints
Generation 80+: agents smoothly complete full laps
```

Each generation follows this cycle:

1. **Spawn**: 60 agents are placed at the start/finish line, each with its own neural network
2. **Simulate**: All agents drive simultaneously at 60 FPS. Sensor data feeds into each agent's neural network, which outputs steering/throttle commands
3. **Evaluate**: Agents accumulate fitness based on checkpoint progress, speed, and survival
4. **Evolve**: NEAT performs selection (top 20% survive), crossover, mutation, and speciation to create the next generation
5. **Repeat** for up to 150 generations

### Agent Perception (Radar Sensors)

Each car agent perceives its environment through **5 radar beams** cast at different angles relative to its heading direction:

```
          Radar Front (0°)
               |
 Radar FL (-45°)   Radar FR (+45°)
           \   |   /
            \  |  /
  Radar L    [CAR]    Radar R
  (-90°)               (+90°)
```

- Each beam extends pixel-by-pixel up to **200 pixels** until it hits a wall (black border)
- Returns **normalized distances** (0.0 = touching wall, 1.0 = max range, no wall detected)
- Combined with **normalized speed** (-1.0 to 1.0), this produces a **6-value input vector** for the neural network

### Neural Network Architecture

The neural network is a **feedforward** network that evolves through NEAT:

```
INPUTS (6)                    OUTPUTS (4)
─────────────                 ────────────
Radar Left  ──┐          ┌── Gas       (> 0.5 → accelerate)
Radar F-Left ─┤          ├── Brake     (> 0.5 → decelerate)
Radar Front ──┼── [??] ──┼── Turn Left (> 0.5 → steer left)
Radar F-Right ┤          └── Turn Right(> 0.5 → steer right)
Radar Right ──┤
Speed ────────┘

[??] = Hidden layer topology evolves over generations
       (starts with 0 hidden nodes, grows as needed)
```

- **Activation function**: `tanh` (output range: -1 to +1, thresholded at 0.5 for decisions)
- **Initial topology**: Direct input-to-output connections, no hidden nodes
- **Topology evolution**: New nodes and connections are added/removed through mutation
- The network complexity grows organically to match the problem difficulty

### Fitness Function

The fitness function is the compass of evolution — it determines which behaviors are rewarded and which are punished. A carefully designed multi-component fitness drives meaningful learning:

| Component | Value | Purpose |
|---|---|---|
| **Checkpoint Passed** | +100 per checkpoint | Core reward for forward progress |
| **Speed Bonus** | +0 to +30 per checkpoint | Rewards fast checkpoint traversal (bonus = (150 - frames_used) * 0.2) |
| **Proximity Reward** | +0.3 * distance_closed per frame | Continuous gradient signal toward the next checkpoint |
| **Lap Completion** | +500 | Major reward for finishing an entire lap |
| **Crash Penalty** | -25 | Discourages wall collisions |
| **Timeout Penalty** | -50 | Kills agents stuck for ~5 seconds (300 frames) without progress |
| **Wrong Direction** | -100 (one-time) | Penalizes going backwards around the track |

**Why this design matters:**
- The **proximity reward** provides a continuous gradient signal, preventing the "sparse reward" problem where agents get no feedback between checkpoints
- The **speed bonus** pushes evolved agents beyond mere survival into efficient, fast driving
- The **timeout mechanic** prevents agents from endlessly circling without making progress
- **Checkpoint skipping detection** (look-ahead of 5) ensures fast agents aren't penalized for crossing multiple gates in a single frame

## Track System

### Color-Coded Design

Tracks are simple PNG images with color-coded elements that can be created in any image editor (Paint, Photoshop, GIMP, etc.):

```
┌──────────────────────────────────┐
│                                  │
│   ██████████████████████████     │  ██ = Black pixels (walls/borders)
│   █                        █    │
│   █   ════════════════     █    │  ── = Blue lines (checkpoints)
│   █   ║              ║     █    │
│   █   ║              ║     █    │  ▓▓ = Red line (start/finish line)
│   █   ════════▓▓══════     █    │
│   █                        █    │
│   ██████████████████████████     │
│                                  │
└──────────────────────────────────┘
```

| Color | HSV Range | Meaning |
|---|---|---|
| **Black** | H: 0-180, S: 0-255, V: 0-50 | Walls and track borders (collision boundaries) |
| **Blue** | H: 90-130, S: 50-255, V: 50-255 | Checkpoint gates (intermediate progress markers) |
| **Red** | H: 0-10 or 170-180, S: 70-255, V: 50-255 | Start/finish line |

### Track Processing Pipeline

When a track image is loaded, OpenCV processes it through the following pipeline:

1. **Color Space Conversion**: BGR → HSV for robust color segmentation
2. **Binary Mask Creation**: Three separate masks for borders, checkpoints, and finish line using `cv2.inRange()`
3. **Contour Extraction**: `cv2.findContours()` identifies checkpoint and finish line shapes
4. **Endpoint Detection**: For each contour, the **two most distant points** on its convex hull are found — these define the gate line endpoints
5. **Centroid Calculation**: Image moments (`cv2.moments()`) compute the center of mass for each gate

### Checkpoint Ordering

Checkpoints are automatically sorted into the correct traversal order:

1. **Nearest-neighbor greedy sort**: Starting from the finish line, each next checkpoint is the nearest unvisited one
2. **Winding direction test**: A signed area calculation (shoelace formula) determines if the order is clockwise or counter-clockwise
3. **Direction correction**: If clockwise, the order is reversed to ensure consistent counter-clockwise traversal
4. **Start angle**: The angle from the finish line to the first checkpoint determines the initial heading of all agents

### Crossing Detection

Checkpoint and finish line crossings use a **line segment intersection algorithm** (CCW test from computational geometry) — instead of simple proximity checks, the system detects if the car's movement vector between frames actually crosses the gate line segment, ensuring accurate detection even at high speeds.

## Real-Time Neural Network Visualization

A semi-transparent overlay in the bottom-right corner renders the currently best-performing alive agent's neural network in real-time:

```
┌─────── Neural Network ───────┐
│                               │
│  Radar L  ●━━━━━━━━━━━● Gas  │   ● Node colors:
│  Radar FL ●━━━━━╲━━━━━● Brake│     Blue (-1) → White (0) → Red (+1)
│  Radar F  ●━━━●━━╲━━━● Left │
│  Radar FR ●━━━╱━━━━━━━● Right│   ━ Connection colors:
│  Radar R  ●━━╱                │     Green = positive weight
│  Speed    ●                   │     Red = negative weight
│                               │     Thickness = weight magnitude
└───────────────────────────────┘
```

- **Nodes** change color based on their activation value using the tanh range (-1 to +1)
- **Connections** show weight polarity (green/red) and magnitude (line thickness 1-4 px)
- **Hidden nodes** appear in the center column as topology evolves
- Updates every frame, showing real-time decision-making

## Tech Stack

| Technology | Version | Role |
|---|---|---|
| **Python** | 3.11 | Core programming language |
| **NEAT-Python** | 0.92 | Neuroevolution algorithm — evolves neural network weights and topology |
| **Pygame** | 2.5.2 | Real-time simulation rendering, input handling, and UI |
| **OpenCV** (cv2) | 4.9.0 | Track image processing — color segmentation, contour detection, moment calculation |
| **NumPy** | 1.26.4 | Numerical operations for image mask processing |

## Project Structure

```
TrackLearnerAI/
├── src/
│   ├── main.py                  # Entry point — track selector, NEAT runner, simulation loop, fitness evaluation
│   ├── agent.py                 # Car agent — physics, 5-direction radar sensors, collision, movement
│   ├── track_env.py             # Track environment — OpenCV image processing, checkpoint/border detection
│   ├── network_visualizer.py    # Real-time neural network topology visualization overlay
│   └── manuel_test.py           # Manual keyboard-controlled driving mode for human testing
├── assets/
│   └── tracks/
│       ├── track1.png           # Race track 1
│       ├── track2.png           # Race track 2
│       ├── track3.png           # Race track 3
│       └── track4.png           # Race track 4
├── config-feedforward.txt       # NEAT algorithm configuration
├── .gitignore
└── README.md
```

### File Responsibilities

| File | Description |
|---|---|
| `main.py` | Orchestrates everything: shows the track selector UI, initializes Pygame and the NEAT population, runs the simulation loop where all agents drive simultaneously, and evaluates fitness for each genome |
| `agent.py` | Defines the car agent with physics (position, speed, acceleration, turning), 5 radar sensors that cast rays up to 200px, and rendering. Provides `get_data()` which returns the normalized 6-value input vector for the neural network |
| `track_env.py` | Loads a PNG track image and uses OpenCV to segment borders (black), checkpoints (blue), and finish line (red) via HSV color thresholds. Sorts checkpoints by path order, calculates start angle, and provides collision detection and line-crossing algorithms |
| `network_visualizer.py` | Renders a layered visualization of the best agent's neural network — input/hidden/output nodes with activation-based coloring and weight-colored connections |
| `manuel_test.py` | Standalone mode for human-controlled driving using Arrow Keys or WASD, useful for testing track layout and checkpoint placement |

## NEAT Configuration

Key parameters from `config-feedforward.txt`:

```
Population size:        60 genomes per generation
Max generations:        150
Fitness threshold:      100,000 (stops early if reached)

Network:
  Inputs:               6 (5 radars + speed)
  Outputs:              4 (gas, brake, left, right)
  Initial hidden:       0 (topology grows through evolution)
  Activation:           tanh
  Initial connections:  full (all inputs connected to all outputs)

Mutation rates:
  Node add/delete:      0.2 / 0.2
  Connection add/delete: 0.5 / 0.5
  Weight mutation:      0.8
  Bias mutation:        0.7

Speciation:
  Compatibility threshold: 3.0
  Max stagnation:       20 generations
  Elitism:              2 (top 2 survive unchanged)
  Survival threshold:   0.2 (top 20% reproduce)
```

## Installation

### Prerequisites

- Python 3.10+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/TrackLearnerAI.git
cd TrackLearnerAI
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install pygame==2.5.2 opencv-python==4.9.0.80 neat-python==0.92 numpy==1.26.4
```

## Usage

### AI Training Mode (Main)

```bash
cd src
python main.py
```

1. A track selector window appears — click a track to begin
2. Watch 60 agents evolve in real-time across generations
3. The HUD displays: generation number, best/average fitness, alive agents, and FPS
4. The neural network visualization shows the best agent's brain activity

### Manual Driving Mode

```bash
cd src
python manuel_test.py
```

- **Arrow Keys** or **WASD** to control the car
- Drive around `track1.png` manually to test the track

## Creating Custom Tracks

1. Open any image editor (Paint, Photoshop, GIMP, etc.)
2. Draw the track using three colors:
   - **Black** for walls and borders
   - **Blue** lines across the track for checkpoints (draw them perpendicular to the driving direction)
   - **Red** line for the start/finish line
3. Save as `trackN.png` in `assets/tracks/`
4. The system will automatically detect all elements, sort checkpoints, and calculate the start angle

**Tips for good tracks:**
- Make borders clearly thick (at least 3-5 pixels)
- Space checkpoints reasonably — too close together is fine (the system handles it), but too far apart may slow down learning
- Ensure the red start/finish line is placed at a logical starting position
- Leave enough white/light space between walls for the car to navigate
