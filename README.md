# TrackLearnerAI v1.0

**AI agents learning to race using NEAT (NeuroEvolution of Augmenting Topologies)**

A 2D top-down racing simulation where neural network-based cars learn to navigate a track through evolutionary algorithms. Watch as populations of AI drivers gradually improve their lap times through generations of training.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Training

```bash
# Default: 50 generations, 30 population, 30 FPS
python3 main.py

# Custom configuration
python3 main.py --generations 100 --population 50 --fps 60 --steps 5000
```

### CLI Options

```
--generations N    Number of training generations (default: 50)
--population N     Population size (default: 30)
--steps N          Max simulation steps per generation (default: 10000)
--fps N            Target frames per second (default: 30)
--track PATH       Path to track PNG file (default: assets/tracks/track1.png)
```

### Controls During Training

- **SPACE** - Toggle sensor visualization (debug mode)
- **P** - Pause/Resume simulation
- **R** - Reset in best lap mode
- **↑/↓** - Speed up/slow down simulation
- **Q** or close window - Quit

## 🎮 How It Works

### Training Process

1. **Population**: 30 cars (controlled by NEAT-evolved neural networks)
2. **Each Generation**: All cars drive simultaneously for up to 10,000 steps
3. **Fitness Evaluation**:
   - Progress on track (distance traveled)
   - Average speed maintained
   - Penalty for leaving the track
4. **Evolution**: NEAT selects best performers, creates offspring through crossover/mutation
5. **Best Lap Mode**: After training, replays the best genome

### Vehicle Physics

- **Throttle**: -1 (brake) → 0 (neutral) → +1 (accelerate)
- **Steering**: -1 (full left) → 0 (straight) → +1 (full right)
- **Max Speed**: 300 pixels/frame
- **Acceleration**: 10 pixels/frame²
- **Friction**: Natural velocity decay when coasting

### Sensors

5 raycasting rays detect obstacles:
```
        -45°    -22.5°    0°    +22.5°    +45°
          \       |       |       |       /
           \      |       |       |      /
            ●-----●-------●-------●-----●  (car)
           /      |       |       |      \
          /       |       |       |       \
```

Each ray outputs normalized distance (0..1) to track boundary.

### AI Learning

- **Input**: 5 sensor values
- **Output**: 2 values (steering, throttle)
- **Network**: Feedforward neural network evolved by NEAT
- **Generations**: Population improves over time through natural selection

## 📊 File Structure

```
TrackLearnerAI/
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── config/
│   └── config-feedforward.txt    # NEAT configuration
├── src/
│   ├── __init__.py
│   ├── track.py            # Track loading & collision
│   ├── car.py              # Vehicle physics & sensors
│   ├── ai_manager.py       # NEAT population management
│   ├── core.py             # Main simulation loop
│   ├── render.py           # Visualization & HUD
│   └── utils.py            # Math & raycasting utilities
├── assets/
│   └── tracks/
│       └── track1.png      # Racing track
├── data/
│   ├── best_brains/        # Saved models (output)
│   │   └── best_genome.pkl
│   └── logs/
│       └── training_summary.json
└── Documents/
    └── TrackLearnerAI_SRS_v1.txt  # Design documentation
```

## 🧠 Key Modules

### `track.py`
- Loads PNG track images
- Generates collision masks
- Provides raycasting interface

### `car.py`
Vehicle simulation with:
- 2D kinematic physics
- 5-angle raycasting sensor system
- Collision detection
- State tracking (position, velocity, alive/dead)

### `ai_manager.py`
NEAT population management:
- Genome creation and evaluation
- Fitness calculation
- Best model persistence
- Neural network generation

### `core.py`
Main simulation engine:
- Training loop coordination
- Generation advancement
- State machine (training/best lap modes)
- Metrics tracking

### `render.py`
Real-time visualization:
- Track and vehicle rendering
- HUD with training metrics
- Debug sensor visualization
- Event handling

### `utils.py`
Utility functions:
- Vector math
- Raycasting algorithm
- Monotonic progress tracking (50 checkpoints)
- Fitness calculation

## 📈 Fitness Function

```
fitness = progress + (avg_speed × 0.5) - 50 (if crashed)
```

Where:
- **progress**: Checkpoint progress (0-50)
- **avg_speed**: Average speed maintained
- **Collision penalty**: -50 points for leaving track

## 🔧 Technical Details

### NEAT Configuration

- **Inputs**: 5 (sensors)
- **Outputs**: 2 (steering, throttle)
- **Activation**: tanh
- **Population Size**: 30
- **Connection Probability**: 0.5
- **Mutation Rates**: Configurable per genome attribute

### Progress Tracking

Uses checkpoint-based system to prevent exploitation:
- Track divided into 50 checkpoints
- Progress is monotonically increasing
- Prevents infinite loops ("doughnut driving")

### Collision Detection

- Pixel-based mask collision (pygame.mask)
- Efficient raycasting for sensors
- Automatic dead-car removal

## 📝 Training Output

After training completes:

1. **Console**: Shows generation progress and final metrics
2. **best_genome.pkl**: Serialized best neural network
3. **Visual**: Best lap mode displays winning genome's drive

## 🎯 Expected Results

- **Generation 1-10**: Random movement, many crashes
- **Generation 10-30**: Learning basic steering, following track
- **Generation 30-50**: Smooth drives, optimizing speed
- **Generation 50+**: Near-optimal racing performance

Time to complete training varies based on:
- Population size
- Steps per generation
- Computer performance
- FPS setting

## 🚧 Future Enhancements (v1.1+)

- [ ] Track Designer: Draw custom tracks
- [ ] Save/Load System: 5 track slots
- [ ] Leaderboard: Compare lap times
- [ ] Replay System: Save/load best runs
- [ ] Statistics: Training graphs and metrics
- [ ] Multi-track Training: Evolve generalists

## 📚 References

- [NEAT Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf)
- [neat-python Documentation](https://neat-python.readthedocs.io/)
- [Pygame Documentation](https://www.pygame.org/wiki/)

## 🐛 Troubleshooting

### "No video mode has been set"
- Ensure pygame initialized (pip install requirements.txt)
- Run with `python3 main.py` (not directly)

### Track not loading
- Check `assets/tracks/track1.png` exists
- Ensure correct file path in arguments

### Slow performance
- Reduce FPS: `--fps 20`
- Reduce steps: `--steps 5000`
- Reduce population: `--population 20`

### NEAT config errors
- Verify `config/config-feedforward.txt` exists
- Check config syntax matches neat-python requirements

## 📄 License

Educational project - TrackLearnerAI v1.0

## 🙋 Authors

- Kaan Güner (Project Lead)
- Claude/Anthropic (Implementation)

---

**Created**: 2026-03-07
**Status**: v1.0 Complete - Ready for Training and Best Lap Playback
