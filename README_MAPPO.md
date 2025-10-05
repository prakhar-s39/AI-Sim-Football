# AI Sim Football - MAPPO Implementation

This document describes the Multi-Agent Proximal Policy Optimization (MAPPO) implementation for the AI Sim Football project.

## 🚀 Features

- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Parallel Simulations**: Run multiple training sessions simultaneously
- **Comprehensive Logging**: Detailed game state logging for visualization
- **MAPPO Algorithm**: Multi-agent reinforcement learning with centralized training
- **Visualization Tools**: Generate trajectory plots and training curves
- **Model Persistence**: Save and load trained models

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Demo Mode (Simple AIs)
```bash
python main.py --mode demo
```

### Training Mode (MAPPO)
```bash
# Basic training
python main.py --mode train --episodes 1000 --simulations 4

# Advanced training with custom parameters
python main.py --mode train --episodes 2000 --simulations 8 --log-dir custom_logs
```

### Evaluation Mode
```bash
# Evaluate trained agents
python main.py --mode eval --model1 logs/sim_0_agent1.pt --model2 logs/sim_0_agent2.pt --games 10 --render

# Evaluate without rendering (faster)
python main.py --mode eval --games 50
```

### Visualization Mode
```bash
# Create trajectory plots from logged games
python main.py --mode visualize --log-dir logs --game-id 0
```

## 🏗️ Architecture

### MAPPO Agent
- **Actor-Critic Network**: Shared layers with separate actor and critic heads
- **Experience Buffer**: Stores (state, action, reward, next_state, done, log_prob, value)
- **PPO Updates**: Clipped surrogate objective with value function loss
- **GPU Support**: Automatic device detection and tensor operations

### Parallel Simulation Manager
- **Threading**: Multiple simulations run in parallel
- **Logging**: Each simulation has its own logger
- **Results Collection**: Centralized result aggregation

### Game Logger
- **Step-by-step Logging**: Records all game states and actions
- **JSON Format**: Human-readable log files
- **Visualization Ready**: Easy conversion to plots and animations

## 📊 Training Process

1. **Initialization**: Create MAPPO agents with random policies
2. **Experience Collection**: Agents play games and collect experiences
3. **Policy Updates**: PPO algorithm updates both actor and critic networks
4. **Parallel Training**: Multiple simulations run simultaneously
5. **Logging**: All game data is logged for analysis
6. **Model Saving**: Trained models are saved automatically

## 📈 Monitoring Training

The system provides comprehensive monitoring:

- **Real-time Progress**: Episode completion and win rates
- **Training Curves**: Reward curves, win rates, episode lengths
- **Model Persistence**: Automatic saving of trained models
- **Log Analysis**: Detailed game logs for post-analysis

## 🎯 Key Components

### Neural Network Architecture
```python
# Shared layers
shared = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU()
)

# Actor head (policy)
actor = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, action_dim)
)

# Critic head (value function)
critic = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)
```

### Observation Space
- **12 dimensions**: [p1x, p1y, p1vx, p1vy, p2x, p2y, p2vx, p2vy, ballx, bally, ballvx, ballvy]
- **Player positions**: x, y coordinates
- **Player velocities**: vx, vy velocities
- **Ball state**: position and velocity

### Action Space
- **5 discrete actions**: 0=up, 1=down, 2=left, 3=right, 4=no-op

### Reward Structure
- **Sparse rewards**: +1 for scoring, -1 for being scored on
- **Terminal rewards**: Only at game end
- **No intermediate rewards**: Encourages goal-oriented behavior

## 🔧 Configuration

### Training Parameters
- **Learning Rate**: 3e-4 (Adam optimizer)
- **Gamma**: 0.99 (discount factor)
- **Epsilon Clip**: 0.2 (PPO clipping)
- **K Epochs**: 4 (policy update iterations)
- **Batch Size**: 32 (experience sampling)

### Simulation Parameters
- **Max Steps**: 1000 per episode
- **Parallel Simulations**: 4 (configurable)
- **Episode Limit**: 1000 (configurable)

## 📁 File Structure

```
logs/
├── sim_0/                    # Simulation 0 logs
│   ├── game_0_*.json        # Individual game logs
│   └── training_summary_*.json
├── sim_1/                    # Simulation 1 logs
├── ...
├── sim_0_agent1.pt         # Trained model files
├── sim_0_agent2.pt
└── training_curves.png      # Training visualization
```

## 🎨 Visualization

### Training Curves
- Episode rewards over time
- Moving average rewards
- Episode lengths
- Win rates over time

### Game Trajectories
- Player movement paths
- Ball trajectory
- Start/end positions
- Field boundaries

## 🚀 Performance

### GPU Acceleration
- Automatic CUDA detection
- Tensor operations on GPU
- Parallel neural network training

### Parallel Processing
- Multiple simulations simultaneously
- Thread-based parallelism
- Efficient resource utilization

### Memory Management
- Experience buffer with capacity limits
- Automatic garbage collection
- Efficient tensor operations

## 🔍 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or number of parallel simulations
2. **Slow Training**: Ensure GPU is being used (check device output)
3. **No Learning**: Check reward structure and learning parameters

### Performance Tips
- Use GPU for faster training
- Increase parallel simulations for more data
- Monitor training curves for learning progress
- Save models regularly for evaluation

## 📚 Example Workflows

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python main.py --mode demo

# Train agents
python main.py --mode train --episodes 500

# Evaluate results
python main.py --mode eval --render
```

### Advanced Training
```bash
# Long training run
python main.py --mode train --episodes 2000 --simulations 8

# Custom log directory
python main.py --mode train --episodes 1000 --log-dir experiments/exp1

# Evaluate specific models
python main.py --mode eval --model1 experiments/exp1/sim_0_agent1.pt --model2 experiments/exp1/sim_0_agent2.pt
```

### Visualization
```bash
# Create trajectory plots
python main.py --mode visualize --log-dir logs --game-id 0

# Multiple game visualizations
for i in {0..9}; do
    python main.py --mode visualize --log-dir logs --game-id $i
done
```

This implementation provides a complete MAPPO training system with GPU acceleration, parallel simulations, and comprehensive logging for the AI Sim Football project.
