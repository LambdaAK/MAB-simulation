# Multi-Armed Bandit Reinforcement Learning Project

A simple implementation of Multi-Armed Bandit (MAB) environments for reinforcement learning experimentation.

## Features

- **Flexible Arm Design**: Each arm can have different reward distributions (normal, bernoulli, uniform)
- **Environment Class**: Complete MAB environment with pull mechanics and statistics
- **Interactive Play**: Play against the bandit manually
- **Algorithm Demo**: See epsilon-greedy in action
- **Reproducible**: Seed-based randomization for consistent experiments

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test the environment:
```bash
python mab_environment.py
```

3. Play with the bandit interactively:
```bash
python play_with_bandit.py
```

## Usage Examples

### Basic Environment Usage

```python
from mab_environment import MultiArmedBandit, Arm

# Create arms with different reward distributions
arms = [
    Arm("Poor", mean=0.2, std=0.3),
    Arm("Medium", mean=0.5, std=0.3),
    Arm("Good", mean=0.8, std=0.3)
]

# Create the bandit
bandit = MultiArmedBandit(arms, seed=42)

# Pull an arm
reward, valid = bandit.pull_arm(0)
print(f"Reward: {reward}")

# Get bandit information
info = bandit.get_info()
print(f"Optimal arm: {info['optimal_arm']}")
```

### Random Bandit Creation

```python
# Create a random 5-armed bandit
bandit = MultiArmedBandit.create_random_bandit(
    n_arms=5, 
    mean_range=(0, 1), 
    std=0.3, 
    seed=123
)
```

## Project Structure

- `mab_environment.py`: Core MAB environment and Arm classes
- `play_with_bandit.py`: Interactive script to play with the bandit
- `requirements.txt`: Python dependencies
- `components.md`: Project component planning

## Next Steps

This is the foundation for your RL project. Next components to implement:

1. **Policy Interface**: Abstract base class for algorithms
2. **Algorithm Implementations**: 
   - Random selection
   - Epsilon-greedy
   - Upper Confidence Bound (UCB)
   - Thompson Sampling
3. **Visualization**: Real-time plotting and comparison tools
4. **Evaluation Framework**: Regret analysis and performance metrics

## Playing with the Environment

The interactive script (`play_with_bandit.py`) offers two modes:

1. **Interactive Play**: You manually choose which arm to pull
2. **Algorithm Demo**: Watch an epsilon-greedy algorithm play automatically

This lets you:
- Experience the exploration vs exploitation trade-off
- See how algorithms learn over time
- Understand regret and optimal performance
- Experiment with different bandit configurations 