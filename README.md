# Multi-Armed Bandit and Contextual Bandit Implementation

This project implements various Multi-Armed Bandit (MAB) and Contextual Multi-Armed Bandit (CMAB) algorithms for reinforcement learning.

## Features

### Multi-Armed Bandit (MAB)
- **Random Policy**: Baseline random selection
- **Epsilon-Greedy Policy**: Probabilistic exploration/exploitation
- **Epsilon-Greedy with Decay**: Adaptive exploration rate
- **Upper Confidence Bound (UCB) Policy**: Balances estimated reward with uncertainty
- **Softmax (Boltzmann Exploration) Policy**: Probabilistic selection based on estimated values
- **Thompson Sampling (Beta)**: Bayesian approach using Beta distributions for Bernoulli-like rewards
- **Thompson Sampling (Normal)**: Bayesian approach using Normal-Inverse-Gamma conjugate priors for continuous rewards

### Contextual Multi-Armed Bandit (CMAB)
- **Random Policy**: Baseline random selection
- **Epsilon-Greedy Linear**: Linear models with epsilon-greedy exploration
- **Linear Thompson Sampling**: Bayesian linear regression with Thompson Sampling
- **Neural Network Contextual Bandit**: Deep learning approach with experience replay

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Multi-Armed Bandit Experiments

```bash
python play_with_bandit.py
```

### Contextual Bandit Experiments

```bash
python run_contextual_experiments.py
```

Or with specific experiment types:

```bash
python run_contextual_experiments.py --experiment comparison --iterations 2000
python run_contextual_experiments.py --experiment thompson --iterations 1000
```

### Interactive Contextual Bandit

```bash
python play_with_contextual_bandit.py
```

## Algorithm Details

### Multi-Armed Bandit Algorithms

#### Random Policy
- **Description**: Selects arms randomly with equal probability
- **Pros**: Simple, unbiased exploration
- **Cons**: No learning, poor performance
- **Use Case**: Baseline comparison

#### Epsilon-Greedy Policy
- **Description**: With probability ε, selects random arm; otherwise selects best estimated arm
- **Mathematical Formulation**: 
  - Action selection: `a_t = argmax_i Q_i` with probability 1-ε, random with probability ε
  - Value update: `Q_i = Q_i + α(r_t - Q_i)` where α is learning rate
- **Pros**: Simple, effective exploration/exploitation balance
- **Cons**: Fixed exploration rate, may not be optimal
- **Use Case**: Good baseline for simple problems

#### Upper Confidence Bound (UCB) Policy
- **Description**: Balances estimated reward with uncertainty using confidence bounds
- **Mathematical Formulation**: 
  - UCB value: `UCB_i = Q_i + c * sqrt(ln(t) / N_i)`
  - Action selection: `a_t = argmax_i UCB_i`
  - Where c=2.0 (theoretical constant), t is time step, N_i is number of pulls for arm i
- **Pros**: No parameters to tune, theoretical guarantees
- **Cons**: Assumes bounded rewards, may be conservative
- **Use Case**: When theoretical guarantees are important

#### Thompson Sampling
- **Description**: Bayesian approach that samples from posterior distributions
- **Mathematical Formulation**:
  - For Beta posterior: `θ_i ~ Beta(α_i, β_i)`
  - Action selection: `a_t = argmax_i θ_i`
  - Update: `α_i += r_t`, `β_i += (1 - r_t)` for binary rewards
- **Pros**: Natural exploration/exploitation balance, good empirical performance
- **Cons**: Computationally more expensive, requires conjugate priors
- **Use Case**: When good empirical performance is desired

### Contextual Multi-Armed Bandit Algorithms

#### Linear Thompson Sampling
- **Description**: Models reward as linear function of context with Bayesian uncertainty
- **Mathematical Formulation**:
  - Reward model: `r = bias + context @ weights + noise`
  - Posterior update: Normal-Inverse-Gamma conjugate prior
  - Action selection: Sample from posterior, select argmax
- **Pros**: Handles context, theoretical guarantees, uncertainty quantification
- **Cons**: Assumes linear relationship, may not capture complex patterns
- **Use Case**: When context-reward relationship is approximately linear

#### Neural Network Contextual Bandit
- **Description**: Deep learning approach using neural networks with experience replay
- **Architecture**:
  - Input: Context vector
  - Hidden layers: Configurable (default: [64, 32] with ReLU and Dropout)
  - Output: Q-values for each arm
- **Key Features**:
  - **Experience Replay**: Stores (context, action, reward) tuples in buffer
  - **Batch Learning**: Updates network using mini-batches from replay buffer
  - **Target Network**: Stable learning with periodic target network updates
  - **Exploration**: Epsilon-greedy exploration strategy
  - **Uncertainty Estimation**: Bayesian posterior over linear parameters for uncertainty
- **Mathematical Formulation**:
  - Q-function: `Q(context, arm) = f_θ(context)[arm]` where f_θ is neural network
  - Loss: `L = MSE(Q(context, action), reward)`
  - Update: `θ = θ - α * ∇_θ L` using Adam optimizer
- **Pros**: Can learn complex non-linear patterns, handles high-dimensional contexts
- **Cons**: More hyperparameters, requires more data, computationally expensive
- **Use Case**: When context-reward relationship is complex or non-linear

## Environment Types

### Multi-Armed Bandit Environment
- **Arms**: Each arm has a fixed reward distribution
- **Rewards**: Sampled from arm's distribution when pulled
- **Goal**: Maximize cumulative reward over time

### Contextual Multi-Armed Bandit Environment
- **Arms**: Each arm has a parameterized reward distribution
- **Context**: Vector that parameterizes arm rewards
- **Rewards**: `r = f_arm(context) + noise` where f_arm is arm-specific function
- **Goal**: Learn context-reward mapping and maximize cumulative reward

#### Parameterized Distributions
- **LinearNormalDistribution**: `r ~ N(context @ weights + bias, std)`
- **LinearBernoulliDistribution**: `P(r=1) = sigmoid(context @ weights + bias)`
- **CustomDistribution**: Arbitrary function mapping context to distribution

## Experiment Framework

### Multi-Armed Bandit Experiments
- Policy comparison with regret analysis
- Individual policy analysis
- Interactive play mode

### Contextual Bandit Experiments
- Policy comparison (Random, Epsilon-Greedy, Linear TS, Neural Network)
- Thompson Sampling detailed analysis
- Neural Network specific analysis
- Train and watch mode for interactive decision observation

## Visualization

The framework provides comprehensive visualizations:
- Cumulative and average rewards over time
- Regret analysis
- Action distribution analysis
- Context visualization
- Policy-specific metrics (prediction errors, uncertainty, etc.)
- Neural network specific plots (learning curves, replay buffer usage)

## Example Usage

### Running a Policy Comparison
```python
from run_contextual_experiments import run_policy_comparison_experiment
results = run_policy_comparison_experiment(n_iterations=2000)
```

### Training and Watching a Model
```python
from run_contextual_experiments import train_and_watch_experiment
train_and_watch_experiment(n_train=2000, n_watch=20)
```

### Creating Custom Environments
```python
from contextual_mab_environment import ContextualMultiArmedBandit, LinearNormalDistribution

# Create custom bandit
arms = []
for i in range(5):
    weights = np.random.normal(0, 1, 5)
    distribution = LinearNormalDistribution(context_dim=5, weights=weights, std=0.5)
    arms.append(ContextualArm(f"Arm_{i}", distribution))

bandit = ContextualMultiArmedBandit(arms, seed=42)
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Colorama
- SciPy
- PyTorch (for Neural Network Contextual Bandit)

## Project Structure

```
├── bandit.py                    # Multi-armed bandit environment
├── policies.py                  # MAB policies
├── experiment.py                # MAB experiment framework
├── play_with_bandit.py         # MAB interactive script
├── contextual_mab_environment.py # CMAB environment
├── contextual_policies.py       # CMAB policies
├── run_contextual_experiments.py # CMAB experiment framework
├── play_with_contextual_bandit.py # CMAB interactive script
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## Contributing

Feel free to add new policies, environments, or experiment types. The modular design makes it easy to extend the framework.

