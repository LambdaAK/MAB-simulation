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
- **Improved Neural Network Contextual Bandit**: Enhanced neural network with better training strategies

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

## Theory and Background

### What is a Contextual Multi-Armed Bandit?

A **Contextual Multi-Armed Bandit (CMAB)** is an extension of the traditional Multi-Armed Bandit problem where the reward distributions of arms depend on a **context vector** that is revealed before each decision.

#### Key Differences from Traditional MAB:

1. **Context Dependence**: Arm rewards are not fixed but depend on the current context
2. **Dynamic Environment**: The optimal arm can change based on the context
3. **Learning Objective**: Learn the mapping from context to expected reward for each arm
4. **Real-world Applications**: Online advertising, recommendation systems, clinical trials, etc.

#### Mathematical Formulation:

- **Context**: $x_t \in \mathbb{R}^d$ (d-dimensional context vector at time t)
- **Arms**: $K$ arms, each with parameterized reward distribution
- **Reward Model**: $r_{i,t} = f_i(x_t) + \epsilon_t$ where $f_i$ is arm-specific function
- **Goal**: Maximize $\sum_{t=1}^T r_{a_t,t}$ where $a_t$ is the selected arm

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
  - Action selection: $a_t = \arg\max_i Q_i$ with probability $1-\epsilon$, random with probability $\epsilon$
  - Value update: $Q_i = Q_i + \alpha(r_t - Q_i)$ where $\alpha$ is learning rate
- **Pros**: Simple, effective exploration/exploitation balance
- **Cons**: Fixed exploration rate, may not be optimal
- **Use Case**: Good baseline for simple problems

#### Upper Confidence Bound (UCB) Policy
- **Description**: Balances estimated reward with uncertainty using confidence bounds
- **Mathematical Formulation**: 
  - UCB value: $\text{UCB}_i = Q_i + c \sqrt{\frac{\ln t}{N_i}}$
  - Action selection: $a_t = \arg\max_i \text{UCB}_i$
  - Where $c=2.0$ (theoretical constant), $t$ is time step, $N_i$ is number of pulls for arm $i$
- **Pros**: No parameters to tune, theoretical guarantees
- **Cons**: Assumes bounded rewards, may be conservative
- **Use Case**: When theoretical guarantees are important

#### Thompson Sampling
- **Description**: Bayesian approach that samples from posterior distributions
- **Mathematical Formulation**:
  - For Beta posterior: $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$
  - Action selection: $a_t = \arg\max_i \theta_i$
  - Update: $\alpha_i \leftarrow \alpha_i + r_t$, $\beta_i \leftarrow \beta_i + (1 - r_t)$ for binary rewards
- **Pros**: Natural exploration/exploitation balance, good empirical performance
- **Cons**: Computationally more expensive, requires conjugate priors
- **Use Case**: When good empirical performance is desired

### Contextual Multi-Armed Bandit Algorithms

#### Random Contextual Policy
- **Description**: Selects arms randomly regardless of context
- **Mathematical Formulation**: $P(a_t = i) = \frac{1}{K}$ for all arms $i$
- **Pros**: Simple baseline, ensures exploration
- **Cons**: Ignores context information completely
- **Use Case**: Baseline comparison for contextual bandits

#### Epsilon-Greedy Linear
- **Description**: Linear models with epsilon-greedy exploration strategy
- **Mathematical Formulation**:
  - Reward model: $r = \text{bias} + \mathbf{x}^T \mathbf{w} + \epsilon$
  - Action selection: $a_t = \arg\max_i (\text{bias}_i + \mathbf{x}^T \mathbf{w}_i)$ with probability $1-\epsilon$
  - Update: Gradient descent on squared error loss
- **Pros**: Simple, handles context, easy to implement
- **Cons**: Fixed exploration rate, assumes linear relationship
- **Use Case**: When context-reward relationship is approximately linear

#### Linear Thompson Sampling
- **Description**: Models reward as linear function of context with Bayesian uncertainty
- **Mathematical Formulation**:
  - Reward model: $r = \text{bias} + \mathbf{x}^T \mathbf{w} + \epsilon$
  - Posterior update: Normal-Inverse-Gamma conjugate prior
  - Action selection: Sample from posterior, select argmax
  - Bayesian update: $\boldsymbol{\mu}_{\text{new}} = \boldsymbol{\mu}_{\text{old}} + \frac{\text{error}}{\text{denominator}} \boldsymbol{\Sigma} \mathbf{x}_{\text{with bias}}$
- **Key Features**:
  - **Bias Term**: Includes intercept term for better modeling
  - **Uncertainty Quantification**: Tracks uncertainty in parameter estimates
  - **Adaptive Exploration**: Uncertainty multipliers adjust based on prediction errors
  - **Numerical Stability**: Regularization and symmetry enforcement
- **Pros**: Handles context, theoretical guarantees, uncertainty quantification
- **Cons**: Assumes linear relationship, may not capture complex patterns
- **Use Case**: When context-reward relationship is approximately linear

#### Neural Network Contextual Bandit
- **Description**: Deep learning approach using neural networks with experience replay
- **Architecture**:
  - Input: Context vector ($d$-dimensional)
  - Hidden layers: $[64, 32]$ with ReLU activation and Dropout(0.1)
  - Output: Q-values for each arm ($K$-dimensional)
- **Key Features**:
  - **Experience Replay**: Stores $(\text{context}, \text{action}, \text{reward})$ tuples in buffer
  - **Batch Learning**: Updates network using mini-batches from replay buffer
  - **Target Network**: Stable learning with periodic target network updates
  - **Exploration**: Epsilon-greedy exploration strategy
  - **Uncertainty Estimation**: Bayesian posterior over linear parameters for uncertainty
- **Mathematical Formulation**:
  - Q-function: $Q(\mathbf{x}, a) = f_\theta(\mathbf{x})[a]$ where $f_\theta$ is neural network
  - Loss: $\mathcal{L} = \text{MSE}(Q(\mathbf{x}, a), r)$
  - Update: $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$ using Adam optimizer
- **Training Process**:
  - **Experience Collection**: Store experiences in replay buffer
  - **Batch Updates**: Sample batches and update network every 10 steps
  - **Target Network**: Update target network every 100 steps
  - **Posterior Updates**: Update Bayesian posteriors for uncertainty estimation
- **Pros**: Can learn complex non-linear patterns, handles high-dimensional contexts
- **Cons**: More hyperparameters, requires more data, computationally expensive
- **Use Case**: When context-reward relationship is complex or non-linear

#### Improved Neural Network Contextual Bandit
- **Description**: Enhanced neural network with better training strategies
- **Improvements over Basic Neural Network**:
  - **Larger Architecture**: $[128, 64]$ vs $[64, 32]$ (more capacity)
  - **Better Optimization**: Lower learning rate $(0.0005)$, weight decay $(10^{-5})$, learning rate scheduler
  - **Batch Normalization**: Stabilizes training and accelerates convergence
  - **Gradient Clipping**: Prevents exploding gradients $(\max\_\text{norm}=1.0)$
  - **More Frequent Updates**: Every 5 steps vs 10 steps
  - **Larger Replay Buffer**: 2000 vs 1000 experiences
  - **Better Initialization**: Xavier/Glorot initialization for weights
  - **Higher Dropout**: 0.2 vs 0.1 for better regularization
- **Mathematical Formulation**: Same as basic neural network but with enhanced training
- **Pros**: Faster learning, better stability, improved performance
- **Cons**: Even more computationally expensive
- **Use Case**: When you need the best possible neural network performance

## Environment Types

### Multi-Armed Bandit Environment
- **Arms**: Each arm has a fixed reward distribution
- **Rewards**: Sampled from arm's distribution when pulled
- **Goal**: Maximize cumulative reward over time

### Contextual Multi-Armed Bandit Environment
- **Arms**: Each arm has a parameterized reward distribution
- **Context**: Vector that parameterizes arm rewards
- **Rewards**: $r = f_{\text{arm}}(\mathbf{x}) + \epsilon$ where $f_{\text{arm}}$ is arm-specific function
- **Goal**: Learn context-reward mapping and maximize cumulative reward

#### Parameterized Distributions
- **LinearNormalDistribution**: $r \sim \mathcal{N}(\mathbf{x}^T \mathbf{w} + \text{bias}, \sigma)$
- **LinearBernoulliDistribution**: $P(r=1) = \sigma(\mathbf{x}^T \mathbf{w} + \text{bias})$ where $\sigma$ is sigmoid
- **CustomDistribution**: Arbitrary function mapping context to distribution

## Experiment Framework

### Multi-Armed Bandit Experiments
- Policy comparison with regret analysis
- Individual policy analysis
- Interactive play mode

### Contextual Bandit Experiments
- **Policy Comparison**: Compare all algorithms (Random, Epsilon-Greedy, Linear TS, Neural Network)
- **Thompson Sampling Analysis**: Detailed analysis of Linear Thompson Sampling behavior
- **Neural Network Analysis**: Specific analysis of neural network learning dynamics
- **Train and Watch**: Interactive mode to observe model decision-making
- **Long-term Training Comparison**: Extended experiments to compare Linear TS vs Neural Network over time

## Visualization

The framework provides comprehensive visualizations:
- **Performance Metrics**: Cumulative and average rewards over time
- **Regret Analysis**: Regret curves for all algorithms
- **Action Distribution**: How often each arm is selected
- **Context Visualization**: Heatmap of recent contexts
- **Policy-specific Metrics**: Prediction errors, uncertainty, replay buffer usage
- **Learning Analysis**: Learning rates, training phases, neural network specific plots

## Key Insights and Findings

### Neural Network Training Dynamics
- **Parameter Complexity**: Neural networks have $\sim 565\times$ more parameters than linear models
- **Learning Speed**: Neural networks learn more slowly but can capture complex patterns
- **Data Requirements**: Need significantly more training data to reach optimal performance
- **Environment Mismatch**: Linear models perform well when environment is approximately linear

### Algorithm Performance Characteristics
- **Linear Thompson Sampling**: Best performance for linear environments, fast convergence
- **Neural Networks**: Slower initial learning but potential for complex pattern recognition
- **Epsilon-Greedy**: Good baseline, simple and effective
- **Random**: Baseline for comparison

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

### Long-term Training Comparison
```python
from run_contextual_experiments import run_long_training_comparison
results = run_long_training_comparison(n_iterations=5000)
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

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms
- Agrawal, S., & Goyal, N. (2013). Thompson Sampling for Contextual Bandits with Linear Payoffs
- Riquelme, C., et al. (2018). Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling

