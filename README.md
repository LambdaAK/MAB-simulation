# Multi-Armed Bandit and Contextual Bandit Implementation

This project implements various Multi-Armed Bandit (MAB) and Contextual Multi-Armed Bandit (CMAB) algorithms for reinforcement learning.

## Introduction

### What is Multi-Armed Bandit?

A **Multi-Armed Bandit (MAB)** is a sequential decision-making problem where an agent must choose between K actions (arms) at each time step to maximize cumulative reward over time. The agent faces the fundamental trade-off between **exploration** (trying different arms to learn their rewards) and **exploitation** (choosing the arm believed to yield the highest reward).

#### Mathematical Formulation

**Problem Setup:**
- **Arms**: K arms indexed by i ∈ {1, 2, ..., K}
- **Time Horizon**: T time steps
- **Reward Distribution**: Each arm i has an unknown reward distribution with mean μᵢ
- **Action**: At time t, the agent selects arm aₜ ∈ {1, 2, ..., K}
- **Reward**: The agent receives reward rₜ ~ D_{aₜ} where D_{aₜ} is the reward distribution of the selected arm

**Objective:**
The goal is to maximize the expected cumulative reward:

```
max E[Σ(t=1 to T) rₜ]
 π
```

where π is the policy that maps history to actions.

**Regret:**
The performance is measured in terms of regret, which is the difference between the optimal cumulative reward and the actual cumulative reward:

```
R(T) = T μ* - E[Σ(t=1 to T) rₜ]
```

where μ* = max μᵢ is the expected reward of the optimal arm.
           i

**Exploration vs Exploitation Trade-off:**
The agent must balance:
- **Exploration**: Gathering information about arm rewards
- **Exploitation**: Using current knowledge to maximize immediate reward

### What is Contextual Multi-Armed Bandit?

A **Contextual Multi-Armed Bandit (CMAB)** extends the traditional MAB problem by introducing context-dependent reward distributions. The reward of each arm depends on a context vector that is revealed before each decision.

#### Mathematical Formulation

**Problem Setup:**
- **Context Space**: X ⊆ ℝᵈ (d-dimensional context space)
- **Context**: At time t, context xₜ ∈ X is revealed
- **Arms**: K arms indexed by i ∈ {1, 2, ..., K}
- **Reward Function**: Each arm i has a reward function fᵢ: X → ℝ
- **Action**: At time t, the agent selects arm aₜ ∈ {1, 2, ..., K} based on context xₜ
- **Reward**: The agent receives reward rₜ = f_{aₜ}(xₜ) + εₜ where εₜ is noise

**Objective:**
Maximize the expected cumulative reward:

```
max E[Σ(t=1 to T) rₜ]
 π
```

where π: X → {1, 2, ..., K} is a context-dependent policy.

**Regret:**
The contextual regret is:

```
R(T) = Σ(t=1 to T) max fᵢ(xₜ) - E[Σ(t=1 to T) rₜ]
                     i
```

**Key Differences from Traditional MAB:**
1. **Context Dependence**: Arm rewards depend on the current context
2. **Dynamic Optimal Arm**: The optimal arm can change based on context
3. **Learning Objective**: Learn the mapping from context to expected reward for each arm
4. **Real-world Applications**: Online advertising, recommendation systems, clinical trials

## Algorithms

### Random Policy

**Description**: Selects arms randomly with equal probability regardless of context.

**Mathematical Formulation**:
```
P(aₜ = i) = 1/K  ∀ i ∈ {1, 2, ..., K}
```

**Pros**: Simple baseline, ensures exploration  
**Cons**: Ignores context information completely, poor performance  
**Use Case**: Baseline comparison

### Epsilon-Greedy Linear

**Description**: Linear models with epsilon-greedy exploration strategy.

**Mathematical Formulation**:

**Reward Model**: For each arm i, the reward is modeled as:
```
rᵢ = wᵢᵀx + bᵢ + ε
```

where wᵢ ∈ ℝᵈ are weights, bᵢ ∈ ℝ is bias, and ε ~ N(0, σ²) is noise.

**Action Selection**:
```
aₜ = { random arm                    with probability ε
     { argmax(wᵢᵀxₜ + bᵢ)           with probability 1 - ε
         i
```

**Parameter Update**: Using gradient descent on squared error loss:
```
Lᵢ = ½(rₜ - wᵢᵀxₜ - bᵢ)²
```

**Pros**: Simple, handles context, easy to implement  
**Cons**: Fixed exploration rate, assumes linear relationship  
**Use Case**: When context-reward relationship is approximately linear

### Linear Thompson Sampling

**Description**: Bayesian approach that models reward as linear function of context with uncertainty quantification.

**Mathematical Formulation**:

**Reward Model**: For each arm i:
```
rᵢ = wᵢᵀx + bᵢ + ε
```

where ε ~ N(0, σ²).

**Prior Distribution**: Normal conjugate prior:
```
wᵢ, bᵢ ~ N(μ₀, σ²Σ₀)
```

**Posterior Update**: After observing (xₜ, rₜ), the posterior is updated using Bayesian linear regression:

```
Σ_new = (Σ_old⁻¹ + (1/σ²) x_aug x_augᵀ)⁻¹

μ_new = Σ_new (Σ_old⁻¹ μ_old + (1/σ²) rₜ x_aug)
```

where x_aug = [1, xᵀ]ᵀ includes the bias term.

**Action Selection**: Sample parameters and select best arm:
```
θᵢ ~ N(μᵢ, Σᵢ)
aₜ = argmax θᵢᵀ x_aug,t
      i
```

**Uncertainty Quantification**: The uncertainty for arm i given context x is:
```
uncertainty_i(x) = √(x_augᵀ Σᵢ x_aug)
```

**Pros**: Handles context, theoretical guarantees, uncertainty quantification  
**Cons**: Assumes linear relationship, may not capture complex patterns  
**Use Case**: When context-reward relationship is approximately linear

### Neural Network Contextual Bandit

**Description**: Deep learning approach using neural networks with experience replay.

**Mathematical Formulation**:

**Network Architecture**: Feedforward neural network f_θ: ℝᵈ → ℝᴷ:
```
f_θ(x) = W_L σ(W_{L-1} σ(...σ(W₁x + b₁)...) + b_{L-1}) + b_L
```

where σ is the ReLU activation function, W_l and b_l are weights and biases of layer l.

**Q-Function**: The Q-value for arm i given context x is:
```
Q(x, i) = f_θ(x)_i
```

**Action Selection**:
```
aₜ = { random arm           with probability ε
     { argmax Q(xₜ, i)      with probability 1 - ε
         i
```

**Loss Function**: Mean squared error loss:
```
L(θ) = (1/|B|) Σ_{(x,a,r) ∈ B} (r - Q(x, a))²
```

where B is a batch of experiences from the replay buffer.

**Parameter Update**: Using Adam optimizer:
```
θ ← θ - α ∇_θ L(θ)
```

**Experience Replay**: Store experiences (xₜ, aₜ, rₜ) in buffer D and sample batches for training.

**Pros**: Can learn complex non-linear patterns, handles high-dimensional contexts  
**Cons**: More hyperparameters, requires more data, computationally expensive  
**Use Case**: When context-reward relationship is complex or non-linear

### Improved Neural Network Contextual Bandit

**Description**: Enhanced neural network with better training strategies.

**Mathematical Formulation**: Same as basic neural network but with enhanced training:

**Batch Normalization**: For each layer l:
```
BN(h_l) = γ_l (h_l - μ_l)/√(σ_l² + ε) + β_l
```

where μ_l and σ_l² are computed over the batch.

**Gradient Clipping**: Prevent exploding gradients:
```
g ← clip(g, -max_norm, max_norm)
```

**Learning Rate Scheduling**: Exponential decay:
```
αₜ = α₀ · γ^(t/step_size)
```

**Weight Decay**: L2 regularization:
```
L_reg(θ) = L(θ) + λ Σᵢ ||θᵢ||²
```

**Pros**: Faster learning, better stability, improved performance  
**Cons**: Even more computationally expensive  
**Use Case**: When you need the best possible neural network performance

## Results

*This section will contain experimental results and comparisons between algorithms.*

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

### Interactive Contextual Bandit

```bash
python play_with_contextual_bandit.py
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

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms
- Agrawal, S., & Goyal, N. (2013). Thompson Sampling for Contextual Bandits with Linear Payoffs
- Riquelme, C., et al. (2018). Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling