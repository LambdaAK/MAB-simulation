# Multi-Armed Bandit and Contextual Bandit Implementation

This project implements various Multi-Armed Bandit (MAB) and Contextual Multi-Armed Bandit (CMAB) algorithms for reinforcement learning.

## Introduction

### What is Multi-Armed Bandit?

A **Multi-Armed Bandit (MAB)** is a sequential decision-making problem where an agent must choose between $K$ actions (arms) at each time step to maximize cumulative reward over time. The agent faces the fundamental trade-off between **exploration** (trying different arms to learn their rewards) and **exploitation** (choosing the arm believed to yield the highest reward).

#### Mathematical Formulation

**Problem Setup:**
- **Arms**: $K$ arms indexed by $i \in \{1, 2, \ldots, K\}$
- **Time Horizon**: $T$ time steps
- **Reward Distribution**: Each arm $i$ has an unknown reward distribution with mean $\mu_i$
- **Action**: At time $t$, the agent selects arm $a_t \in \{1, 2, \ldots, K\}$
- **Reward**: The agent receives reward $r_t \sim D_{a_t}$ where $D_{a_t}$ is the reward distribution of the selected arm

**Objective:**
The goal is to maximize the expected cumulative reward:

```math
\max_{\pi} \mathbb{E}\left[\sum_{t=1}^T r_t\right]
```

where $\pi$ is the policy that maps history to actions.

**Regret:**
The performance is measured in terms of regret, which is the difference between the optimal cumulative reward and the actual cumulative reward:

```math
R(T) = T \mu^* - \mathbb{E}\left[\sum_{t=1}^T r_t\right]
```

where $\mu^* = \max_{i} \mu_i$ is the expected reward of the optimal arm.

**Exploration vs Exploitation Trade-off:**
The agent must balance:
- **Exploration**: Gathering information about arm rewards
- **Exploitation**: Using current knowledge to maximize immediate reward

### What is Contextual Multi-Armed Bandit?

A **Contextual Multi-Armed Bandit (CMAB)** extends the traditional MAB problem by introducing context-dependent reward distributions. The reward of each arm depends on a context vector that is revealed before each decision.

#### Mathematical Formulation

**Problem Setup:**
- **Context Space**: $\mathcal{X} \subseteq \mathbb{R}^d$ (d-dimensional context space)
- **Context**: At time $t$, context $x_t \in \mathcal{X}$ is revealed
- **Arms**: $K$ arms indexed by $i \in \{1, 2, \ldots, K\}$
- **Reward Function**: Each arm $i$ has a reward function $f_i: \mathcal{X} \rightarrow \mathbb{R}$
- **Action**: At time $t$, the agent selects arm $a_t \in \{1, 2, \ldots, K\}$ based on context $x_t$
- **Reward**: The agent receives reward $r_t = f_{a_t}(x_t) + \epsilon_t$ where $\epsilon_t$ is noise

**Objective:**
Maximize the expected cumulative reward:

```math
\max_{\pi} \mathbb{E}\left[\sum_{t=1}^T r_t\right]
```

where $\pi: \mathcal{X} \rightarrow \{1, 2, \ldots, K\}$ is a context-dependent policy.

**Regret:**
The contextual regret is:

```math
R(T) = \sum_{t=1}^T \max_{i} f_i(x_t) - \mathbb{E}\left[\sum_{t=1}^T r_t\right]
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

```math
P(a_t = i) = \frac{1}{K} \quad \forall i \in \{1, 2, \ldots, K\}
```

**Pros**: Simple baseline, ensures exploration
**Cons**: Ignores context information completely, poor performance
**Use Case**: Baseline comparison

### Epsilon-Greedy Linear

**Description**: Linear models with epsilon-greedy exploration strategy.

**Mathematical Formulation**:

**Reward Model**: For each arm $i$, the reward is modeled as:

```math
r_i = \mathbf{w}_i^T \mathbf{x} + b_i + \epsilon
```

where $\mathbf{w}_i \in \mathbb{R}^d$ are weights, $b_i \in \mathbb{R}$ is bias, and $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is noise.

**Action Selection**:

```math
a_t = \begin{cases}
\text{random arm} & \text{with probability } \epsilon \\
\arg\max_{i} (\mathbf{w}_i^T \mathbf{x}_t + b_i) & \text{with probability } 1 - \epsilon
\end{cases}
```

**Parameter Update**: Using gradient descent on squared error loss:

```math
\mathcal{L}_i = \frac{1}{2}(r_t - \mathbf{w}_i^T \mathbf{x}_t - b_i)^2
```

```math
\mathbf{w}_i \leftarrow \mathbf{w}_i - \alpha \nabla_{\mathbf{w}_i} \mathcal{L}_i
```

```math
b_i \leftarrow b_i - \alpha \nabla_{b_i} \mathcal{L}_i
```

where $\alpha$ is the learning rate.

**Pros**: Simple, handles context, easy to implement
**Cons**: Fixed exploration rate, assumes linear relationship
**Use Case**: When context-reward relationship is approximately linear

### Linear Thompson Sampling

**Description**: Bayesian approach that models reward as linear function of context with uncertainty quantification.

**Mathematical Formulation**:

**Reward Model**: For each arm $i$:

```math
r_i = \mathbf{w}_i^T \mathbf{x} + b_i + \epsilon
```

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

**Prior Distribution**: Normal conjugate prior:

```math
\mathbf{w}_i, b_i \sim \mathcal{N}(\mu_0, \sigma^2\Sigma_0)
```

**Posterior Update**: After observing $(\mathbf{x}_t, r_t)$, the posterior is updated using Bayesian linear regression:

```math
\Sigma_{\text{new}} = \left(\Sigma_{\text{old}}^{-1} + \frac{1}{\sigma^2} \mathbf{x}_{\text{aug}} \mathbf{x}_{\text{aug}}^T\right)^{-1}
```

```math
\mu_{\text{new}} = \Sigma_{\text{new}} \left(\Sigma_{\text{old}}^{-1} \mu_{\text{old}} + \frac{1}{\sigma^2} r_t \mathbf{x}_{\text{aug}}\right)
```

where $\mathbf{x}_{\text{aug}} = [1, \mathbf{x}^T]^T$ includes the bias term.

**Action Selection**: Sample parameters and select best arm:

```math
\theta_i \sim \mathcal{N}(\mu_i, \Sigma_i)
```

```math
a_t = \arg\max_{i} \theta_i^T \mathbf{x}_{\text{aug}, t}
```

**Uncertainty Quantification**: The uncertainty for arm $i$ given context $\mathbf{x}$ is:

```math
\text{uncertainty}_i(\mathbf{x}) = \sqrt{\mathbf{x}_{\text{aug}}^T \Sigma_i \mathbf{x}_{\text{aug}}}
```

**Pros**: Handles context, theoretical guarantees, uncertainty quantification
**Cons**: Assumes linear relationship, may not capture complex patterns
**Use Case**: When context-reward relationship is approximately linear

### Neural Network Contextual Bandit

**Description**: Deep learning approach using neural networks with experience replay.

**Mathematical Formulation**:

**Network Architecture**: Feedforward neural network $f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^K$:

```math
f_\theta(\mathbf{x}) = \mathbf{W}_L \sigma(\mathbf{W}_{L-1} \sigma(\ldots \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \ldots) + \mathbf{b}_{L-1}) + \mathbf{b}_L
```

where $\sigma$ is the ReLU activation function, $\mathbf{W}_l$ and $\mathbf{b}_l$ are weights and biases of layer $l$.

**Q-Function**: The Q-value for arm $i$ given context $\mathbf{x}$ is:

```math
Q(\mathbf{x}, i) = f_\theta(\mathbf{x})_i
```

**Action Selection**:

```math
a_t = \begin{cases}
\text{random arm} & \text{with probability } \epsilon \\
\arg\max_{i} Q(\mathbf{x}_t, i) & \text{with probability } 1 - \epsilon
\end{cases}
```

**Loss Function**: Mean squared error loss:

```math
\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{x}, a, r) \in \mathcal{B}} (r - Q(\mathbf{x}, a))^2
```

where $\mathcal{B}$ is a batch of experiences from the replay buffer.

**Parameter Update**: Using Adam optimizer:

```math
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)
```

**Experience Replay**: Store experiences $(\mathbf{x}_t, a_t, r_t)$ in buffer $\mathcal{D}$ and sample batches for training.

**Pros**: Can learn complex non-linear patterns, handles high-dimensional contexts
**Cons**: More hyperparameters, requires more data, computationally expensive
**Use Case**: When context-reward relationship is complex or non-linear

### Improved Neural Network Contextual Bandit

**Description**: Enhanced neural network with better training strategies.

**Mathematical Formulation**: Same as basic neural network but with enhanced training:

**Batch Normalization**: For each layer $l$:

```math
\text{BN}(\mathbf{h}_l) = \gamma_l \frac{\mathbf{h}_l - \boldsymbol{\mu}_l}{\sqrt{\boldsymbol{\sigma}_l^2 + \epsilon}} + \boldsymbol{\beta}_l
```

where $\boldsymbol{\mu}_l$ and $\boldsymbol{\sigma}_l^2$ are computed over the batch.

**Gradient Clipping**: Prevent exploding gradients:

```math
\mathbf{g} \leftarrow \text{clip}(\mathbf{g}, -\text{max\_norm}, \text{max\_norm})
```

**Learning Rate Scheduling**: Exponential decay:

```math
\alpha_t = \alpha_0 \cdot \gamma^{t/\text{step\_size}}
```

**Weight Decay**: L2 regularization:

```math
\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \lambda \sum_{i} \|\theta_i\|^2
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

## Experiment 1: Context-Aware vs. Context-Ignorant Bandit Algorithms

This experiment demonstrates the importance of using context in multi-armed bandit problems. We compare several algorithms on a 5-armed contextual bandit with 5-dimensional context vectors, where each arm's expected reward is a different linear function of the context.

**Algorithms compared:**
- **Linear Thompson Sampling** (context-aware, Bayesian)
- **Epsilon-Greedy Linear** (context-aware, linear model)
- **Neural Network Contextual** (context-aware, nonlinear model)
- **Contextless Epsilon-Greedy** (ignores context, classic bandit)

**Results:**
- The left plot shows **cumulative regret**: how much reward each algorithm missed out on compared to the optimal policy.
- The right plot shows **average reward** over time, with a dashed line for the optimal average reward.

![Context-Aware vs. Context-Ignorant Bandit Algorithms](https://github.com/LambdaAK/MAB-simulation/blob/main/Experiment%20pictures/exp1.png)

**Key takeaway:**  
Context-aware algorithms (especially Linear Thompson Sampling and Neural Network Contextual) achieve much lower regret and higher average reward than context-ignorant algorithms. The Contextless Epsilon-Greedy policy, which ignores context, performs significantly worse, highlighting the value of leveraging contextual information in decision-making.

## Experiment 2: Epsilon-Greedy Linear Hyperparameter Comparison

This experiment investigates how the choice of the epsilon parameter in the Epsilon-Greedy Linear policy affects performance in a contextual bandit setting. We compare several values of epsilon (ε = 0.01, 0.05, 0.1, 0.2, 0.5) on a 5-armed contextual bandit with 5-dimensional context vectors.

**What is being compared:**
- **Epsilon-Greedy Linear** with different values of ε (exploration rate).

**Results:**
- The left plot shows **cumulative regret** for each epsilon value. Lower regret indicates better performance.
- The right plot shows **average reward** over time. The dashed line represents the optimal average reward.

![Epsilon-Greedy Linear Hyperparameter Comparison](https://github.com/LambdaAK/MAB-simulation/blob/main/Experiment%20pictures/exp2.png)

**Key takeaway:**  
Smaller values of epsilon (less random exploration) generally lead to lower regret and higher average reward in this linear contextual bandit setting. However, too little exploration (very small epsilon) can risk missing out on learning about better arms early on, while too much exploration (large epsilon) leads to consistently worse performance.

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms
- Agrawal, S., & Goyal, N. (2013). Thompson Sampling for Contextual Bandits with Linear Payoffs
- Riquelme, C., et al. (2018). Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling