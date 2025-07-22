# Multi-Armed Bandit Reinforcement Learning Project

## Table of Contents
- [Introduction](#introduction)
- [Theory: The Multi-Armed Bandit Problem](#theory-the-multi-armed-bandit-problem)
  - [Mathematical Formulation](#mathematical-formulation)
  - [Algorithmic Approaches](#algorithmic-approaches)
    - [Random Policy](#random-policy)
    - [Epsilon-Greedy Policy](#epsilon-greedy-policy)
    - [Epsilon-Greedy with Decay](#epsilon-greedy-with-decay)
    - [Softmax (Boltzmann) Policy](#softmax-boltzmann-policy)
    - [Upper Confidence Bound (UCB) Policy](#upper-confidence-bound-ucb-policy)
    - [Thompson Sampling](#thompson-sampling)
- [Code Structure and Usage](#code-structure-and-usage)
  - [Project Structure](#project-structure)
  - [How to Run Experiments](#how-to-run-experiments)
  - [Adding New Algorithms or Experiments](#adding-new-algorithms-or-experiments)
- [References](#references)

---

## Introduction

This project implements and compares algorithms for the Multi-Armed Bandit (MAB) problem, a foundational scenario in reinforcement learning and online decision-making. The codebase is modular, extensible, and includes interactive demos, experiment classes, and plotting utilities.

---

## Theory: The Multi-Armed Bandit Problem

### What is the Multi-Armed Bandit Problem?

The Multi-Armed Bandit (MAB) problem models a scenario where an agent must choose between multiple options ("arms"), each with an unknown reward distribution. The agent's goal is to maximize cumulative reward over time by balancing:
- **Exploration:** Trying different arms to learn their rewards.
- **Exploitation:** Choosing the arm believed to yield the highest reward.

This models real-world problems like online advertising, clinical trials, and recommendation systems.

### Mathematical Formulation

- Let there be $K$ arms, indexed by $i = 1, 2, ..., K$.
- Each arm $i$ provides a reward $r_t$ at time $t$, drawn from an unknown distribution with mean $\mu_i$.
- The agent selects an arm $a_t$ at each time step $t$, observes the reward, and updates its strategy.

**Objective:**

$$
\max_{a_1, ..., a_T} \mathbb{E}\left[ \sum_{t=1}^T r_t \right]
$$

**Regret:**

$$
\text{Regret}(T) = T \mu^* - \mathbb{E}\left[ \sum_{t=1}^T r_t \right], \quad \mu^* = \max_i \mu_i
$$

---

### Algorithmic Approaches

#### Random Policy
- **Description:** Select an arm uniformly at random at each step.
- **Math:**

$$
P(a_t = i) = \frac{1}{K} \quad \forall i
$$

- **Pros:** Simple, ensures exploration.
- **Cons:** Ignores past information, high regret.

#### Epsilon-Greedy Policy
- **Description:** With probability $\epsilon$, select a random arm (exploration); with probability $1-\epsilon$, select the arm with the highest estimated mean reward (exploitation).
- **Math:**

$$
Q_t(i) = \frac{1}{N_t(i)} \sum_{s=1}^{t} r_s \cdot \mathbb{I}[a_s = i]
$$

$$
a_t = \begin{cases}
\text{random arm} & \text{with probability } \epsilon \\
\arg\max_i Q_t(i) & \text{with probability } 1 - \epsilon
\end{cases}
$$

- **Pros:** Balances exploration and exploitation, simple.
- **Cons:** Fixed $\epsilon$ may not be optimal, does not account for uncertainty.

#### Epsilon-Greedy with Decay
- **Description:** Same as Epsilon-Greedy, but $\epsilon$ decreases over time.
- **Math:**

$$
\epsilon_t = \max(\epsilon_{\text{min}}, \epsilon_0 \cdot \text{decay}^t)
$$

- **Pros:** Adapts exploration rate, often achieves lower regret.
- **Cons:** Requires tuning decay schedule.

#### Softmax (Boltzmann) Policy
- **Description:** Selects arms with probability proportional to $\exp(Q_i / \tau)$, where $Q_i$ is the estimated value and $\tau$ is the temperature parameter.
- **Math:**

$$
P(a = i) = \frac{\exp(Q_i / \tau)}{\sum_{j=1}^K \exp(Q_j / \tau)}
$$

- **Pros:** Smoothly interpolates between exploration and exploitation; all arms are always explored, but better arms are favored.
- **Cons:** Requires tuning of $\tau$; can be sensitive to the scale of $Q_i$.
- **Usage Example (Pseudocode):**
  ```python
  import numpy as np
  def softmax_policy(Q, tau):
      exp_Q = np.exp(Q / tau)
      probs = exp_Q / np.sum(exp_Q)
      return np.random.choice(len(Q), p=probs)
  ```

#### Upper Confidence Bound (UCB) Policy
- **Description:** Selects the arm with the highest upper confidence bound on the estimated reward.
- **Math:**

$$
a_t = \arg\max_i \left( Q_t(i) + 2.0 \sqrt{\frac{\ln t}{N_t(i)}} \right)
$$

where $Q_t(i)$ is the estimated mean reward for arm $i$ at time $t$, and $N_t(i)$ is the number of times arm $i$ has been pulled.

- **Pros:** Theoretically optimal regret bounds, automatically balances exploration and exploitation.
- **Cons:** Requires knowledge of the total number of pulls, can be conservative in practice.

#### Thompson Sampling
- **Description:** Bayesian approach that samples from posterior distributions to select arms. Maintains uncertainty about arm rewards and naturally balances exploration and exploitation.

**Thompson Sampling with Beta Prior (for Bernoulli rewards):**
- **Math:**

$$
\theta_i \sim \text{Beta}(\alpha_i, \beta_i)
$$

$$
a_t = \arg\max_i \theta_i
$$

where $\alpha_i = 1 + \text{successes}_i$ and $\beta_i = 1 + \text{failures}_i$.

**Thompson Sampling with Normal-Inverse-Gamma Prior (for continuous rewards):**
- **Math:**

$$
\mu_i, \sigma_i^2 \sim \text{Normal-Inverse-Gamma}(\mu_0, \lambda_0, \alpha_0, \beta_0)
$$

$$
a_t = \arg\max_i \mu_i
$$

- **Pros:** Excellent empirical performance, naturally handles uncertainty, often outperforms UCB in practice.
- **Cons:** Requires prior assumptions, can be computationally more expensive than simpler methods.

---

## Code Structure and Usage

### Project Structure

- `mab_environment.py`: Core MAB environment and Arm classes.
- `policies.py`: Policy (algorithm) classes, including EpsilonGreedy, RandomPolicy, etc.
- `experiment.py`: Modular Experiment class for running and comparing policies.
- `play_with_bandit.py`: Main script for interactive play, demos, and running experiments.
- `requirements.txt`: Python dependencies.
- `README.md`: This documentation.
- `components.md`: Project planning notes.

### How to Run Experiments

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the main script:**
   ```bash
   python play_with_bandit.py
   ```
   - Choose from interactive play, policy demos, or sample experiments.
   - Use `--mode` to run a specific experiment directly.

3. **Sample Multi-Policy Experiment:**
   - Compares Random and several EpsilonGreedy configurations on a 10-armed bandit.
   - Plots average accumulated reward for each policy.

### Adding New Algorithms or Experiments

- **To add a new policy:**
  - Implement a new class in `policies.py` inheriting from `Policy`.
  - Add it to the `policies` list in your experiment setup.
- **To add a new experiment:**
  - Create a new function in `play_with_bandit.py` using the `Experiment` class.
  - Specify arms, policies, and parameters as needed.

---

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. (Ch. 2)
- Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms. Cambridge University Press.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning, 47(2-3), 235–256.

---

*This README is self-contained and provides both the theoretical background and practical instructions for using and extending the Multi-Armed Bandit project.* 

