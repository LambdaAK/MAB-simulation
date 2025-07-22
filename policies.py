import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from mab_environment import MultiArmedBandit


class Policy(ABC):
    """
    Abstract base class for bandit policies.
    
    All bandit algorithms should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, n_arms: int, **kwargs):
        """
        Initialize the policy.
        
        Args:
            n_arms: Number of arms in the bandit
            **kwargs: Additional parameters specific to the policy
        """
        self.n_arms = n_arms
        self.step_count = 0
        self.action_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        
    @abstractmethod
    def select_action(self) -> int:
        """
        Select an action (arm) to pull.
        
        Returns:
            Index of the selected arm
        """
        pass
    
    def update(self, action: int, reward: float):
        """
        Update the policy with the observed reward.
        
        Args:
            action: The arm that was pulled
            reward: The reward received
        """
        self.step_count += 1
        self.action_counts[action] += 1
        
        # Update action value using incremental average
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]
    
    def reset(self):
        """Reset the policy state."""
        self.step_count = 0
        self.action_values = np.zeros(self.n_arms)
        self.action_counts = np.zeros(self.n_arms)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the policy state."""
        return {
            'step_count': self.step_count,
            'action_values': self.action_values.copy(),
            'action_counts': self.action_counts.copy(),
            'best_action': np.argmax(self.action_values) if self.step_count > 0 else None
        }


class EpsilonGreedy(Policy):
    """
    Epsilon-Greedy policy with optional epsilon decay.
    
    With probability epsilon, selects a random action (exploration).
    With probability 1-epsilon, selects the action with highest estimated value (exploitation).
    """
    
    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: Optional[int] = None, decay: float = 1.0, min_epsilon: float = 0.01):
        """
        Initialize epsilon-greedy policy.
        
        Args:
            n_arms: Number of arms
            epsilon: Initial probability of random exploration (0 <= epsilon <= 1)
            seed: Random seed for reproducibility
            decay: Multiplicative decay factor for epsilon (default 1.0 = no decay)
            min_epsilon: Minimum value for epsilon (default 0.01)
        """
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        
        if seed is not None:
            np.random.seed(seed)
    
    def select_action(self) -> int:
        """
        Select action using epsilon-greedy strategy.
        
        Returns:
            Index of the selected arm
        """
        if np.random.random() < self.epsilon:
            # Explore: select random action
            return np.random.randint(self.n_arms)
        else:
            # Exploit: select action with highest estimated value
            # Break ties randomly
            best_actions = np.where(self.action_values == np.max(self.action_values))[0]
            return np.random.choice(best_actions)
    
    def update(self, action: int, reward: float):
        super().update(action, reward)
        # Decay epsilon after each update
        if self.decay < 1.0 or self.min_epsilon > 0.0:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
    
    def reset(self):
        super().reset()
        self.epsilon = self.initial_epsilon
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the policy state."""
        info = super().get_info()
        info['epsilon'] = self.epsilon
        info['exploration_rate'] = self.epsilon
        info['decay'] = self.decay
        info['min_epsilon'] = self.min_epsilon
        return info


class RandomPolicy(Policy):
    """
    Random policy (baseline).
    
    Always selects actions uniformly at random.
    """
    
    def __init__(self, n_arms: int, seed: Optional[int] = None):
        """
        Initialize random policy.
        
        Args:
            n_arms: Number of arms
            seed: Random seed for reproducibility
        """
        super().__init__(n_arms)
        
        if seed is not None:
            np.random.seed(seed)
    
    def select_action(self) -> int:
        """
        Select random action.
        
        Returns:
            Index of the selected arm
        """
        return np.random.randint(self.n_arms)


class GreedyPolicy(Policy):
    """
    Greedy policy (always exploit).
    
    Always selects the action with highest estimated value.
    """
    
    def select_action(self) -> int:
        """
        Select action with highest estimated value.
        
        Returns:
            Index of the selected arm
        """
        if self.step_count == 0:
            # If no actions have been taken, select randomly
            return np.random.randint(self.n_arms)
        else:
            # Select action with highest estimated value
            # Break ties randomly
            best_actions = np.where(self.action_values == np.max(self.action_values))[0]
            return np.random.choice(best_actions)


class UCBPolicy(Policy):
    """
    Upper Confidence Bound (UCB1) policy.
    Selects the arm with the highest upper confidence bound on the estimated reward.
    Uses the standard theoretical value c = 2.0 for the exploration bonus.
    """
    def __init__(self, n_arms: int, seed: Optional[int] = None):
        super().__init__(n_arms)
        if seed is not None:
            np.random.seed(seed)

    def select_action(self) -> int:
        # If any arm hasn't been selected yet, select it first
        for i in range(self.n_arms):
            if self.action_counts[i] == 0:
                return i
        total_counts = np.sum(self.action_counts)
        ucb_values = self.action_values + 2.0 * np.sqrt(np.log(total_counts) / self.action_counts)
        # Break ties randomly
        best_actions = np.where(ucb_values == np.max(ucb_values))[0]
        return np.random.choice(best_actions)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info['c'] = 2.0  # Fixed theoretical value
        return info


class SoftmaxPolicy(Policy):
    """
    Softmax (Boltzmann) exploration policy.
    Selects arms with probability proportional to exp(Q_i / tau).
    """
    def __init__(self, n_arms: int, tau: float = 0.1, seed: Optional[int] = None):
        super().__init__(n_arms)
        self.tau = tau
        if seed is not None:
            np.random.seed(seed)

    def select_action(self) -> int:
        # Avoid division by zero
        if self.tau <= 0:
            # Greedy selection
            best_actions = np.where(self.action_values == np.max(self.action_values))[0]
            return np.random.choice(best_actions)
        # Compute softmax probabilities
        max_q = np.max(self.action_values)  # for numerical stability
        exp_q = np.exp((self.action_values - max_q) / self.tau)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.n_arms, p=probs)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info['tau'] = self.tau
        return info


class ThompsonSamplingPolicy(Policy):
    """
    Thompson Sampling policy using Beta distributions.
    Assumes rewards are in [0,1] (can be Bernoulli or normalized continuous rewards).
    For each arm, maintains Beta(α, β) posterior where:
    - α = 1 + number of successes (rewards = 1)
    - β = 1 + number of failures (rewards = 0)
    """
    def __init__(self, n_arms: int, seed: Optional[int] = None):
        super().__init__(n_arms)
        # Initialize Beta(1,1) priors (uniform distribution)
        self.alpha = np.ones(n_arms)  # α = 1 + successes
        self.beta = np.ones(n_arms)   # β = 1 + failures
        if seed is not None:
            np.random.seed(seed)

    def select_action(self) -> int:
        # Sample from Beta distributions for each arm
        samples = np.random.beta(self.alpha, self.beta)
        # Select arm with highest sampled value
        best_actions = np.where(samples == np.max(samples))[0]
        return np.random.choice(best_actions)

    def update(self, action: int, reward: float):
        super().update(action, reward)
        # Update Beta parameters
        # For Bernoulli rewards: reward should be 0 or 1
        # For continuous rewards in [0,1]: treat as success/failure based on threshold
        if reward > 0.5:  # Success
            self.alpha[action] += 1
        else:  # Failure
            self.beta[action] += 1

    def reset(self):
        super().reset()
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info['alpha'] = self.alpha.copy()
        info['beta'] = self.beta.copy()
        info['posterior_means'] = self.alpha / (self.alpha + self.beta)
        return info


class ThompsonSamplingNormalPolicy(Policy):
    """
    Thompson Sampling policy using Normal distributions.
    Assumes rewards follow Normal(μ, σ²) distributions.
    Uses Normal-Inverse-Gamma conjugate prior.
    """
    def __init__(self, n_arms: int, seed: Optional[int] = None):
        super().__init__(n_arms)
        # Initialize Normal-Inverse-Gamma priors
        # μ₀ = 0, λ₀ = 1, α₀ = 1, β₀ = 1
        self.mu_0 = np.zeros(n_arms)      # Prior mean
        self.lambda_0 = np.ones(n_arms)   # Prior precision multiplier
        self.alpha_0 = np.ones(n_arms)    # Prior shape
        self.beta_0 = np.ones(n_arms)     # Prior scale
        
        # Posterior parameters
        self.mu_n = np.zeros(n_arms)      # Posterior mean
        self.lambda_n = np.ones(n_arms)   # Posterior precision multiplier
        self.alpha_n = np.ones(n_arms)    # Posterior shape
        self.beta_n = np.ones(n_arms)     # Posterior scale
        
        if seed is not None:
            np.random.seed(seed)

    def select_action(self) -> int:
        # Sample from Student's t-distribution for each arm
        # This is the posterior predictive distribution
        samples = np.random.standard_t(2 * self.alpha_n)
        # Scale and shift to get samples from the posterior predictive
        samples = self.mu_n + samples * np.sqrt(self.beta_n * (self.lambda_n + 1) / (self.alpha_n * self.lambda_n))
        
        # Select arm with highest sampled value
        best_actions = np.where(samples == np.max(samples))[0]
        return np.random.choice(best_actions)

    def update(self, action: int, reward: float):
        super().update(action, reward)
        
        # Update posterior parameters for the selected arm
        n = self.action_counts[action]
        
        # Update lambda (precision multiplier)
        self.lambda_n[action] = self.lambda_0[action] + n
        
        # Update mu (mean)
        if n == 1:
            self.mu_n[action] = reward
        else:
            # Incremental update of mean
            self.mu_n[action] = (self.mu_n[action] * (n-1) + reward) / n
        
        # Update alpha and beta
        self.alpha_n[action] = self.alpha_0[action] + n/2
        
        # Update beta (scale parameter)
        if n == 1:
            self.beta_n[action] = self.beta_0[action] + 0.5 * (reward - self.mu_0[action])**2
        else:
            # Incremental update of sum of squared differences
            old_mean = (self.mu_n[action] * n - reward) / (n-1)
            self.beta_n[action] = self.beta_0[action] + 0.5 * (n-1) * (self.mu_n[action] - old_mean)**2 + 0.5 * (reward - self.mu_n[action])**2

    def reset(self):
        super().reset()
        self.mu_n = np.zeros(self.n_arms)
        self.lambda_n = np.ones(self.n_arms)
        self.alpha_n = np.ones(self.n_arms)
        self.beta_n = np.ones(self.n_arms)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info['posterior_means'] = self.mu_n.copy()
        info['posterior_variances'] = self.beta_n / (self.alpha_n - 0.5)  # Variance of posterior
        return info


# Example usage and testing
if __name__ == "__main__":
    from mab_environment import MultiArmedBandit, Arm
    
    # Create a test bandit
    arms = [
        Arm("Poor", mean=0.2, std=0.3),
        Arm("Medium", mean=0.5, std=0.3),
        Arm("Good", mean=0.8, std=0.3)
    ]
    bandit = MultiArmedBandit(arms, seed=42)
    
    print("Testing Epsilon-Greedy Policy")
    print("="*40)
    
    # Test different epsilon values
    for epsilon in [0.0, 0.1, 0.3]:
        print(f"\nEpsilon = {epsilon}")
        
        policy = EpsilonGreedy(n_arms=3, epsilon=epsilon, seed=123)
        total_reward = 0
        
        for step in range(50):
            action = policy.select_action()
            reward, valid = bandit.pull_arm(action)
            
            if valid:
                policy.update(action, reward)
                total_reward += reward
        
        info = policy.get_info()
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward: {total_reward/50:.3f}")
        print(f"Action counts: {info['action_counts']}")
        print(f"Action values: {info['action_values']}")
        print(f"Best action estimate: {info['best_action']}")
        print(f"Optimal arm: {bandit.get_optimal_arm()}")
        
        # Reset for next test
        bandit.reset()
    
    print("\n" + "="*40)
    print("Testing Random Policy")
    print("="*40)
    
    policy = RandomPolicy(n_arms=3, seed=123)
    total_reward = 0
    
    for step in range(50):
        action = policy.select_action()
        reward, valid = bandit.pull_arm(action)
        
        if valid:
            policy.update(action, reward)
            total_reward += reward
    
    info = policy.get_info()
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward: {total_reward/50:.3f}")
    print(f"Action counts: {info['action_counts']}")
    print(f"Action values: {info['action_values']}")
    print(f"Best action estimate: {info['best_action']}")
    print(f"Optimal arm: {bandit.get_optimal_arm()}") 