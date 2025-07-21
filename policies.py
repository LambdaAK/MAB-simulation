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
    """
    def __init__(self, n_arms: int, c: float = 2.0, seed: Optional[int] = None):
        super().__init__(n_arms)
        self.c = c
        if seed is not None:
            np.random.seed(seed)

    def select_action(self) -> int:
        # If any arm hasn't been selected yet, select it first
        for i in range(self.n_arms):
            if self.action_counts[i] == 0:
                return i
        total_counts = np.sum(self.action_counts)
        ucb_values = self.action_values + self.c * np.sqrt(np.log(total_counts) / self.action_counts)
        # Break ties randomly
        best_actions = np.where(ucb_values == np.max(ucb_values))[0]
        return np.random.choice(best_actions)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info['c'] = self.c
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