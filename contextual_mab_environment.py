import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Tuple, Optional
from scipy import stats


class ParameterizedDistribution(ABC):
    """Abstract base class for distributions that can be parameterized by context."""
    
    def __init__(self, context_dim: int):
        self.context_dim = context_dim
    
    @abstractmethod
    def get_parameters(self, context: np.ndarray) -> Tuple:
        """Convert context to distribution parameters."""
        pass
    
    @abstractmethod
    def sample(self, context: np.ndarray) -> float:
        """Sample from the distribution given context."""
        pass
    
    @abstractmethod
    def get_mean(self, context: np.ndarray) -> float:
        """Get the expected value given context."""
        pass


class LinearNormalDistribution(ParameterizedDistribution):
    """Normal distribution with mean parameterized linearly by context."""
    
    def __init__(self, context_dim: int, weights: np.ndarray, std: float = 1.0):
        super().__init__(context_dim)
        self.weights = weights  # Linear weights: mean = context @ weights
        self.std = std
    
    def get_parameters(self, context: np.ndarray) -> Tuple[float, float]:
        mean = np.dot(context, self.weights)
        return mean, self.std
    
    def sample(self, context: np.ndarray) -> float:
        mean, std = self.get_parameters(context)
        return np.random.normal(mean, std)
    
    def get_mean(self, context: np.ndarray) -> float:
        mean, _ = self.get_parameters(context)
        return mean


class LinearBernoulliDistribution(ParameterizedDistribution):
    """Bernoulli distribution with probability parameterized by sigmoid of linear context."""
    
    def __init__(self, context_dim: int, weights: np.ndarray):
        super().__init__(context_dim)
        self.weights = weights
    
    def get_parameters(self, context: np.ndarray) -> Tuple[float]:
        logit = np.dot(context, self.weights)
        prob = 1 / (1 + np.exp(-logit))  # sigmoid
        return (prob,)
    
    def sample(self, context: np.ndarray) -> float:
        prob, = self.get_parameters(context)
        return np.random.binomial(1, prob)
    
    def get_mean(self, context: np.ndarray) -> float:
        prob, = self.get_parameters(context)
        return prob


class CustomDistribution(ParameterizedDistribution):
    """Custom distribution using a lambda function."""
    
    def __init__(self, context_dim: int, distribution_func: Callable[[np.ndarray], float]):
        super().__init__(context_dim)
        self.distribution_func = distribution_func
    
    def get_parameters(self, context: np.ndarray) -> Tuple:
        # For custom distributions, we might not have explicit parameters
        return (context,)
    
    def sample(self, context: np.ndarray) -> float:
        return self.distribution_func(context)
    
    def get_mean(self, context: np.ndarray) -> float:
        # For custom distributions, we might need to estimate the mean
        # This is a simple Monte Carlo estimate
        samples = [self.distribution_func(context) for _ in range(100)]
        return np.mean(samples)


class ContextualArm:
    """An arm in a contextual bandit with parameterized reward distribution."""
    
    def __init__(self, name: str, distribution: ParameterizedDistribution):
        self.name = name
        self.distribution = distribution
        self.context_dim = distribution.context_dim
        self.pull_count = 0
    
    def pull(self, context: np.ndarray) -> float:
        """Pull the arm and get a reward given the context."""
        if len(context) != self.context_dim:
            raise ValueError(f"Context dimension {len(context)} doesn't match arm's expected dimension {self.context_dim}")
        
        self.pull_count += 1
        return self.distribution.sample(context)
    
    def get_expected_reward(self, context: np.ndarray) -> float:
        """Get the expected reward for a given context."""
        return self.distribution.get_mean(context)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the arm."""
        return {
            'name': self.name,
            'context_dim': self.context_dim,
            'pull_count': self.pull_count,
            'distribution_type': type(self.distribution).__name__
        }


class ContextualMultiArmedBandit:
    """Contextual multi-armed bandit environment."""
    
    def __init__(self, arms: List[ContextualArm], context_generator: Optional[Callable] = None, seed: Optional[int] = None):
        """
        Initialize the contextual bandit.
        
        Args:
            arms: List of ContextualArm objects
            context_generator: Function that generates context vectors. If None, uses random normal.
            seed: Random seed for reproducibility
        """
        self.arms = arms
        self.n_arms = len(arms)
        self.context_generator = context_generator or self._default_context_generator
        self.step_count = 0
        self.context_history = []
        self.action_history = []
        self.reward_history = []
        
        # Check that all arms have the same context dimension
        context_dims = [arm.context_dim for arm in arms]
        if len(set(context_dims)) > 1:
            raise ValueError("All arms must have the same context dimension")
        self.context_dim = context_dims[0]
        
        if seed is not None:
            np.random.seed(seed)
    
    def _default_context_generator(self) -> np.ndarray:
        """Default context generator: random normal distribution."""
        return np.random.normal(0, 1, self.context_dim)
    
    def generate_context(self) -> np.ndarray:
        """Generate a new context vector."""
        return self.context_generator()
    
    def pull_arm(self, arm_idx: int, context: Optional[np.ndarray] = None) -> Tuple[float, bool]:
        """
        Pull an arm and get a reward.
        
        Args:
            arm_idx: Index of the arm to pull
            context: Context vector. If None, generates a new one.
        
        Returns:
            Tuple of (reward, valid_action)
        """
        if arm_idx < 0 or arm_idx >= self.n_arms:
            return 0.0, False
        
        if context is None:
            context = self.generate_context()
        
        reward = self.arms[arm_idx].pull(context)
        
        # Record history
        self.step_count += 1
        self.context_history.append(context.copy())
        self.action_history.append(arm_idx)
        self.reward_history.append(reward)
        
        return reward, True
    
    def get_optimal_arm(self, context: np.ndarray) -> int:
        """Get the arm with highest expected reward for a given context."""
        expected_rewards = [arm.get_expected_reward(context) for arm in self.arms]
        return np.argmax(expected_rewards)
    
    def get_optimal_reward(self, context: np.ndarray) -> float:
        """Get the expected reward of the optimal arm for a given context."""
        optimal_arm = self.get_optimal_arm(context)
        return self.arms[optimal_arm].get_expected_reward(context)
    
    def get_arm_info(self, arm_idx: int) -> Dict[str, Any]:
        """Get information about a specific arm."""
        if arm_idx < 0 or arm_idx >= self.n_arms:
            return {}
        return self.arms[arm_idx].get_info()
    
    def reset(self):
        """Reset the bandit state."""
        self.step_count = 0
        self.context_history = []
        self.action_history = []
        self.reward_history = []
        for arm in self.arms:
            arm.pull_count = 0
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the bandit."""
        return {
            'n_arms': self.n_arms,
            'context_dim': self.context_dim,
            'step_count': self.step_count,
            'arms': [arm.get_info() for arm in self.arms]
        }
    
    @classmethod
    def create_linear_bandit(cls, n_arms: int, context_dim: int, seed: Optional[int] = None) -> 'ContextualMultiArmedBandit':
        """Create a contextual bandit with linear parameterization."""
        if seed is not None:
            np.random.seed(seed)
        
        arms = []
        for i in range(n_arms):
            # Random weights for each arm
            weights = np.random.normal(0, 1, context_dim)
            distribution = LinearNormalDistribution(context_dim, weights, std=0.1)
            arm = ContextualArm(f"Arm_{i}", distribution)
            arms.append(arm)
        
        return cls(arms, seed=seed)
    
    @classmethod
    def create_bernoulli_bandit(cls, n_arms: int, context_dim: int, seed: Optional[int] = None) -> 'ContextualMultiArmedBandit':
        """Create a contextual bandit with Bernoulli rewards."""
        if seed is not None:
            np.random.seed(seed)
        
        arms = []
        for i in range(n_arms):
            # Random weights for each arm
            weights = np.random.normal(0, 1, context_dim)
            distribution = LinearBernoulliDistribution(context_dim, weights)
            arm = ContextualArm(f"Arm_{i}", distribution)
            arms.append(arm)
        
        return cls(arms, seed=seed)


# Example usage and testing
if __name__ == "__main__":
    # Test linear contextual bandit
    print("Testing Linear Contextual Bandit")
    print("="*50)
    
    bandit = ContextualMultiArmedBandit.create_linear_bandit(n_arms=3, context_dim=2, seed=42)
    
    for step in range(10):
        context = bandit.generate_context()
        optimal_arm = bandit.get_optimal_arm(context)
        optimal_reward = bandit.get_optimal_reward(context)
        
        print(f"Step {step + 1}:")
        print(f"  Context: {context}")
        print(f"  Optimal arm: {optimal_arm}")
        print(f"  Optimal expected reward: {optimal_reward:.3f}")
        
        # Pull a random arm
        arm_idx = np.random.randint(3)
        reward, valid = bandit.pull_arm(arm_idx, context)
        print(f"  Pulled arm {arm_idx}: reward = {reward:.3f}")
        print()
    
    print("Bandit info:")
    print(bandit.get_info()) 