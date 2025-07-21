import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Arm:
    """Represents a single arm in the multi-armed bandit."""
    name: str
    mean: float
    std: float = 1.0
    distribution: str = "normal"  # "normal", "bernoulli", "uniform"

    # TODO: add better distribution support with different parameters
    
    def pull(self) -> float:
        """Pull the arm and return the reward."""
        if self.distribution == "normal":
            return np.random.normal(self.mean, self.std)
        elif self.distribution == "bernoulli":
            return np.random.binomial(1, self.mean)
        elif self.distribution == "uniform":
            return np.random.uniform(self.mean - self.std, self.mean + self.std)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class MultiArmedBandit:
    """
    Multi-Armed Bandit environment.
    
    This class represents a bandit with K arms, each with its own reward distribution.
    """
    
    def __init__(self, arms: List[Arm], seed: Optional[int] = None):
        """
        Initialize the multi-armed bandit.
        
        Args:
            arms: List of Arm objects representing the bandit arms
            seed: Random seed for reproducibility
        """
        self.arms = arms
        self.n_arms = len(arms)
        self.step_count = 0
        
        if seed is not None:
            np.random.seed(seed)
    
    @classmethod
    def create_random_bandit(cls, n_arms: int, mean_range: Tuple[float, float] = (0, 1), 
                           std: float = 1.0, seed: Optional[int] = None) -> 'MultiArmedBandit':
        """
        Create a random bandit with specified number of arms.
        
        Args:
            n_arms: Number of arms
            mean_range: Range for arm means (min, max)
            std: Standard deviation for all arms
            seed: Random seed
            
        Returns:
            MultiArmedBandit instance
        """
        if seed is not None:
            np.random.seed(seed)
            
        arms = []
        for i in range(n_arms):
            mean = np.random.uniform(mean_range[0], mean_range[1])
            arm = Arm(name=f"Arm_{i}", mean=mean, std=std)
            arms.append(arm)
            
        return cls(arms, seed)
    
    def pull_arm(self, arm_idx: int) -> Tuple[float, bool]:
        """
        Pull a specific arm and return the reward.
        
        Args:
            arm_idx: Index of the arm to pull (0-based)
            
        Returns:
            Tuple of (reward, valid_pull) where valid_pull is True if arm_idx is valid
        """
        if arm_idx < 0 or arm_idx >= self.n_arms:
            return 0.0, False
            
        self.step_count += 1
        reward = self.arms[arm_idx].pull()
        return reward, True
    
    def get_optimal_arm(self) -> int:
        """Return the index of the arm with the highest expected reward."""
        means = [arm.mean for arm in self.arms]
        return np.argmax(means)
    
    def get_optimal_reward(self) -> float:
        """Return the expected reward of the optimal arm."""
        means = [arm.mean for arm in self.arms]
        return max(means)
    
    def get_arm_info(self, arm_idx: int) -> Optional[dict]:
        """Get information about a specific arm."""
        if arm_idx < 0 or arm_idx >= self.n_arms:
            return None
            
        arm = self.arms[arm_idx]
        return {
            'name': arm.name,
            'mean': arm.mean,
            'std': arm.std,
            'distribution': arm.distribution
        }
    
    def reset(self):
        """Reset the environment (reset step counter)."""
        self.step_count = 0
    
    def get_info(self) -> dict:
        """Get information about the bandit."""
        return {
            'n_arms': self.n_arms,
            'step_count': self.step_count,
            'optimal_arm': self.get_optimal_arm(),
            'optimal_reward': self.get_optimal_reward(),
            'arms': [self.get_arm_info(i) for i in range(self.n_arms)]
        }


# Example usage and testing
if __name__ == "__main__":
    # Create a simple 3-armed bandit
    arms = [
        Arm("Arm_0", mean=0.2, std=0.5),
        Arm("Arm_1", mean=0.5, std=0.5),
        Arm("Arm_2", mean=0.8, std=0.5)
    ]
    
    bandit = MultiArmedBandit(arms, seed=42)
    
    print("Bandit Info:")
    info = bandit.get_info()
    print(f"Number of arms: {info['n_arms']}")
    print(f"Optimal arm: {info['optimal_arm']}")
    print(f"Optimal reward: {info['optimal_reward']:.3f}")
    
    print("\nArm details:")
    for i, arm_info in enumerate(info['arms']):
        print(f"Arm {i}: {arm_info}")
    
    print("\nTesting pulls:")
    for i in range(5):
        arm_idx = i % 3
        reward, valid = bandit.pull_arm(arm_idx)
        print(f"Pulled arm {arm_idx}: reward = {reward:.3f}")
    
    # Test random bandit creation
    print("\nCreating random bandit:")
    random_bandit = MultiArmedBandit.create_random_bandit(5, seed=123)
    random_info = random_bandit.get_info()
    print(f"Random bandit optimal arm: {random_info['optimal_arm']}")
    print(f"Random bandit optimal reward: {random_info['optimal_reward']:.3f}") 