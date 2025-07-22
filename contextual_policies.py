#!/usr/bin/env python3
"""
Contextual bandit policies for contextual multi-armed bandit problems.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import copy


class ContextualPolicy(ABC):
    """Abstract base class for contextual bandit policies."""
    
    def __init__(self, n_arms: int, context_dim: int, **kwargs):
        """
        Initialize the contextual policy.
        
        Args:
            n_arms: Number of arms
            context_dim: Dimension of context vectors
            **kwargs: Additional parameters
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.step_count = 0
        
    @abstractmethod
    def select_action(self, context: np.ndarray) -> int:
        """
        Select an action given the context.
        
        Args:
            context: Context vector
            
        Returns:
            Index of the selected arm
        """
        pass
    
    def update(self, context: np.ndarray, action: int, reward: float):
        """
        Update the policy with observed reward.
        
        Args:
            context: Context vector
            action: Selected arm
            reward: Observed reward
        """
        self.step_count += 1
        self._update_impl(context, action, reward)
    
    @abstractmethod
    def _update_impl(self, context: np.ndarray, action: int, reward: float):
        """Implementation of the update method."""
        pass
    
    def reset(self):
        """Reset the policy state."""
        self.step_count = 0
        self._reset_impl()
    
    @abstractmethod
    def _reset_impl(self):
        """Implementation of the reset method."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the policy state."""
        return {
            'step_count': self.step_count,
            'n_arms': self.n_arms,
            'context_dim': self.context_dim
        }


class LinearThompsonSampling(ContextualPolicy):
    """
    Linear Thompson Sampling with bias term (intercept).
    
    Models each arm's reward as: r = bias + context @ weights + noise
    where bias is the intercept term.
    """
    
    def __init__(self, n_arms: int, context_dim: int, 
                 prior_mean: float = 0.0, prior_std: float = 1.0,
                 noise_std: float = 1.0, seed: Optional[int] = None):
        """
        Initialize Linear Thompson Sampling.
        
        Args:
            n_arms: Number of arms
            context_dim: Dimension of context vectors
            prior_mean: Prior mean for parameters (default: 0.0)
            prior_std: Prior standard deviation for parameters (default: 1.0)
            noise_std: Standard deviation of reward noise (default: 1.0)
            seed: Random seed
        """
        super().__init__(n_arms, context_dim)
        
        # Add bias term: total parameters = context_dim + 1 (bias + weights)
        self.param_dim = context_dim + 1
        self.noise_std = noise_std
        
        # Initialize posterior parameters for each arm
        # μ_i: posterior mean vector [bias, weight_1, weight_2, ..., weight_d]
        # Σ_i: posterior covariance matrix
        self.posterior_means = np.full((n_arms, self.param_dim), prior_mean)
        self.posterior_covs = np.stack([np.eye(self.param_dim) * prior_std**2 for _ in range(n_arms)])
        
        # Store context and reward history for each arm
        self.context_history = [[] for _ in range(n_arms)]
        self.reward_history = [[] for _ in range(n_arms)]
        
        # Bias detection
        self.prediction_errors = []
        self.bias_threshold = 2.0 * noise_std  # Threshold for bias detection
        self.uncertainty_multipliers = np.ones(n_arms)  # Adaptive uncertainty
        
        if seed is not None:
            np.random.seed(seed)
    
    def _add_bias_term(self, context: np.ndarray) -> np.ndarray:
        """Add bias term (intercept) to context vector."""
        return np.concatenate([[1.0], context])  # [1, x_1, x_2, ..., x_d]
    
    def select_action(self, context: np.ndarray) -> int:
        """
        Select action using Thompson Sampling.
        
        Args:
            context: Context vector (without bias term)
            
        Returns:
            Index of the selected arm
        """
        # Add bias term to context
        context_with_bias = self._add_bias_term(context)
        
        # Sample from posterior for each arm
        sampled_rewards = []
        for arm in range(self.n_arms):
            # Sample parameters from posterior
            θ_sample = np.random.multivariate_normal(
                self.posterior_means[arm],
                self.posterior_covs[arm] * self.uncertainty_multipliers[arm]
            )
            
            # Predict reward: bias + context @ weights
            predicted_reward = np.dot(context_with_bias, θ_sample)
            sampled_rewards.append(predicted_reward)
        
        # Select arm with highest sampled reward
        return np.argmax(sampled_rewards)
    
    def _update_impl(self, context: np.ndarray, action: int, reward: float):
        """
        Update posterior for the selected arm using Bayesian linear regression.
        
        Args:
            context: Context vector (without bias term)
            action: Selected arm
            reward: Observed reward
        """
        # Add bias term to context
        context_with_bias = self._add_bias_term(context)
        
        # Store data
        self.context_history[action].append(context_with_bias.copy())
        self.reward_history[action].append(reward)
        
        # Get current posterior parameters
        μ = self.posterior_means[action]
        Σ = self.posterior_covs[action]
        
        # Compute prediction error for bias detection
        predicted_reward = np.dot(context_with_bias, μ)
        prediction_error = abs(reward - predicted_reward)
        self.prediction_errors.append(prediction_error)
        
        # Detect bias and adjust uncertainty
        if prediction_error > self.bias_threshold:
            self.uncertainty_multipliers[action] = min(
                self.uncertainty_multipliers[action] * 1.1,  # Increase uncertainty
                5.0  # Cap at 5x
            )
        
        # Bayesian update using matrix inversion lemma
        # Σ_new = Σ - (Σ * x * x^T * Σ) / (σ² + x^T * Σ * x)
        # μ_new = μ + (r - x^T * μ) / (σ² + x^T * Σ * x) * Σ * x
        
        # Compute intermediate quantities
        Σx = Σ @ context_with_bias
        xTΣx = np.dot(context_with_bias, Σx)
        denominator = self.noise_std**2 + xTΣx
        
        # Update covariance
        self.posterior_covs[action] = Σ - np.outer(Σx, Σx) / denominator
        
        # Update mean
        prediction_error = reward - predicted_reward
        self.posterior_means[action] = μ + (prediction_error / denominator) * Σx
        
        # Ensure numerical stability
        self._ensure_numerical_stability(action)
    
    def _ensure_numerical_stability(self, arm: int):
        """Ensure numerical stability of posterior parameters."""
        # Ensure covariance is symmetric and positive definite
        Σ = self.posterior_covs[arm]
        Σ = (Σ + Σ.T) / 2  # Make symmetric
        
        # Add small regularization to diagonal
        min_eigenval = np.linalg.eigvals(Σ).min()
        if min_eigenval < 1e-6:
            self.posterior_covs[arm] = Σ + np.eye(self.param_dim) * 1e-6
    
    def _reset_impl(self):
        """Reset the policy state."""
        # Reset posterior parameters
        prior_std = 1.0
        self.posterior_means = np.zeros((self.n_arms, self.param_dim))
        self.posterior_covs = np.stack([np.eye(self.param_dim) * prior_std**2 for _ in range(self.n_arms)])
        
        # Reset history
        self.context_history = [[] for _ in range(self.n_arms)]
        self.reward_history = [[] for _ in range(self.n_arms)]
        
        # Reset bias detection
        self.prediction_errors = []
        self.uncertainty_multipliers = np.ones(self.n_arms)
    
    def get_expected_reward(self, context: np.ndarray, arm: int) -> float:
        """
        Get expected reward for an arm given context.
        
        Args:
            context: Context vector (without bias term)
            arm: Arm index
            
        Returns:
            Expected reward
        """
        context_with_bias = self._add_bias_term(context)
        return np.dot(context_with_bias, self.posterior_means[arm])
    
    def get_uncertainty(self, context: np.ndarray, arm: int) -> float:
        """
        Get uncertainty (standard deviation) for an arm given context.
        
        Args:
            context: Context vector (without bias term)
            arm: Arm index
            
        Returns:
            Uncertainty measure
        """
        context_with_bias = self._add_bias_term(context)
        variance = context_with_bias @ self.posterior_covs[arm] @ context_with_bias
        return np.sqrt(variance)
    
    def get_info(self) -> Dict[str, Any]:
        """Get detailed information about the policy state."""
        info = super().get_info()
        info.update({
            'policy_type': 'LinearThompsonSampling',
            'param_dim': self.param_dim,
            'noise_std': self.noise_std,
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'uncertainty_multipliers': self.uncertainty_multipliers.copy(),
            'posterior_means': self.posterior_means.copy(),
            'arm_pull_counts': [len(history) for history in self.reward_history]
        })
        return info


class EpsilonGreedyLinear(ContextualPolicy):
    """
    Epsilon-Greedy policy with linear models and bias term.
    """
    
    def __init__(self, n_arms: int, context_dim: int, 
                 epsilon: float = 0.1, learning_rate: float = 0.01,
                 seed: Optional[int] = None):
        """
        Initialize Epsilon-Greedy with linear models.
        
        Args:
            n_arms: Number of arms
            context_dim: Dimension of context vectors
            epsilon: Exploration probability
            learning_rate: Learning rate for gradient descent
            seed: Random seed
        """
        super().__init__(n_arms, context_dim)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        # Linear parameters for each arm: [bias, weight_1, weight_2, ..., weight_d]
        self.param_dim = context_dim + 1
        self.parameters = np.zeros((n_arms, self.param_dim))
        
        if seed is not None:
            np.random.seed(seed)
    
    def _add_bias_term(self, context: np.ndarray) -> np.ndarray:
        """Add bias term to context vector."""
        return np.concatenate([[1.0], context])
    
    def select_action(self, context: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best predicted action
            context_with_bias = self._add_bias_term(context)
            predicted_rewards = [np.dot(context_with_bias, self.parameters[arm]) 
                               for arm in range(self.n_arms)]
            return np.argmax(predicted_rewards)
    
    def _update_impl(self, context: np.ndarray, action: int, reward: float):
        """Update parameters using gradient descent."""
        context_with_bias = self._add_bias_term(context)
        
        # Compute prediction error
        predicted_reward = np.dot(context_with_bias, self.parameters[action])
        error = reward - predicted_reward
        
        # Gradient descent update
        gradient = -error * context_with_bias
        self.parameters[action] -= self.learning_rate * gradient
    
    def _reset_impl(self):
        """Reset parameters."""
        self.parameters = np.zeros((self.n_arms, self.param_dim))
    
    def get_expected_reward(self, context: np.ndarray, arm: int) -> float:
        """Get expected reward for an arm."""
        context_with_bias = self._add_bias_term(context)
        return np.dot(context_with_bias, self.parameters[arm])
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info.update({
            'policy_type': 'EpsilonGreedyLinear',
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'parameters': self.parameters.copy()
        })
        return info


class RandomContextualPolicy(ContextualPolicy):
    """Random policy for contextual bandits."""
    
    def select_action(self, context: np.ndarray) -> int:
        """Select random action."""
        return np.random.randint(self.n_arms)
    
    def _update_impl(self, context: np.ndarray, action: int, reward: float):
        """No update needed for random policy."""
        pass
    
    def _reset_impl(self):
        """No reset needed for random policy."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get policy information."""
        info = super().get_info()
        info['policy_type'] = 'Random'
        return info


class NeuralNetworkContextualPolicy(ContextualPolicy):
    """Neural network contextual bandit policy using PyTorch."""
    
    def __init__(self, n_arms: int, context_dim: int, 
                 hidden_dims: list = [64, 32], learning_rate: float = 0.001,
                 exploration_rate: float = 0.1, batch_size: int = 32,
                 buffer_size: int = 1000, update_frequency: int = 10,
                 seed: Optional[int] = None):
        super().__init__(n_arms, context_dim)
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.step_count = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.reset()
    
    def reset(self):
        """Reset the policy to initial state."""
        super().reset()  # Call parent reset to reset step_count
        self._reset_impl()
    
    def _reset_impl(self):
        """Implementation of the reset method."""
        # Initialize neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.replay_buffer = []
        self.action_counts = np.zeros(self.n_arms)
        self.prediction_errors = []
        
        # Thompson Sampling parameters
        self.posterior_means = {}
        self.posterior_covs = {}
        self._initialize_posteriors()
    
    def _build_network(self) -> nn.Module:
        """Build the neural network architecture."""
        layers = []
        input_dim = self.context_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer (one output per arm)
        layers.append(nn.Linear(input_dim, self.n_arms))
        
        return nn.Sequential(*layers)
    
    def _initialize_posteriors(self):
        """Initialize posterior distributions for Thompson Sampling."""
        for arm in range(self.n_arms):
            # Initialize with prior
            self.posterior_means[arm] = torch.zeros(self.context_dim + 1).to(self.device)
            self.posterior_covs[arm] = torch.eye(self.context_dim + 1).to(self.device)
    
    def _add_bias_term(self, context: np.ndarray) -> np.ndarray:
        """Add bias term to context."""
        return np.concatenate([[1.0], context])
    
    def select_action(self, context: np.ndarray) -> int:
        """Select action using neural network with exploration."""
        if np.random.random() < self.exploration_rate:
            # Exploration: random action
            return np.random.randint(0, self.n_arms)
        else:
            # Exploitation: best predicted action
            with torch.no_grad():
                context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
                q_values = self.network(context_tensor)
                return q_values.argmax().item()
    
    def _update_impl(self, context: np.ndarray, action: int, reward: float):
        """Update the neural network and posterior distributions."""
        # Store experience in replay buffer
        self.replay_buffer.append((context.copy(), action, reward))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        
        # Update action counts
        self.action_counts[action] += 1
        
        # Update neural network periodically
        self.step_count += 1
        if self.step_count % self.update_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
            self._update_network()
        
        # Update posterior distributions for Thompson Sampling
        self._update_posterior(context, action, reward)
    
    def _update_network(self):
        """Update neural network using experience replay."""
        # Sample batch from replay buffer
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        contexts, actions, rewards = zip(*batch)
        contexts = torch.FloatTensor(np.array(contexts)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Forward pass
        q_values = self.network(contexts)
        predicted_rewards = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute loss
        loss = self.criterion(predicted_rewards, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if self.step_count % 100 == 0:
            self.target_network.load_state_dict(self.network.state_dict())
    
    def _update_posterior(self, context: np.ndarray, action: int, reward: float):
        """Update posterior distribution for Thompson Sampling."""
        context_with_bias = self._add_bias_term(context)
        context_tensor = torch.FloatTensor(context_with_bias).to(self.device)
        
        # Get current prediction from neural network
        with torch.no_grad():
            context_input = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            predicted_reward = self.network(context_input)[0, action].item()
        
        prediction_error = reward - predicted_reward
        self.prediction_errors.append(abs(prediction_error))
        
        # Bayesian update for posterior
        mean = self.posterior_means[action]
        cov = self.posterior_covs[action]
        
        # Compute update terms
        cov_context = cov @ context_tensor
        denominator = context_tensor @ cov_context + 1.0  # noise_std = 1.0
        
        # Update mean
        self.posterior_means[action] = mean + (prediction_error / denominator) * cov_context
        
        # Update covariance
        outer_product = torch.outer(cov_context, cov_context)
        self.posterior_covs[action] = cov - outer_product / denominator
        
        # Ensure numerical stability
        self.posterior_covs[action] = (self.posterior_covs[action] + self.posterior_covs[action].T) / 2
    
    def get_expected_reward(self, context: np.ndarray, arm: int) -> float:
        """Get expected reward for an arm given context."""
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            q_values = self.network(context_tensor)
            return q_values[0, arm].item()
    
    def get_uncertainty(self, context: np.ndarray, arm: int) -> float:
        """Get uncertainty for an arm given context."""
        context_with_bias = self._add_bias_term(context)
        context_tensor = torch.FloatTensor(context_with_bias).to(self.device)
        cov = self.posterior_covs[arm]
        return torch.sqrt(context_tensor @ cov @ context_tensor).item()
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "policy_type": "Neural Network Contextual Bandit",
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "arm_pull_counts": self.action_counts.tolist(),
            "avg_prediction_error": np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            "replay_buffer_size": len(self.replay_buffer),
            "step_count": self.step_count,
            "device": str(self.device)
        }


class ImprovedNeuralNetworkContextualPolicy(ContextualPolicy):
    """Improved neural network contextual bandit policy with better learning strategies."""
    
    def __init__(self, n_arms: int, context_dim: int, 
                 hidden_dims: list = [128, 64], learning_rate: float = 0.0005,
                 exploration_rate: float = 0.1, batch_size: int = 64,
                 buffer_size: int = 2000, update_frequency: int = 5,
                 target_update_frequency: int = 50, seed: Optional[int] = None):
        super().__init__(n_arms, context_dim)
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.step_count = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.reset()
    
    def reset(self):
        """Reset the policy to initial state."""
        super().reset()  # Call parent reset to reset step_count
        self._reset_impl()
    
    def _reset_impl(self):
        """Implementation of the reset method."""
        # Initialize neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer with better settings
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.replay_buffer = []
        self.action_counts = np.zeros(self.n_arms)
        self.prediction_errors = []
        self.training_losses = []
        
        # Thompson Sampling parameters
        self.posterior_means = {}
        self.posterior_covs = {}
        self._initialize_posteriors()
    
    def _build_network(self) -> nn.Module:
        """Build an improved neural network architecture."""
        layers = []
        input_dim = self.context_dim
        
        # Hidden layers with better initialization
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Add batch normalization
                nn.ReLU(),
                nn.Dropout(0.2)  # Slightly higher dropout
            ])
            input_dim = hidden_dim
        
        # Output layer (one output per arm)
        layers.append(nn.Linear(input_dim, self.n_arms))
        
        network = nn.Sequential(*layers)
        
        # Initialize weights with Xavier/Glorot initialization
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        return network
    
    def _initialize_posteriors(self):
        """Initialize posterior distributions for Thompson Sampling."""
        for arm in range(self.n_arms):
            # Initialize with prior
            self.posterior_means[arm] = torch.zeros(self.context_dim + 1).to(self.device)
            self.posterior_covs[arm] = torch.eye(self.context_dim + 1).to(self.device)
    
    def _add_bias_term(self, context: np.ndarray) -> np.ndarray:
        """Add bias term to context."""
        return np.concatenate([[1.0], context])
    
    def select_action(self, context: np.ndarray) -> int:
        """Select action using neural network with exploration."""
        if np.random.random() < self.exploration_rate:
            # Exploration: random action
            return np.random.randint(0, self.n_arms)
        else:
            # Exploitation: best predicted action
            with torch.no_grad():
                context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
                q_values = self.network(context_tensor)
                return q_values.argmax().item()
    
    def _update_impl(self, context: np.ndarray, action: int, reward: float):
        """Update the neural network and posterior distributions."""
        # Store experience in replay buffer
        self.replay_buffer.append((context.copy(), action, reward))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        
        # Update action counts
        self.action_counts[action] += 1
        
        # Update neural network more frequently
        self.step_count += 1
        if self.step_count % self.update_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
            self._update_network()
        
        # Update posterior distributions for Thompson Sampling
        self._update_posterior(context, action, reward)
    
    def _update_network(self):
        """Update neural network using experience replay with improvements."""
        # Sample batch from replay buffer
        batch_indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        contexts, actions, rewards = zip(*batch)
        contexts = torch.FloatTensor(np.array(contexts)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Forward pass
        q_values = self.network(contexts)
        predicted_rewards = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute loss
        loss = self.criterion(predicted_rewards, rewards)
        self.training_losses.append(loss.item())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network more frequently
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.network.state_dict())
    
    def _update_posterior(self, context: np.ndarray, action: int, reward: float):
        """Update posterior distribution for Thompson Sampling."""
        context_with_bias = self._add_bias_term(context)
        context_tensor = torch.FloatTensor(context_with_bias).to(self.device)
        
        # Get current prediction from neural network
        with torch.no_grad():
            context_input = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            predicted_reward = self.network(context_input)[0, action].item()
        
        prediction_error = reward - predicted_reward
        self.prediction_errors.append(abs(prediction_error))
        
        # Bayesian update for posterior
        mean = self.posterior_means[action]
        cov = self.posterior_covs[action]
        
        # Compute update terms
        cov_context = cov @ context_tensor
        denominator = context_tensor @ cov_context + 1.0  # noise_std = 1.0
        
        # Update mean
        self.posterior_means[action] = mean + (prediction_error / denominator) * cov_context
        
        # Update covariance
        outer_product = torch.outer(cov_context, cov_context)
        self.posterior_covs[action] = cov - outer_product / denominator
        
        # Ensure numerical stability
        self.posterior_covs[action] = (self.posterior_covs[action] + self.posterior_covs[action].T) / 2
    
    def get_expected_reward(self, context: np.ndarray, arm: int) -> float:
        """Get expected reward for an arm given context."""
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            q_values = self.network(context_tensor)
            return q_values[0, arm].item()
    
    def get_uncertainty(self, context: np.ndarray, arm: int) -> float:
        """Get uncertainty for an arm given context."""
        context_with_bias = self._add_bias_term(context)
        context_tensor = torch.FloatTensor(context_with_bias).to(self.device)
        cov = self.posterior_covs[arm]
        return torch.sqrt(context_tensor @ cov @ context_tensor).item()
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "policy_type": "Improved Neural Network Contextual Bandit",
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "arm_pull_counts": self.action_counts.tolist(),
            "avg_prediction_error": np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            "avg_training_loss": np.mean(self.training_losses) if self.training_losses else 0.0,
            "replay_buffer_size": len(self.replay_buffer),
            "step_count": self.step_count,
            "device": str(self.device)
        }


class ContextlessEpsilonGreedy(ContextualPolicy):
    """
    Epsilon-Greedy policy that ignores the context vector and learns a mean reward for each arm.
    """
    def __init__(self, n_arms: int, context_dim: int, epsilon: float = 0.1, seed: Optional[int] = None):
        super().__init__(n_arms, context_dim)
        self.epsilon = epsilon
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self._reset_impl()

    def select_action(self, context: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return int(np.argmax(self.arm_means))

    def _update_impl(self, context: np.ndarray, action: int, reward: float):
        # Incremental mean update
        self.arm_counts[action] += 1
        n = self.arm_counts[action]
        old_mean = self.arm_means[action]
        self.arm_means[action] += (reward - old_mean) / n

    def _reset_impl(self):
        self.arm_means = np.zeros(self.n_arms)
        self.arm_counts = np.zeros(self.n_arms, dtype=int)

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            'policy_type': 'ContextlessEpsilonGreedy',
            'epsilon': self.epsilon,
            'arm_means': self.arm_means.copy(),
            'arm_counts': self.arm_counts.copy(),
        })
        return info


# Example usage and testing
if __name__ == "__main__":
    from contextual_mab_environment import ContextualMultiArmedBandit
    
    # Create a test contextual bandit
    bandit = ContextualMultiArmedBandit.create_linear_bandit(n_arms=3, context_dim=2, seed=42)
    
    # Test Linear Thompson Sampling
    print("Testing Linear Thompson Sampling")
    print("="*50)
    
    policy = LinearThompsonSampling(n_arms=3, context_dim=2, seed=123)
    total_reward = 0
    
    for step in range(100):
        context = bandit.generate_context()
        action = policy.select_action(context)
        reward, _ = bandit.pull_arm(action, context)
        policy.update(context, action, reward)
        total_reward += reward
        
        if (step + 1) % 20 == 0:
            avg_reward = total_reward / (step + 1)
            info = policy.get_info()
            print(f"Step {step + 1}: Avg reward = {avg_reward:.3f}, "
                  f"Avg prediction error = {info['avg_prediction_error']:.3f}")
    
    print(f"Final total reward: {total_reward:.3f}")
    print(f"Policy info: {policy.get_info()}") 

