#!/usr/bin/env python3
"""
Experiment script for testing contextual bandit algorithms.
"""

import numpy as np
import argparse
from contextual_mab_environment import ContextualMultiArmedBandit, ContextualArm, LinearNormalDistribution, CustomDistribution
from contextual_policies import LinearThompsonSampling, EpsilonGreedyLinear, RandomContextualPolicy, NeuralNetworkContextualPolicy, ImprovedNeuralNetworkContextualPolicy, ContextlessEpsilonGreedy
from colorama import init, Fore, Style
import matplotlib.pyplot as plt
from typing import List, Dict, Any

init(autoreset=True)


def create_experiment_bandit(n_arms: int = 5, context_dim: int = 5, seed: int = 42):
    """Create a contextual bandit for experiments."""
    print(f"{Fore.CYAN}Creating experiment bandit: {n_arms} arms, {context_dim}D context")
    
    # Create arms with different weight patterns
    arms = []
    for i in range(n_arms):
        # Different weight patterns for each arm
        if i == 0:
            # Arm 0: Strong bias, weak context dependence
            weights = np.array([2.0] + [0.1] * context_dim)
        elif i == 1:
            # Arm 1: Moderate bias, strong feature 1 dependence
            weights = np.array([1.0] + [0.0, 1.5] + [0.1] * (context_dim - 2))
        elif i == 2:
            # Arm 2: Low bias, balanced context dependence
            weights = np.array([0.5] + [0.8] * context_dim)
        elif i == 3:
            # Arm 3: Negative bias, feature 3 dependence
            weights = np.array([-0.5] + [0.1, 0.1, 0.1, 1.2] + [0.1] * max(0, context_dim - 4))
        else:
            # Arm 4+: Random pattern
            np.random.seed(seed + i)
            weights = np.random.normal(0, 0.5, context_dim + 1)
            weights[0] = np.random.normal(0, 1)  # Bias term
        
        # Create distribution with varying noise
        std = 0.2 + 0.1 * i
        distribution = LinearNormalDistribution(context_dim, weights[1:], std)
        arm = ContextualArm(f"Arm_{i}", distribution)
        arms.append(arm)
        
        print(f"{Fore.GREEN}  Arm {i}: bias={weights[0]:.2f}, weights={weights[1:context_dim+1]}, std={std:.2f}")
    
    return ContextualMultiArmedBandit(arms, seed=seed)


def create_nonlinear_experiment_bandit(n_arms: int = 5, context_dim: int = 5, seed: int = 123):
    """Create a contextual bandit with at least one nonlinear arm."""
    np.random.seed(seed)
    arms = []
    for i in range(n_arms):
        if i == 0:
            # Arm 0: Linear
            weights = np.array([0.5] * context_dim)
            dist = LinearNormalDistribution(context_dim, weights, std=0.3)
        elif i == 1:
            # Arm 1: Quadratic (nonlinear)
            def quad_func(context):
                return 1.0 + 0.5 * np.sum(context ** 2) + np.random.normal(0, 0.3)
            dist = CustomDistribution(context_dim, quad_func)
        elif i == 2:
            # Arm 2: Sinusoidal (nonlinear)
            def sin_func(context):
                return np.sin(np.sum(context)) + np.random.normal(0, 0.3)
            dist = CustomDistribution(context_dim, sin_func)
        elif i == 3:
            # Arm 3: Linear with different weights
            weights = np.array([-0.5] * context_dim)
            dist = LinearNormalDistribution(context_dim, weights, std=0.3)
        else:
            # Arm 4: Random nonlinear
            def rand_func(context):
                return np.cos(context[0]) + 0.2 * np.sum(context) + np.random.normal(0, 0.3)
            dist = CustomDistribution(context_dim, rand_func)
        arm = ContextualArm(f"Arm_{i}", dist)
        arms.append(arm)
    return ContextualMultiArmedBandit(arms, seed=seed)


def run_single_policy_experiment(bandit, policy, n_iterations: int = 1000, name: str = "Policy"):
    """Run experiment with a single policy."""
    print(f"\n{Fore.BLUE}Running {name} for {n_iterations} iterations...")
    
    bandit.reset()
    policy.reset()
    
    total_reward = 0
    rewards = []
    actions = []
    contexts = []
    optimal_rewards = []
    
    for step in range(n_iterations):
        context = bandit.generate_context()
        action = policy.select_action(context)
        reward, _ = bandit.pull_arm(action, context)
        
        # Get optimal reward for regret calculation
        optimal_reward = bandit.get_optimal_reward(context)
        
        # Store data
        total_reward += reward
        rewards.append(reward)
        actions.append(action)
        contexts.append(context.copy())
        optimal_rewards.append(optimal_reward)
        
        # Update policy
        policy.update(context, action, reward)
        
        # Progress report
        if (step + 1) % 200 == 0:
            avg_reward = total_reward / (step + 1)
            info = policy.get_info()
            print(f"{Fore.YELLOW}Step {step + 1}: Avg reward = {avg_reward:.3f}")
            if 'avg_prediction_error' in info:
                print(f"{Fore.YELLOW}  Avg prediction error = {info['avg_prediction_error']:.3f}")
    
    # Calculate metrics
    cumulative_rewards = np.cumsum(rewards)
    cumulative_optimal = np.cumsum(optimal_rewards)
    regret = cumulative_optimal - cumulative_rewards
    
    return {
        'name': name,
        'total_reward': total_reward,
        'avg_reward': total_reward / n_iterations,
        'final_regret': regret[-1],
        'rewards': rewards,
        'actions': actions,
        'contexts': contexts,
        'optimal_rewards': optimal_rewards,
        'cumulative_rewards': cumulative_rewards,
        'regret': regret,
        'policy_info': policy.get_info()
    }


def run_policy_comparison_experiment(n_iterations: int = 2000):
    """Run comparison experiment with multiple policies."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}CONTEXTUAL BANDIT POLICY COMPARISON")
    print(f"{Fore.CYAN}{'='*70}")
    
    # Create bandit
    bandit = create_experiment_bandit(n_arms=5, context_dim=5, seed=42)
    
    # Define policies
    policies = [
        ("Random", RandomContextualPolicy(n_arms=5, context_dim=5, seed=123)),
        ("Epsilon-Greedy (ε=0.1)", EpsilonGreedyLinear(n_arms=5, context_dim=5, epsilon=0.1, seed=123)),
        ("Epsilon-Greedy (ε=0.05)", EpsilonGreedyLinear(n_arms=5, context_dim=5, epsilon=0.05, seed=123)),
        ("Linear Thompson Sampling", LinearThompsonSampling(n_arms=5, context_dim=5, seed=123)),
        ("Linear TS (High Prior)", LinearThompsonSampling(n_arms=5, context_dim=5, prior_mean=1.0, seed=123)),
        ("Neural Network (ε=0.1)", NeuralNetworkContextualPolicy(n_arms=5, context_dim=5, exploration_rate=0.1, seed=123)),
        ("Neural Network (ε=0.05)", NeuralNetworkContextualPolicy(n_arms=5, context_dim=5, exploration_rate=0.05, seed=123)),
    ]
    
    # Run experiments
    results = []
    for name, policy in policies:
        result = run_single_policy_experiment(bandit, policy, n_iterations, name)
        results.append(result)
    
    # Print comparison table
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}RESULTS COMPARISON")
    print(f"{Fore.CYAN}{'='*70}")
    print(f"{Fore.MAGENTA}{'Policy':<30} {'Total Reward':<15} {'Avg Reward':<15} {'Final Regret':<15}")
    print(f"{Fore.MAGENTA}{'-' * 75}")
    
    for result in results:
        print(f"{result['name']:<30} {result['total_reward']:<15.3f} "
              f"{result['avg_reward']:<15.3f} {result['final_regret']:<15.3f}")
    
    # Plot results
    plot_comparison_results(results, n_iterations)
    
    return results


def plot_comparison_results(results: List[Dict[str, Any]], n_iterations: int):
    """Plot comparison results."""
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Cumulative rewards
    plt.subplot(2, 4, 1)
    for result in results:
        plt.plot(result['cumulative_rewards'], label=result['name'], alpha=0.8)
    plt.xlabel('Time step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Average rewards
    plt.subplot(2, 4, 2)
    for result in results:
        avg_rewards = result['cumulative_rewards'] / (np.arange(n_iterations) + 1)
        plt.plot(avg_rewards, label=result['name'], alpha=0.8)
    plt.xlabel('Time step')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Regret
    plt.subplot(2, 4, 3)
    for result in results:
        plt.plot(result['regret'], label=result['name'], alpha=0.8)
    plt.xlabel('Time step')
    plt.ylabel('Regret')
    plt.title('Regret')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Action distribution
    plt.subplot(2, 4, 4)
    action_counts = []
    labels = []
    for result in results:
        if result['name'] != 'Random':  # Skip random for clarity
            counts = np.bincount(result['actions'], minlength=5)
            action_counts.append(counts)
            labels.append(result['name'])
    
    if action_counts:
        action_counts = np.array(action_counts)
        x = np.arange(5)
        width = 0.8 / len(action_counts)
        
        for i, (counts, label) in enumerate(zip(action_counts, labels)):
            plt.bar(x + i * width, counts, width, label=label, alpha=0.8)
        
        plt.xlabel('Arm')
        plt.ylabel('Number of pulls')
        plt.title('Action Distribution')
        plt.xticks(x + width * (len(action_counts) - 1) / 2, [f'Arm {i}' for i in range(5)])
        plt.legend(fontsize=8)
    
    # Plot 5: Context visualization (last 100 contexts)
    plt.subplot(2, 4, 5)
    if results:
        recent_contexts = np.array(results[0]['contexts'][-100:])
        plt.imshow(recent_contexts.T, aspect='auto', cmap='viridis')
        plt.xlabel('Time step (last 100)')
        plt.ylabel('Context dimension')
        plt.title('Recent Contexts')
        plt.colorbar()
    
    # Plot 6: Policy-specific metrics
    plt.subplot(2, 4, 6)
    policy_names = []
    avg_errors = []
    
    for result in results:
        if 'avg_prediction_error' in result['policy_info']:
            policy_names.append(result['name'])
            avg_errors.append(result['policy_info']['avg_prediction_error'])
    
    if policy_names:
        plt.bar(range(len(policy_names)), avg_errors, alpha=0.8)
        plt.xlabel('Policy')
        plt.ylabel('Avg Prediction Error')
        plt.title('Prediction Error Comparison')
        plt.xticks(range(len(policy_names)), policy_names, rotation=45, ha='right')
    
    # Plot 7: Neural Network specific metrics
    plt.subplot(2, 4, 7)
    nn_policies = [r for r in results if 'Neural Network' in r['name']]
    if nn_policies:
        replay_sizes = []
        nn_names = []
        for result in nn_policies:
            if 'replay_buffer_size' in result['policy_info']:
                replay_sizes.append(result['policy_info']['replay_buffer_size'])
                nn_names.append(result['name'])
        
        if replay_sizes:
            plt.bar(range(len(nn_names)), replay_sizes, alpha=0.8)
            plt.xlabel('Neural Network Policy')
            plt.ylabel('Replay Buffer Size')
            plt.title('NN Replay Buffer Usage')
            plt.xticks(range(len(nn_names)), nn_names, rotation=45, ha='right')
    
    # Plot 8: Learning curves comparison
    plt.subplot(2, 4, 8)
    # Show learning curves for different policy types
    policy_types = ['Linear Thompson Sampling', 'Neural Network']
    colors = ['blue', 'red']
    
    for policy_type, color in zip(policy_types, colors):
        matching_results = [r for r in results if policy_type in r['name']]
        if matching_results:
            result = matching_results[0]  # Take first one
            avg_rewards = result['cumulative_rewards'] / (np.arange(n_iterations) + 1)
            plt.plot(avg_rewards, label=policy_type, color=color, alpha=0.8)
    
    plt.xlabel('Time step')
    plt.ylabel('Average Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_thompson_sampling_analysis(n_iterations: int = 1000):
    """Detailed analysis of Thompson Sampling behavior."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}THOMPSON SAMPLING DETAILED ANALYSIS")
    print(f"{Fore.CYAN}{'='*70}")
    
    # Create bandit
    bandit = create_experiment_bandit(n_arms=5, context_dim=5, seed=42)
    
    # Run Thompson Sampling
    policy = LinearThompsonSampling(n_arms=5, context_dim=5, seed=123)
    result = run_single_policy_experiment(bandit, policy, n_iterations, "Thompson Sampling")
    
    # Analyze posterior parameters
    print(f"\n{Fore.BLUE}Posterior Analysis:")
    info = result['policy_info']
    posterior_means = info['posterior_means']
    arm_pull_counts = info['arm_pull_counts']
    
    print(f"{Fore.MAGENTA}{'Arm':<8} {'Pulls':<8} {'Bias':<10} {'Weights':<30}")
    print(f"{Fore.MAGENTA}{'-' * 60}")
    
    for i in range(5):
        bias = posterior_means[i][0]
        weights = posterior_means[i][1:6]  # First 5 weights
        weights_str = f"[{', '.join([f'{w:.2f}' for w in weights])}]"
        print(f"{i:<8} {arm_pull_counts[i]:<8} {bias:<10.3f} {weights_str:<30}")
    
    # Plot Thompson Sampling specific visualizations
    plot_thompson_analysis(result, policy)
    
    return result


def plot_thompson_analysis(result: Dict[str, Any], policy: LinearThompsonSampling):
    """Plot Thompson Sampling specific analysis."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Uncertainty over time
    plt.subplot(2, 3, 1)
    uncertainties = []
    for i, context in enumerate(result['contexts'][::10]):  # Sample every 10th context
        arm_uncertainties = [policy.get_uncertainty(context, arm) for arm in range(5)]
        uncertainties.append(arm_uncertainties)
    
    uncertainties = np.array(uncertainties)
    for arm in range(5):
        plt.plot(uncertainties[:, arm], label=f'Arm {arm}', alpha=0.8)
    
    plt.xlabel('Time step (x10)')
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Expected rewards vs actual rewards
    plt.subplot(2, 3, 2)
    expected_rewards = []
    for i, context in enumerate(result['contexts'][::10]):
        arm_expected = [policy.get_expected_reward(context, arm) for arm in range(5)]
        expected_rewards.append(arm_expected)
    
    expected_rewards = np.array(expected_rewards)
    actual_rewards = np.array(result['rewards'][::10])
    
    plt.scatter(expected_rewards.flatten(), actual_rewards, alpha=0.6)
    plt.plot([expected_rewards.min(), expected_rewards.max()], 
             [expected_rewards.min(), expected_rewards.max()], 'r--', alpha=0.8)
    plt.xlabel('Expected Reward')
    plt.ylabel('Actual Reward')
    plt.title('Expected vs Actual Rewards')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty multipliers
    plt.subplot(2, 3, 3)
    info = result['policy_info']
    multipliers = info['uncertainty_multipliers']
    plt.bar(range(5), multipliers, alpha=0.8)
    plt.xlabel('Arm')
    plt.ylabel('Uncertainty Multiplier')
    plt.title('Adaptive Uncertainty Multipliers')
    plt.xticks(range(5), [f'Arm {i}' for i in range(5)])
    
    # Plot 4: Prediction errors over time
    plt.subplot(2, 3, 4)
    if 'prediction_errors' in info:
        errors = info.get('prediction_errors', [])
        if errors:
            plt.plot(errors, alpha=0.8)
            plt.xlabel('Time step')
            plt.ylabel('Prediction Error')
            plt.title('Prediction Errors Over Time')
            plt.grid(True, alpha=0.3)
    
    # Plot 5: Posterior means evolution (bias terms)
    plt.subplot(2, 3, 5)
    bias_evolution = []
    for i in range(0, len(result['contexts']), 50):  # Sample every 50th step
        if i < len(result['contexts']):
            # This would require storing posterior means over time
            # For now, just show final values
            pass
    
    # Plot 6: Action selection frequency
    plt.subplot(2, 3, 6)
    action_counts = np.bincount(result['actions'], minlength=5)
    plt.bar(range(5), action_counts, alpha=0.8)
    plt.xlabel('Arm')
    plt.ylabel('Number of selections')
    plt.title('Action Selection Frequency')
    plt.xticks(range(5), [f'Arm {i}' for i in range(5)])
    
    plt.tight_layout()
    plt.show()


def train_and_watch_experiment(n_train: int = 2000, n_watch: int = 20):
    """Train a contextual bandit model, then watch it make decisions step by step."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}TRAIN AND WATCH CONTEXTUAL BANDIT")
    print(f"{Fore.CYAN}{'='*70}")
    
    # Create bandit and policy (now with neural network option)
    bandit = create_experiment_bandit(n_arms=5, context_dim=5, seed=42)
    
    # Let user choose policy type
    print(f"{Fore.BLUE}Choose policy type:")
    print("1. Linear Thompson Sampling")
    print("2. Neural Network Contextual Bandit")
    
    while True:
        try:
            choice = input(f"{Fore.YELLOW}Enter choice (1-2): {Style.RESET_ALL}").strip()
            if choice == "1":
                policy = LinearThompsonSampling(n_arms=5, context_dim=5, seed=123)
                policy_name = "Linear Thompson Sampling"
                break
            elif choice == "2":
                policy = NeuralNetworkContextualPolicy(n_arms=5, context_dim=5, seed=123)
                policy_name = "Neural Network Contextual Bandit"
                break
            else:
                print(f"{Fore.RED}Please enter 1 or 2")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Goodbye!")
            return
    
    # Show model/distribution type for each arm
    print(f"{Fore.CYAN}Model/distribution type for each arm:")
    for i, arm in enumerate(bandit.arms):
        dist_type = type(arm.distribution).__name__
        print(f"  Arm {i}: {dist_type}")
    
    print(f"{Fore.CYAN}Policy type: {policy_name}")

    # Training phase
    print(f"{Fore.BLUE}Training for {n_train} steps...")
    bandit.reset()
    policy.reset()
    for step in range(n_train):
        context = bandit.generate_context()
        action = policy.select_action(context)
        reward, _ = bandit.pull_arm(action, context)
        policy.update(context, action, reward)
        if (step + 1) % 500 == 0:
            print(f"{Fore.YELLOW}Training step {step + 1}/{n_train}")
    print(f"{Fore.GREEN}Training complete!\n")
    
    # Watch phase
    print(f"{Fore.BLUE}Watch mode: Model will make decisions on new contexts.")
    print(f"{Fore.BLUE}Showing {n_watch} steps. Press Enter to step, or 'q' to quit.")
    
    for step in range(n_watch):
        context = bandit.generate_context()
        action = policy.select_action(context)
        expected_rewards = [policy.get_expected_reward(context, arm) for arm in range(5)]
        uncertainty = [policy.get_uncertainty(context, arm) for arm in range(5)]
        optimal_arm = bandit.get_optimal_arm(context)
        optimal_reward = bandit.get_optimal_reward(context)
        reward, _ = bandit.pull_arm(action, context)
        
        print(f"\n{Fore.CYAN}Step {step + 1}")
        print(f"{Fore.YELLOW}Context: {context}")
        print(f"{Fore.GREEN}Model chose arm: {action}")
        print(f"{Fore.GREEN}  Expected reward: {expected_rewards[action]:.3f} (uncertainty: {uncertainty[action]:.3f})")
        print(f"{Fore.GREEN}  Actual reward: {reward:.3f}")
        print(f"{Fore.MAGENTA}Optimal arm: {optimal_arm} (expected: {optimal_reward:.3f})")
        print(f"{Fore.MAGENTA}Expected rewards for all arms:")
        for i in range(5):
            marker = " ← CHOSEN" if i == action else (" ← OPTIMAL" if i == optimal_arm else "")
            print(f"  Arm {i}: {expected_rewards[i]:.3f} (uncertainty: {uncertainty[i]:.3f}){marker}")
        
        user_input = input(f"{Fore.BLUE}Press Enter to continue, or 'q' to quit: {Style.RESET_ALL}").strip()
        if user_input.lower() == 'q':
            print(f"{Fore.YELLOW}Exiting watch mode.")
            break


def run_long_training_comparison(n_iterations: int = 5000):
    """Run a longer experiment to compare Linear TS vs Neural Network over time."""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}LONG-TERM TRAINING COMPARISON: LINEAR TS vs NEURAL NETWORK")
    print(f"{Fore.CYAN}{'='*70}")
    
    # Create bandit
    bandit = create_experiment_bandit(n_arms=5, context_dim=5, seed=42)
    
    # Define policies for comparison
    policies = [
        ("Linear Thompson Sampling", LinearThompsonSampling(n_arms=5, context_dim=5, seed=123)),
        ("Neural Network (ε=0.05)", NeuralNetworkContextualPolicy(n_arms=5, context_dim=5, exploration_rate=0.05, seed=123)),
        ("Improved Neural Network", ImprovedNeuralNetworkContextualPolicy(n_arms=5, context_dim=5, exploration_rate=0.05, seed=123)),
    ]
    
    # Run experiments
    results = []
    for name, policy in policies:
        print(f"\n{Fore.BLUE}Running {name} for {n_iterations} iterations...")
        result = run_single_policy_experiment(bandit, policy, n_iterations, name)
        results.append(result)
    
    # Print comparison table
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}LONG-TERM RESULTS COMPARISON")
    print(f"{Fore.CYAN}{'='*70}")
    print(f"{Fore.MAGENTA}{'Policy':<30} {'Total Reward':<15} {'Avg Reward':<15} {'Final Regret':<15}")
    print(f"{Fore.MAGENTA}{'-' * 75}")
    
    for result in results:
        print(f"{result['name']:<30} {result['total_reward']:<15.3f} "
              f"{result['avg_reward']:<15.3f} {result['final_regret']:<15.3f}")
    
    # Plot long-term comparison
    plot_long_term_comparison(results, n_iterations)
    
    return results


def plot_long_term_comparison(results: List[Dict[str, Any]], n_iterations: int):
    """Plot long-term comparison between Linear TS and Neural Network."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Cumulative rewards over time
    plt.subplot(2, 3, 1)
    for result in results:
        plt.plot(result['cumulative_rewards'], label=result['name'], alpha=0.8, linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Cumulative Reward')
    plt.title('Long-term Cumulative Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Average rewards over time
    plt.subplot(2, 3, 2)
    for result in results:
        avg_rewards = result['cumulative_rewards'] / (np.arange(n_iterations) + 1)
        plt.plot(avg_rewards, label=result['name'], alpha=0.8, linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Average Reward')
    plt.title('Long-term Average Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Regret comparison
    plt.subplot(2, 3, 3)
    for result in results:
        plt.plot(result['regret'], label=result['name'], alpha=0.8, linewidth=2)
    plt.xlabel('Time step')
    plt.ylabel('Regret')
    plt.title('Long-term Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate analysis (slope of average rewards)
    plt.subplot(2, 3, 4)
    window_size = 100
    for result in results:
        avg_rewards = result['cumulative_rewards'] / (np.arange(n_iterations) + 1)
        # Compute rolling slope
        slopes = []
        for i in range(window_size, len(avg_rewards)):
            slope = (avg_rewards[i] - avg_rewards[i-window_size]) / window_size
            slopes.append(slope)
        plt.plot(range(window_size, len(avg_rewards)), slopes, label=result['name'], alpha=0.8)
    plt.xlabel('Time step')
    plt.ylabel('Learning Rate (slope)')
    plt.title('Learning Rate Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Performance in different phases
    plt.subplot(2, 3, 5)
    phases = [
        (0, n_iterations//4, "Early (0-25%)"),
        (n_iterations//4, n_iterations//2, "Mid (25-50%)"),
        (n_iterations//2, 3*n_iterations//4, "Late (50-75%)"),
        (3*n_iterations//4, n_iterations, "Final (75-100%)")
    ]
    
    phase_performance = {result['name']: [] for result in results}
    for start, end, phase_name in phases:
        for result in results:
            phase_rewards = result['rewards'][start:end]
            avg_phase_reward = np.mean(phase_rewards) if phase_rewards else 0
            phase_performance[result['name']].append(avg_phase_reward)
    
    x = np.arange(len(phases))
    width = 0.8 / len(results)
    for i, (name, performance) in enumerate(phase_performance.items()):
        plt.bar(x + i * width, performance, width, label=name, alpha=0.8)
    
    plt.xlabel('Training Phase')
    plt.ylabel('Average Reward')
    plt.title('Performance by Training Phase')
    plt.xticks(x + width * (len(results) - 1) / 2, [phase[2] for phase in phases])
    plt.legend()
    
    # Plot 6: Neural Network specific metrics
    plt.subplot(2, 3, 6)
    nn_results = [r for r in results if 'Neural Network' in r['name']]
    if nn_results:
        for result in nn_results:
            info = result['policy_info']
            if 'replay_buffer_size' in info:
                plt.bar([result['name']], [info['replay_buffer_size']], alpha=0.8)
        plt.ylabel('Replay Buffer Size')
        plt.title('Neural Network Replay Buffer Usage')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def run_epsilon_greedy_hyperparam_experiment(n_iterations: int = 1000, seed: int = 42):
    """Compare different epsilon values for Epsilon-Greedy Linear."""
    print("\n===== Epsilon-Greedy Linear Hyperparameter Comparison =====\n")
    n_arms = 5
    context_dim = 5
    bandit = create_experiment_bandit(n_arms=n_arms, context_dim=context_dim, seed=seed)
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
    policies = [
        (EpsilonGreedyLinear(n_arms, context_dim, epsilon=eps, seed=seed), f"Epsilon-Greedy (ε={eps})")
        for eps in epsilons
    ]
    results = []
    for policy, name in policies:
        res = run_single_policy_experiment(bandit, policy, n_iterations=n_iterations, name=name)
        results.append(res)
    # Compute optimal average reward
    optimal_rewards = np.array(results[0]['optimal_rewards'])
    optimal_avg = np.cumsum(optimal_rewards) / (np.arange(len(optimal_rewards)) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res['regret'], label=res['name'])
    plt.title("Cumulative Regret (Epsilon-Greedy Linear)")
    plt.xlabel("Step")
    plt.ylabel("Regret")
    plt.legend()
    plt.subplot(1, 2, 2)
    for res in results:
        avg_reward = np.cumsum(res['rewards']) / (np.arange(len(res['rewards'])) + 1)
        plt.plot(avg_reward, label=res['name'])
    plt.plot(optimal_avg, label="Optimal Avg Reward", linestyle="--", color="black")
    plt.title("Average Reward (Epsilon-Greedy Linear)")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.suptitle("Epsilon-Greedy Linear: Hyperparameter Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def interactive_experiment():
    print("Contextual Bandit Demo Menu:")
    print("1. Context-aware vs. context-ignorant policies")
    print("2. Nonlinear context bandit demo (neural network advantage)")
    print("3. Epsilon-Greedy Linear: Hyperparameter Comparison")
    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            if choice == "1":
                n_iter = int(input("Number of iterations (default 1000): ") or "1000")
                run_context_aware_vs_ignorant_experiment(n_iter)
                break
            elif choice == "2":
                n_iter = int(input("Number of iterations (default 1000): ") or "1000")
                run_nonlinear_experiment(n_iter)
                break
            elif choice == "3":
                n_iter = int(input("Number of iterations (default 1000): ") or "1000")
                run_epsilon_greedy_hyperparam_experiment(n_iter)
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("Goodbye!")
            break


def run_context_aware_vs_ignorant_experiment(n_iterations: int = 1000, seed: int = 42):
    """Context-aware vs. context-ignorant policies experiment."""
    print("\n===== Context-Aware vs. Context-Ignorant Policies =====\n")
    n_arms = 5
    context_dim = 5
    bandit = create_experiment_bandit(n_arms=n_arms, context_dim=context_dim, seed=seed)

    policies = [
        (LinearThompsonSampling(n_arms, context_dim, seed=seed), "Linear Thompson Sampling"),
        (EpsilonGreedyLinear(n_arms, context_dim, epsilon=0.1, seed=seed), "Epsilon-Greedy Linear"),
        (NeuralNetworkContextualPolicy(n_arms, context_dim, seed=seed), "Neural Network Contextual"),
        (ContextlessEpsilonGreedy(n_arms, context_dim, epsilon=0.1, seed=seed), "Contextless Epsilon-Greedy"),
    ]

    results = []
    for policy, name in policies:
        res = run_single_policy_experiment(bandit, policy, n_iterations=n_iterations, name=name)
        results.append(res)

    # Compute optimal average reward
    optimal_rewards = np.array(results[0]['optimal_rewards'])
    optimal_avg = np.cumsum(optimal_rewards) / (np.arange(len(optimal_rewards)) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res['regret'], label=res['name'])
    plt.title("Cumulative Regret")
    plt.xlabel("Step")
    plt.ylabel("Regret")
    plt.legend()
    plt.subplot(1, 2, 2)
    for res in results:
        avg_reward = np.cumsum(res['rewards']) / (np.arange(len(res['rewards'])) + 1)
        plt.plot(avg_reward, label=res['name'])
    plt.plot(optimal_avg, label="Optimal Avg Reward", linestyle="--", color="black")
    plt.title("Average Reward")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.suptitle("Context Matters: Context-Aware vs. Context-Ignorant Bandit Algorithms")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def run_nonlinear_experiment(n_iterations: int = 1000, seed: int = 123):
    """Nonlinear context-reward relationships experiment."""
    print("\n===== Nonlinear Contextual Bandit Demo =====\n")
    n_arms = 5
    context_dim = 5
    bandit = create_nonlinear_experiment_bandit(n_arms=n_arms, context_dim=context_dim, seed=seed)

    policies = [
        (LinearThompsonSampling(n_arms, context_dim, seed=seed), "Linear Thompson Sampling"),
        (EpsilonGreedyLinear(n_arms, context_dim, epsilon=0.1, seed=seed), "Epsilon-Greedy Linear"),
        (NeuralNetworkContextualPolicy(n_arms, context_dim, seed=seed), "Neural Network Contextual"),
        (ContextlessEpsilonGreedy(n_arms, context_dim, epsilon=0.1, seed=seed), "Contextless Epsilon-Greedy"),
        (RandomContextualPolicy(n_arms, context_dim), "Random Policy"),
    ]

    results = []
    for policy, name in policies:
        res = run_single_policy_experiment(bandit, policy, n_iterations=n_iterations, name=name)
        results.append(res)

    # Compute optimal average reward
    optimal_rewards = np.array(results[0]['optimal_rewards'])
    optimal_avg = np.cumsum(optimal_rewards) / (np.arange(len(optimal_rewards)) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res['regret'], label=res['name'])
    plt.title("Cumulative Regret (Nonlinear Bandit)")
    plt.xlabel("Step")
    plt.ylabel("Regret")
    plt.legend()
    plt.subplot(1, 2, 2)
    for res in results:
        avg_reward = np.cumsum(res['rewards']) / (np.arange(len(res['rewards'])) + 1)
        plt.plot(avg_reward, label=res['name'])
    plt.plot(optimal_avg, label="Optimal Avg Reward", linestyle="--", color="black")
    plt.title("Average Reward (Nonlinear Bandit)")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.suptitle("Nonlinear Contextual Bandit: Neural Network Advantage")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Contextual Bandit Experiments')
    parser.add_argument('--experiment', '-e', type=str, 
                       choices=['comparison', 'thompson', 'interactive'],
                       default='interactive',
                       help='Experiment type: comparison, thompson, interactive')
    parser.add_argument('--iterations', '-i', type=int, default=2000,
                       help='Number of iterations (default: 2000)')
    args = parser.parse_args()
    print("Contextual Bandit Experiments")
    print(f"Number of iterations: {args.iterations}")
    if args.experiment == 'comparison':
        run_policy_comparison_experiment(args.iterations)
    elif args.experiment == 'thompson':
        run_thompson_sampling_analysis(args.iterations)
    elif args.experiment == 'interactive':
        interactive_experiment()


if __name__ == "__main__":
    main() 