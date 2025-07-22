#!/usr/bin/env python3
"""
Interactive script to play with the Contextual Multi-Armed Bandit environment.
"""

import numpy as np
import argparse
from contextual_mab_environment import (
    ContextualMultiArmedBandit, ContextualArm, 
    LinearNormalDistribution, LinearBernoulliDistribution, CustomDistribution
)
from colorama import init, Fore, Style
import matplotlib.pyplot as plt

init(autoreset=True)


def create_demo_contextual_bandit():
    """Create a demo contextual bandit with 5 arms and 5D context vectors."""
    print(f"{Fore.CYAN}Creating contextual bandit with 5 arms and 5D context vectors...")
    
    # Define interesting weight patterns for each arm
    # Each arm will respond differently to different features
    weights_patterns = [
        # Arm 0: Responds strongly to feature 0, weakly to others
        np.array([2.0, 0.1, 0.1, 0.1, 0.1]),
        
        # Arm 1: Responds to features 1 and 2
        np.array([0.1, 1.5, 1.5, 0.1, 0.1]),
        
        # Arm 2: Responds to all features equally
        np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        
        # Arm 3: Responds to features 3 and 4
        np.array([0.1, 0.1, 0.1, 1.5, 1.5]),
        
        # Arm 4: Responds to feature 4 and feature 0
        np.array([1.0, 0.1, 0.1, 0.1, 1.0])
    ]
    
    arms = []
    for i, weights in enumerate(weights_patterns):
        # Create normal distribution with different standard deviations
        std = 0.2 + 0.1 * i  # Varying noise levels
        distribution = LinearNormalDistribution(context_dim=5, weights=weights, std=std)
        arm = ContextualArm(f"Arm_{i}", distribution)
        arms.append(arm)
        print(f"{Fore.GREEN}  Arm {i}: weights = {weights}, std = {std:.2f}")
    
    return ContextualMultiArmedBandit(arms, seed=42)


def print_contextual_bandit_info(bandit):
    """Print information about the contextual bandit."""
    info = bandit.get_info()
    print(f"\n{Fore.CYAN + Style.BRIGHT}{'='*60}")
    print(f"{Fore.CYAN + Style.BRIGHT}CONTEXTUAL BANDIT INFORMATION")
    print(f"{Fore.CYAN + Style.BRIGHT}{'='*60}")
    print(f"{Fore.YELLOW}Number of arms: {info['n_arms']}")
    print(f"{Fore.YELLOW}Context dimension: {info['context_dim']}")
    print(f"{Fore.YELLOW}Total pulls: {info['step_count']}")
    
    print(f"\n{Fore.MAGENTA}Arm details:")
    for i, arm_info in enumerate(info['arms']):
        print(f"  {Fore.MAGENTA}Arm {i} ({arm_info['name']}): {arm_info['distribution_type']}, pulls={arm_info['pull_count']}")


def interactive_play():
    """Interactive mode to play with the contextual bandit."""
    print(f"{Fore.GREEN}Welcome to the Contextual Multi-Armed Bandit Simulator!")
    print(f"{Fore.GREEN}You'll be playing against a 5-armed bandit with 5D context vectors.")
    
    bandit = create_demo_contextual_bandit()
    total_reward = 0
    pulls = 0
    
    print_contextual_bandit_info(bandit)
    
    while True:
        print(f"\n{Fore.CYAN}{'='*40}")
        print(f"{Fore.CYAN}Step {pulls + 1}")
        print(f"{Fore.YELLOW}Total reward so far: {total_reward:.3f}")
        print(f"{Fore.YELLOW}Average reward: {total_reward/max(1, pulls):.3f}")
        
        # Generate context
        context = bandit.generate_context()
        print(f"\n{Fore.BLUE}Generated context: {context}")
        
        # Show which arm would be optimal
        optimal_arm = bandit.get_optimal_arm(context)
        optimal_reward = bandit.get_optimal_reward(context)
        print(f"{Fore.GREEN}Optimal arm: {optimal_arm} (expected reward: {optimal_reward:.3f})")
        
        # Show expected rewards for all arms
        print(f"\n{Fore.MAGENTA}Expected rewards for each arm:")
        for i in range(bandit.n_arms):
            expected = bandit.arms[i].get_expected_reward(context)
            marker = " ← OPTIMAL" if i == optimal_arm else ""
            print(f"  {Fore.MAGENTA}Arm {i}: {expected:.3f}{marker}")
        
        try:
            choice = input(f"\n{Fore.BLUE}Choose an arm (0-{bandit.n_arms-1}) or 'q' to quit: {Style.RESET_ALL}").strip()
            
            if choice.lower() == 'q':
                break
                
            arm_idx = int(choice)
            if arm_idx < 0 or arm_idx >= bandit.n_arms:
                print(f"{Fore.RED}Invalid arm! Choose between 0 and {bandit.n_arms-1}")
                continue
                
            reward, valid = bandit.pull_arm(arm_idx, context)
            if valid:
                total_reward += reward
                pulls += 1
                print(f"{Fore.GREEN}Pulled arm {arm_idx}: reward = {reward:.3f}")
                print(f"{Fore.GREEN}Expected reward was: {bandit.arms[arm_idx].get_expected_reward(context):.3f}")
            else:
                print(f"{Fore.RED}Invalid pull!")
                
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Goodbye!")
            break
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}FINAL RESULTS")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.YELLOW}Total pulls: {pulls}")
    print(f"{Fore.YELLOW}Total reward: {total_reward:.3f}")
    print(f"{Fore.YELLOW}Average reward: {total_reward/max(1, pulls):.3f}")
    
    # Calculate regret
    if pulls > 0:
        total_optimal_reward = sum([
            bandit.get_optimal_reward(bandit.context_history[i]) 
            for i in range(pulls)
        ])
        regret = total_optimal_reward - total_reward
        print(f"{Fore.YELLOW}Total optimal reward: {total_optimal_reward:.3f}")
        print(f"{Fore.YELLOW}Regret: {regret:.3f}")


def demo_automatic_play(n_iterations=1000):
    """Demo automatic play with different strategies."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}DEMO: Automatic Play with Different Strategies")
    print(f"{Fore.CYAN}{'='*60}")
    
    bandit = create_demo_contextual_bandit()
    
    # Strategy 1: Random
    print(f"\n{Fore.BLUE}Strategy 1: Random Selection")
    bandit.reset()
    total_reward_random = 0
    rewards_random = []
    
    for step in range(n_iterations):
        context = bandit.generate_context()
        arm_idx = np.random.randint(bandit.n_arms)
        reward, _ = bandit.pull_arm(arm_idx, context)
        total_reward_random += reward
        rewards_random.append(reward)
        
        if (step + 1) % 200 == 0:
            avg_reward = total_reward_random / (step + 1)
            print(f"{Fore.YELLOW}Step {step + 1}: Avg reward = {avg_reward:.3f}")
    
    # Strategy 2: Greedy (always pick optimal based on current estimates)
    print(f"\n{Fore.BLUE}Strategy 2: Greedy Selection")
    bandit.reset()
    total_reward_greedy = 0
    rewards_greedy = []
    
    # Simple greedy: track average rewards for each arm
    arm_rewards = [[] for _ in range(bandit.n_arms)]
    
    for step in range(n_iterations):
        context = bandit.generate_context()
        
        # Select arm with highest average reward so far
        if step < bandit.n_arms:
            # Initial exploration: try each arm once
            arm_idx = step
        else:
            # Greedy selection
            avg_rewards = [np.mean(rewards) if rewards else 0 for rewards in arm_rewards]
            arm_idx = np.argmax(avg_rewards)
        
        reward, _ = bandit.pull_arm(arm_idx, context)
        arm_rewards[arm_idx].append(reward)
        total_reward_greedy += reward
        rewards_greedy.append(reward)
        
        if (step + 1) % 200 == 0:
            avg_reward = total_reward_greedy / (step + 1)
            print(f"{Fore.YELLOW}Step {step + 1}: Avg reward = {avg_reward:.3f}")
    
    # Strategy 3: Optimal (oracle)
    print(f"\n{Fore.BLUE}Strategy 3: Oracle (Optimal)")
    bandit.reset()
    total_reward_optimal = 0
    rewards_optimal = []
    
    for step in range(n_iterations):
        context = bandit.generate_context()
        optimal_arm = bandit.get_optimal_arm(context)
        reward, _ = bandit.pull_arm(optimal_arm, context)
        total_reward_optimal += reward
        rewards_optimal.append(reward)
        
        if (step + 1) % 200 == 0:
            avg_reward = total_reward_optimal / (step + 1)
            print(f"{Fore.YELLOW}Step {step + 1}: Avg reward = {avg_reward:.3f}")
    
    # Results comparison
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}RESULTS COMPARISON")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.MAGENTA}{'Strategy':<15} {'Total Reward':<15} {'Avg Reward':<15} {'Regret':<15}")
    print(f"{Fore.MAGENTA}{'-' * 60}")
    
    # Calculate regrets
    regret_random = total_reward_optimal - total_reward_random
    regret_greedy = total_reward_optimal - total_reward_greedy
    
    print(f"{'Random':<15} {total_reward_random:<15.3f} {total_reward_random/n_iterations:<15.3f} {regret_random:<15.3f}")
    print(f"{'Greedy':<15} {total_reward_greedy:<15.3f} {total_reward_greedy/n_iterations:<15.3f} {regret_greedy:<15.3f}")
    print(f"{'Oracle':<15} {total_reward_optimal:<15.3f} {total_reward_optimal/n_iterations:<15.3f} {'0.000':<15}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Cumulative rewards
    plt.subplot(2, 2, 1)
    cum_rewards_random = np.cumsum(rewards_random)
    cum_rewards_greedy = np.cumsum(rewards_greedy)
    cum_rewards_optimal = np.cumsum(rewards_optimal)
    
    plt.plot(cum_rewards_random, label='Random', alpha=0.7)
    plt.plot(cum_rewards_greedy, label='Greedy', alpha=0.7)
    plt.plot(cum_rewards_optimal, label='Oracle', alpha=0.7)
    plt.xlabel('Time step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Average rewards
    plt.subplot(2, 2, 2)
    avg_rewards_random = cum_rewards_random / (np.arange(n_iterations) + 1)
    avg_rewards_greedy = cum_rewards_greedy / (np.arange(n_iterations) + 1)
    avg_rewards_optimal = cum_rewards_optimal / (np.arange(n_iterations) + 1)
    
    plt.plot(avg_rewards_random, label='Random', alpha=0.7)
    plt.plot(avg_rewards_greedy, label='Greedy', alpha=0.7)
    plt.plot(avg_rewards_optimal, label='Oracle', alpha=0.7)
    plt.xlabel('Time step')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Regret
    plt.subplot(2, 2, 3)
    regret_random_curve = cum_rewards_optimal - cum_rewards_random
    regret_greedy_curve = cum_rewards_optimal - cum_rewards_greedy
    
    plt.plot(regret_random_curve, label='Random', alpha=0.7)
    plt.plot(regret_greedy_curve, label='Greedy', alpha=0.7)
    plt.xlabel('Time step')
    plt.ylabel('Regret')
    plt.title('Regret')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Context visualization (last 100 contexts)
    plt.subplot(2, 2, 4)
    if len(bandit.context_history) >= 100:
        recent_contexts = np.array(bandit.context_history[-100:])
        plt.imshow(recent_contexts.T, aspect='auto', cmap='viridis')
        plt.xlabel('Time step (last 100)')
        plt.ylabel('Context dimension')
        plt.title('Recent Contexts')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def analyze_arm_behavior():
    """Analyze how different arms respond to different context features."""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}ARM BEHAVIOR ANALYSIS")
    print(f"{Fore.CYAN}{'='*60}")
    
    bandit = create_demo_contextual_bandit()
    
    # Test each arm with different context patterns
    test_contexts = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # Strong feature 0
        np.array([0.0, 1.0, 0.0, 0.0, 0.0]),  # Strong feature 1
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),  # Strong feature 2
        np.array([0.0, 0.0, 0.0, 1.0, 0.0]),  # Strong feature 3
        np.array([0.0, 0.0, 0.0, 0.0, 1.0]),  # Strong feature 4
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # All features
        np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),  # Negative features
    ]
    
    context_names = [
        "Feature 0 strong", "Feature 1 strong", "Feature 2 strong", 
        "Feature 3 strong", "Feature 4 strong", "All features", "Negative features"
    ]
    
    print(f"\n{Fore.MAGENTA}{'Context':<20} {'Arm 0':<10} {'Arm 1':<10} {'Arm 2':<10} {'Arm 3':<10} {'Arm 4':<10}")
    print(f"{Fore.MAGENTA}{'-' * 70}")
    
    for context, name in zip(test_contexts, context_names):
        expected_rewards = [bandit.arms[i].get_expected_reward(context) for i in range(bandit.n_arms)]
        print(f"{name:<20} {expected_rewards[0]:<10.3f} {expected_rewards[1]:<10.3f} {expected_rewards[2]:<10.3f} {expected_rewards[3]:<10.3f} {expected_rewards[4]:<10.3f}")
    
    # Visualize arm responses
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Arm responses to individual features
    plt.subplot(2, 2, 1)
    feature_responses = []
    for i in range(5):  # 5 features
        context = np.zeros(5)
        context[i] = 1.0
        responses = [bandit.arms[j].get_expected_reward(context) for j in range(5)]
        feature_responses.append(responses)
    
    feature_responses = np.array(feature_responses)
    plt.imshow(feature_responses, aspect='auto', cmap='viridis')
    plt.xlabel('Arm')
    plt.ylabel('Feature')
    plt.title('Arm Responses to Individual Features')
    plt.colorbar()
    plt.xticks(range(5), [f'Arm {i}' for i in range(5)])
    plt.yticks(range(5), [f'Feature {i}' for i in range(5)])
    
    # Plot 2: Arm weight vectors
    plt.subplot(2, 2, 2)
    weights_matrix = []
    for arm in bandit.arms:
        if hasattr(arm.distribution, 'weights'):
            weights_matrix.append(arm.distribution.weights)
    
    weights_matrix = np.array(weights_matrix)
    plt.imshow(weights_matrix, aspect='auto', cmap='RdBu_r', center=0)
    plt.xlabel('Feature')
    plt.ylabel('Arm')
    plt.title('Arm Weight Vectors')
    plt.colorbar()
    plt.xticks(range(5), [f'F{i}' for i in range(5)])
    plt.yticks(range(5), [f'Arm {i}' for i in range(5)])
    
    # Plot 3: Expected rewards vs context magnitude
    plt.subplot(2, 2, 3)
    context_magnitudes = np.linspace(-2, 2, 50)
    for i in range(5):
        rewards = []
        for mag in context_magnitudes:
            context = np.array([mag, mag, mag, mag, mag])  # All features same magnitude
            rewards.append(bandit.arms[i].get_expected_reward(context))
        plt.plot(context_magnitudes, rewards, label=f'Arm {i}')
    
    plt.xlabel('Context magnitude')
    plt.ylabel('Expected reward')
    plt.title('Expected Rewards vs Context Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Optimal arm for different contexts
    plt.subplot(2, 2, 4)
    x_range = np.linspace(-2, 2, 50)
    optimal_arms = []
    for x in x_range:
        context = np.array([x, 0, 0, 0, 0])  # Only feature 0 varies
        optimal_arm = bandit.get_optimal_arm(context)
        optimal_arms.append(optimal_arm)
    
    plt.plot(x_range, optimal_arms, 'o-', markersize=3)
    plt.xlabel('Feature 0 value')
    plt.ylabel('Optimal arm')
    plt.title('Optimal Arm vs Feature 0')
    plt.grid(True, alpha=0.3)
    plt.yticks(range(5))
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Contextual Multi-Armed Bandit Simulator')
    parser.add_argument('--iterations', '-i', type=int, default=1000,
                       help='Number of iterations for automatic demo (default: 1000)')
    parser.add_argument('--mode', '-m', type=int, choices=[1, 2, 3], default=None,
                       help='Mode: 1=Interactive, 2=Automatic demo, 3=Arm analysis')
    
    args = parser.parse_args()
    
    print("Contextual Multi-Armed Bandit Simulator")
    print(f"Number of iterations: {args.iterations}")
    
    if args.mode is None:
        print("1. Interactive play")
        print("2. Automatic demo (compare strategies)")
        print("3. Arm behavior analysis")
        
        while True:
            try:
                choice = input("\nChoose an option (1-3): ").strip()
                if choice == "1":
                    interactive_play()
                    break
                elif choice == "2":
                    demo_automatic_play(args.iterations)
                    break
                elif choice == "3":
                    analyze_arm_behavior()
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        if args.mode == 1:
            interactive_play()
        elif args.mode == 2:
            demo_automatic_play(args.iterations)
        elif args.mode == 3:
            analyze_arm_behavior()


if __name__ == "__main__":
    main() 