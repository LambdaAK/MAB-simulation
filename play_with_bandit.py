#!/usr/bin/env python3
"""
Interactive script to play with the Multi-Armed Bandit environment.
"""

import numpy as np
import argparse
from mab_environment import MultiArmedBandit, Arm
from policies import EpsilonGreedy, RandomPolicy, GreedyPolicy, UCBPolicy, SoftmaxPolicy, ThompsonSamplingPolicy, ThompsonSamplingNormalPolicy
from colorama import init, Fore, Style
import matplotlib.pyplot as plt
from experiment import Experiment

init(autoreset=True)


def create_demo_bandit():
    """Create a demo bandit with 3 arms of different qualities."""
    arms = [
        Arm("Poor", mean=0.2, std=0.3),
        Arm("Medium", mean=0.5, std=0.3),
        Arm("Good", mean=5.8, std=0.3)
    ]
    return MultiArmedBandit(arms, seed=42)


def create_10_arm_bandit():
    """Create a bandit with 10 arms, each label describing its mean and std."""
    arms = [
        Arm(
            name=f"mean={mean:.2f}, std={std:.2f}",
            mean=mean,
            std=std
        )
        for mean, std in [(0.1 * i, 0.2 + 0.05 * (i % 3)) for i in range(10)]
    ]
    return MultiArmedBandit(arms, seed=123)


def print_bandit_info(bandit):
    """Print information about the bandit."""
    info = bandit.get_info()
    print(f"\n{Fore.CYAN + Style.BRIGHT}{'='*50}")
    print(f"{Fore.CYAN + Style.BRIGHT}BANDIT INFORMATION")
    print(f"{Fore.CYAN + Style.BRIGHT}{'='*50}")
    print(f"{Fore.YELLOW}Number of arms: {info['n_arms']}")
    print(f"{Fore.YELLOW}Optimal arm: {info['optimal_arm']}")
    print(f"{Fore.YELLOW}Optimal expected reward: {info['optimal_reward']:.3f}")
    print(f"{Fore.YELLOW}Total pulls: {info['step_count']}")
    
    print(f"\n{Fore.MAGENTA}Arm details:")
    for i, arm_info in enumerate(info['arms']):
        print(f"  {Fore.MAGENTA}Arm {i} ({arm_info['name']}): mean={arm_info['mean']:.3f}, std={arm_info['std']:.3f}")


def interactive_play():
    """Interactive mode to play with the bandit."""
    print(f"{Fore.GREEN}Welcome to the Multi-Armed Bandit Simulator!")
    print(f"{Fore.GREEN}You'll be playing against a 10-armed bandit.")
    
    bandit = create_10_arm_bandit()
    total_reward = 0
    pulls = 0
    
    print_bandit_info(bandit)
    
    while True:
        print(f"\n{Fore.CYAN}{'='*30}")
        print(f"{Fore.CYAN}Step {pulls + 1}")
        print(f"{Fore.YELLOW}Total reward so far: {total_reward:.3f}")
        print(f"{Fore.YELLOW}Average reward: {total_reward/max(1, pulls):.3f}")
        
        # Show arm details for easier selection
        print(f"\n{Fore.MAGENTA}Arm details:")
        bandit_info = bandit.get_info()
        for i, arm_info in enumerate(bandit_info['arms']):
            print(f"  {Fore.MAGENTA}{i}: {arm_info['name']}")
        
        try:
            choice = input(f"\n{Fore.BLUE}Choose an arm (0-{bandit.n_arms-1}) or 'q' to quit: {Style.RESET_ALL}").strip()
            
            if choice.lower() == 'q':
                break
                
            arm_idx = int(choice)
            if arm_idx < 0 or arm_idx >= bandit.n_arms:
                print(f"{Fore.RED}Invalid arm! Choose between 0 and {bandit.n_arms-1}")
                continue
                
            reward, valid = bandit.pull_arm(arm_idx)
            if valid:
                total_reward += reward
                pulls += 1
                print(f"{Fore.GREEN}Pulled arm {arm_idx}: reward = {reward:.3f}")
            else:
                print(f"{Fore.RED}Invalid pull!")
                
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Goodbye!")
            break
    
    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"{Fore.CYAN}FINAL RESULTS")
    print(f"{Fore.CYAN}{'='*50}")
    print(f"{Fore.YELLOW}Total pulls: {pulls}")
    print(f"{Fore.YELLOW}Total reward: {total_reward:.3f}")
    print(f"{Fore.YELLOW}Average reward: {total_reward/max(1, pulls):.3f}")
    print(f"{Fore.YELLOW}Optimal average reward: {bandit.get_optimal_reward():.3f}")
    print(f"{Fore.YELLOW}Regret: {bandit.get_optimal_reward() * pulls - total_reward:.3f}")


def demo_policies(n_iterations):
    """Demo different policies."""
    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"{Fore.CYAN}DEMO: Policy Comparison")
    print(f"{Fore.CYAN}{'='*50}")
    
    bandit = create_10_arm_bandit()
    
    # Ask user for epsilon-greedy parameters
    print(f"\n{Fore.BLUE}Epsilon-Greedy parameters:")
    try:
        epsilon = float(input(f"  {Fore.YELLOW}Initial epsilon (e.g. 0.1): {Style.RESET_ALL}"))
        decay = float(input(f"  {Fore.YELLOW}Decay rate (e.g. 0.99): {Style.RESET_ALL}"))
        min_epsilon = float(input(f"  {Fore.YELLOW}Minimum epsilon (e.g. 0.01): {Style.RESET_ALL}"))
    except Exception:
        print(f"{Fore.RED}Invalid input, using defaults: epsilon=0.1, decay=1.0, min_epsilon=0.01")
        epsilon = 0.1
        decay = 1.0
        min_epsilon = 0.01
    
    policies = {
        f"Random": RandomPolicy(n_arms=10, seed=123),
        f"Greedy": GreedyPolicy(n_arms=10),
        f"Epsilon-Greedy (ε={epsilon}, decay={decay}, min={min_epsilon})": EpsilonGreedy(n_arms=10, epsilon=epsilon, decay=decay, min_epsilon=min_epsilon, seed=123),
        f"Epsilon-Greedy (ε={epsilon*2}, decay={decay}, min={min_epsilon})": EpsilonGreedy(n_arms=10, epsilon=epsilon*2, decay=decay, min_epsilon=min_epsilon, seed=123)
    }
    
    results = {}
    reward_histories = {}
    
    for name, policy in policies.items():
        print(f"\n{Fore.CYAN}Testing {name}...")
        
        # Reset environment and policy
        bandit.reset()
        policy.reset()
        
        total_reward = 0
        rewards = []
        cumulative_rewards = []
        
        for step in range(n_iterations):
            action = policy.select_action()
            reward, valid = bandit.pull_arm(action)
            
            if valid:
                policy.update(action, reward)
                total_reward += reward
                rewards.append(reward)
                cumulative_rewards.append(total_reward)
            if (step + 1) % 40 == 0:
                avg_reward = total_reward / (step + 1)
                info = policy.get_info()
                print(f"{Fore.YELLOW}Step {step + 1}: Avg reward = {avg_reward:.3f}, Best arm estimate = {info['best_action']}, Epsilon = {info.get('epsilon', '-'):.4f}")
        
        info = policy.get_info()
        results[name] = {
            'total_reward': total_reward,
            'avg_reward': total_reward / n_iterations,
            'action_counts': info['action_counts'],
            'action_values': info['action_values'],
            'best_action': info['best_action']
        }
        reward_histories[name] = cumulative_rewards
        
        print(f"  {Fore.GREEN}Total reward: {total_reward:.3f}")
        print(f"  {Fore.GREEN}Average reward: {total_reward/n_iterations:.3f}")
        print(f"  {Fore.GREEN}Action counts: {info['action_counts']}")
        print(f"  {Fore.GREEN}Best action estimate: {info['best_action']}")
    
    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"{Fore.CYAN}COMPARISON SUMMARY")
    print(f"{Fore.CYAN}{'='*50}")
    print(f"{Fore.YELLOW}Optimal arm: {bandit.get_optimal_arm()}")
    print(f"{Fore.YELLOW}Optimal expected reward: {bandit.get_optimal_reward():.3f}")
    
    print(f"\n{Fore.MAGENTA}{'Policy':<45} {'Avg Reward':<12} {'Best Action':<12}")
    print(f"{Fore.MAGENTA}{'-' * 70}")
    for name, result in results.items():
        print(f"{name:<45} {result['avg_reward']:<12.3f} {result['best_action']:<12}")

    # Plot cumulative reward vs time for each policy
    plt.figure(figsize=(10, 6))
    for name, cum_rewards in reward_histories.items():
        plt.plot(cum_rewards, label=name)
    plt.xlabel('Time step')
    plt.ylabel('Total Accumulated Reward')
    plt.title('Total Accumulated Reward vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def demo_single_policy(n_iterations):
    """Demo a single policy with detailed output."""
    print(f"\n{Fore.CYAN}{'='*50}")
    print(f"{Fore.CYAN}DEMO: Single Policy")
    print(f"{Fore.CYAN}{'='*50}")
    
    bandit = create_10_arm_bandit()
    
    # Let user choose policy
    print(f"{Fore.BLUE}Choose a policy:")
    print(f"{Fore.YELLOW}1. Random")
    print(f"{Fore.YELLOW}2. Greedy")
    print(f"{Fore.YELLOW}3. Epsilon-Greedy")
    
    while True:
        try:
            choice = input(f"\n{Fore.BLUE}Enter choice (1-3): {Style.RESET_ALL}").strip()
            if choice == "1":
                policy = RandomPolicy(n_arms=10, seed=123)
                policy_name = f"{Fore.MAGENTA}Random{Style.RESET_ALL}"
                break
            elif choice == "2":
                policy = GreedyPolicy(n_arms=10)
                policy_name = f"{Fore.MAGENTA}Greedy{Style.RESET_ALL}"
                break
            elif choice == "3":
                print(f"{Fore.BLUE}Epsilon-Greedy parameters:")
                epsilon = float(input(f"  {Fore.YELLOW}Initial epsilon (e.g. 0.1): {Style.RESET_ALL}"))
                decay = float(input(f"  {Fore.YELLOW}Decay rate (e.g. 0.99): {Style.RESET_ALL}"))
                min_epsilon = float(input(f"  {Fore.YELLOW}Minimum epsilon (e.g. 0.01): {Style.RESET_ALL}"))
                policy = EpsilonGreedy(n_arms=10, epsilon=epsilon, decay=decay, min_epsilon=min_epsilon, seed=123)
                policy_name = f"{Fore.MAGENTA}Epsilon-Greedy (ε={epsilon}, decay={decay}, min={min_epsilon}){Style.RESET_ALL}"
                break
            else:
                print(f"{Fore.RED}Please enter 1, 2, or 3")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number")
    
    print(f"\n{Fore.CYAN}Running {policy_name} for {n_iterations} pulls...")
    
    total_reward = 0
    rewards = []
    
    for step in range(n_iterations):
        action = policy.select_action()
        reward, valid = bandit.pull_arm(action)
        
        if valid:
            policy.update(action, reward)
            total_reward += reward
            rewards.append(reward)
        
        # Print progress every 40 steps (less frequent for longer runs)
        if (step + 1) % 40 == 0:
            avg_reward = total_reward / (step + 1)
            info = policy.get_info()
            epsilon_val = info.get('epsilon', '-')
            if isinstance(epsilon_val, float):
                epsilon_str = f"{epsilon_val:.4f}"
            else:
                epsilon_str = str(epsilon_val)
            print(f"{Fore.YELLOW}Step {step + 1}: Avg reward = {avg_reward:.3f}, Best arm estimate = {info['best_action']}, Epsilon = {epsilon_str}")
    
    info = policy.get_info()
    epsilon_val = info.get('epsilon', '-')
    if isinstance(epsilon_val, float):
        epsilon_str = f"{epsilon_val:.4f}"
    else:
        epsilon_str = str(epsilon_val)
    print(f"\n{Fore.GREEN}Final results:")
    print(f"{Fore.GREEN}Total reward: {total_reward:.3f}")
    print(f"{Fore.GREEN}Average reward: {total_reward/n_iterations:.3f}")
    print(f"{Fore.GREEN}Action counts: {info['action_counts']}")
    print(f"{Fore.GREEN}Action values: {info['action_values']}")
    print(f"{Fore.GREEN}Optimal arm: {bandit.get_optimal_arm()}")
    print(f"{Fore.GREEN}Best estimated arm: {info['best_action']}")
    print(f"{Fore.GREEN}Final epsilon: {epsilon_str}")
    
    # Show arm details
    print(f"\n{Fore.MAGENTA}Arm details:")
    bandit_info = bandit.get_info()
    for i, arm_info in enumerate(bandit_info['arms']):
        count = info['action_counts'][i]
        value = info['action_values'][i]
        print(f"  {Fore.MAGENTA}Arm {i}: {arm_info['name']}, pulled {count} times, estimated value {value:.3f}")

    # Plot average accumulated reward vs time
    avg_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards, label="")  # No label for the line
    plt.xlabel('Time step')
    plt.ylabel('Average Accumulated Reward')
    plt.title('Average Accumulated Reward vs Time')
    plt.legend().set_visible(False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_sample_experiment():
    exp_name = "Sample: 10-Armed Bandit, EpsilonGreedy (eps=0.1, decay=0.99, min=0.01)"
    print(f"{Fore.CYAN}Running experiment: {exp_name}")
    # Create 10 arms with means from 0.1 to 1.0
    arms = [Arm(name=f"Arm_{i}", mean=0.1 * (i + 1), std=0.1) for i in range(10)]
    bandit = MultiArmedBandit(arms, seed=42)
    policy = EpsilonGreedy(n_arms=10, epsilon=0.1, decay=0.99, min_epsilon=0.01, seed=42)
    exp = Experiment(bandit, policy, n_iterations=10000, seed=42, name=exp_name)
    exp.run()
    exp.plot(show_avg=True)
    print(f"{Fore.GREEN}Sample experiment complete!")


def run_sample_multi_policy_experiment():
    exp_name = "Sample: 10-Armed Bandit, Multi-Policy Comparison"
    print(f"{Fore.CYAN}Running experiment: {exp_name}")
    # Create 10 arms with means from 0.1 to 1.0
    arms = [Arm(name=f"Arm_{i}", mean=0.1 * (i + 1), std=0.1) for i in range(10)]
    bandit = MultiArmedBandit(arms, seed=42)
    policies = [
        ("Random", RandomPolicy(n_arms=10, seed=42)),
        ("EpsilonGreedy (eps=0.1, decay=0.99, min=0.01)", EpsilonGreedy(n_arms=10, epsilon=0.1, decay=0.99, min_epsilon=0.01, seed=42)),
        ("EpsilonGreedy (eps=0.3, decay=0.99, min=0.01)", EpsilonGreedy(n_arms=10, epsilon=0.3, decay=0.99, min_epsilon=0.01, seed=42)),
        ("EpsilonGreedy (eps=0.8, decay=0.999, min=0.01)", EpsilonGreedy(n_arms=10, epsilon=0.8, decay=0.999, min_epsilon=0.01, seed=42)),
        ("EpsilonGreedy (eps=0.9, decay=0.9999, min=0.01)", EpsilonGreedy(n_arms=10, epsilon=0.9, decay=0.9999, min_epsilon=0.01, seed=42))
    ]
    exp = Experiment(bandit, policies, n_iterations=10000, seed=42, name=exp_name)
    exp.run()
    exp.plot(show_avg=True)
    print(f"{Fore.GREEN}Sample multi-policy experiment complete!")


def run_ucb_vs_epsilon_experiment():
    exp_name = "UCB vs EpsilonGreedy vs Softmax: 10-Armed Bandit Comparison"
    print(f"{Fore.CYAN}Running experiment: {exp_name}")
    arms = [Arm(name=f"Arm_{i}", mean=0.1 * (i + 1), std=0.1) for i in range(10)]
    bandit = MultiArmedBandit(arms, seed=42)
    policies = [
        ("UCB", UCBPolicy(n_arms=10, seed=42)),
        ("UCB (seed=123)", UCBPolicy(n_arms=10, seed=123)),
        ("EpsilonGreedy (eps=0.1, decay=0.99, min=0.01)", EpsilonGreedy(n_arms=10, epsilon=0.1, decay=0.99, min_epsilon=0.01, seed=42)),
        ("EpsilonGreedy (eps=0.3, decay=0.995, min=0.01)", EpsilonGreedy(n_arms=10, epsilon=0.3, decay=0.995, min_epsilon=0.01, seed=42)),
        ("Softmax (tau=0.1)", SoftmaxPolicy(n_arms=10, tau=0.1, seed=42)),
        ("Softmax (tau=0.5)", SoftmaxPolicy(n_arms=10, tau=0.5, seed=42)),
    ]
    exp = Experiment(bandit, policies, n_iterations=10000, seed=42, name=exp_name)
    exp.run()
    exp.plot(show_avg=True)
    print(f"{Fore.GREEN}UCB vs EpsilonGreedy vs Softmax experiment complete!")


def run_thompson_sampling_experiment():
    exp_name = "Thompson Sampling vs Other Algorithms: 10-Armed Bandit Comparison"
    print(f"{Fore.CYAN}Running experiment: {exp_name}")
    arms = [Arm(name=f"Arm_{i}", mean=0.1 * (i + 1), std=0.1) for i in range(10)]
    bandit = MultiArmedBandit(arms, seed=42)
    policies = [
        ("Random", RandomPolicy(n_arms=10, seed=42)),
        ("EpsilonGreedy (eps=0.1)", EpsilonGreedy(n_arms=10, epsilon=0.1, seed=42)),
        ("UCB", UCBPolicy(n_arms=10, seed=42)),
        ("Softmax (tau=0.1)", SoftmaxPolicy(n_arms=10, tau=0.1, seed=42)),
        ("Thompson Sampling (Beta)", ThompsonSamplingPolicy(n_arms=10, seed=42)),
        ("Thompson Sampling (Normal)", ThompsonSamplingNormalPolicy(n_arms=10, seed=42)),
    ]
    exp = Experiment(bandit, policies, n_iterations=10000, seed=42, name=exp_name)
    exp.run()
    exp.plot(show_avg=True)
    print(f"{Fore.GREEN}Thompson Sampling experiment complete!")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit Simulator')
    parser.add_argument('--iterations', '-i', type=int, default=200,
                       help='Number of iterations to simulate (default: 200)')
    parser.add_argument('--mode', '-m', type=int, choices=[1, 2, 3, 4, 5, 6, 7], default=None,
                       help='Mode: 1=Interactive, 2=Policy comparison, 3=Single policy, 4=Sample experiment (10 arms, EpsilonGreedy), 5=Sample multi-policy experiment, 6=UCB vs EpsilonGreedy, 7=Thompson Sampling')
    
    args = parser.parse_args()
    
    print("Multi-Armed Bandit Simulator")
    print(f"Number of iterations: {args.iterations}")
    
    if args.mode is None:
        print("1. Interactive play")
        print("2. Policy comparison demo")
        print("3. Single policy demo")
        print("4. Sample experiment (10 arms, EpsilonGreedy)")
        print("5. Sample multi-policy experiment")
        print("6. UCB vs EpsilonGreedy experiment")
        print("7. Thompson Sampling experiment")
        
        while True:
            try:
                choice = input("\nChoose an option (1-7): ").strip()
                if choice == "1":
                    interactive_play()
                    break
                elif choice == "2":
                    demo_policies(args.iterations)
                    break
                elif choice == "3":
                    demo_single_policy(args.iterations)
                    break
                elif choice == "4":
                    run_sample_experiment()
                    break
                elif choice == "5":
                    run_sample_multi_policy_experiment()
                    break
                elif choice == "6":
                    run_ucb_vs_epsilon_experiment()
                    break
                elif choice == "7":
                    run_thompson_sampling_experiment()
                    break
                else:
                    print("Please enter 1, 2, 3, 4, 5, 6, or 7")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        if args.mode == 1:
            interactive_play()
        elif args.mode == 2:
            demo_policies(args.iterations)
        elif args.mode == 3:
            demo_single_policy(args.iterations)
        elif args.mode == 4:
            run_sample_experiment()
        elif args.mode == 5:
            run_sample_multi_policy_experiment()
        elif args.mode == 6:
            run_ucb_vs_epsilon_experiment()
        elif args.mode == 7:
            run_thompson_sampling_experiment()


if __name__ == "__main__":
    main() 