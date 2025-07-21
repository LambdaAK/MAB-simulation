import numpy as np

class Experiment:
    def __init__(self, bandit, policies, n_iterations=1000, seed=None, name=None):
        self.bandit = bandit
        # Accept either a single policy or a list/dict of policies
        if isinstance(policies, (list, tuple, dict)):
            if isinstance(policies, dict):
                self.policies = list(policies.items())
            else:
                self.policies = list(policies)
        else:
            self.policies = [(getattr(policies, 'name', 'Policy'), policies)]
        self.n_iterations = n_iterations
        self.seed = seed
        self.name = name
        self.reset_history()
        if seed is not None:
            np.random.seed(seed)

    def reset_history(self):
        self.results = {}

    def run(self):
        self.reset_history()
        for policy_name, policy in self.policies:
            self.bandit.reset()
            policy.reset()
            rewards = []
            actions = []
            cumulative_rewards = []
            total_reward = 0
            for t in range(self.n_iterations):
                action = policy.select_action()
                reward, valid = self.bandit.pull_arm(action)
                if not valid:
                    raise ValueError(f"Invalid action {action}")
                policy.update(action, reward)
                total_reward += reward
                rewards.append(reward)
                actions.append(action)
                cumulative_rewards.append(total_reward)
            self.results[policy_name] = {
                "rewards": rewards,
                "actions": actions,
                "cumulative_rewards": cumulative_rewards,
            }

    def get_results(self):
        return self.results

    def plot(self, show_avg=True):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for policy_name, result in self.results.items():
            if show_avg:
                avg_rewards = np.cumsum(result["rewards"]) / (np.arange(len(result["rewards"])) + 1)
                plt.plot(avg_rewards, label=policy_name)
                plt.ylabel('Average Accumulated Reward')
            else:
                plt.plot(result["cumulative_rewards"], label=policy_name)
                plt.ylabel('Total Accumulated Reward')
        plt.xlabel('Time step')
        title = self.name if self.name else 'Bandit Experiment Results'
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend()
        plt.show() 