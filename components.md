# Project structure

- A class that represents the environment - MAB
- Arm class that contains a distribution. This is the distribution from which rewards are sampled from.
- A class that represents a policy
- Classes that represent different algorithms for computing a policy
- An interface that is inherited by classes that represent algorithms for computing policies

# Algorithms

- Random selection (baseline)
- Epsilon-greedy
- UCB
- Thompson sampling

# Visualization

- A simulator that visualizes algorithms. It should show which action is taken by the algorithm at different times.