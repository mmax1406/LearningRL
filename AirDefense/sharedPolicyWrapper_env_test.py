import numpy as np
from sharedPolicyWrapper import SharedPolicyWrapper
from intercept_env import env

# Create base PettingZoo environment
base_env = env(N_adversaries=3, M_good=5, width_ratio=5.0)

# Wrap with shared policy wrapper
wrapped_env = SharedPolicyWrapper(base_env)

# Reset environment
obs, info = wrapped_env.reset()
print("Initial observation shape:", obs.shape)  # Should be (N_agents, obs_dim)

for step in range(200):
    # Sample random actions for all agents
    actions = np.stack([wrapped_env.action_space.sample() for _ in range(wrapped_env.num_agents)])

    obs, reward, done, infos = wrapped_env.step(actions)
    print(f"Step {step} | obs shape: {obs.shape}, reward: {reward.shape}, done: {done}")

    # Render underlying env
    base_env.render()

    if done:
        print("Game over")
        break
