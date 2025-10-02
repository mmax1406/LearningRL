import gymnasium as gym
from stable_baselines3 import PPO

# Load best model
model = PPO.load("ppo_cartpole_best")

# Make environment for visualization (normal physics)
env = gym.make("CartPole-v1", render_mode="human")

obs, info = env.reset()
done = False
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        obs, info = env.reset()
