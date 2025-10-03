import gymnasium as gym
from world import GridCleanEnvGym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

# ------------------- Load Model -------------------
model = PPO.load("ppo_gridclean_best")

# ------------------- Visualization Function -------------------
def plot_paths(env, model, max_steps=20*20):
    """Visualize the agent's trajectory in the GridCleanEnvGym environment."""

    # Reset environment
    obs, info = env.reset()
    rewards = 0
    x_vals, y_vals = [], []
    isDone = []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Agent Trajectory")

    def render_state():
        """Render environment state as RGB image."""
        n, m, _ = obs.shape
        img = np.ones((n, m, 3))  # start as white
        # obstacles = black
        img[env.obstacles == 1] = [0, 0, 0]
        # cleaned = light gray
        img[env.cleaned == 1] = [0.8, 0.8, 0.8]
        # robot = blue
        img[env.r, env.c] = [0.0, 0.0, 1.0]
        return img

    mat = ax.imshow(render_state(), interpolation="none")
    plt.ion()
    plt.show()

    # Run agent and record trajectory
    for stepCount in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        rewards += reward
        x_vals.append(env.c)
        y_vals.append(env.r)
        isDone.append(done)

        # Update display
        mat.set_data(render_state())
        ax.plot(x_vals, y_vals, "-r")  # path in red
        plt.pause(0.1)

        if done:
            break

    plt.pause(1)
    plt.ioff()
    plt.close(fig)
    print(f"Total reward: {rewards:.2f}")

# ------------------- Run Visualization -------------------
if __name__ == "__main__":
    while True:
        env = GridCleanEnvGym(grid_size=10, num_obstacles=3, obstacles_max_size=3, max_steps=200)
        plot_paths(env, model)
