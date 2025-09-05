import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_rewards(rewards):
    plt.figure(figsize=(8,4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.grid(True)
    plt.show()

def animate_agent(env, agent, max_steps=50):
    grid = env.grid.copy()
    fig, ax = plt.subplots()

    mat = ax.matshow(grid, cmap="coolwarm")
    ax.set_title("Agent Trajectory")

    def update(frame):
        ax.clear()
        ax.matshow(grid, cmap="coolwarm")
        ax.plot(env.c, env.r, "yo")

    s = env.reset()
    frames = []
    for _ in range(max_steps):
        a = agent.act(s)
        s, _, done, _ = env.step(a)
        frames.append((env.r, env.c))
        if done:
            break

    ani = animation.FuncAnimation(fig, update, frames=len(frames), repeat=False)
    plt.show()