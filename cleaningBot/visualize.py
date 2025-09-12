import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

def plot_rewards(rewards):
    plt.figure(figsize=(8,4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.grid(True)
    plt.show(block=False)

def plotMap(env):
    grid = env.grid.copy()
    fig, ax = plt.subplots()

    mat = ax.matshow(np.logical_not(grid), cmap="gray")
    ax.set_title("Generated Map")
    plt.show(block=False)

def plot_paths(env, agent, max_steps=20*20):
    grid = env.grid.copy()
    fig, ax = plt.subplots()
    mat = ax.matshow(np.logical_not(grid), cmap="gray")
    ax.set_title("Agent Trajectory")

    s = env.reset()
    x_vals = []
    y_vals = []
    rewards = 0
    isDone = []

    # Run agent and record trajectory
    for stepCount in range(max_steps):
        # Run a step
        a = agent.act(s)
        s, r, done, _ = env.step(a, stepCount == max_steps -1)
        # record position
        isDone.append(done)
        x_vals.append(env.c)
        y_vals.append(env.r)
        rewards += r
        if done:
            break

    # create an empty line object that we will update
    path_line, = ax.plot([], [], "-ob")

    plt.ion()
    plt.show()

    # Animate the path
    for stepCount in range(len(x_vals)):
        path_line.set_data(x_vals[:stepCount + 1], y_vals[:stepCount + 1])
        plt.pause(0.5)

    print(f"Reward={rewards:.2f}")
    plt.ioff()
    plt.show()
