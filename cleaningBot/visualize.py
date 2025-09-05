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

def plotMap(env):
    grid = env.grid.copy()
    fig, ax = plt.subplots()

    mat = ax.matshow(np.logical_not(grid), cmap="gray")
    ax.set_title("Generated Map")
    plt.show()

def plot_paths(env, agent, max_steps=50):
    grid = env.grid.copy()
    fig, ax = plt.subplots()
    mat = ax.matshow(np.logical_not(grid), cmap="gray")
    ax.set_title("Agent Trajectory")

    s = env.reset()
    positions_r = []
    positions_c = []
    rewards = 0

    # Run agent and record trajectory
    for stepCount in range(max_steps):
        # Run a step
        a = agent.act(s)
        s, r, done, _ = env.step(a, stepCount == max_steps -1)
        # record position
        positions_r.append(env.r)
        positions_c.append(env.c)
        rewards += r

    ax.plot(positions_r, positions_c, '-r')
    plt.show()
    print(f"Reward={rewards:.2f}")

def animate_agent(env, agent, max_steps=50):
    grid = env.grid.copy()
    fig, ax = plt.subplots()
    mat = ax.matshow(np.logical_not(grid), cmap="gray")
    trail, = ax.plot([], [], "r.", markersize=6)  # trail for past positions
    agent_dot, = ax.plot([], [], "ro", markersize=12)
    ax.set_title("Agent Trajectory")
    plt.ion()  # turn on interactive mode
    plt.show()

    s = env.reset()
    positions_r = []
    positions_c = []

    # Run agent and record trajectory
    for stepCount in range(max_steps):
        # Run a step
        a = agent.act(s)
        s, _, done, _ = env.step(a, stepCount == max_steps -1)
        # record position
        positions_r.append(env.r)
        positions_c.append(env.c)

        # update trail and agent dot
        trail.set_data(positions_c, positions_r)
        agent_dot.set_data(env.c, env.r)
        # ax.plot(positions_r, positions_c,'-r')

        # Pause for visulaiztion
        plt.pause(0.5)
        if done:
            break

    plt.ioff()
    plt.show()