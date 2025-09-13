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
    # Reset environment
    s = env.reset()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Agent Trajectory")

    def render_state():
        """Render environment state as RGB image."""
        n, m, _ = s.shape
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

def plot_paths(env, agent, max_steps=20*20):
    # Reset environment
    s = env.reset()
    rewards, x_vals, y_vals, isDone = 0, [], [], []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Agent Trajectory")

    def render_state():
        """Render environment state as RGB image."""
        n, m, _ = s.shape
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
        a = agent.act(s)
        s, r, done, _ = env.step(a, stepCount == max_steps - 1)

        x_vals.append(env.c)
        y_vals.append(env.r)
        isDone.append(done)
        rewards += r

        # update display
        mat.set_data(render_state())
        ax.plot(x_vals, y_vals, "-r")  # path in red
        plt.pause(0.05)

        if done:
            break

    print(f"Reward={rewards:.2f}")
    plt.ioff()
    plt.show()

