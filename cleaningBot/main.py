from visualize import *
from qnn import *
from world import *

if __name__ == "__main__":
    grid = np.array([
        [1,1,1,1,1,1],
        [1,0,0,0,0,1],
        [1,0,1,0,0,1],
        [1,0,0,0,0,1],
        [1,1,1,1,1,1],
    ], dtype=np.float32)

    start = (1,1)
    env = GridCleanEnv(grid, start)

    trainer = DQNTrainer(env)
    rewards = trainer.train()

    plot_rewards(rewards)

    agent = DQNAgent(trainer.qnet)
    animate_agent(env, agent)