from visualize import *
from qnn import *
from world import *
import pickle

if __name__ == "__main__":

    env = GridCleanEnv(20, 5, 6)
    plotMap(env)
    # Save the env
    with open("env.pkl", "wb") as f:
        pickle.dump(env, f)

    # Train model and save the weights
    trainer = DQNTrainer(env)
    rewards = trainer.train()
    plot_rewards(rewards)

    # Load a model with the trained weights
    agent = DQNAgent()
    # Recreate Env
    with open("env.pkl", "rb") as f:
        env = pickle.load(f)

    # Plot
    # plotMap(env)
    # animate_agent(env, agent)
    plot_paths(env, agent)