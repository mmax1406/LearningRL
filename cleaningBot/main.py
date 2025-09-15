from visualize import *
from qnn import *
from world import *
from A2C import *
import pickle

if __name__ == "__main__":

    test = 0

    if not test:
        env = GridCleanEnv(20, 5, 6)
        plotMap(env)
        # Save the env
        with open("env.pkl", "wb") as f:
            pickle.dump(env, f)

        # Train model and save the weights
        # trainer = DQNTrainer(env)
        trainer = A2CTrainer(env)

        # plot the rewards
        rewards = trainer.train()
        plot_rewards(rewards)

    # Recreate Env
    with open("env.pkl", "rb") as f:
        env = pickle.load(f)

    # Load a model with the trained weights
    # agent = DQNAgent(env)
    agent = A2C_Agent(env)

    # Plot
    plot_paths(env, agent)