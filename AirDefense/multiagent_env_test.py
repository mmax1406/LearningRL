from intercept_env import env
import numpy as np

environment = env(N_adversaries=2, M_good=2, render_mode="human")
observations = environment.reset()

for step in range(100):
    # random actions
    actions = {agent: environment.action_spaces[agent].sample() for agent in environment.agents}
    obs, rewards, term, trunc, infos = environment.step(actions)
    if not any(environment.active.values()):
        print("All agents removed — game over")
        break

environment.close()
