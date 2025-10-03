import optuna
import gymnasium as gym
import torch as th
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------- Custom Env with Domain Randomization ----------
class RandomizedCartPole(gym.Wrapper):
    def __init__(self, env, gravity, pole_length):
        super().__init__(env)
        self.unwrapped.gravity = gravity
        self.unwrapped.length = pole_length

    def reset(self, **kwargs):
        return super().reset(**kwargs)


# ---------- Objective Function ----------
def optimize_ppo(trial):

    # --- Domain randomization parameters ---
    gravity = trial.suggest_float("gravity", 8.0, 12.0)
    pole_length = trial.suggest_float("pole_length", 0.3, 0.8)

    # --- PPO hyperparams ---
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.7, 1.0])

    net_arch = trial.suggest_categorical("net_arch", [
        [64, 64], [128, 128], [256, 256], [256, 128], [400, 300]])
    activation_fn = trial.suggest_categorical("activation_fn", [th.nn.Tanh, th.nn.ReLU])

    # --- Make environment with domain randomization ---
    def make_env():
        env = gym.make("CartPole-v1")
        env = RandomizedCartPole(env, gravity=gravity, pole_length=pole_length)
        return Monitor(env)

    env = DummyVecEnv([make_env])

    # --- Policy ---
    policy_kwargs = dict(net_arch=net_arch, activation_fn=activation_fn)

    # --- Train PPO ---
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    model.learn(total_timesteps=30_000)

    # --- Evaluate ---
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

    env.close()
    return mean_reward


if __name__ == "__main__":
    # Run study
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=20, n_jobs=1)

    # Save results to CSV
    df = study.trials_dataframe()
    df.to_csv("ppo_cartpole_optuna_results.csv", index=False)

    # Best trial
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print(f"  Params: {best_trial.params}")

    # Retrain final model with best params
    best_params = best_trial.params

    def make_best_env():
        env = gym.make("CartPole-v1")
        env = RandomizedCartPole(env,
                                 gravity=best_params["gravity"],
                                 pole_length=best_params["pole_length"])
        return Monitor(env)

    env = DummyVecEnv([make_best_env])

    policy_kwargs = dict(
        net_arch=best_params["net_arch"],
        activation_fn=best_params["activation_fn"]
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        gamma=best_params["gamma"],
        learning_rate=best_params["learning_rate"],
        clip_range=best_params["clip_range"],
        gae_lambda=best_params["gae_lambda"],
        ent_coef=best_params["ent_coef"],
        vf_coef=best_params["vf_coef"],
        max_grad_norm=best_params["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    model.learn(total_timesteps=10_000)
    model.save("ppo_cartpole_best")
    env.close()
