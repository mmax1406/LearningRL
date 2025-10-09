import optuna
import gymnasium as gym
import torch as th
import pandas as pd
from world import GridCleanEnvGym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Create a custom class for me
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, conv_arch=[(32,3,1),(64,3,1)], fc_arch=[64,64], activation_fn=th.nn.ReLU):
        # Compute dummy features_dim, will overwrite later
        super().__init__(observation_space, features_dim=1)

        n_channels = observation_space.shape[2]  # NHWC (H,W,C)
        self.conv_layers = []
        in_ch = n_channels
        for out_ch, k, s in conv_arch:
            self.conv_layers.append(th.nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s))
            self.conv_layers.append(activation_fn())
            in_ch = out_ch
        self.conv = th.nn.Sequential(*self.conv_layers)

        # compute conv output size
        with th.no_grad():
            x = th.zeros(1, n_channels, observation_space.shape[0], observation_space.shape[1])
            conv_out = self.conv(x)
        conv_out_size = int(np.prod(conv_out.shape))

        # Fully connected layers
        layers = []
        in_features = conv_out_size
        for h in fc_arch:
            layers.append(th.nn.Linear(in_features, h))
            layers.append(activation_fn())
            in_features = h
        self.fc = th.nn.Sequential(*layers)

        self._features_dim = fc_arch[-1]

    def forward(self, x):
        x = x.permute(0,3,1,2)  # NHWC → NCHW
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

# ---------- Objective Function ----------
def optimize_ppo_cnn(trial):
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

    # --- CNN architecture parameters ---
    # List of conv layers: [(out_channels, kernel, stride), ...]
    conv_arch = trial.suggest_categorical("conv_arch", [
        [(32, 3, 1), (64, 3, 1)],
        [(16, 3, 1), (32, 3, 1), (64, 3, 1)],
        [(64, 3, 1), (64, 3, 1), (128, 3, 1)]
    ])
    # FC layers after CNN
    fc_arch = trial.suggest_categorical("fc_arch", [[64, 64], [128, 128], [256, 256]])
    activation_fn = trial.suggest_categorical("activation_fn", [th.nn.Tanh, th.nn.ReLU])

    # --- Make environment ---
    def make_env():
        env = GridCleanEnvGym(grid_size=10, num_obstacles=3, obstacles_max_size=3, max_steps=200)
        return Monitor(env)

    env = DummyVecEnv([make_env for _ in range(5)])

    # --- CNN Policy kwargs ---
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(conv_arch=conv_arch, fc_arch=fc_arch),
        activation_fn=activation_fn
    )

    # --- Train PPO ---
    model = PPO(
        "CnnPolicy",
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
    model.learn(total_timesteps=50_000)

    # --- Evaluate ---
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    env.close()
    return mean_reward


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo_cnn, n_trials=30, n_jobs=1)

    df = study.trials_dataframe()
    df.to_csv("ppo_gridclean_optuna_results.csv", index=False)

    # --- Retrain final model with best params ---
    best_trial = study.best_trial
    best_params = best_trial.params

    def make_best_env():
        env = GridCleanEnvGym(grid_size=10, num_obstacles=3, obstacles_max_size=3, max_steps=200)
        return Monitor(env)

    env = DummyVecEnv([make_best_env])

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(conv_arch=best_params["conv_arch"],
                                       fc_arch=best_params["fc_arch"]),
        activation_fn=best_params["activation_fn"]
    )

    model = PPO(
        "CnnPolicy",
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
    model.save("ppo_gridclean_best")
    env.close()
