import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from helperClass import MultiAgentWrapper
from intercept_env import PursuitEvasionEnv

GOOD_MODEL_PATH = "good_policy.zip"
ADVERSARY_MODEL_PATH = "adversary_policy.zip"

# ------------------- ENV FACTORY -------------------
def make_env(N_good=2, N_adv=2):
    """Create an environment instance wrapped for training good agents."""
    env = PursuitEvasionEnv(N_adversaries=N_adv, M_good=N_good, width_ratio=2.0)

    # Load adversary agent policy (if exists) as fixed opponents
    if os.path.exists(ADVERSARY_MODEL_PATH):
        print("✅ Loading adversary model as opponents...")
        adv_model = PPO.load(ADVERSARY_MODEL_PATH)

        # Wrap env step() dynamically with opponent model
        original_step = env.step

        def wrapped_step(actions):
            obs = env._get_obs()
            # Let adversary agents act using their trained policy
            for agent in env.agents:
                if "adversary" in agent and agent not in actions:
                    obs_vec = obs[agent].reshape(1, -1)
                    act, _ = adv_model.predict(obs_vec, deterministic=True)
                    actions[agent] = act[0]
            return original_step(actions)

        env.step = wrapped_step

    # Wrap environment for good team
    env = MultiAgentWrapper(env, agent_type="good")
    return env

# ------------------- OPTUNA OBJECTIVE -------------------
def optimize_ppo(trial, N_good=2, N_adv=2):
    """Run one Optuna trial."""
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    env = DummyVecEnv([make_env(N_good, N_adv)])
    model = PPO("MlpPolicy", env,
                learning_rate=lr, gamma=gamma,
                n_steps=n_steps, batch_size=batch_size,
                verbose=0)

    model.learn(total_timesteps=200_000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()
    return mean_reward

# ------------------- MAIN TRAINING -------------------
def train_good(N_good=2, N_adv=2):
    # Optuna hyperparameter search
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_ppo(trial, N_good, N_adv), n_trials=5)
    print("Best hyperparameters:", study.best_params)

    # Train final model using best params
    vec_env = SubprocVecEnv([make_env for _ in range(4)])
    model = PPO("MlpPolicy", vec_env, verbose=1, **study.best_params)
    model.learn(total_timesteps=1_000_000)
    model.save(GOOD_MODEL_PATH)
    print("✅ Saved good model to:", GOOD_MODEL_PATH)


if __name__ == "__main__":
    train_good(N_good=3, N_adv=2)
