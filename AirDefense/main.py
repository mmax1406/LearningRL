import os
import shutil
from train_good_agent import train_good_agent, TRAINED_GOOD_PATH
from train_adversary_agent import train_adversary_agent, TRAINED_ADVERSARY_PATH

# -----------------------------
# CONFIG
# -----------------------------
SELF_PLAY_ITERATIONS = 5      # number of alternating training rounds

# -----------------------------
# Self-play loop
# -----------------------------
def self_play():
    for i in range(1, SELF_PLAY_ITERATIONS + 1):
        print(f"\n==================== SELF-PLAY ROUND {i} ====================")

        # ---------------------
        # Train GOOD agent
        # ---------------------
        print("\n🎯 Training Good Agent...")
        avg_reward_good = train_good_agent(use_optuna=False)
        print(f"✅ Good agent average reward this round: {avg_reward_good:.2f}")
        if os.path.exists(TRAINED_GOOD_PATH):
            shutil.copy(TRAINED_GOOD_PATH, f"good_policy_round{i}.pt")

        # ---------------------
        # Train ADVERSARY agent
        # ---------------------
        print("\n👾 Training Adversary Agent...")
        avg_reward_adv = train_adversary_agent(use_optuna=False)
        print(f"✅ Adversary agent average reward this round: {avg_reward_adv:.2f}")
        if os.path.exists(TRAINED_ADVERSARY_PATH):
            shutil.copy(TRAINED_ADVERSARY_PATH, f"adversary_policy_round{i}.pt")

    print("\n==================== SELF-PLAY FINISHED ====================")

if __name__ == "__main__":
    self_play()
