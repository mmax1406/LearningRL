from sharedPolicyWrapper import SharedPolicyWrapper
from intercept_env import env
# ---------------- PARAMETERS ----------------
N_ITERATIONS = 1
MIN_GOOD, MAX_GOOD = 5, 10
MIN_ADV, MAX_ADV = 1, 5

# Paths to store trained models
GOOD_MODEL_PATH = "good_policy.zip"
ADVERSARY_MODEL_PATH = "adversary_policy.zip"

# ---------------- MAIN LOOP ----------------
for i in range(N_ITERATIONS):
    print(f"\n==== Iteration {i + 1}/{N_ITERATIONS} ====")
    multiEnv = SharedPolicyWrapper(env(N_adversaries=3, M_good=5, width_ratio=3.0))
    observations, info = multiEnv.reset()


    # # Train good agents
    # good_model_path = train_good(N_good=N_good, N_adv=N_adv)
    # # good_model_path = train_good(N_good=N_good, N_adv=N_adv, opponent_model_path=ADVERSARY_MODEL_PATH)
    # print(f"✅ Saved good model: {good_model_path}")
    #
    # # Train adversary agents
    # adv_model_path = train_adversary(N_adv=N_adv, N_good=N_good)
    # # adv_model_path = train_adversary(N_adv=N_adv, N_good=N_good, opponent_model_path=GOOD_MODEL_PATH)
    # print(f"✅ Saved adversary model: {adv_model_path}")

print("\n🎉 Curriculum training completed!")
