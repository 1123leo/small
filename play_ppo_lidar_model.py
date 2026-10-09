import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO

MODEL_PATH = "ppo_flappybird_final_v2"
TOTAL_EPISODES = 3
AI_SPEED_MULTIPLIER = 100
HUMAN_TARGET_FPS = 120


def make_env():
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    try:
        env.metadata["render_fps"] = HUMAN_TARGET_FPS
    except Exception:
        pass
    return env


def run_human(env, model):
    episode_scores = []

    for episode_idx in range(TOTAL_EPISODES):
        obs, _ = env.reset()
        score = 0
        done = False

        while not done:
            for _ in range(AI_SPEED_MULTIPLIER):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                score = info.get("score", 0)

                if terminated or truncated:
                    done = True
                    break

        episode_scores.append(score)
        print(f"Episode {episode_idx + 1}: Score = {score}")

    return episode_scores


def print_summary(scores):
    best_score = max(scores) if scores else 0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    print(f"Best Score: {best_score}")
    print(f"Average Score: {avg_score:.2f}")


def main():
    model = PPO.load(MODEL_PATH)
    env = make_env()

    try:
        scores = run_human(env, model)
        print_summary(scores)
    finally:
        env.close()


if __name__ == "__main__":
    main()
