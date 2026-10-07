import argparse
import importlib
import time
from collections import deque


RUNTIME_MODULES = [
    ("stable_baselines3", "stable-baselines3==2.8.0"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("mss", "mss"),
    ("pydirectinput", "pydirectinput"),
]

BASELINE_EXTRA_MODULES = [
    ("gymnasium", "gymnasium"),
    ("flappy_bird_gymnasium", "flappy-bird-gymnasium"),
]

BASELINE_PACKAGES = [
    "stable-baselines3==2.8.0",
    "opencv-python",
    "numpy",
    "mss",
    "pydirectinput",
    "gymnasium",
    "flappy-bird-gymnasium",
]

FEATURE_NAMES = [
    "bird_y",
    "bird_vel",
    "target_x",
    "bird_minus_gap_center",
    "target_top",
    "target_bottom",
    "bird_vel_mul_target_x",
    "falling_flag",
    "p2_x_if_p1_behind",
]

GAME_REGION = {"top": 50, "left": 100, "width": 500, "height": 800}
STACK_SIZE = 4
FEATURE_DIM = 9
EXPECTED_OBS_DIM = STACK_SIZE * FEATURE_DIM

# HSV thresholds
BIRD_LOWER = (20, 100, 100)
BIRD_UPPER = (40, 255, 255)
PIPE_LOWER = (35, 50, 50)
PIPE_UPPER = (85, 255, 255)


def check_dependencies():
    loaded = {}
    missing_runtime = []
    for module_name, package_name in RUNTIME_MODULES:
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing_runtime.append((module_name, package_name))

    if missing_runtime:
        print("Missing required modules:")
        for module_name, package_name in missing_runtime:
            print(f"  - {module_name} (pip package: {package_name})")
        print("")
        print("Install baseline packages with:")
        print(f"  pip install {' '.join(BASELINE_PACKAGES)}")
        return None

    missing_baseline = []
    for module_name, package_name in BASELINE_EXTRA_MODULES:
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing_baseline.append((module_name, package_name))

    if missing_baseline:
        print("Warning: some baseline parity modules are missing:")
        for module_name, package_name in missing_baseline:
            print(f"  - {module_name} (pip package: {package_name})")
        print("The screen player can still run, but full test_model parity checks may be limited.")

    sb3_version = getattr(loaded["stable_baselines3"], "__version__", "unknown")
    if sb3_version != "2.8.0":
        print(
            f"Warning: stable_baselines3 version is {sb3_version}, "
            "recommended is 2.8.0."
        )

    return loaded


class FeatureBuilder:
    def __init__(self, np_module, cv2_module, game_region, stack_size=4, max_missing_frames=5):
        self.np = np_module
        self.cv2 = cv2_module
        self.game_region = game_region
        self.stack_size = stack_size
        self.max_missing_frames = max_missing_frames

        self.bird_lower = self.np.array(BIRD_LOWER, dtype=self.np.uint8)
        self.bird_upper = self.np.array(BIRD_UPPER, dtype=self.np.uint8)
        self.pipe_lower = self.np.array(PIPE_LOWER, dtype=self.np.uint8)
        self.pipe_upper = self.np.array(PIPE_UPPER, dtype=self.np.uint8)
        self.kernel = self.np.ones((5, 5), self.np.uint8)

        self.frames = deque(maxlen=self.stack_size)
        self.y_history = deque(maxlen=5)

        self.last_features9 = None
        self.last_stacked_obs = None
        self.last_obs_reused = False
        self.missing_count = 0

        self.feature_mins = self.np.full(FEATURE_DIM, self.np.inf, dtype=self.np.float32)
        self.feature_maxs = self.np.full(FEATURE_DIM, -self.np.inf, dtype=self.np.float32)

    def get_observation(self, sct):
        frame = self.np.array(sct.grab(self.game_region))
        features9 = self.extract_features9(frame)

        if features9 is None:
            self.missing_count += 1
            self.last_obs_reused = False
            if self.last_stacked_obs is not None and self.missing_count <= self.max_missing_frames:
                self.last_obs_reused = True
                return self.last_stacked_obs
            return None

        self.missing_count = 0
        self.last_obs_reused = False
        return self.build_stacked_obs(features9)

    def extract_features9(self, frame_bgra):
        bgr = self.cv2.cvtColor(frame_bgra, self.cv2.COLOR_BGRA2BGR)
        hsv = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2HSV)

        bird = self._detect_bird(hsv)
        if bird is None:
            return None

        bird_x, bird_y = bird
        height = float(self.game_region["height"])
        width = float(self.game_region["width"])
        bird_y_norm = bird_y / height

        self.y_history.append(bird_y_norm)
        if len(self.y_history) >= 3:
            bird_vel = (self.y_history[-1] - self.y_history[-3]) / 2.0
        elif len(self.y_history) >= 2:
            bird_vel = self.y_history[-1] - self.y_history[-2]
        else:
            bird_vel = 0.0

        pipe_groups = self._build_pipe_groups(hsv, bird_x)
        if not pipe_groups:
            return None

        p1 = pipe_groups[0]
        if len(pipe_groups) > 1:
            p2 = pipe_groups[1]
        else:
            p2 = {
                "x_rel": p1["x_rel"] + 0.25,
                "gap_top_norm": p1["gap_top_norm"],
                "gap_bottom_norm": p1["gap_bottom_norm"],
            }

        use_p2_as_target = p1["x_rel"] < -0.1
        target = p2 if use_p2_as_target else p1

        target_top = target["gap_top_norm"]
        target_bottom = target["gap_bottom_norm"]
        gap_center = (target_top + target_bottom) / 2.0

        features = self.np.array(
            [
                bird_y_norm,
                bird_vel,
                target["x_rel"],
                bird_y_norm - gap_center,
                target_top,
                target_bottom,
                bird_vel * target["x_rel"],
                1.0 if bird_vel < -0.5 else 0.0,
                p2["x_rel"] if p1["x_rel"] < 0.0 else 0.0,
            ],
            dtype=self.np.float32,
        )

        features = self.np.clip(features, -10.0, 10.0).astype(self.np.float32)
        self._update_feature_range(features)
        self.last_features9 = features
        return features

    def build_stacked_obs(self, features9):
        if len(self.frames) == 0:
            for _ in range(self.stack_size):
                self.frames.append(features9.copy())
        else:
            self.frames.append(features9.copy())
            while len(self.frames) < self.stack_size:
                self.frames.append(features9.copy())

        stacked = self.np.concatenate(list(self.frames), axis=0).astype(self.np.float32)
        self.last_stacked_obs = stacked
        return stacked

    def _detect_bird(self, hsv):
        bird_mask = self.cv2.inRange(hsv, self.bird_lower, self.bird_upper)
        bird_mask = self.cv2.morphologyEx(bird_mask, self.cv2.MORPH_OPEN, self.kernel)

        contours, _ = self.cv2.findContours(
            bird_mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        bird_cnt = max(contours, key=self.cv2.contourArea)
        if self.cv2.contourArea(bird_cnt) < 100:
            return None

        bx, by, bw, bh = self.cv2.boundingRect(bird_cnt)
        return bx + (bw // 2), by + (bh // 2)

    def _build_pipe_groups(self, hsv, bird_x):
        pipe_mask = self.cv2.inRange(hsv, self.pipe_lower, self.pipe_upper)
        pipe_mask = self.cv2.morphologyEx(pipe_mask, self.cv2.MORPH_OPEN, self.kernel)

        contours, _ = self.cv2.findContours(
            pipe_mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
        )
        segments = []
        for cnt in contours:
            x, y, w, h = self.cv2.boundingRect(cnt)
            if w > 20 and h > 50 and self.cv2.contourArea(cnt) > 500:
                segments.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "top": y,
                        "bottom": y + h,
                        "center_x": x + (w / 2.0),
                    }
                )

        if not segments:
            return []

        segments.sort(key=lambda item: item["center_x"])
        grouped = []
        tolerance = 30.0
        for seg in segments:
            if not grouped:
                grouped.append({"x_center": seg["center_x"], "members": [seg]})
                continue

            last_group = grouped[-1]
            if abs(seg["center_x"] - last_group["x_center"]) <= tolerance:
                members = last_group["members"]
                members.append(seg)
                last_group["x_center"] = sum(m["center_x"] for m in members) / len(members)
            else:
                grouped.append({"x_center": seg["center_x"], "members": [seg]})

        height = float(self.game_region["height"])
        width = float(self.game_region["width"])
        candidates = []
        for group in grouped:
            members = sorted(group["members"], key=lambda m: m["y"])
            gap_top = None
            gap_bottom = None

            if len(members) >= 2:
                gap_top = float(members[0]["bottom"])
                gap_bottom = float(members[-1]["top"])
            else:
                single = members[0]
                if single["y"] < (height * 0.5):
                    gap_top = float(single["bottom"])
                    gap_bottom = min(height, gap_top + 120.0)
                else:
                    gap_bottom = float(single["top"])
                    gap_top = max(0.0, gap_bottom - 120.0)

            if gap_bottom <= gap_top + 5.0:
                continue

            x_rel = (group["x_center"] - float(bird_x)) / width
            candidates.append(
                {
                    "x_rel": x_rel,
                    "gap_top_norm": gap_top / height,
                    "gap_bottom_norm": gap_bottom / height,
                }
            )

        if not candidates:
            return []

        candidates.sort(key=lambda item: item["x_rel"])
        near_candidates = [c for c in candidates if c["x_rel"] > -0.5]
        if near_candidates:
            return near_candidates

        return candidates[-2:] if len(candidates) >= 2 else [candidates[-1]]

    def _update_feature_range(self, features9):
        self.feature_mins = self.np.minimum(self.feature_mins, features9)
        self.feature_maxs = self.np.maximum(self.feature_maxs, features9)

    def print_feature_ranges(self):
        if self.last_features9 is None:
            print("No valid 9D features were extracted yet.")
            return

        print("\nFeature ranges (min -> max):")
        for idx, name in enumerate(FEATURE_NAMES):
            fmin = float(self.feature_mins[idx])
            fmax = float(self.feature_maxs[idx])
            print(f"  [{idx}] {name}: {fmin:+.4f} -> {fmax:+.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Flappy Bird screen player using PPO model.")
    parser.add_argument("--model-path", default="best_model", help="Path to PPO model.")
    parser.add_argument("--dry-run", action="store_true", help="Predict only, do not press keys.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N loop frames. 0 means run forever.",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=5,
        help="Reuse last observation for this many missing frames.",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="Main loop target FPS.")
    parser.add_argument(
        "--warmup-captures",
        type=int,
        default=5,
        help="Initial captures to fill velocity/stack buffers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    loaded = check_dependencies()
    if loaded is None:
        return 1

    np = loaded["numpy"]
    mss = loaded["mss"]
    cv2 = loaded["cv2"]
    pydirectinput = loaded["pydirectinput"]
    PPO = loaded["stable_baselines3"].PPO

    model = PPO.load(args.model_path)
    print(f"model.observation_space = {model.observation_space}")
    print(f"model.action_space = {model.action_space}")

    obs_shape = tuple(getattr(model.observation_space, "shape", ()))
    if obs_shape != (EXPECTED_OBS_DIM,):
        print(
            "Model observation shape mismatch. "
            f"Expected {(EXPECTED_OBS_DIM,)}, got {obs_shape}."
        )
        return 2

    sleep_seconds = 1.0 / max(args.fps, 1.0)
    extractor = FeatureBuilder(
        np_module=np,
        cv2_module=cv2,
        game_region=GAME_REGION,
        stack_size=STACK_SIZE,
        max_missing_frames=max(0, args.max_missing_frames),
    )

    print("Starting loop...")
    print(f"GAME_REGION = {GAME_REGION}")
    print(f"dry_run = {args.dry_run}")
    time.sleep(1.0)

    loop_idx = 0
    try:
        with mss.MSS() as sct:
            for _ in range(max(0, args.warmup_captures)):
                extractor.get_observation(sct)
                time.sleep(0.03)

            while True:
                loop_idx += 1
                obs = extractor.get_observation(sct)

                if obs is None:
                    if extractor.missing_count == extractor.max_missing_frames + 1:
                        print(
                            "Detection missing too long. "
                            "Pausing action until new features are found."
                        )
                    time.sleep(sleep_seconds)
                    if args.max_frames > 0 and loop_idx >= args.max_frames:
                        break
                    continue

                if obs.shape != (EXPECTED_OBS_DIM,):
                    raise ValueError(f"Unexpected observation shape: {obs.shape}")
                if not np.isfinite(obs).all():
                    raise ValueError("Observation contains non-finite values.")

                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

                if extractor.last_features9 is not None and not extractor.last_obs_reused:
                    feat = extractor.last_features9
                    print(
                        "f9 "
                        + ", ".join(
                            f"{name}={value:+.3f}" for name, value in zip(FEATURE_NAMES, feat)
                        )
                    )

                if args.dry_run:
                    reuse_tag = " [reused_obs]" if extractor.last_obs_reused else ""
                    print(f"frame={loop_idx:05d} action={action}{reuse_tag}")
                else:
                    if action == 1:
                        pydirectinput.press("space")
                        print("pressed space")

                time.sleep(sleep_seconds)

                if args.max_frames > 0 and loop_idx >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")

    extractor.print_feature_ranges()
    print(f"Finished after {loop_idx} loop frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
