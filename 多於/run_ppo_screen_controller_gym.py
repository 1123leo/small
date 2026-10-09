import argparse
import ctypes
import importlib
import time
from collections import deque
from ctypes import wintypes


RUNTIME_MODULES = [
    ("stable_baselines3", "stable-baselines3==2.8.0"),
    ("gymnasium", "gymnasium"),
    ("flappy_bird_gymnasium", "flappy-bird-gymnasium"),
    ("pygame", "pygame"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("mss", "mss"),
]

BASELINE_PACKAGES = [
    "stable-baselines3==2.8.0",
    "gymnasium",
    "flappy-bird-gymnasium",
    "pygame",
    "opencv-python",
    "numpy",
    "mss",
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

TARGET_FRAME_WIDTH = 500
TARGET_FRAME_HEIGHT = 800

STACK_SIZE = 4
FEATURE_DIM = 9
EXPECTED_OBS_DIM = STACK_SIZE * FEATURE_DIM

# HSV thresholds
BIRD_LOWER = (15, 80, 80)
BIRD_UPPER = (30, 200, 200)
PIPE_LOWER = (0, 0, 80)
PIPE_UPPER = (180, 30, 150)


def check_dependencies():
    loaded = {}
    missing = []
    for module_name, package_name in RUNTIME_MODULES:
        try:
            loaded[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append((module_name, package_name))

    if missing:
        print("Missing required modules:")
        for module_name, package_name in missing:
            print(f"  - {module_name} (pip package: {package_name})")
        print("")
        print("Install baseline packages with:")
        print(f"  pip install {' '.join(BASELINE_PACKAGES)}")
        return None

    sb3_version = getattr(loaded["stable_baselines3"], "__version__", "unknown")
    if sb3_version != "2.8.0":
        print(
            f"Warning: stable_baselines3 version is {sb3_version}, "
            "recommended is 2.8.0."
        )

    return loaded


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _get_window_title(user32, hwnd):
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value.strip()


def _get_window_client_region(user32, hwnd):
    client_rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
        return None

    width = int(client_rect.right - client_rect.left)
    height = int(client_rect.bottom - client_rect.top)
    if width <= 0 or height <= 0:
        return None

    top_left = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None

    return {
        "left": int(top_left.x),
        "top": int(top_left.y),
        "width": width,
        "height": height,
    }


def find_window_region(title_keyword):
    user32 = ctypes.windll.user32
    keyword = title_keyword.lower()
    matches = []

    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    @enum_proc
    def _callback(hwnd, lparam):
        if not user32.IsWindowVisible(hwnd):
            return True

        title = _get_window_title(user32, hwnd)
        if not title:
            return True

        if keyword not in title.lower():
            return True

        region = _get_window_client_region(user32, hwnd)
        if region is None:
            return True

        area = region["width"] * region["height"]
        if area < 10_000:
            return True

        matches.append((area, hwnd, title, region))
        return True

    user32.EnumWindows(_callback, 0)

    if not matches:
        return None, None, None

    matches.sort(key=lambda item: item[0], reverse=True)
    _, hwnd, title, region = matches[0]
    return hwnd, title, region


def wait_for_window_region(title_keyword, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        hwnd, title, region = find_window_region(title_keyword)
        if hwnd is not None:
            return hwnd, title, region
        time.sleep(0.2)
    raise RuntimeError(
        f"Cannot find visible window with title containing '{title_keyword}' "
        f"within {timeout_seconds:.1f}s."
    )


def get_pygame_window_region(pygame_module):
    try:
        wm_info = pygame_module.display.get_wm_info()
    except Exception:
        return None, None, None

    hwnd = wm_info.get("window")
    if not hwnd:
        return None, None, None

    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd):
        return None, None, None

    region = _get_window_client_region(user32, hwnd)
    if region is None:
        return None, None, None

    title = _get_window_title(user32, hwnd) or "pygame-window"
    return hwnd, title, region


def prime_gym_window(env, pygame_module, attempts=30, sleep_s=0.05):
    hwnd = None
    title = None
    region = None
    for _ in range(max(1, attempts)):
        try:
            env.render()
        except Exception:
            pass
        hwnd, title, region = get_pygame_window_region(pygame_module)
        if hwnd is not None:
            return hwnd, title, region
        time.sleep(sleep_s)
    return None, None, None


class FeatureBuilder:
    def __init__(self, np_module, cv2_module, stack_size=4, max_missing_frames=5):
        self.np = np_module
        self.cv2 = cv2_module
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

    def get_observation_from_bgra(self, frame_bgra):
        features9 = self.extract_features9(frame_bgra)

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
        height = float(TARGET_FRAME_HEIGHT)
        width = float(TARGET_FRAME_WIDTH)
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

        target = p2 if p1["x_rel"] < -0.1 else p1
        gap_center = (target["gap_top_norm"] + target["gap_bottom_norm"]) / 2.0

        features = self.np.array(
            [
                bird_y_norm,
                bird_vel,
                target["x_rel"],
                bird_y_norm - gap_center,
                target["gap_top_norm"],
                target["gap_bottom_norm"],
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

        height = float(TARGET_FRAME_HEIGHT)
        width = float(TARGET_FRAME_WIDTH)
        candidates = []
        for group in grouped:
            members = sorted(group["members"], key=lambda m: m["y"])
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
    parser = argparse.ArgumentParser(
        description="Gym Flappy Bird: capture with MSS, infer with PPO, act in gym."
    )
    parser.add_argument("--model-path", default="best_model", help="Path to PPO model.")
    parser.add_argument("--fps", type=float, default=60.0, help="Loop FPS.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call env.step().")
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=5,
        help="Reuse last valid obs for this many missing frames.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Stop after N finished episodes. 0 means no limit.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=0,
        help="Stop/reset after N steps in one episode. 0 means no limit.",
    )
    parser.add_argument(
        "--window-title",
        default="Flappy",
        help="Keyword for auto-detecting the gym window title.",
    )
    parser.add_argument(
        "--capture-retry-seconds",
        type=float,
        default=10.0,
        help="Timeout for auto-detecting gym window.",
    )
    return parser.parse_args()


def _ensure_target_size(np_module, cv2_module, frame_bgra):
    h, w = frame_bgra.shape[:2]
    if w == TARGET_FRAME_WIDTH and h == TARGET_FRAME_HEIGHT:
        return frame_bgra
    return cv2_module.resize(
        frame_bgra,
        (TARGET_FRAME_WIDTH, TARGET_FRAME_HEIGHT),
        interpolation=cv2_module.INTER_LINEAR,
    )


def _put_overlay(cv2_module, frame_bgr, episode_no, step_no, score, action, reused):
    line1 = f"episode={episode_no} step={step_no} score={score}"
    line2 = f"action={action} reused_obs={reused}"
    cv2_module.putText(
        frame_bgr, line1, (10, 24), cv2_module.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    cv2_module.putText(
        frame_bgr, line2, (10, 50), cv2_module.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2
    )


def create_mss_session(mss_module):
    if hasattr(mss_module, "MSS"):
        return mss_module.MSS()
    if hasattr(mss_module, "mss"):
        return mss_module.mss()
    raise RuntimeError("Unsupported mss module: no MSS() or mss() factory available.")


def main():
    args = parse_args()
    loaded = check_dependencies()
    if loaded is None:
        return 1

    np = loaded["numpy"]
    cv2 = loaded["cv2"]
    mss = loaded["mss"]
    gym = loaded["gymnasium"]
    pygame = loaded["pygame"]
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

    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env.reset(seed=1019)
    time.sleep(0.2)

    # Prefer direct pygame window handle (more reliable than title matching).
    hwnd, window_name, region = prime_gym_window(env, pygame_module=pygame, attempts=40, sleep_s=0.05)

    # Fallback: search by title keyword if pygame handle is unavailable.
    if hwnd is None:
        try:
            _, window_name, region = wait_for_window_region(
                args.window_title, args.capture_retry_seconds
            )
        except RuntimeError as exc:
            env.close()
            print(str(exc))
            print(
                "Tip: try a broader --window-title keyword, "
                "or ensure the gym window is visible (not minimized/background)."
            )
            return 3

    print(f"Captured gym window: {window_name}")
    print(f"MSS region: {region}")

    extractor = FeatureBuilder(
        np_module=np,
        cv2_module=cv2,
        stack_size=STACK_SIZE,
        max_missing_frames=max(0, args.max_missing_frames),
    )

    sleep_seconds = 1.0 / max(args.fps, 1.0)
    current_episode = 1
    current_step = 0
    finished_episodes = 0
    current_score = 0
    global_frame = 0

    try:
        with create_mss_session(mss) as sct:
            while True:
                loop_start = time.monotonic()
                global_frame += 1

                raw_bgra = np.array(sct.grab(region))
                frame_bgra = _ensure_target_size(np, cv2, raw_bgra)

                obs = extractor.get_observation_from_bgra(frame_bgra)
                action = 0

                if obs is not None:
                    if obs.shape != (EXPECTED_OBS_DIM,):
                        raise ValueError(f"Unexpected observation shape: {obs.shape}")
                    if not np.isfinite(obs).all():
                        raise ValueError("Observation contains non-finite values.")
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    if extractor.missing_count == extractor.max_missing_frames + 1:
                        print(
                            "Detection missing too long. "
                            "Sending neutral action until features recover."
                        )

                if not args.dry_run:
                    _, _, terminated, truncated, info = env.step(action)
                    current_step += 1
                    current_score = int(info.get("score", 0))

                    force_reset = (
                        args.max_steps_per_episode > 0
                        and current_step >= args.max_steps_per_episode
                    )
                    if terminated or truncated or force_reset:
                        finished_episodes += 1
                        print(
                            f"episode {current_episode} done: "
                            f"score={current_score}, steps={current_step}, "
                            f"reason={'limit' if force_reset else 'env'}"
                        )

                        if args.max_episodes > 0 and finished_episodes >= args.max_episodes:
                            break

                        env.reset(seed=1019 + finished_episodes)
                        current_episode += 1
                        current_step = 0
                        current_score = 0
                else:
                    current_step += 1
                    if args.max_steps_per_episode > 0 and current_step >= args.max_steps_per_episode:
                        print(f"dry-run reached max steps: {current_step}")
                        break

                frame_bgr = cv2.cvtColor(raw_bgra, cv2.COLOR_BGRA2BGR)
                _put_overlay(
                    cv2_module=cv2,
                    frame_bgr=frame_bgr,
                    episode_no=current_episode,
                    step_no=current_step,
                    score=current_score,
                    action=action,
                    reused=extractor.last_obs_reused,
                )
                cv2.imshow("Gym MSS Monitor", frame_bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("Stopped by user keypress.")
                    break

                elapsed = time.monotonic() - loop_start
                if elapsed < sleep_seconds:
                    time.sleep(sleep_seconds - elapsed)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        extractor.print_feature_ranges()
        cv2.destroyAllWindows()
        env.close()
        print(
            f"Finished. frames={global_frame}, "
            f"episodes={finished_episodes}, current_episode={current_episode}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
