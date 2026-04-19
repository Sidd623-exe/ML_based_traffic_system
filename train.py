"""
train.py — Task 4, File 1.

Main training entry-point for the Adaptive Traffic Signal Control project.

Runs Linear Actor-Critic on the CN+ Bremen network for ``num_episodes``
episodes and logs per-episode statistics.  A live Matplotlib dashboard
updates after each episode.

Usage::
    python train.py                           # headless, 200 episodes
    python train.py --gui                     # open SUMO GUI (slower)
    python train.py --episodes 5 --seconds 300  # quick smoke-test
"""

from __future__ import annotations

import argparse
import sys
import os
import io

# Fix Windows cp1252 encoding for Unicode characters in print statements
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")   # headless backend — no GUI event-loop overhead

# ── Dynamic SUMO_HOME Fix ───────────────────────────────────────────────────
if not os.environ.get("SUMO_HOME") or not os.path.exists(os.environ["SUMO_HOME"]):
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Project imports ─────────────────────────────────────────────────────────
from environment.observation_fn import CNPlusObservation
from environment.reward_fn import custom_reward
from agent.multi_agent import MultiAgentLinearAC
from dashboard.live_monitor import LiveDashboard

# ── sumo_rl ──────────────────────────────────────────────────────────────────
try:
    from sumo_rl import SumoEnvironment
    import traci
except ImportError as exc:
    sys.exit(f"ERROR: sumo_rl not installed. Run: pip install sumo-rl\n{exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Patch SumoEnvironment to fix the hardcoded --time-to-teleport -1
# sumo-rl injects '--time-to-teleport -1' into every SUMO command, which
# causes vehicles to wait forever when routes are bad, gridlocking the sim.
# We subclass and override _start_simulation to replace -1 with 300.
# ──────────────────────────────────────────────────────────────────────────────

class PatchedSumoEnvironment(SumoEnvironment):
    """SumoEnvironment with --time-to-teleport overridden to 300s."""

    def _start_simulation(self):
        """Override time_to_teleport before delegating to parent."""
        # sumo-rl defaults time_to_teleport to -1 (wait forever), which
        # gridlocks the simulation.  We override to 300s so stuck vehicles
        # get teleported and the sim keeps moving.
        self.time_to_teleport = 300
        super()._start_simulation()


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

ACTOR_LR:  float = 0.01
CRITIC_LR: float = 0.05
GAMMA:     float = 0.95


# ──────────────────────────────────────────────────────────────────────────────
# Helper — episode metric aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_metrics(
    info_history: List[Dict],
) -> Tuple[float, float, float]:
    """Average queue, wait, and delay across all steps in one episode."""
    queues:  List[float] = []
    waits:   List[float] = []
    delays:  List[float] = []

    for info in info_history:
        if "system_total_stopped" in info:
            queues.append(float(info["system_total_stopped"]))
        if "system_mean_waiting_time" in info:
            waits.append(float(info["system_mean_waiting_time"]))
        if "system_mean_speed" in info:
            avg_speed = float(info["system_mean_speed"])
            max_speed = 13.9
            delays.append(1.0 - min(avg_speed / max_speed, 1.0))

    mean_queue = float(np.mean(queues)) if queues else 0.0
    mean_wait  = float(np.mean(waits))  if waits  else 0.0
    mean_delay = float(np.mean(delays)) if delays  else 0.0
    return mean_queue, mean_wait, mean_delay


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(num_episodes: int = 200, use_gui: bool = False, num_seconds: int = 900, gui_delay: int = 200) -> None:
    """Run multi-agent Linear Actor-Critic training on the CN+ Bremen network."""

    # ── Paths ────────────────────────────────────────────────────────────────
    project_dir: Path = Path(__file__).resolve().absolute().parent
    net_file:   str  = str(project_dir / "data" / "bremen.net.xml")
    route_file: str  = str(project_dir / "data" / "bremen.rou.xml")
    add_file:   str  = str(project_dir / "data" / "bremen.add.xml")

    # ── Step limit: prevents infinite loop if SUMO gridlocks ─────────────────
    max_steps_per_episode: int = int((num_seconds // 5) * 1.2) + 10
    print(f"[train.py] Step limit per episode: {max_steps_per_episode} steps")

    # ── Create SUMO environment ──────────────────────────────────────────────
    print("[train.py] Creating SUMO environment …")
    print(f"[train.py] Simulation duration: {num_seconds}s "
          f"({num_seconds/60:.0f} min, {num_seconds//5} steps)")

    env = PatchedSumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        num_seconds=num_seconds,
        delta_time=5,
        min_green=10,
        max_green=60,
        use_gui=use_gui,
        reward_fn=custom_reward,
        observation_class=CNPlusObservation,
        sumo_warnings=False,
        additional_sumo_cmd=(
            f"--additional-files {add_file} "
            "--ignore-route-errors "
            "--no-step-log"
            + (f" --delay {gui_delay}" if use_gui else "")
            # NOTE: do NOT add --time-to-teleport here;
            # PatchedSumoEnvironment handles it above.
        ),
    )

    # ── Create multi-agent controller ────────────────────────────────────────
    print(f"[train.py] Intersections found: {env.ts_ids}")
    agents = MultiAgentLinearAC(
        env,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
    )

    # ── Live dashboard ────────────────────────────────────────────────────────
    dashboard = LiveDashboard(agent_ids=env.ts_ids)

    # ── History buffers ───────────────────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "reward": [], "queue": [], "wait": [], "delay": [],
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════════
    for episode in range(1, num_episodes + 1):
        t_start: float = time.time()
        obs_res        = env.reset()
        obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

        episode_reward: float      = 0.0
        info_history:   List[Dict] = []
        terminated:     Dict       = {a: False for a in env.ts_ids}
        truncated:      Dict       = {a: False for a in env.ts_ids}
        step_count:     int        = 0

        # ── Inner step loop ──────────────────────────────────────────────────
        while step_count < max_steps_per_episode:

            step_count += 1

            actions: Dict[str, int] = agents.act(obs)
            step_res = env.step(actions)

            if len(step_res) == 5:
                next_obs, rewards, terminated, truncated, info = step_res
            else:
                next_obs, rewards, dones, info = step_res
                terminated = truncated = dones

            # Check proper episode termination via __all__ key
            done_all: bool = (
                terminated.get("__all__", False)
                or truncated.get("__all__", False)
                or all(terminated.get(a, False) for a in env.ts_ids)
                or all(truncated.get(a, False) for a in env.ts_ids)
            )

            dones: Dict[str, bool] = {
                a: terminated.get(a, False) or truncated.get(a, False)
                for a in env.ts_ids
            }
            agents.learn(obs, actions, rewards, next_obs, dones)
            episode_reward += float(sum(rewards.values()))
            info_history.append(info)
            obs = next_obs

            if done_all:
                break

        # ── Episode summary ───────────────────────────────────────────────────
        mean_queue, mean_wait, mean_delay = _aggregate_metrics(info_history)
        elapsed: float = time.time() - t_start

        history["reward"].append(episode_reward)
        history["queue"].append(mean_queue)
        history["wait"].append(mean_wait)
        history["delay"].append(mean_delay)

        dashboard.update(
            reward=episode_reward,
            queue=mean_queue,
            wait=mean_wait,
            delay=mean_delay,
        )

        print(
            f"Episode {episode:>4d}/{num_episodes}  "
            f"reward={episode_reward:>8.2f}  "
            f"queue={mean_queue:>6.2f}  "
            f"wait={mean_wait:>6.2f}s  "
            f"delay={mean_delay:>5.3f}  "
            f"steps={step_count:>4d}  "
            f"({elapsed:.1f}s)"
        )

    # ── Post-training save ────────────────────────────────────────────────────
    print("\n[train.py] Training complete. Saving ...")

    weights_dir: Path = project_dir / "weights"
    results_dir: Path = project_dir / "results"
    weights_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    agents.save_all(str(weights_dir))
    np.save(str(results_dir / "training_history.npy"), history)
    print(f"  History  -> {results_dir / 'training_history.npy'}")

    dashboard.save_figure(str(results_dir / "training_dashboard.png"))
    print(f"  Dashboard -> {results_dir / 'training_dashboard.png'}")

    try:
        env.close()
    except OSError:
        pass  # suppress WinError 10038 socket noise on Windows


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Linear Actor-Critic on CN+ Bremen traffic network."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seconds",  type=int, default=900)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--delay", type=int, default=200,
                        help="GUI step delay in ms (higher = slower, default 200)")
    args = parser.parse_args()
    if args.gui:
        matplotlib.use("TkAgg")
    train(num_episodes=args.episodes, use_gui=args.gui,
          num_seconds=args.seconds, gui_delay=args.delay)