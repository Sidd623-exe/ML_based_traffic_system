"""
train.py — Task 4, File 1.

Main training entry-point for the Adaptive Traffic Signal Control project.

Runs Linear Actor-Critic on the CN+ Bremen network for ``num_episodes``
episodes and logs per-episode statistics.  A live Matplotlib dashboard
updates after each episode.

Usage::
    python train.py                      # headless, 200 episodes
    python train.py --gui                # open SUMO GUI (slower)
    python train.py --episodes 50        # quick test run
"""

from __future__ import annotations

import argparse
import sys
import os
import matplotlib
matplotlib.use("Agg")   # headless backend — no GUI event-loop overhead

# ── Dynamic SUMO_HOME Fix ───────────────────────────────────────────────────
# Override SUMO_HOME if it's missing or points to a non-existent directory.
# This prevents FileNotFoundError from sumolib.
if not os.environ.get("SUMO_HOME") or not os.path.exists(os.environ["SUMO_HOME"]):
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
from environment.observation_fn import CNPlusObservation
from environment.reward_fn import custom_reward
from agent.multi_agent import MultiAgentLinearAC
from dashboard.live_monitor import LiveDashboard

# ── sumo_rl ───────────────────────────────────────────────────────────────────
try:
    from sumo_rl import SumoEnvironment
except ImportError as exc:
    sys.exit(f"ERROR: sumo_rl not installed. Run: pip install sumo-rl\n{exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters (all in one place for easy tuning)
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
    """
    Average queue, wait, and delay across all steps and agents in one episode.

    SUMO-RL stores per-step info in the info dict returned by env.step().
    Keys used: 'system_mean_waiting_time', 'system_mean_speed', and
    individual agent queue info accessible via the TrafficSignal objects.

    For simplicity the function averages the reward-component estimates that
    are already computed by sumo_rl's system-level metrics.

    Args:
        info_history: List of info dicts from each env.step() call.

    Returns:
        Tuple (mean_queue, mean_wait, mean_delay) for the episode.
    """
    queues:  List[float] = []
    waits:   List[float] = []
    delays:  List[float] = []

    for info in info_history:
        # sumo_rl provides system-level metrics in the info dict
        if "system_total_stopped" in info:
            queues.append(float(info["system_total_stopped"]))
        if "system_mean_waiting_time" in info:
            waits.append(float(info["system_mean_waiting_time"]))
        if "system_mean_speed" in info:
            avg_speed = float(info["system_mean_speed"])
            max_speed = 13.9   # 50 km/h
            delays.append(1.0 - min(avg_speed / max_speed, 1.0))

    mean_queue = float(np.mean(queues)) if queues else 0.0
    mean_wait  = float(np.mean(waits))  if waits  else 0.0
    mean_delay = float(np.mean(delays)) if delays  else 0.0
    return mean_queue, mean_wait, mean_delay


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(num_episodes: int = 200, use_gui: bool = False, num_seconds: int = 900) -> None:
    """
    Run multi-agent Linear Actor-Critic training on the CN+ Bremen network.

    For each episode:
        1. Reset the SUMO environment.
        2. Step through the simulation, collecting transitions.
        3. Call agents.learn() at every step (online Actor-Critic).
        4. Aggregate episode metrics.
        5. Update the live Matplotlib dashboard.
        6. Print a per-episode summary line.

    After all episodes:
        * Weights are saved to weights/
        * Training history is saved to results/training_history.npy

    Args:
        num_episodes: Number of full simulation episodes to run.
        use_gui:      If True, open the SUMO GUI window (slows training).
    """
    # ── Paths ────────────────────────────────────────────────────────────────
    project_dir: Path = Path(__file__).resolve().absolute().parent
    net_file:   str  = str(project_dir / "data" / "bremen.net.xml")
    route_file: str  = str(project_dir / "data" / "bremen.rou.xml")
    add_file:   str  = str(project_dir / "data" / "bremen.add.xml")  # bus stops

    # ── Create SUMO environment ──────────────────────────────────────────────
    print("[train.py] Creating SUMO environment …")
    print(f"[train.py] Simulation duration per episode: {num_seconds}s  "
          f"({num_seconds/60:.0f} min, {num_seconds//5} steps)")
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        num_seconds=num_seconds,   # configurable via --seconds
        delta_time=5,              # agent decision every 5 simulated seconds
        min_green=10,              # minimum time the agent must keep a green phase
        max_green=60,              # maximum green phase duration
        use_gui=use_gui,
        reward_fn=custom_reward,
        observation_class=CNPlusObservation,
        sumo_warnings=False,       # suppress non-critical SUMO console output
        additional_sumo_cmd=(
            f"--additional-files {add_file} "   # bus stops (correct SUMO flag)
            "--ignore-route-errors "             # don't crash on bad routes
            "--no-step-log"                      # suppress per-step console spam
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

    # ── Create live dashboard ─────────────────────────────────────────────────
    dashboard = LiveDashboard(agent_ids=env.ts_ids)

    # ── History buffers (saved to disk after training) ────────────────────────
    history: Dict[str, List[float]] = {
        "reward": [],
        "queue":  [],
        "wait":   [],
        "delay":  [],
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════════
    for episode in range(1, num_episodes + 1):
        t_start: float = time.time()
        obs_res        = env.reset()
        obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

        episode_reward: float       = 0.0
        info_history:   List[Dict]  = []
        terminated:     Dict        = {a: False for a in env.ts_ids}
        truncated:      Dict        = {a: False for a in env.ts_ids}

        # ── Inner step loop ──────────────────────────────────────────────────
        while not all(terminated.values()) and not all(truncated.values()):
            # 1. Select actions from stochastic policy
            actions: Dict[str, int] = agents.act(obs)

            # 2. Step the simulation
            step_res = env.step(actions)

            # 3. Unpack step result (sumo-rl may return 4 or 5 values)
            if len(step_res) == 5:
                next_obs, rewards, terminated, truncated, info = step_res
            else:
                next_obs, rewards, dones, info = step_res
                terminated = truncated = dones

            # 4. Online Actor-Critic update for every agent
            dones: Dict[str, bool] = {
                a: terminated.get(a, False) or truncated.get(a, False)
                for a in env.ts_ids
            }
            agents.learn(obs, actions, rewards, next_obs, dones)

            # 5. Accumulate episode reward (sum across agents and steps)
            episode_reward += float(sum(rewards.values()))

            # 6. Store info for metric aggregation
            info_history.append(info)

            obs = next_obs   # advance state

        # ── Episode summary ───────────────────────────────────────────────────
        mean_queue, mean_wait, mean_delay = _aggregate_metrics(info_history)
        elapsed: float = time.time() - t_start

        # Store in history
        history["reward"].append(episode_reward)
        history["queue"].append(mean_queue)
        history["wait"].append(mean_wait)
        history["delay"].append(mean_delay)

        # Update live dashboard
        dashboard.update(
            reward=episode_reward,
            queue=mean_queue,
            wait=mean_wait,
            delay=mean_delay,
        )

        # Console summary line
        print(
            f"Episode {episode:>4d}/{num_episodes}  "
            f"reward={episode_reward:>8.2f}  "
            f"queue={mean_queue:>6.2f}  "
            f"wait={mean_wait:>6.2f}s  "
            f"delay={mean_delay:>5.3f}  "
            f"({elapsed:.1f}s)"
        )

    # ── Post-training persistence ─────────────────────────────────────────────
    print("\n[train.py] Training complete.  Saving …")

    weights_dir: Path  = project_dir / "weights"
    results_dir: Path  = project_dir / "results"
    weights_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    agents.save_all(str(weights_dir))

    np.save(str(results_dir / "training_history.npy"), history)
    print(f"  History  → {results_dir / 'training_history.npy'}")

    # Save final dashboard snapshot
    dashboard.save_figure(str(results_dir / "training_dashboard.png"))
    print(f"  Dashboard → {results_dir / 'training_dashboard.png'}")

    env.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Linear Actor-Critic on CN+ Bremen traffic network."
    )
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Number of training episodes (default: 200)."
    )
    parser.add_argument(
        "--seconds", type=int, default=900,
        help="Simulated seconds per episode (default: 900 = 15 min). "
             "Use 300 for a quick smoke-test, 3600 for full fidelity."
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Open SUMO GUI during training (slower but visual)."
    )
    args = parser.parse_args()
    if args.gui:
        matplotlib.use("TkAgg")   # need a real backend when GUI is requested
    train(num_episodes=args.episodes, use_gui=args.gui, num_seconds=args.seconds)
