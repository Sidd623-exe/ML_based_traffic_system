"""
baselines/run_baselines.py — Task 4, File 2.

Runs two baseline traffic control strategies and prints a comparison table:
  1. Fixed-Time (90 s cycle, equal phase durations)
  2. Actuated TLS (SUMO built-in vehicle-detector-based control, 5–60 s)

Neither baseline uses any reinforcement learning; they serve as reference
points for the comparison table in evaluate.py and the paper.

Usage::
    python baselines/run_baselines.py
"""

from __future__ import annotations

import sys
import os

if not os.environ.get("SUMO_HOME") or not os.path.exists(os.environ["SUMO_HOME"]):
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"

import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── sumo_rl / traci ──────────────────────────────────────────────────────────
try:
    import traci
    from sumo_rl import SumoEnvironment
except ImportError as exc:
    sys.exit(
        f"ERROR: SUMO / sumo_rl not installed correctly.\n{exc}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Paths (relative to project root — adjust if you run from elsewhere)
# ──────────────────────────────────────────────────────────────────────────────

_PROJECT: Path = Path(__file__).resolve().parent.parent
_NET:     str  = str(_PROJECT / "data" / "bremen.net.xml")
_ROUTE:   str  = str(_PROJECT / "data" / "bremen.rou.xml")
_CFG:     str  = str(_PROJECT / "data" / "bremen.sumocfg")
_ADD:     str  = str(_PROJECT / "data" / "bremen.add.xml")  # bus stops

# Duration of each simulated episode (seconds)
_SIM_SECONDS: int = 3600
_DELTA_TIME:  int = 5

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_metrics(env: SumoEnvironment, episodes: int) -> Dict[str, float]:
    """
    Run ``episodes`` episodes with a fixed policy (no learning) and return
    averaged performance metrics.

    At each step the baseline selects action 0 for every agent (keeps current
    phase, relying on the fixed-time or actuated TLS program set in the net).

    Args:
        env:      Already-configured SumoEnvironment for this baseline.
        episodes: Number of full simulation episodes to average over.

    Returns:
        dict with keys 'queue', 'wait', 'delay' — episode-averaged means.
    """
    all_queues: List[float] = []
    all_waits:  List[float] = []
    all_delays: List[float] = []

    for _ in range(episodes):
        obs_res     = env.reset()
        obs         = obs_res[0] if isinstance(obs_res, tuple) else obs_res
        terminated  = {a: False for a in env.ts_ids}
        truncated   = {a: False for a in env.ts_ids}
        step_q: List[float] = []
        step_w: List[float] = []
        step_d: List[float] = []

        max_steps = (_SIM_SECONDS // _DELTA_TIME) + 50
        step = 0
        while step < max_steps:
            step += 1
            # For baselines: always pass action 0 (do not change phase), so
            # the SUMO TLS program defined in the net file takes full control.
            actions = {a: 0 for a in obs}
            step_res = env.step(actions)
            if len(step_res) == 5:
                obs, _, terminated, truncated, info = step_res
            else:
                obs, _, dones, info = step_res
                terminated = truncated = dones

            done_all = (
                terminated.get("__all__", False)
                or truncated.get("__all__", False)
                or all(terminated.get(a, False) for a in env.ts_ids)
                or all(truncated.get(a, False) for a in env.ts_ids)
            )

            if "system_total_stopped" in info:
                step_q.append(float(info["system_total_stopped"]))
            if "system_mean_waiting_time" in info:
                step_w.append(float(info["system_mean_waiting_time"]))
            if "system_mean_speed" in info:
                spd = float(info["system_mean_speed"])
                step_d.append(1.0 - min(spd / 13.9, 1.0))

            if done_all:
                break

        all_queues.append(np.mean(step_q) if step_q else 0.0)
        all_waits.append( np.mean(step_w) if step_w else 0.0)
        all_delays.append(np.mean(step_d) if step_d else 0.0)

    return {
        "queue": float(np.mean(all_queues)),
        "wait":  float(np.mean(all_waits)),
        "delay": float(np.mean(all_delays)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 1 — Fixed Time (90 s cycle)
# ──────────────────────────────────────────────────────────────────────────────

def run_fixed_time(episodes: int = 10) -> Dict[str, float]:
    """
    Simulate a fixed-time traffic light plan with a 90-second cycle.

    SUMO's 'static' TLS type is used.  The phase durations are set uniformly
    so that one full cycle lasts 90 seconds (the industry standard benchmarked
    in most traffic engineering studies).

    Because the SUMO-RL wrapper is used with ``reward_fn=None`` and
    ``observation_class=None``, it simply cycles through its built-in fixed
    program — the RL agent (action 0) never requests a phase change.

    Args:
        episodes: Number of simulation episodes to average.

    Returns:
        dict: averaged {queue, wait, delay} metrics.
    """
    print("[Baseline] Fixed Time (90 s cycle) …")
    env = SumoEnvironment(
        net_file=_NET,
        route_file=_ROUTE,
        num_seconds=_SIM_SECONDS,
        delta_time=_DELTA_TIME,
        # Fixed-time: set min_green = max_green = 45 s  so the RL wrapper
        # cannot change phases; SUMO's own static program runs freely.
        min_green=45,
        max_green=45,
        use_gui=False,
        sumo_warnings=False,
        time_to_teleport=300,
        additional_sumo_cmd=f"-a {_ADD}",
    )
    results = _collect_metrics(env, episodes)
    env.close()
    print(f"  → queue={results['queue']:.2f}  wait={results['wait']:.2f}s  delay={results['delay']:.3f}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 2 — Actuated TLS (SUMO built-in vehicle detector logic)
# ──────────────────────────────────────────────────────────────────────────────

def run_actuated_tls(episodes: int = 10) -> Dict[str, float]:
    """
    Simulate SUMO's built-in actuated (vehicle-detector-driven) TLS control.

    Actuated control extends green phases while vehicles are detected and
    cuts phases short when the detector is empty.  It is widely used as a
    practical baseline in traffic engineering research.

    The SumoEnvironment is configured with a wide min/max green range (5–60 s)
    to give the SUMO actuated program room to adapt — but the RL agent still
    only sends action 0 (no phase change override).

    Note: For full actuated TLS, the network's TLS type must be 'actuated' in
    the .net.xml file.  If the Bremen network uses 'static', the behaviour will
    be equivalent to fixed-time.  You can convert the TLS type offline with:
        netconvert --tls.default-type actuated -s bremen.net.xml -o bremen_actuated.net.xml

    Args:
        episodes: Number of simulation episodes to average.

    Returns:
        dict: averaged {queue, wait, delay} metrics.
    """
    print("[Baseline] Actuated TLS (5–60 s) …")
    env = SumoEnvironment(
        net_file=_NET,
        route_file=_ROUTE,
        num_seconds=_SIM_SECONDS,
        delta_time=_DELTA_TIME,
        min_green=5,
        max_green=60,
        use_gui=False,
        sumo_warnings=False,
        time_to_teleport=300,
        additional_sumo_cmd=f"-a {_ADD}",
    )
    results = _collect_metrics(env, episodes)
    env.close()
    print(f"  → queue={results['queue']:.2f}  wait={results['wait']:.2f}s  delay={results['delay']:.3f}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Formatted comparison table
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(
    fixed:    Dict[str, float],
    actuated: Dict[str, float],
    ac:       Dict[str, float] | None = None,
) -> None:
    """
    Print a formatted side-by-side comparison of all evaluated methods.

    Args:
        fixed:    Results dict from run_fixed_time().
        actuated: Results dict from run_actuated_tls().
        ac:       Optional results dict from the Linear AC agent (evaluate.py).
    """
    sep  = "-" * 65
    head = f"{'Algorithm':<20} {'Queue':>7} {'Wait(s)':>9} {'Delay':>7} {'Interpretable':>14}"
    print("\n" + sep)
    print("  BASELINE COMPARISON")
    print(sep)
    print(head)
    print(sep)
    print(f"  {'Fixed Time (90s)':<18} {fixed['queue']:>7.2f} {fixed['wait']:>9.2f} {fixed['delay']:>7.3f}   {'No':>12}")
    print(f"  {'Actuated TLS':<18} {actuated['queue']:>7.2f} {actuated['wait']:>9.2f} {actuated['delay']:>7.3f}   {'No':>12}")
    if ac is not None:
        print(f"  {'Linear AC (ours)':<18} {ac['queue']:>7.2f} {ac['wait']:>9.2f} {ac['delay']:>7.3f}   {'Yes':>12}")
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fixed_results    = run_fixed_time(episodes=10)
    actuated_results = run_actuated_tls(episodes=10)
    print_comparison_table(fixed=fixed_results, actuated=actuated_results)
