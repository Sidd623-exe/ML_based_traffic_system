"""
environment/reward_fn.py — Task 2, File 2.

Weighted multi-metric reward function for the CN+ SUMO-RL environment.

The reward definition:
    reward = -(w1 * queue + w2 * wait + w3 * delay)

All terms are non-negative, so the reward is always ≤ 0.
A reward closer to 0 means better performance across all three metrics.

Weights are declared at the TOP of the function body for easy tuning.
"""

from __future__ import annotations

# No deep-learning imports — only NumPy for array math
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_SPEED_MS: float = 13.9  # 50 km/h ≈ 13.9 m/s — used for delay normalization


def custom_reward(traffic_signal) -> float:
    """
    Compute the weighted multi-metric reward for a single traffic signal.

    This function is the core of the project's reward shaping.  Every term
    contributes to teaching the agent to simultaneously minimise:
        • queue  — congestion at the intersection
        • wait   — accumulated wait time (fairness to individual drivers)
        • delay  — speed-loss relative to free-flow speed (network efficiency)

    Weighting scheme (must sum to 1.0):
        w1 = 0.40  → queue is the primary metric (most visible bottleneck)
        w2 = 0.35  → wait time is secondary (comfort and fairness)
        w3 = 0.25  → delay has the least direct weight but captures throughput

    Args:
        traffic_signal: A sumo_rl ``TrafficSignal`` object.
            Provides the following attributes / methods used here:
              • get_lanes_queue()      → List[float]  (halted vehicle counts)
              • get_accumulated_waiting_time_per_lane()  → List[float]  (accumulated wait, seconds)
              • get_average_speed()    → float        (mean speed m/s on all lanes)

    Returns:
        float: The instant reward (always ≤ 0).  Higher (less negative) is better.
    """
    # ── Weights — CHANGE THESE to tune the agent's priorities ────────────────
    w1: float = 0.40   # weight for normalized queue length
    w2: float = 0.35   # weight for normalized accumulated wait time
    w3: float = 0.25   # weight for delay (1 − speed ratio)

    # ── Term 1: Queue length ─────────────────────────────────────────────────
    #   Sum of halted-vehicle counts across all lanes feeding this intersection.
    #   Raw counts can be large; they are not normalized here because the weight
    #   itself controls the scale contribution.  If you want normalized queues,
    #   divide by MAX_QUEUE * num_lanes.
    raw_queues: list = traffic_signal.get_lanes_queue()
    queue: float = float(np.sum(raw_queues))   # total vehicles waiting

    # ── Term 2: Wait time ────────────────────────────────────────────────────
    #   Sum of per-lane accumulated wait times (seconds), divided by 100.0 to
    #   bring it into a comparable order-of-magnitude with the other two terms.
    raw_wait: list = traffic_signal.get_accumulated_waiting_time_per_lane()
    wait: float = float(np.sum(raw_wait)) / 100.0   # normalized sum

    # ── Term 3: Delay ────────────────────────────────────────────────────────
    #   Delay = 1 − (average_speed / max_free_flow_speed).
    #   When vehicles travel at free-flow speed, delay ≈ 0.
    #   When vehicles are nearly stationary (stop-and-go), delay ≈ 1.
    avg_speed: float = traffic_signal.get_average_speed()   # m/s
    #   Clip to [0, MAX_SPEED_MS] in case SUMO returns occasional outliers
    avg_speed_clipped: float = float(np.clip(avg_speed, 0.0, MAX_SPEED_MS))
    delay: float = 1.0 - (avg_speed_clipped / MAX_SPEED_MS)   # in [0, 1]

    # ── Weighted combination ─────────────────────────────────────────────────
    reward: float = -(w1 * queue + w2 * wait + w3 * delay)
    return reward
