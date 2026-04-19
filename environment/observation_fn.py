
"""
environment/observation_fn.py — Task 2, File 1.

Custom observation function for the CN+ SUMO-RL environment.

The observation per intersection is a flat float32 numpy array built from
four feature groups:

  [a] Queue length per lane    — normalized to [0, 1]  (max assumed = 10 veh)
  [b] Wait time per lane       — normalized by dividing by 120 s
  [c] Current phase            — one-hot encoded
  [d] Time since last change   — normalized by dividing by 60 s

All four groups are concatenated in that order.
"""

from __future__ import annotations

from typing import List

import numpy as np
import gymnasium as gym

# sumo_rl imports — only available when SUMO is installed
try:
    from sumo_rl.environment.observations import ObservationFunction
except ImportError as exc:  # graceful degradation for static analysis
    raise ImportError(
        "sumo_rl is not installed.  Run: pip install sumo-rl"
    ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Constants for normalization
# ──────────────────────────────────────────────────────────────────────────────

MAX_QUEUE: float = 10.0    # assumed maximum vehicles waiting per lane
MAX_WAIT:  float = 120.0   # seconds — cap for wait-time normalization
MAX_PHASE_TIME: float = 60.0  # seconds — cap for time-since-change normalization


class CNPlusObservation(ObservationFunction):
    """
    Observation function tailored to the CN+ (Bremen) dataset intersections.

    Extends sumo_rl's ObservationFunction base class and is passed directly
    to SumoEnvironment via the ``observation_class`` argument.

    Feature layout (index ranges assume N lanes and P phases):
        [0 : N]         queue length per lane (normalized)
        [N : 2N]        wait time per lane    (normalized)
        [2N : 2N+P]     one-hot current phase
        [2N+P]          time since last phase change (normalized)
    """

    def __init__(self, ts) -> None:  # ts = TrafficSignal object from sumo_rl
        """
        Initialise observation function.

        Args:
            ts: The sumo_rl TrafficSignal this observation is attached to.
                Provides access to lane data, current phase, etc.
        """
        super().__init__(ts)
        # During initialization, self.ts hasn't instantiated self.ts.lanes yet.
        # We query the number of lanes dynamically from the first API call,
        # or from the controlled lanes directly via traci / sumo_rl.
        self._num_lanes: int = len(
            list(dict.fromkeys(self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)))
        )
        # self.ts.green_phases is populated in _build_phases(), which runs AFTER
        # observation_fn initialization in newer sumo_rl versions.
        # Use the standard TraCI method to get phase definitions:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            logic = self.ts.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.ts.id)[0]
        self._num_phases: int = len(logic.getPhases())

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — required by ObservationFunction base class
    # ─────────────────────────────────────────────────────────────────────────

    def __call__(self) -> np.ndarray:
        """
        Build and return the current observation vector.

        Returns:
            np.ndarray: float32 vector of shape (obs_dim,).
        """
        return np.concatenate([
            self._queue_features(),       # [a] queue length, normalized
            self._wait_features(),        # [b] wait time, normalized
            self._phase_onehot(),         # [c] one-hot phase
            self._time_since_change(),    # [d] time since last change
        ]).astype(np.float32)

    def observation_space(self) -> gym.spaces.Box:
        """
        Return the gymnasium Box describing the observation shape and bounds.

        Returns:
            gym.spaces.Box with shape (obs_dim,) and all values in [0, 1].
        """
        obs_dim: int = (
            self._num_lanes     # queue
            + self._num_lanes   # wait
            + self._num_phases  # one-hot phase
            + 1                 # time since change
        )
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private feature extractors
    # ─────────────────────────────────────────────────────────────────────────

    def _queue_features(self) -> np.ndarray:
        """
        Query SUMO for the halted-vehicle count on each lane and normalize.

        sumo_rl's TrafficSignal.get_lanes_queue() returns a list of raw counts.

        Returns:
            np.ndarray: shape (num_lanes,), values in [0, 1].
        """
        raw_queues: List[float] = self.ts.get_lanes_queue()
        # Clip at MAX_QUEUE so extreme jam spikes don't break normalization
        normalized = np.clip(np.array(raw_queues, dtype=np.float32) / MAX_QUEUE,
                             0.0, 1.0)
        return normalized

    def _wait_features(self) -> np.ndarray:
        """
        Query SUMO for per-lane accumulated wait time and normalize.

        Returns:
            np.ndarray: shape (num_lanes,), values in [0, 1].
        """
        raw_wait: List[float] = self.ts.get_accumulated_waiting_time_per_lane()
        normalized = np.clip(np.array(raw_wait, dtype=np.float32) / MAX_WAIT,
                             0.0, 1.0)
        return normalized

    def _phase_onehot(self) -> np.ndarray:
        """
        One-hot encode the current green phase index.

        Returns:
            np.ndarray: shape (num_phases,), exactly one element is 1.0.
        """
        onehot = np.zeros(self._num_phases, dtype=np.float32)
        current_phase: int = self.ts.green_phase  # 0-indexed phase id
        onehot[current_phase] = 1.0
        return onehot

    def _time_since_change(self) -> np.ndarray:
        """
        Compute how long (seconds) the current phase has been active, normalized.

        ts.time_since_last_phase_change is maintained by sumo_rl internally.

        Returns:
            np.ndarray: shape (1,), value in [0, 1].
        """
        time_s: float = self.ts.time_since_last_phase_change
        normalized = float(np.clip(time_s / MAX_PHASE_TIME, 0.0, 1.0))
        return np.array([normalized], dtype=np.float32)
