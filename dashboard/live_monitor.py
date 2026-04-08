"""
dashboard/live_monitor.py — Task 5.

Real-time 2×2 Matplotlib dashboard that updates after every training episode.

Layout:
    ┌──────────────────────┬──────────────────────┐
    │  Cumulative Reward   │  Queue Length (veh)  │
    │  (blue line)         │  (red line)          │
    ├──────────────────────┼──────────────────────┤
    │  Wait Time (s)       │  Delay               │
    │  (orange line)       │  (purple line)       │
    └──────────────────────┴──────────────────────┘
"""

from __future__ import annotations

from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class LiveDashboard:
    """
    Real-time 2×2 Matplotlib figure that refreshes after each training episode.

    Call ``update()`` once per episode from the training loop.  The figure
    stays open because plt.ion() enables interactive (non-blocking) mode.

    Attributes:
        agent_ids (List[str]): Agent identifiers from the environment.
        _rewards  (List[float]): Cumulative reward history.
        _queues   (List[float]): Mean queue length per episode.
        _waits    (List[float]): Mean wait time per episode.
        _delays   (List[float]): Mean delay per episode.
    """

    def __init__(self, agent_ids: List[str]) -> None:
        """
        Create the 2×2 figure and enable interactive mode.

        Args:
            agent_ids: List of intersection IDs from the SUMO environment.
                       Used only for the window subtitle at this stage.
        """
        self.agent_ids: List[str] = agent_ids

        # Internal history lists — one entry per episode
        self._rewards: List[float] = []
        self._queues:  List[float] = []
        self._waits:   List[float] = []
        self._delays:  List[float] = []

        # ── Matplotlib setup ──────────────────────────────────────────────────
        plt.ion()   # non-blocking interactive mode so training is not paused

        self._fig, self._axes = plt.subplots(
            nrows=2, ncols=2,
            figsize=(12, 7),
        )
        self._fig.suptitle(
            "Adaptive Traffic Signal Control — Live Monitor",
            fontsize=14,
            fontweight="bold",
        )

        # Unpack axes for clarity
        self._ax_reward = self._axes[0, 0]
        self._ax_queue  = self._axes[0, 1]
        self._ax_wait   = self._axes[1, 0]
        self._ax_delay  = self._axes[1, 1]

        # Show the (empty) figure immediately so the window appears on launch
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # Public update method — call once per episode
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        reward: float,
        queue:  float,
        wait:   float,
        delay:  float,
    ) -> None:
        """
        Append new episode data and redraw all four panels.

        Args:
            reward: Cumulative episode reward (sum across agents and steps).
            queue:  Mean queue length in vehicles for this episode.
            wait:   Mean wait time in seconds for this episode.
            delay:  Mean delay (0–1) for this episode.
        """
        # 1. Append values to histories
        self._rewards.append(reward)
        self._queues.append(queue)
        self._waits.append(wait)
        self._delays.append(delay)

        episodes = list(range(1, len(self._rewards) + 1))

        # 2. Redraw each panel
        self._draw_panel(
            ax=self._ax_reward,
            x=episodes,
            y=self._rewards,
            ylabel="Cumulative Reward",
            color="royalblue",
        )
        self._draw_panel(
            ax=self._ax_queue,
            x=episodes,
            y=self._queues,
            ylabel="Queue Length (veh)",
            color="crimson",
        )
        self._draw_panel(
            ax=self._ax_wait,
            x=episodes,
            y=self._waits,
            ylabel="Wait Time (s)",
            color="darkorange",
        )
        self._draw_panel(
            ax=self._ax_delay,
            x=episodes,
            y=self._delays,
            ylabel="Delay",
            color="mediumpurple",
        )

        # 3. Refresh layout and flush to screen
        plt.tight_layout()
        plt.pause(0.01)   # tiny pause required by Matplotlib's event loop

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_figure(self, path: str) -> None:
        """
        Save the current dashboard state as a PNG image.

        Args:
            path: Destination file path (e.g. 'results/training_dashboard.png').
        """
        plt.tight_layout()
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[LiveDashboard] Figure saved → {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_panel(
        ax,
        x:       List[int],
        y:       List[float],
        ylabel:  str,
        color:   str,
    ) -> None:
        """
        Clear an axes object and redraw a single time-series line.

        Args:
            ax:     Matplotlib Axes to draw on.
            x:      Episode indices (x-axis).
            y:      Metric values (y-axis).
            ylabel: y-axis label string.
            color:  Line and marker colour.
        """
        ax.cla()   # clear previous contents (faster than fig.clear())
        ax.plot(x, y, color=color, linewidth=1.8, marker=".", markersize=4)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)   # subtle grid lines as specified
        # Show last value in title for quick reading without zooming
        if y:
            ax.set_title(f"{ylabel}  (latest: {y[-1]:.2f})", fontsize=9)
