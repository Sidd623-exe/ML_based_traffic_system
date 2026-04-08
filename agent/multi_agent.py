"""
agent/multi_agent.py — Task 3, File 2.

Coordinates one LinearActorCritic agent per intersection.

The sumo_rl multi-agent API delivers observations as a dict keyed by
intersection/agent ID.  MultiAgentLinearAC maps each agent_id to its own
LinearActorCritic instance so every intersection learns its own policy.

All agents share the same hyper-parameters but NEVER share weights — each
intersection must adapt to its local geometry and traffic demand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from agent.linear_actor_critic import LinearActorCritic


class MultiAgentLinearAC:
    """
    Container that holds one LinearActorCritic per intersection.

    Usage::
        env = SumoEnvironment(...)
        agents = MultiAgentLinearAC(env)
        obs, _ = env.reset()
        actions = agents.act(obs)
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        agents.learn(obs, actions, rewards, next_obs, terms)
    """

    def __init__(
        self,
        env,
        actor_lr:  float = 0.01,
        critic_lr: float = 0.05,
        gamma:     float = 0.95,
    ) -> None:
        """
        Build one LinearActorCritic for each agent reported by the env.

        The state_dim and action_dim are inferred from the env's observation
        and action spaces so this class works with any intersection geometry.

        Args:
            env:       A sumo_rl SumoEnvironment (multi-agent, PettingZoo API).
            actor_lr:  Shared actor learning rate for all sub-agents.
            critic_lr: Shared critic learning rate for all sub-agents.
            gamma:     Shared discount factor for all sub-agents.
        """
        self.agents: Dict[str, LinearActorCritic] = {}

        for agent_id in env.ts_ids:
            # Observation and action spaces may differ per intersection (
            # different lane counts / phase sets); infer dimensions per agent.
            obs_space = env.observation_spaces(agent_id)
            act_space = env.action_spaces(agent_id)

            state_dim:  int = int(np.prod(obs_space.shape))   # flatten
            action_dim: int = int(act_space.n)

            self.agents[agent_id] = LinearActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Act
    # ─────────────────────────────────────────────────────────────────────────

    def act(self, observations: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Sample an action for every active agent.

        Only agents present in ``observations`` are queried; agents not yet
        or no longer active are skipped automatically.

        Args:
            observations: dict mapping agent_id → flat observation array.

        Returns:
            dict mapping agent_id → chosen action integer.
        """
        actions: Dict[str, int] = {}
        for agent_id, obs in observations.items():
            if agent_id in self.agents:
                actions[agent_id] = self.agents[agent_id].choose_action(obs)
        return actions

    # ─────────────────────────────────────────────────────────────────────────
    # Learn
    # ─────────────────────────────────────────────────────────────────────────

    def learn(
        self,
        obs:       Dict[str, np.ndarray],
        actions:   Dict[str, int],
        rewards:   Dict[str, float],
        next_obs:  Dict[str, np.ndarray],
        dones:     Dict[str, bool],
    ) -> None:
        """
        Call update() on each agent with its own transition tuple.

        Only intersections present in ALL five dicts are updated; this guards
        against partial observations at episode boundaries.

        Args:
            obs:      Observations at time t.
            actions:  Actions taken at time t.
            rewards:  Rewards received after time t.
            next_obs: Observations at time t+1.
            dones:    Whether each agent's episode ended.
        """
        for agent_id in self.agents:
            # Skip if any part of the transition is missing for this agent
            if not all(
                agent_id in d
                for d in (obs, actions, rewards, next_obs, dones)
            ):
                continue
            self.agents[agent_id].update(
                state=obs[agent_id],
                action=actions[agent_id],
                reward=rewards[agent_id],
                next_state=next_obs[agent_id],
                done=dones[agent_id],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_all(self, folder: str) -> None:
        """
        Save each agent's weights to <folder>/<agent_id>.npz.

        Args:
            folder: Directory to write weight files into (created if absent).
        """
        Path(folder).mkdir(parents=True, exist_ok=True)
        for agent_id, agent in self.agents.items():
            # Replace characters that are invalid in filenames on Windows
            safe_id: str = agent_id.replace(":", "_").replace("/", "_")
            path: str = str(Path(folder) / f"{safe_id}.npz")
            agent.save(path)
            print(f"  Saved weights: {path}")

    def load_all(self, folder: str) -> None:
        """
        Load each agent's weights from <folder>/<agent_id>.npz.

        Args:
            folder: Directory containing weight files saved by save_all().

        Raises:
            FileNotFoundError: if the folder does not contain a required file.
        """
        for agent_id, agent in self.agents.items():
            safe_id: str = agent_id.replace(":", "_").replace("/", "_")
            path: str = str(Path(folder) / f"{safe_id}.npz")
            agent.load(path)
            print(f"  Loaded weights: {path}")
