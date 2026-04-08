"""
agent/linear_actor_critic.py — Task 3, File 1.

Linear Actor-Critic RL agent — fully interpretable, zero neural networks.

Architecture
────────────
  Actor  : linear map  state → action_logits → softmax probabilities
  Critic : linear map  state → scalar value estimate (baseline)

Both actor_w and critic_w are plain numpy arrays that can be inspected,
printed, or saved at any time — no black-box layers.

Key novelty: explain_decision() shows EXACTLY which features drove each
phase selection, enabling human-readable interpretability of every decision.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


class LinearActorCritic:
    """
    Online Linear Actor-Critic (Policy Gradient + TD(0) Critic).

    Attributes:
        state_dim  (int):  dimensionality of the observation vector.
        action_dim (int):  number of discrete actions (green phases).
        actor_lr   (float): learning rate for actor weight updates.
        critic_lr  (float): learning rate for critic weight updates.
        gamma      (float): discount factor for future rewards.
        actor_w    (np.ndarray): shape (state_dim, action_dim) — interpretable.
        critic_w   (np.ndarray): shape (state_dim,)            — interpretable.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 0.01,
        critic_lr: float = 0.05,
        gamma: float = 0.95,
    ) -> None:
        """
        Initialise the agent with zero weights (can warm-start with load()).

        Args:
            state_dim:  Length of the flat observation vector.
            action_dim: Number of discrete phase choices.
            actor_lr:   Step size for the policy gradient update.
            critic_lr:  Step size for the TD-error critic update.
            gamma:      Discount factor applied to next-state value.
        """
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.actor_lr   = actor_lr
        self.critic_lr  = critic_lr
        self.gamma      = gamma

        # ── Weight matrices — NO hidden layers, no NN ─────────────────────
        # actor_w[s, a] = weight connecting feature s to action a's logit
        self.actor_w:  np.ndarray = np.zeros((state_dim, action_dim), dtype=np.float64)
        # critic_w[s]  = weight for feature s in the value estimate
        self.critic_w: np.ndarray = np.zeros(state_dim, dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass methods
    # ─────────────────────────────────────────────────────────────────────────

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the softmax policy distribution over all actions.

        Score for each action = state · actor_w[:, action] (linear).
        Subtracting max before exp() prevents floating-point overflow
        (numerically identical to standard softmax).

        Args:
            state: float array of shape (state_dim,).

        Returns:
            np.ndarray: probability vector of shape (action_dim,), sums to 1.
        """
        scores: np.ndarray = state @ self.actor_w   # shape (action_dim,)
        # Numerical stability: shift by max so largest exp() is exp(0) = 1
        scores -= scores.max()
        exp_scores: np.ndarray = np.exp(scores)
        probs: np.ndarray = exp_scores / exp_scores.sum()
        return probs

    def choose_action(self, state: np.ndarray) -> int:
        """
        Sample an action from the stochastic policy.

        Stochastic sampling is critical during training because it encourages
        exploration.  At evaluation time you may want argmax instead, but
        sampling is used here for consistency.

        Args:
            state: float array of shape (state_dim,).

        Returns:
            int: selected action index in [0, action_dim).
        """
        probs: np.ndarray = self.action_probs(state)
        # np.random.choice samples proportionally to probs
        action: int = int(np.random.choice(self.action_dim, p=probs))
        return action

    def value(self, state: np.ndarray) -> float:
        """
        Estimate the state value with a single dot product (linear critic).

        V(s) = critic_w · state

        Args:
            state: float array of shape (state_dim,).

        Returns:
            float: scalar value estimate.
        """
        return float(self.critic_w @ state)

    # ─────────────────────────────────────────────────────────────────────────
    # Update (online, called after every environment step)
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> float:
        """
        Perform one Actor-Critic update using the TD(0) error as the advantage.

        Critic update (semi-gradient TD):
            critic_w += critic_lr * δ * state
            where δ = r + γ·V(s') - V(s)   (TD error / advantage estimate)

        Actor update (REINFORCE with baseline):
            For the log-gradient of the softmax policy:
                ∂ log π(a|s) / ∂ actor_w = outer(s, -probs); then
                add s to the action-a column (score-function trick).
            Full update:
                actor_w += actor_lr * δ * grad

        Using TD error as the advantage reduces variance relative to Monte Carlo
        returns while keeping the algorithm fully online (no replay buffer).

        Args:
            state:      observation at time t.
            action:     action chosen at time t.
            reward:     reward received at time t+1.
            next_state: observation at time t+1.
            done:       True if the episode ended at t+1.

        Returns:
            float: the TD error δ (useful for logging).
        """
        # ── TD error ──────────────────────────────────────────────────────
        v_current: float = self.value(state)
        v_next:    float = self.value(next_state) if not done else 0.0
        td_error:  float = reward + self.gamma * v_next - v_current

        # ── Critic update (gradient ascent on TD target) ──────────────────
        self.critic_w += self.critic_lr * td_error * state

        # ── Actor update (policy gradient) ───────────────────────────────
        probs: np.ndarray = self.action_probs(state)   # current distribution

        # Build the log-policy gradient for all actions simultaneously:
        # grad[:, a] = state * (1[a==chosen] - probs[a])
        # Equivalently: start with outer(-probs), then add state to chosen col.
        grad: np.ndarray = np.outer(state, -probs)          # (state_dim, action_dim)
        grad[:, action] += state                             # score-function term

        # Scale gradient by TD error (prevents updates when prediction is exact)
        self.actor_w += self.actor_lr * td_error * grad

        return td_error

    # ─────────────────────────────────────────────────────────────────────────
    # Interpretability — THE KEY NOVELTY
    # ─────────────────────────────────────────────────────────────────────────

    def explain_decision(
        self,
        state:         np.ndarray,
        feature_names: List[str],
        phase_names:   List[str],
        top_k:         int = 5,
    ) -> None:
        """
        Explain EXACTLY why the agent chose the current best phase.

        This method demonstrates full linear interpretability:
        the contribution of each feature to the chosen action's logit can be
        computed as a simple element-wise product:
            contribution[i] = actor_w[i, action] * state[i]

        Sorting by |contribution| gives the features that most influenced the
        decision — something impossible with a deep neural network.

        Output format (printed to stdout):
            Chosen phase: [phase_name]  confidence: XX.X%
            Top 5 driving features:
              [feature_name]    val=X.XXX  weight=+X.XXX  contrib=+X.XXX

        Args:
            state:         Current observation vector, shape (state_dim,).
            feature_names: Human-readable names for each state dimension.
            phase_names:   Human-readable names for each action / phase.
            top_k:         How many top features to print (default 5).
        """
        probs:          np.ndarray = self.action_probs(state)
        chosen_action:  int        = int(np.argmax(probs))   # greedy best action
        confidence:     float      = float(probs[chosen_action]) * 100.0

        # Per-feature contribution to the chosen action's score
        weights_for_action: np.ndarray = self.actor_w[:, chosen_action]
        contributions:      np.ndarray = weights_for_action * state

        # Sort by absolute contribution (most influential first)
        sorted_indices: np.ndarray = np.argsort(np.abs(contributions))[::-1]
        top_indices:    np.ndarray = sorted_indices[:top_k]

        # ── Print formatted report ────────────────────────────────────────
        phase_name: str = (
            phase_names[chosen_action]
            if chosen_action < len(phase_names)
            else f"Phase {chosen_action}"
        )
        print(f"\nChosen phase: {phase_name}  confidence: {confidence:.1f}%")
        print(f"Top {top_k} driving features:")
        for idx in top_indices:
            fname   = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            val     = float(state[idx])
            weight  = float(weights_for_action[idx])
            contrib = float(contributions[idx])
            # Align columns for readability
            print(
                f"  {fname:<25s}  "
                f"val={val:+.3f}  "
                f"weight={weight:+.3f}  "
                f"contrib={contrib:+.3f}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Persist actor and critic weights to a compressed .npz file.

        Args:
            path: File path (should end in .npz).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            actor_w=self.actor_w,
            critic_w=self.critic_w,
        )

    def load(self, path: str) -> None:
        """
        Restore actor and critic weights from a previously saved .npz file.

        Args:
            path: File path created by save().

        Raises:
            FileNotFoundError: if path does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Weight file not found: {path}")
        data: np.lib.npyio.NpzFile = np.load(path)
        self.actor_w  = data["actor_w"]
        self.critic_w = data["critic_w"]
