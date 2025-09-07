import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# Your external functions
from utils.uniswap import swap_fee, LVR


class UniswapV3LiquidityEnv(gym.Env):
    """
    Uniswap V3 Liquidity Management Environment (Gymnasium API)

    Action space:  Discrete allocation ratios [0.0, 0.25, 0.5, 0.75, 1.0]
    Observation:   Feature vector at current timestep (optionally includes previous action)

    reset() -> (obs, info)
    step(action) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        numeric_data,
        time_data,
        total_liquidity: float = 1e18,
        gas_cost: float = 5.0,
        fee_tier: float = 0.0005,
        max_steps: Optional[int] = None,
        start_index: int = 0,
    ):
        super().__init__()

        # --- data & shape checks ---
        data = np.asarray(numeric_data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"`data` must be 2D, got shape={data.shape}")
        if data.shape[0] < 2:
            raise ValueError("`data` must have at least 2 rows to create transitions.")
        self.data: np.ndarray = data
        self.time_data: np.ndarray = time_data
        self.n_steps_total: int = data.shape[0]
        self.feature_dim: int = data.shape[1]

        # --- core parameters ---
        self.total_liquidity = int(total_liquidity)
        self.gas_cost = float(gas_cost)
        self.fee_tier = float(fee_tier)  # respect the argument

        # Horizon = number of transitions allowed from the starting index
        horizon = int(max_steps) if max_steps is not None else (self.n_steps_total - 1)
        if horizon < 1:
            raise ValueError("`max_steps` must allow at least 1 transition.")
        # End index is the last valid index we can step *to* (for next state)
        # We will allow transitions while next_idx <= end_index
        self.start_index = int(start_index)
        self.end_index: int = min(self.start_index + horizon, self.n_steps_total - 1)

        # --- action/observation spaces ---
        self.action_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_values))

        obs_dim = self.feature_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- state trackers ---
        self._t: int = 0                        # current index in `data`
        self._last_action_idx: Optional[int] = None
        self._last_action_ratio: float = 0.0
        self.total_reward: float = 0.0

        # Gymnasium-friendly RNG (updated in reset with the provided seed)
        self.np_random = np.random.default_rng()

    # ================== Gymnasium API ==================

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Return (obs, info). Resets pointer, totals, and RNG."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
     
        self._t = self.start_index
        self._last_action_idx = None
        self._last_action_ratio = 0.0
        self.total_reward = 0.0

        obs = self._get_obs()

        return obs

    def step(self, action_idx: int):
        """
        Return (obs, reward, terminated, truncated, info)
        - terminated: task-natured end (e.g., liquidation/target hit)
        - truncated:  time-limit end (e.g., horizon exhausted)
        """
        # Validate action
        if not self.action_space.contains(action_idx):
            raise ValueError(f"action_idx={action_idx} not in action space.")

        t = self._t
        next_idx = t + 1

        # Guard against stepping beyond available horizon
        if next_idx > self.end_index:
            raise RuntimeError("Step called beyond episode end. Call reset() to start a new episode.")

        # Allocation ratio and rebalancing cost
        allocation_ratio = float(self.action_values[int(action_idx)])
        liquidity_allocated = allocation_ratio * self.total_liquidity

        if int(action_idx) != 0:
            gas_applied = self.gas_cost
        else:
            gas_applied = 0

        # Build reward using transition t -> next_idx
        price_in = float(self.data[t, 0])
        price_out = float(self.data[next_idx, 0])
        x_scaled_vol = float(self.data[next_idx, 1]) if self.feature_dim >= 2 else 0.0
        y_scaled_vol = float(self.data[next_idx, 2]) if self.feature_dim >= 3 else 0.0

        fee_component = float(swap_fee(price_out, x_scaled_vol, y_scaled_vol, liquidity_allocated)[2])
        lvr_component = float(LVR(price_in, price_out, liquidity_allocated))
        reward = fee_component + lvr_component - gas_applied

        self.total_reward += reward

        # Advance time and save action
        self._t = next_idx
        self._last_action_idx = int(action_idx)
        self._last_action_ratio = allocation_ratio

        # No termination condition as requested

        # Time-limit truncation at horizon end
        done = (self._t >= self.end_index)

        obs = self._get_obs()
        info = {
            "t": self._t,
            "time": self.time_data[t][0],
            "allocation_idx": int(action_idx),
            "allocation_ratio": allocation_ratio,
            "liquidity_allocated": liquidity_allocated,
            "price_in": price_in,
            "price_out": price_out,
            "fee": fee_component,
            "lvr": lvr_component,
            "gas_applied": gas_applied,
            "step_reward": float(reward),
            "total_reward": float(self.total_reward),
            "terminated_reason": None,
        }
        return obs, float(reward), bool(done), info

    # ================== Helpers ==================

    def _get_obs(self) -> np.ndarray:
        """Build observation vector (optionally append previous action ratio)."""

        return self.data[self._t].astype(np.float32)

    def render(self):
        pass

    def close(self):
        pass
