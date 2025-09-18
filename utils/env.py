import numpy as np
import pandas as pd
import gym
import os
import math
import json
from gym import spaces
from gym.utils import seeding
from typing import Optional

# External functions (assumed to return values in account currency, e.g., USDC)
from utils.uniswap import swap_fee, LVR, liquidity_multiplier

"""
Global State
"""

path = os.path.dirname(os.path.abspath(__file__)) + '/..' + '/config' + '/config.json'

with open(path, "r") as f:
    config = json.load(f)

pool_address = config["pool_address"]
chain = config["chain"]
token0 = config["token0"]
token1 = config["token1"]
decimal_0 = int(config["decimal_0"])
decimal_1 = int(config["decimal_1"])
fee_tier = float(config["fee_tier"])
tickspacing = int(config["tickspacing"])


class UniswapV3LiquidityEnv(gym.Env):
    """
    Uniswap V3 Liquidity Management Environment (old Gym API)

    - Action space:   Discrete allocation ratios [0.0, 0.25, 0.5, 0.75, 1.0]
    - Observation:    numeric_data[t] concatenated with current equity at the END
                      obs = [ price, x_scaled_vol, y_scaled_vol, ..., <other features>, equity ]
                      IMPORTANT: The first three columns of numeric_data MUST remain
                      in the exact order [price, x_scaled_vol, y_scaled_vol] because
                      they are used to compute swap fee / LVR internally.
    - reset() -> obs
    - step(action) -> (obs, reward, done, info)

    Capital-based variant:
    - 'init_value' is the starting equity (e.g., 10000).
    - Each step, the chosen action is interpreted as the fraction of current equity
      to be put into the LP position (position_notional = equity * allocation_ratio).
    - If equity <= liquidation_value, the episode terminates (done=True).
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        numeric_data,
        time_data,
        *,
        init_value: float = 10_000.0,    # starting capital
        liquidation_value: float = 0.0,  # liquidation threshold; equity <= this ends the episode
        gas_cost: float = 5.0,           # gas cost charged only when allocation ratio changes (first allocation free)
        fee_tier: float = 0.0005,        # kept for compatibility if your fee fn uses it
        max_steps: Optional[int] = None, # horizon in number of transitions
        start_index: int = 0,            # starting index in the time series
    ):
        super().__init__()

        # -- Data & shape checks --
        data = np.asarray(numeric_data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"`numeric_data` must be 2D, got shape={data.shape}")
        if data.shape[0] < 2:
            raise ValueError("`numeric_data` must have at least 2 rows to create transitions.")
        # We require at least the first 3 columns in the strict order:
        #   [price, x_scaled_vol, y_scaled_vol]
        if data.shape[1] < 3:
            raise ValueError(
                "`numeric_data` must have at least 3 columns in order: "
                "[price, x_scaled_vol, y_scaled_vol]."
            )

        self.data: np.ndarray = data
        self.time_data: np.ndarray = time_data
        self.n_steps_total: int = data.shape[0]
        self.feature_dim: int = data.shape[1]

        # -- Core parameters --
        self.init_value = float(init_value)
        self.liquidation_value = float(liquidation_value)
        self.gas_cost = float(gas_cost)
        self.fee_tier = float(fee_tier)

        # Horizon: number of allowed transitions from the starting index
        horizon = int(max_steps) if max_steps is not None else (self.n_steps_total - 1)
        if horizon < 1:
            raise ValueError("`max_steps` must allow at least 1 transition.")
        self.start_index = int(start_index)
        # Allow transitions while next_idx <= end_index
        self.end_index: int = min(self.start_index + horizon, self.n_steps_total - 1)

        # -- Spaces --
        self.action_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        self.action_space  = spaces.Discrete(len(self.action_values))

        # Observation contains numeric_data features PLUS current equity appended at the END
        # shape = (feature_dim + 1,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim + 1,), dtype=np.float32
        )

        # -- State trackers --
        self._t: int = 0                           # current index in `data`
        self._last_action_idx: Optional[int] = None
        self._last_action_ratio: float = 0.0
        self.total_reward: float = 0.0
        self.equity: float = self.init_value       # current capital (internal state, appended to obs)
        self.prev_equity: float = self.init_value  # for info['prev_equity']

        # Old-gym seeding helper
        self.np_random, _ = seeding.np_random(None)

    # ================== Old Gym API ==================

    def seed(self, seed: Optional[int] = None):
        """Old Gym-style RNG seeding."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset environment and return initial observation (obs includes equity as last element)."""
        self._t = self.start_index
        self._last_action_idx = None
        self._last_action_ratio = 0.0
        self.total_reward = 0.0
        self.equity = float(self.init_value)
        self.prev_equity = float(self.init_value)  # keep in sync
        return self._get_obs()

    def step(self, action_idx: int):
        """
        Perform one environment step.
        Returns (obs, reward, done, info) in old Gym API.
        - done is True if either:
          (a) equity <= liquidation_value (liquidation), or
          (b) time horizon is reached (episode time limit).
        """
        # Validate action
        if not self.action_space.contains(action_idx):
            raise ValueError(f"action_idx={action_idx} not in action space.")

        t = self._t
        next_idx = t + 1

        # Guard against stepping beyond available horizon
        if next_idx > self.end_index:
            raise RuntimeError("Step called beyond episode end. Call reset() to start a new episode.")

        # Build transition t -> next_idx
        # !! Strictly use columns [0,1,2] as [price, x_scaled_vol, y_scaled_vol]
        price_in  = float(self.data[t, 0])
        price_out = float(self.data[next_idx, 0])
        x_scaled_vol = float(self.data[next_idx, 1]) if self.feature_dim >= 2 else 0.0
        y_scaled_vol = float(self.data[next_idx, 2]) if self.feature_dim >= 3 else 0.0

        tick_in = int(round(math.log(price_in * 10**(decimal_1 - decimal_0), 1.0001)))

        # Allocation ratio and position sizing
        allocation_ratio = float(self.action_values[int(action_idx)])
        position_notional = self.equity * allocation_ratio
        liquidity = liquidity_multiplier(tick_in, 500) * position_notional * 10 ** decimal_1

        # Gas only when allocation changes; first allocation is free
        if (self._last_action_idx is None) or (int(action_idx) == self._last_action_idx):
            gas_applied = 0.0
        else:
            gas_applied = self.gas_cost

        # PnL components (assumed in account currency)
        fee_component = float(swap_fee(price_out, x_scaled_vol, y_scaled_vol, liquidity)[2])
        lvr_component = float(LVR(price_in, price_out, liquidity))
        step_pnl = fee_component + lvr_component - gas_applied

        # Update totals and equity
        self.total_reward += step_pnl
        self.prev_equity = float(self.equity)             # snapshot before update
        self.equity = max(0.0, self.equity + step_pnl)
        after_equity = float(self.equity)

        # Advance time and save action
        self._t = next_idx
        self._last_action_idx = int(action_idx)
        self._last_action_ratio = allocation_ratio

        # Termination conditions
        terminated = bool(self.equity <= self.liquidation_value)
        truncated = bool(self._t >= self.end_index)
        done = terminated or truncated

        # Safe time extraction for 1D / 2D time_data
        try:
            if hasattr(self.time_data, "__len__") and len(self.time_data) > t:
                time_val = self.time_data[t][0] if np.ndim(self.time_data) == 2 else self.time_data[t]
            else:
                time_val = None
        except Exception:
            time_val = None

        # Per-step return (relative pnl to previous equity)
        step_ret = float(step_pnl / self.prev_equity) if self.prev_equity > 0 else 0.0

        obs = self._get_obs()
        info = {
            "t": self._t,
            "time": time_val,

            # equity reporting
            "prev_equity": float(self.prev_equity),
            "after_equity": after_equity,        # primary key used by tune/eval
            "equity": after_equity,              # kept for backward compatibility
            "return": step_ret,                  # per-step return = pnl / prev_equity

            # action / position
            "allocation_idx": int(action_idx),
            "allocation_ratio": float(allocation_ratio),
            "position_notional": float(position_notional),
            "liquidity": float(liquidity),

            # pricing / components
            "price_in": float(price_in),
            "price_out": float(price_out),
            "fee": float(fee_component),
            "lvr": float(lvr_component),
            "gas_applied": float(gas_applied),

            # rewards
            "step_reward": float(step_pnl),
            "total_reward": float(self.total_reward),

            "terminated_reason": ("liquidation" if terminated else ("time_limit" if truncated else None)),
        }
        return obs, float(step_pnl), bool(done), info

    # ================== Helpers ==================

    def _get_obs(self) -> np.ndarray:
        """
        Build observation vector:
        - Take numeric_data[t] (unchanged order; the first 3 columns are [price, x_scaled_vol, y_scaled_vol])
        - Append current equity at the END
        Result shape = (feature_dim + 1,)
        """
        base = self.data[self._t].astype(np.float32)             # (feature_dim,)
        eq   = np.array([self.equity], dtype=np.float32)         # (1,)
        return np.concatenate([base, eq], axis=0)                # (feature_dim + 1,)

    def render(self, mode="human"):
        """No-op renderer (extend if needed)."""
        pass

    def close(self):
        """Resource cleanup if you allocate any external handles."""
        pass
