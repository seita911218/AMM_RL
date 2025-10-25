# ===== Standard library =====
import os
import json
import math
from typing import Optional, Sequence

# ===== Third-party =====
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ===== Project-specific helpers (AMM math / fee & LVR components) =====
from utils.uniswap import swap_fee, LVR, total_y_amount_2_liquidity

# ===== Load pool/config defaults =====
_cfg_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "config",
    "config.json",
)
with open(_cfg_path, "r") as f:
    _cfg = json.load(f)

pool_address = _cfg["pool_address"]
chain = _cfg["chain"]
token0 = _cfg["token0"]
token1 = _cfg["token1"]
decimal_0 = int(_cfg["decimal_0"])
decimal_1 = int(_cfg["decimal_1"])
fee_tier_default = float(_cfg["fee_tier"])
tickspacing_default = int(_cfg["tickspacing"])


class UniswapV3LiquidityEnv(gym.Env):
    """
    Uniswap V3 Liquidity Management Environment (Gymnasium API)

    Action space (configurable):
        - Discrete (default): allocation ratios from a fixed set, e.g. [0.0, 0.25, 0.5, 0.75, 1.0]
        - Continuous: a real-valued allocation ratio in [0, 1], suitable for SAC/TD3

    Observation:
        obs_t = [ <selected features from numeric_data>, equity_t ]
        By default, first three columns [price, x_scaled_vol, y_scaled_vol] are included.
        Use `include_price`, `include_x_vol`, `include_y_vol` to control individually.

    Reward:
        log return (default) or step PnL.

    Episode termination/truncation:
        - terminated=True if equity <= liquidation_value
        - truncated=True when horizon consumed OR max_episode_steps reached
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        numeric_data,
        time_data,
        *,
        # --- Capital settings ---
        init_value: float = 100_000.0,
        randomize_init_value: bool = False,
        init_value_low: float = 50_000.0,
        init_value_high: float = 200_000.0,
        liquidation_value: float = 0.0,

        # --- Costs & fees ---
        gas_cost: float = 5.0,
        fee_tier: Optional[float] = None,

        # --- Horizon & indexing ---
        max_steps: Optional[int] = None,
        max_episode_steps: Optional[int] = 2000,
        start_index: int = 0,
        randomize_start: bool = False,

        # --- Action config ---
        action_mode: str = "discrete",
        action_ratios: Optional[Sequence[float]] = None,
        continuous_low: float = 0.0,
        continuous_high: float = 1.0,
        gas_change_tol: float = 1e-8,

        # --- Reward ---
        reward_mode: str = "log_return",
        log_eps: float = 1e-9,
        reward_scale: float = 100.0,
        
        # ★ NEW: Inaction penalty
        penalize_inaction: bool = False,
        inaction_penalty: float = 10.0,
        inaction_threshold: float = 0.05,

        # ★ NEW: State feature selection
        include_price: bool = True,
        include_x_vol: bool = True,
        include_y_vol: bool = True,
        additional_features_start_col: int = 3,  # Column index where additional features start

        # --- Market constants ---
        decimal0: Optional[int] = None,
        decimal1: Optional[int] = None,
        tickspacing: Optional[int] = None,
    ):
        super().__init__()

        # ===== Data & shape checks =====
        data = np.asarray(numeric_data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"`numeric_data` must be 2D, got shape={data.shape}")
        if data.shape[0] < 2:
            raise ValueError("`numeric_data` must have at least 2 rows to create transitions.")
        if data.shape[1] < 3:
            raise ValueError(
                "`numeric_data` must have at least 3 columns in order: "
                "[price, x_scaled_vol, y_scaled_vol, ...]."
            )

        self.data: np.ndarray = data
        self.time_data = time_data
        self.n_steps_total: int = data.shape[0]
        self.feature_dim_raw: int = data.shape[1]

        # ★ NEW: State feature selection
        self.include_price = bool(include_price)
        self.include_x_vol = bool(include_x_vol)
        self.include_y_vol = bool(include_y_vol)
        self.additional_features_start_col = int(additional_features_start_col)

        # Build list of column indices to include in observation
        self.state_col_indices = []
        if self.include_price:
            self.state_col_indices.append(0)
        if self.include_x_vol:
            self.state_col_indices.append(1)
        if self.include_y_vol:
            self.state_col_indices.append(2)
        
        # Add any additional features beyond the first 3 columns
        if self.additional_features_start_col < self.feature_dim_raw:
            for i in range(self.additional_features_start_col, self.feature_dim_raw):
                self.state_col_indices.append(i)
        
        self.state_col_indices = np.array(self.state_col_indices, dtype=np.int32)
        self.feature_dim = len(self.state_col_indices)  # Number of features in observation

        if self.feature_dim == 0:
            raise ValueError(
                "At least one feature must be included in the state. "
                "Set include_price, include_x_vol, or include_y_vol to True, "
                "or ensure additional_features_start_col < data columns."
            )

        # ===== Core parameters =====
        self.init_value_cfg = float(init_value)
        self.randomize_init_value = bool(randomize_init_value)
        self.init_value_low = float(init_value_low)
        self.init_value_high = float(init_value_high)

        self.liquidation_value = float(liquidation_value)
        self.gas_cost = float(gas_cost)
        self.fee_tier = float(fee_tier if fee_tier is not None else fee_tier_default)

        self.decimal_0 = int(decimal0 if decimal0 is not None else decimal_0)
        self.decimal_1 = int(decimal1 if decimal1 is not None else decimal_1)
        self.tickspacing = int(tickspacing if tickspacing is not None else tickspacing_default)

        # ===== Episode length limits =====
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps is not None else None
        self.episode_step_count = 0

        # ===== Horizon & starting index handling =====
        horizon = int(max_steps) if max_steps is not None else (self.n_steps_total - 1)
        if horizon < 1:
            raise ValueError("`max_steps` must allow at least 1 transition.")
        self.horizon: int = horizon

        self._fixed_start_index: int = int(start_index)
        self.randomize_start: bool = bool(randomize_start)
        self.start_index: int = 0
        self.end_index: int = 0

        # ===== Action space =====
        action_mode = str(action_mode).lower().strip()
        if action_mode not in ("discrete", "continuous"):
            raise ValueError("`action_mode` must be 'discrete' or 'continuous'.")
        self.action_mode = action_mode

        if self.action_mode == "discrete":
            if action_ratios is None:
                action_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
            ar = np.asarray(action_ratios, dtype=np.float32).reshape(-1)
            if ar.ndim != 1 or np.any(~np.isfinite(ar)):
                raise ValueError("`action_ratios` must be a 1D finite array/list.")
            ar = np.clip(ar, 0.0, 1.0)
            ar = np.unique(ar)
            if ar.size < 1:
                raise ValueError("`action_ratios` must contain at least one valid ratio.")
            self.action_ratios = ar
            self.action_space = spaces.Discrete(len(self.action_ratios))
        else:  # continuous
            self.action_ratios = None
            lo = float(continuous_low)
            hi = float(continuous_high)
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise ValueError("`continuous_low`/`continuous_high` must be finite and hi > lo.")
            self.cont_low = lo
            self.cont_high = hi
            self.action_space = spaces.Box(
                low=np.array([lo], dtype=np.float32),
                high=np.array([hi], dtype=np.float32),
                shape=(1,),
                dtype=np.float32,
            )

        # Observation space: selected features + equity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim + 1,), dtype=np.float32
        )

        # ===== Reward mode =====
        if reward_mode not in ("log_return", "pnl"):
            raise ValueError("`reward_mode` must be one of {'log_return','pnl'}.")
        self.reward_mode = reward_mode
        self.log_eps = float(log_eps)
        self.reward_scale = float(reward_scale)
        
        # ★ NEW: Inaction penalty parameters
        self.penalize_inaction = bool(penalize_inaction)
        self.inaction_penalty = float(inaction_penalty)
        self.inaction_threshold = float(inaction_threshold)

        # ===== State trackers =====
        self._t: int = 0
        self._last_action_idx: Optional[int] = None
        self._last_action_ratio: float = 0.0
        self.total_pnl: float = 0.0
        self.equity: float = float(self.init_value_cfg)
        self.prev_equity: float = float(self.init_value_cfg)
        self.gas_change_tol: float = float(gas_change_tol)

        # RNG
        self._rng = np.random.default_rng()
        
        # ★ Startup notification
        if self.penalize_inaction:
            print(f"⚠️  Inaction penalty enabled: Position < {self.inaction_threshold*100:.1f}% "
                  f"will incur penalty of {self.inaction_penalty}")
        
        print(f"📊 State features: {self._get_feature_names()}")

    # ==========================================================================================
    # Gymnasium API
    # ==========================================================================================

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.randomize_start:
            valid_max_start = self.n_steps_total - 1 - self.horizon
            if valid_max_start < 0:
                raise ValueError(
                    f"Data too short for horizon={self.horizon}. Need at least horizon+1 rows."
                )
            start = int(self._rng.integers(0, valid_max_start + 1))
        else:
            start = int(min(self._fixed_start_index, max(0, self.n_steps_total - 1 - self.horizon)))

        self.start_index = start
        self.end_index = start + self.horizon

        self._t = self.start_index
        self._last_action_idx = None
        self._last_action_ratio = 0.0
        self.total_pnl = 0.0
        self.episode_step_count = 0

        if self.randomize_init_value:
            lo, hi = self.init_value_low, self.init_value_high
            if not (hi > lo > 0):
                raise ValueError("init_value_low/high must satisfy 0 < low < high.")
            init_val = float(self._rng.uniform(lo, hi))
        else:
            init_val = float(self.init_value_cfg)

        self.equity = init_val
        self.prev_equity = init_val

        obs = self._get_obs()
        info = {"episode_start_index": self.start_index}
        return obs, info

    def step(self, action):
        """
        Args:
            action: int (discrete) or float/array-like (continuous).
        """
        t = self._t
        next_idx = t + 1
        if next_idx > self.end_index:
            raise RuntimeError("Step beyond episode end. Call reset().")

        # ----- Inputs (always read from raw data columns 0, 1, 2) -----
        price_in = float(self.data[t, 0])
        price_out = float(self.data[next_idx, 0])
        x_scaled_vol = float(self.data[next_idx, 1]) if self.feature_dim_raw >= 2 else 0.0
        y_scaled_vol = float(self.data[next_idx, 2]) if self.feature_dim_raw >= 3 else 0.0

        # ----- Tick -----
        tick_in = int(round(math.log(price_in * 10 ** (self.decimal_1 - self.decimal_0), 1.0001)))

        # ----- Parse action -> allocation_ratio -----
        if self.action_mode == "discrete":
            if not self.action_space.contains(action):
                raise ValueError(f"action={action} not in action space.")
            action_idx = int(action)
            allocation_ratio_raw = float(self.action_ratios[action_idx])
            allocation_ratio = float(np.clip(allocation_ratio_raw, 0.0, 1.0))
        else:
            a = np.asarray(action, dtype=np.float32).reshape(-1)[0]
            allocation_ratio_raw = float(a)
            allocation_ratio = float(np.clip(a, self.cont_low, self.cont_high))

        # Position sizing
        position_notional = max(0.0, self.equity) * allocation_ratio

        # Liquidity proxy
        liquidity = total_y_amount_2_liquidity(position_notional, tick_in, 500)
        
        # Gas cost
        gas_applied = self.gas_cost if allocation_ratio != 0 else 0.0

        # ----- PnL components -----
        fee_component = float(swap_fee(price_out, x_scaled_vol, y_scaled_vol, liquidity)[2])
        lvr_component = float(LVR(price_in, price_out, liquidity))
        step_pnl = fee_component + lvr_component - gas_applied

        # ★ NEW: Inaction penalty
        inaction_penalty_applied = 0.0
        if self.penalize_inaction and allocation_ratio < self.inaction_threshold:
            inaction_penalty_applied = -self.inaction_penalty
            step_pnl += inaction_penalty_applied

        # ----- Equity update -----
        self.total_pnl += step_pnl
        self.prev_equity = float(self.equity)
        self.equity = max(0.0, self.equity + step_pnl)
        after_equity = float(self.equity)

        # ----- Advance time & remember action -----
        self._t = next_idx
        self.episode_step_count += 1
        self._last_action_ratio = allocation_ratio
        self._last_action_idx = int(np.argmin(np.abs(self.action_ratios - allocation_ratio))) if self.action_mode == "discrete" else None

        # ----- Termination / truncation -----
        terminated = bool(self.equity <= self.liquidation_value)
        
        reached_max_steps = False
        if self.max_episode_steps is not None and self.episode_step_count >= self.max_episode_steps:
            reached_max_steps = True
        
        reached_horizon = bool(self._t >= self.end_index)
        truncated = reached_max_steps or reached_horizon

        # ----- Time passthrough -----
        try:
            if hasattr(self.time_data, "__len__") and len(self.time_data) > t:
                time_val = self.time_data[t][0] if np.ndim(self.time_data) == 2 else self.time_data[t]
            else:
                time_val = None
        except Exception:
            time_val = None

        # ----- Reward -----
        if self.reward_mode == "log_return":
            if self.prev_equity > 0.0 and after_equity > 0.0:
                raw_log_return = float(np.log((after_equity + self.log_eps) / (self.prev_equity + self.log_eps)))
                reward = raw_log_return * self.reward_scale
            else:
                reward = float(np.log(self.log_eps)) * self.reward_scale
        else:
            reward = float(step_pnl / self.init_value_cfg)

        obs = self._get_obs()
        
        # Determine termination reason
        terminated_reason = None
        if terminated:
            terminated_reason = "liquidation"
        elif reached_max_steps:
            terminated_reason = "max_episode_steps"
        elif reached_horizon:
            terminated_reason = "horizon"
        
        info = {
            # indexing / time
            "t": self._t,
            "episode_start_index": self.start_index,
            "episode_step_count": self.episode_step_count,
            "time": time_val,

            # equity / reward reporting
            "prev_equity": float(self.prev_equity),
            "after_equity": after_equity,
            "equity": after_equity,
            "log_return": float(np.log((after_equity + self.log_eps) / (self.prev_equity + self.log_eps)))
                          if self.prev_equity > 0 and after_equity > 0 else float(np.log(self.log_eps)),
            "pnl": float(step_pnl),

            # action / position
            "action_mode": self.action_mode,
            "allocation_ratio_raw": float(allocation_ratio_raw),
            "allocation_ratio": float(allocation_ratio),
            "allocation_idx": (int(self._last_action_idx) if self._last_action_idx is not None else None),
            "position_notional": float(position_notional),
            "liquidity": float(liquidity),

            # pricing / components
            "price_in": float(price_in),
            "price_out": float(price_out),
            "fee": float(fee_component),
            "lvr": float(lvr_component),
            "gas_applied": float(gas_applied),
            
            # ★ NEW: Inaction penalty info
            "inaction_penalty": float(inaction_penalty_applied),

            # vols passthrough
            "x_scaled_vol": float(x_scaled_vol),
            "y_scaled_vol": float(y_scaled_vol),

            # accounting
            "total_pnl": float(self.total_pnl),

            "terminated_reason": terminated_reason,
        }

        return obs, reward, terminated, truncated, info

    # ==========================================================================================
    # Helpers
    # ==========================================================================================

    def _get_obs(self) -> np.ndarray:
        """Extract selected features from data and append equity."""
        selected_features = self.data[self._t, self.state_col_indices].astype(np.float32)
        eq = np.array([self.equity], dtype=np.float32)
        return np.concatenate([selected_features, eq], axis=0)

    def _get_feature_names(self) -> list:
        """Return human-readable names of features included in observation."""
        base_names = ["price", "x_scaled_vol", "y_scaled_vol"]
        names = []
        for idx in self.state_col_indices:
            if idx < 3:
                names.append(base_names[idx])
            else:
                names.append(f"feature_{idx}")
        names.append("equity")
        return names

    def render(self):
        pass

    def close(self):
        pass