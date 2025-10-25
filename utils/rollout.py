# --- Simple rollout to CSV with robust time parsing and multi-algo support ---
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO, DQN, SAC, TD3


def _normalize_time_1d(time_data):
    """
    Convert time_data to 1D numpy array and return (time_ns: int64 1D, time_readable: Series).
    Supports various formats: seconds, nanoseconds, datetime64, strings, etc.
    """
    arr = np.asarray(time_data)
    
    # Flatten to 1D (supports (T,1), list of lists, pandas Series, etc.)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    
    # Case A: Numeric data (could be seconds or nanoseconds)
    if np.issubdtype(arr.dtype, np.number):
        arr_i64 = arr.astype("int64", copy=False)
        if np.nanmax(arr_i64) < 10**12:
            # Treat as seconds, convert to readable then to ns
            ser = pd.to_datetime(arr_i64, unit="s")
            time_ns = ser.astype("int64").to_numpy()
            return time_ns, ser
        else:
            # Treat as nanoseconds
            ser = pd.to_datetime(arr_i64)
            return arr_i64, ser
    
    # Case B: datetime64
    if np.issubdtype(arr.dtype, np.datetime64):
        ser = pd.to_datetime(arr)
        return ser.astype("int64").to_numpy(), ser
    
    # Case C: Other (strings / objects / mixed) -> to_datetime with coerce
    ser = pd.to_datetime(arr, errors="coerce")
    return ser.astype("int64").to_numpy(), ser


def rollout(
    model_path,
    progress_path,
    env_class,              # Your environment class, e.g. UniswapV3LiquidityEnv
    env_kw,                 # ENV_KW dictionary
    algo="ppo",             # Algorithm: 'ppo', 'dqn', 'sac', or 'td3'
    out_csv="./result/rollout.csv",
    numeric_data=None,      # If not in env_kw, pass it here
    time_data=None,         # If not in env_kw, pass it here (supports str/datetime/seconds/ns/2D)
    df=None,                # If nothing above provided, use df with price/x_scaled_vol/y_scaled_vol/timestamp
    deterministic=True,     # Use deterministic actions during rollout
    seed=42,                # Random seed for environment reset
):
    """
    Perform a complete rollout of a trained RL model and save results to CSV.
    
    Args:
        model_path: Path to saved model (.zip file)
        env_class: Environment class to instantiate
        env_kw: Dictionary of environment keyword arguments
        algo: Algorithm name - 'ppo', 'dqn', 'sac', or 'td3'
        out_csv: Output CSV file path
        numeric_data: Optional numpy array of [price, x_scaled_vol, y_scaled_vol]
        time_data: Optional time data in various formats
        df: Optional DataFrame containing required columns
        deterministic: Whether to use deterministic actions
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with rollout results
    """
    
    # Map algorithm names to classes
    ALGOS = {"ppo": PPO, "dqn": DQN, "sac": SAC}
    
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {list(ALGOS.keys())}, got: {algo}")
    
    ModelClass = ALGOS[algo]
    
    # 1) Fill in numeric_data / time_data if not in env_kw
    if "numeric_data" not in env_kw or "time_data" not in env_kw:
        if numeric_data is None or time_data is None:
            if df is None:
                raise ValueError(
                    "Please provide numeric_data/time_data in env_kw, "
                    "or pass numeric_data & time_data, or pass df."
                )
            numeric_data = df[["price", "x_scaled_vol", "y_scaled_vol"]].to_numpy(np.float64)
            time_data = df["timestamp"].to_numpy()
        env_kw["numeric_data"] = numeric_data
        env_kw["time_data"] = time_data
    else:
        numeric_data = env_kw["numeric_data"]
        time_data = env_kw["time_data"]
    
    # Convert time_data to 1D nanoseconds + readable format
    time_ns, time_readable = _normalize_time_1d(time_data)
    
    # 2) Disable randomization to run through entire dataset
    env_kw = dict(env_kw)  # Make a copy to avoid modifying original
    env_kw["max_episode_steps"] = int(numeric_data.shape[0]-1)
    env_kw["randomize_start"] = False
    env_kw["randomize_init_value"] = False
    env_kw["penalize_inaction"] = False
    
    # 3) Load model and create environment
    print(f"Loading {algo.upper()} model from: {model_path}")
    model = ModelClass.load(model_path, device="auto", print_system_info=False)
    env = env_class(**env_kw)
    
    rows = []
    try:
        obs, info = env.reset(seed=seed)
        done = False
        step = 0
        T = len(numeric_data)
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            step += 1
            
            # Get current timestep index (with fallback to step-1)
            t = getattr(env, "_t", getattr(env, "t", None))
            if t is None:
                t = step - 1
            t = int(np.clip(t, 0, T - 1))
            
            # Build row with basic data
            row = {
                "step": step,
                "reward": float(reward),
                "action": action.item() if hasattr(action, "item") else action,
                "close": float(numeric_data[t, 0]) if numeric_data is not None else np.nan,
                "x_scaled_vol": float(numeric_data[t, 1]) if numeric_data is not None else np.nan,
                "y_scaled_vol": float(numeric_data[t, 2]) if numeric_data is not None else np.nan,
                "timestamp_ns": int(time_ns[t]) if len(time_ns) > t else np.nan,
                "timestamp_readable": (
                    time_readable.iloc[t] if hasattr(time_readable, "iloc")
                    else (time_readable[t] if len(time_readable) > t else pd.NaT)
                ),
            }
            
            # Add price from environment if available (for cross-checking)
            if hasattr(env, "price"):
                try:
                    row["price_from_env"] = (
                        float(env.price[t]) if hasattr(env.price, "__getitem__") 
                        else float(env.price)
                    )
                except Exception:
                    pass
            
            # Extract common info fields + other scalar values
            if isinstance(info, dict):
                # Known important fields
                for k in ("equity", "fees", "gas_cost", "position_value", "liquidity"):
                    if k in info:
                        v = info[k]
                        row[k] = float(v) if isinstance(v, (int, float, np.number)) else v
                
                # Add any other scalar fields from info
                for k, v in info.items():
                    try:
                        if k not in row and np.isscalar(v):
                            row[k] = float(v) if isinstance(v, (int, float, np.number)) else v
                    except Exception:
                        pass
            
            rows.append(row)
            obs = next_obs
        
        # 4) Save to CSV
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_csv, index=False)
        df_progress = pd.read_csv(progress_path)
        
        print(f"\n{'='*60}")
        print(f"Rollout completed: {step} steps")
        print(f"Saved to: {Path(out_csv).resolve()}")
        print(f"{'='*60}\n")
        print("Last 10 rows:")
        print(df_out.tail(10).to_string(index=False))
        
        return df_out, df_progress
        
    finally:
        try:
            env.close()
        except Exception:
            pass