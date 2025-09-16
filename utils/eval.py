# ==== Best-trial loader + rollout (self-contained; saves to result/<trial_name>/) ====
# Requirements: numpy, pandas, torch, and your utils.env[2].UniswapV3LiquidityEnv.

import os, json, glob, shutil, sys, inspect, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Optional: bridge gymnasium -> gym to avoid version mismatch during env import
try:
    import gymnasium as _gym  # noqa
    sys.modules.setdefault("gym", _gym)
except Exception:
    pass


# ===== Env loader & kwargs filtering =====

def _get_env_class():
    """
    Prefer utils.env2.UniswapV3LiquidityEnv if available, else utils.env.UniswapV3LiquidityEnv.
    Also returns the module object so we can access module-level constants (e.g., decimal_0/1).
    """
    from utils.env import UniswapV3LiquidityEnv as Env
    import utils.env as env_mod
    return Env, env_mod, "utils.env"

# Static fallback list (used only if dynamic inspection fails)
ENV_INIT_KEYS_FALLBACK = {
    "numeric_data", "time_data",
    "init_value", "liquidation_value",
    "total_liquidity",        # for older envs
    "gas_cost", "fee_tier",
    "max_steps", "max_step",  # some envs use max_step
    "start_index",
}

def _env_init_kw(d: dict) -> dict:
    """
    Build kwargs for UniswapV3LiquidityEnv dynamically:
    - Keep only keys present in the env __init__ signature.
    - Map new->old names when needed:
        init_value -> total_liquidity (if env expects total_liquidity)
        max_steps  -> max_step        (if env expects max_step)
    """
    try:
        Env, _, _ = _get_env_class()
        params = set(inspect.signature(Env.__init__).parameters.keys())
        params.discard("self")

        out = {k: d[k] for k in d.keys() if k in params}

        # Remap: init_value -> total_liquidity
        if "init_value" in d and "init_value" not in params and "total_liquidity" in params and "total_liquidity" not in out:
            out["total_liquidity"] = d["init_value"]

        # Remap: max_steps -> max_step
        if "max_steps" in d and "max_steps" not in params and "max_step" in params and "max_step" not in out:
            out["max_step"] = d["max_steps"]

        return out
    except Exception:
        # Fallback to static whitelist if inspection fails
        return {k: d[k] for k in d.keys() if k in ENV_INIT_KEYS_FALLBACK}

def _probe_env_dims(env_kw: dict) -> tuple[int, int]:
    """Instantiate env once to get true obs_dim (flattened) and action_dim."""
    Env, _, _ = _get_env_class()
    env = Env(**_env_init_kw(env_kw))

    s0 = env.reset()
    if isinstance(s0, tuple):  # (obs, info)
        s0 = s0[0]
    s0 = np.asarray(s0, dtype=np.float32)
    obs_dim = int(np.prod(s0.shape))

    act_dim = 5
    if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
        act_dim = int(env.action_space.n)

    try:
        env.close()
    except Exception:
        pass
    return obs_dim, act_dim


#####################################################################################################################################
# DQN
#####################################################################################################################################

# --- Helper A) Locate experiment directory (Ray Tune results) ---
def _find_experiment_dir(explicit: str | Path | None) -> Path:
    """
    REQUIRE an explicit path chosen by the user.
    Accepts either:
      - the top-level experiment directory containing many trials, or
      - a single trial directory, or
      - a direct path to progress.csv / result.json (will use its parent dir).
    """
    if explicit is None:
        raise ValueError("Please pass experiment_dir=... (absolute or relative path). No fallback candidates are used.")
    p = Path(explicit).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {p}")

    if p.is_file():
        if p.name in ("progress.csv", "result.json"):
            return p.parent
        raise FileNotFoundError(f"Expected a directory or a Ray log file (progress.csv/result.json), got file: {p}")

    return p

# --- Helper B) Pick the best trial directory by maximum cum_return_max ---
def _pick_best_trial(exp_dir: Path) -> tuple[Path, float]:
    """
    Scan all trials under exp_dir. Prefer progress.csv; fallback to result.json (JSONL).
    Return (best_trial_dir, best_score).
    """
    best_dir, best_score = None, -np.inf

    # Prefer progress.csv aggregated metrics
    for csv_path in exp_dir.rglob("progress.csv"):
        try:
            df = pd.read_csv(csv_path)
            if "cum_return_max" in df.columns:
                v = pd.to_numeric(df["cum_return_max"], errors="coerce").max()
                if pd.notna(v) and float(v) > best_score:
                    best_score = float(v)
                    best_dir = csv_path.parent
        except Exception:
            pass

    # Fallback: scan result.json lines (newline-delimited JSON)
    if best_dir is None:
        for js in exp_dir.rglob("result.json"):
            try:
                with open(js, "r") as f:
                    local_best = -np.inf
                    for line in f:
                        rec = json.loads(line)
                        v = rec.get("cum_return_max", None)
                        if v is not None and np.isfinite(v) and v > local_best:
                            local_best = float(v)
                    if local_best > best_score:
                        best_score = local_best
                        best_dir = js.parent
            except Exception:
                pass

    if best_dir is None:
        raise RuntimeError("No `cum_return_max` found in any trial logs.")
    return best_dir, best_score

# --- Helper C) Read trial hyperparameters (e.g., net_dims) from params.json ---
def _read_trial_params(trial_dir: Path) -> dict:
    """
    Read parameters like `net_dims` from params.json/configuration.json/param.json.
    """
    params = {}
    for name in ("params.json", "configuration.json", "param.json"):
        p = trial_dir / name
        if p.exists():
            try:
                with open(p, "r") as f:
                    params = json.load(f)
            except Exception:
                pass
            break

    net_dims = params.get("net_dims", (64, 64))
    if isinstance(net_dims, list):
        net_dims = tuple(int(x) for x in net_dims)
    elif isinstance(net_dims, (int, float)):
        net_dims = (int(net_dims), int(net_dims))
    return {"net_dims": net_dims}

# --- Helper D) Build ElegantRL Config for inference ---
def _build_cfg_for_inference(best_trial_dir: Path, env_class, state_dim: int, action_dim: int, net_dims: tuple):
    """
    Build a minimal ElegantRL Config for inference.
    Points cfg.cwd to <trial_dir>/erl if it exists, otherwise the trial dir.
    """
    from elegantrl.train.config import Config
    from elegantrl.agents.AgentDQN import AgentDQN

    erl_dir = best_trial_dir / "erl"
    cwd_dir = erl_dir if erl_dir.exists() else best_trial_dir

    cfg = Config(agent_class=AgentDQN, env_class=env_class, env_args=None)
    cfg.state_dim = int(state_dim)
    cfg.action_dim = int(action_dim)
    cfg.net_dims = tuple(net_dims)
    cfg.gpu_id = 0 if torch.cuda.is_available() else -1
    cfg.cwd = str(cwd_dir)

    # Helpful extras for compatibility
    try:
        cfg.if_discrete = True
    except Exception:
        pass
    cfg.env_num = 1
    try:
        # max_step is not strictly required for inference, but set for completeness
        cfg.max_step = int(1)
    except Exception:
        pass
    return cfg

# --- Helper E) Robust DQN actor loader (prefers actor_latest.pth) ---
def load_dqn_actor(cfg, checkpoint_path: str | None = None):
    """
    Load DQN actor from checkpoint. Accepts:
      - raw state_dict
      - dict with "state_dict"
      - nn.Module (actor), or dict with "act"/"actor" module.
    Searches cfg.cwd recursively, preferring actor_latest.pth, then actor.pth/pt, then actor__*.pt.
    """
    from elegantrl.agents.AgentDQN import AgentDQN
    device = torch.device('cpu' if getattr(cfg, "gpu_id", -1) < 0 else 'cuda')
    agent  = AgentDQN(cfg.net_dims, cfg.state_dim, cfg.action_dim, gpu_id=getattr(cfg, "gpu_id", -1), args=cfg)
    agent.device = device

    def _find_ckpt(root):
        for name in ("actor_latest.pth", "actor.pth", "actor.pt"):
            p = os.path.join(root, name)
            if os.path.isfile(p): return p
        # try snapshots
        cands = sorted(glob.glob(os.path.join(root, "actor__*.pt")))
        if cands: return cands[-1]
        # recurse into subdirs
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d):
                p = _find_ckpt(d)
                if p: return p
        return None

    ckpt = checkpoint_path or _find_ckpt(cfg.cwd)
    if not ckpt:
        raise FileNotFoundError(f"No actor checkpoint found under {cfg.cwd}")

    obj = torch.load(ckpt, map_location=device)

    def _load_module(m: nn.Module):
        try:
            agent.act.load_state_dict(m.state_dict())
        except Exception:
            agent.act = m.to(device)
        agent.act.eval()

    if isinstance(obj, nn.Module):
        _load_module(obj)
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            agent.act.load_state_dict(obj["state_dict"]); agent.act.eval()
        elif "act" in obj and isinstance(obj["act"], nn.Module):
            _load_module(obj["act"])
        elif "actor" in obj and isinstance(obj["actor"], nn.Module):
            _load_module(obj["actor"])
        else:
            # assume it's a raw state_dict
            agent.act.load_state_dict(obj); agent.act.eval()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    return agent, ckpt

# --- Helper F) Greedy rollout with logging to CSV (saves into out_dir) ---
def rollout_dqn(data, time_data, env_kwargs_base: dict, cfg, csv_name="rollout.csv", out_dir="./result"):
    """
    Run greedy actions using the loaded DQN actor on the given dataset.
    - Uses _env_init_kw() to pass only valid kwargs into env
    - Flattens obs to 1D
    - Fallback-computes allocation_ratio/position_notional/equity/liquidity when missing
    - Logs expected_gas (first step free; gas only when action changes) for quick sanity-check
    """
    # Env & module (to access decimals if present)
    Env, env_mod, env_src = _get_env_class()
    decimal_0 = getattr(env_mod, "decimal_0", None)
    decimal_1 = getattr(env_mod, "decimal_1", None)

    # Optional liquidity multiplier for fallback
    try:
        from utils.uniswap import liquidity_multiplier as _max_liq_mult
    except Exception:
        _max_liq_mult = None

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose environment kwargs (whitelist + remap)
    base = dict(env_kwargs_base)
    base.update({
        "numeric_data": data,
        "time_data": time_data,
        "max_steps": min(base.get("max_steps", data.shape[0] - 1), data.shape[0] - 1),
    })
    env_kwargs = _env_init_kw(base)
    env = Env(**env_kwargs)

    # Load actor
    agent, ckpt = load_dqn_actor(cfg)

    # Copy checkpoint for reference
    try:
        shutil.copy2(ckpt, out_dir / os.path.basename(ckpt))
    except Exception:
        pass

    # Reset (old/new Gym compatible)
    s = env.reset()
    if isinstance(s, tuple):  # (obs, info)
        s = s[0]
    s = np.asarray(s, dtype=np.float32).reshape(-1)

    # Initial equity fallback
    try:
        eq_tracker = float(getattr(env, "equity")) if hasattr(env, "equity") else float(
            env_kwargs.get("init_value", env_kwargs.get("total_liquidity", np.nan))
        )
    except Exception:
        eq_tracker = np.nan

    # Default action values
    default_action_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    try:
        if hasattr(env, "action_values"):
            av = np.asarray(getattr(env, "action_values"), dtype=np.float32)
            if av.ndim == 1 and av.size >= 2:
                default_action_values = av
    except Exception:
        pass

    records, done, total, cum_max = [], False, 0.0, 0.0
    last_action = None
    printed_debug_once = False

    while not done:
        st = torch.as_tensor(s, dtype=torch.float32, device=agent.device).view(1, -1)
        with torch.no_grad():
            q = agent.act(st)                # [1, action_dim]
            a = int(q.argmax(dim=1).item())  # greedy action

        out = env.step(a)
        if isinstance(out, tuple) and len(out) == 5:
            s_next, r, term, trunc, info = out
            done = bool(term or trunc)
        else:
            s_next, r, done, info = out

        rr = float(0.0 if (r is None or not np.isfinite(r)) else r)
        total += rr
        if total > cum_max:
            cum_max = total

        s_next = np.asarray(s_next, dtype=np.float32).reshape(-1)
        s = s_next

        info = info or {}

        # ---- Fallbacks ----
        # allocation_ratio
        alloc_ratio = info.get("allocation_ratio", None)
        if alloc_ratio is None or not np.isfinite(alloc_ratio):
            try:
                alloc_ratio = float(default_action_values[a])
            except Exception:
                alloc_ratio = float("nan")

        # equity
        equity = info.get("equity", None)
        if equity is None or not np.isfinite(equity):
            try:
                if hasattr(env, "equity"):
                    equity = float(getattr(env, "equity"))
                else:
                    equity = float(eq_tracker)
            except Exception:
                equity = float("nan")

        # position_notional
        pos_notional = info.get("position_notional", None)
        if pos_notional is None or not np.isfinite(pos_notional):
            try:
                if np.isfinite(equity) and np.isfinite(alloc_ratio):
                    pos_notional = float(equity * alloc_ratio)
                else:
                    pos_notional = float("nan")
            except Exception:
                pos_notional = float("nan")

        # liquidity fallback（需 decimal_0/1 + max_liquidity_multiplier）
        liquidity = info.get("liquidity", None)
        if (liquidity is None or not np.isfinite(liquidity)) and _max_liq_mult and (decimal_0 is not None) and (decimal_1 is not None):
            try:
                price_in = float(info.get("price_in", np.nan))
                if np.isfinite(price_in) and np.isfinite(pos_notional):
                    tick = int(round(math.log(price_in * (10 ** (decimal_0 - decimal_1)), 1.0001)))
                    mult = float(_max_liq_mult(tick, 500))
                    liquidity = float(mult * pos_notional * (10 ** decimal_1))
                else:
                    liquidity = float("nan")
            except Exception:
                liquidity = float("nan")

        # ---- Record ----
        records.append({
            "t": info.get("t", len(records)),
            "time": info.get("time", np.nan),
            "action_idx": info.get("allocation_idx", a),
            "allocation_ratio": alloc_ratio,
            "position_notional": pos_notional,
            "liquidity": liquidity,
            "price_in": info.get("price_in", np.nan),
            "price_out": info.get("price_out", np.nan),
            "fee": info.get("fee", np.nan),
            "lvr": info.get("lvr", np.nan),
            "gas": info.get("gas_applied", np.nan),
            "step_reward": rr,
            "cum_reward": total,
            "cum_reward_max": cum_max,
            "equity": equity,
            "terminated_reason": info.get("terminated_reason", None),
        })

    df = pd.DataFrame(records)
    out_path = out_dir / csv_name
    df.to_csv(out_path, index=False)
    print(f"✅ Rollout saved: {out_path}\nTotal return: {total:.6f}\nCheckpoint: {ckpt}")
    return df, total, str(out_path)

# --- Main: one-call pipeline (find best trial → build cfg → run train/test rollouts) ---
def best_dqn_rollout(
    train_data,
    train_time_data,
    test_data,
    test_time_data,
    TRAIN_ENV_KW: dict,
    TEST_ENV_KW: dict,
    *,
    result_root: str = "./result",
    experiment_dir: str | Path,   # ← REQUIRED
    action_dim: int = 5,
):
    """
    One-call pipeline:
      1) Locate the best Ray Tune trial by `cum_return_max`
      2) Build an ElegantRL Config for inference
      3) Run greedy rollouts on train & test sets
      4) Save CSVs and a copy of the used checkpoint to result/<trial_name>/

    Returns a dict with file paths, metrics, and the cfg used for inference.
    """
    Env, _, _ = _get_env_class()

    # --- Probe true dims from the actual env (dynamic-filtered kwargs) ---
    probe_kw = {
        "numeric_data": train_data,
        "time_data": train_time_data,
        **TRAIN_ENV_KW,
        "max_steps": min(int(TRAIN_ENV_KW.get("max_steps", train_data.shape[0]-1)), int(train_data.shape[0]-1)),
    }
    state_dim, action_dim_probe = _probe_env_dims(probe_kw)
    action_dim_used = int(action_dim_probe if action_dim_probe and action_dim_probe > 0 else action_dim)

    # Locate experiment directory and pick the best trial
    exp_dir = _find_experiment_dir(experiment_dir)
    best_trial_dir, best_score = _pick_best_trial(exp_dir)
    trial_tag = best_trial_dir.name

    # Read params (net_dims) and build cfg for inference
    params = _read_trial_params(best_trial_dir)
    net_dims = tuple(params["net_dims"])
    cfg = _build_cfg_for_inference(
        best_trial_dir=best_trial_dir,
        env_class=Env,
        state_dim=int(state_dim),
        action_dim=int(action_dim_used),
        net_dims=net_dims,
    )

    # Prepare output directory: result/<trial_name>/
    result_root_path = Path(result_root).resolve()
    out_dir = result_root_path / trial_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run rollouts (train / test) and write CSVs into the result folder
    df_train, train_return, train_csv_path = rollout_dqn(
        train_data, train_time_data, TRAIN_ENV_KW, cfg,
        csv_name="train_result.csv", out_dir=str(out_dir)
    )
    df_test, test_return, test_csv_path = rollout_dqn(
        test_data, test_time_data, TEST_ENV_KW, cfg,
        csv_name="test_result.csv", out_dir=str(out_dir)
    )

    # Summary
    print("🏆 Best trial:", best_trial_dir)
    print("   best cum_return_max:", best_score)
    print("   net_dims from params:", net_dims)
    print("📂 Output dir:", out_dir)
    print(f"📈 Train return: {train_return:.6f} | Test return: {test_return:.6f}")
    print(f"📝 Files: {train_csv_path} , {test_csv_path}")

    return {
        "trial_dir": str(best_trial_dir),
        "trial_tag": trial_tag,
        "best_cum_return_max": float(best_score),
        "net_dims": net_dims,
        "out_dir": str(out_dir),
        "train": {
            "csv": str(train_csv_path),
            "total_return": float(train_return),
            "dataframe": df_train,
        },
        "test": {
            "csv": str(test_csv_path),
            "total_return": float(test_return),
            "dataframe": df_test,
        },
        "cfg": cfg,  # you can reuse this cfg and its checkpoint later
    }


#####################################################################################################################################
# PPO
#####################################################################################################################################

# ==== Best-trial loader + rollout for PPO (self-contained; saves to result/<trial_name>/) ====

# --- Helper A) Locate experiment directory (Ray Tune results) ---
def _find_experiment_dir_ppo(explicit: str | Path | None) -> Path:
    """
    REQUIRE an explicit path chosen by the user.
    Accepts either:
      - the top-level experiment directory containing many trials, or
      - a single trial directory, or
      - a direct path to progress.csv / result.json (will use its parent dir).
    """
    if explicit is None:
        raise ValueError("Please pass experiment_dir=... (absolute or relative path). No fallback candidates are used.")
    p = Path(explicit).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"PPO experiment path does not exist: {p}")

    if p.is_file():
        if p.name in ("progress.csv", "result.json"):
            return p.parent
        raise FileNotFoundError(f"Expected a directory or a Ray log file (progress.csv/result.json), got file: {p}")

    return p

# --- Helper B) Pick the best trial directory by maximum cum_return_max ---
def _pick_best_trial_ppo(exp_dir: Path) -> tuple[Path, float]:
    """
    Scan all trials under exp_dir. Prefer progress.csv; fallback to result.json (JSONL).
    Return (best_trial_dir, best_score).
    """
    best_dir, best_score = None, -np.inf

    # Prefer progress.csv aggregated metrics
    for csv_path in exp_dir.rglob("progress.csv"):
        try:
            df = pd.read_csv(csv_path)
            if "cum_return_max" in df.columns:
                v = pd.to_numeric(df["cum_return_max"], errors="coerce").max()
                if pd.notna(v) and float(v) > best_score:
                    best_score = float(v)
                    best_dir = csv_path.parent
        except Exception:
            pass

    # Fallback: scan result.json lines
    if best_dir is None:
        for js in exp_dir.rglob("result.json"):
            try:
                with open(js, "r") as f:
                    local_best = -np.inf
                    for line in f:
                        rec = json.loads(line)
                        v = rec.get("cum_return_max", None)
                        if v is not None and np.isfinite(v) and v > local_best:
                            local_best = float(v)
                    if local_best > best_score:
                        best_score = local_best
                        best_dir = js.parent
            except Exception:
                pass

    if best_dir is None:
        raise RuntimeError("No `cum_return_max` found in any PPO trial logs.")
    return best_dir, best_score

# --- Helper C) Read trial hyperparameters (e.g., net_dims) from params.json ---
def _read_trial_params_ppo(trial_dir: Path) -> dict:
    """
    Read parameters like `net_dims` from params.json/configuration.json/param.json.
    """
    params = {}
    for name in ("params.json", "configuration.json", "param.json"):
        p = trial_dir / name
        if p.exists():
            try:
                with open(p, "r") as f:
                    params = json.load(f)
            except Exception:
                pass
            break

    net_dims = params.get("net_dims", (64, 64))
    if isinstance(net_dims, list):
        net_dims = tuple(int(x) for x in net_dims)
    elif isinstance(net_dims, (int, float)):
        net_dims = (int(net_dims), int(net_dims))
    return {"net_dims": net_dims}

# --- Helper D) Build ElegantRL Config for inference (PPO, discrete) ---
def _build_cfg_for_inference_ppo(best_trial_dir: Path, env_class, state_dim: int, action_dim: int, net_dims: tuple):
    """
    Build a minimal ElegantRL Config for PPO inference.
    Prefers AgentDiscretePPO; falls back to AgentPPO with discrete nets.
    Points cfg.cwd to <trial_dir>/erl_ppo (or erl) if exists, otherwise the trial dir.
    """
    from elegantrl.train.config import Config
    try:
        from elegantrl.agents.AgentPPO import AgentDiscretePPO as PPOAgent
    except Exception:
        from elegantrl.agents.AgentPPO import AgentPPO as PPOAgent

    erl_ppo_dir = best_trial_dir / "erl_ppo"
    erl_dir     = best_trial_dir / "erl"
    if erl_ppo_dir.exists():
        cwd_dir = erl_ppo_dir
    elif erl_dir.exists():
        cwd_dir = erl_dir
    else:
        cwd_dir = best_trial_dir

    cfg = Config(agent_class=PPOAgent, env_class=env_class, env_args=None)
    cfg.state_dim = int(state_dim)
    cfg.action_dim = int(action_dim)
    cfg.net_dims = tuple(net_dims)
    cfg.gpu_id = 0 if torch.cuda.is_available() else -1
    cfg.cwd = str(cwd_dir)
    try:
        cfg.if_discrete = True
    except Exception:
        pass
    cfg.env_num = 1
    try:
        cfg.max_step = int(1)
    except Exception:
        pass
    return cfg

# --- Helper E) Robust PPO actor loader (prefers actor_latest/actor_best) ---
def load_ppo_actor(cfg, checkpoint_path: str | None = None):
    """
    Load PPO actor from checkpoint. Accepts:
      - raw state_dict
      - dict with "state_dict"
      - nn.Module (actor), or dict with "act"/"actor" module.
    Searches cfg.cwd recursively, preferring actor_latest.pth, actor_best.pth, then actor.pth/pt, then actor__*.pt.
    """
    try:
        from elegantrl.agents.AgentPPO import AgentDiscretePPO as PPOAgent
    except Exception:
        from elegantrl.agents.AgentPPO import AgentPPO as PPOAgent

    device = torch.device('cpu' if getattr(cfg, "gpu_id", -1) < 0 else 'cuda')
    agent  = PPOAgent(cfg.net_dims, cfg.state_dim, cfg.action_dim, gpu_id=getattr(cfg, "gpu_id", -1), args=cfg)
    agent.device = device

    def _find_ckpt(root):
        for name in ("actor_latest.pth", "actor_best.pth", "actor.pth", "actor.pt"):
            p = os.path.join(root, name)
            if os.path.isfile(p): return p
        # try snapshots
        cands = sorted(glob.glob(os.path.join(root, "actor__*.pt")))
        if cands: return cands[-1]
        # recurse into subdirs
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d):
                p = _find_ckpt(d)
                if p: return p
        return None

    ckpt = checkpoint_path or _find_ckpt(cfg.cwd)
    if not ckpt:
        raise FileNotFoundError(f"No PPO actor checkpoint found under {cfg.cwd}")

    obj = torch.load(ckpt, map_location=device)

    def _load_module(m: nn.Module):
        try:
            agent.act.load_state_dict(m.state_dict())
        except Exception:
            agent.act = m.to(device)
        agent.act.eval()

    if isinstance(obj, nn.Module):
        _load_module(obj)
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            agent.act.load_state_dict(obj["state_dict"]); agent.act.eval()
        elif "act" in obj and isinstance(obj["act"], nn.Module):
            _load_module(obj["act"])
        elif "actor" in obj and isinstance(obj["actor"], nn.Module):
            _load_module(obj["actor"])
        else:
            # assume it's a raw state_dict
            agent.act.load_state_dict(obj); agent.act.eval()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    return agent, ckpt

# --- Helper F) Convert PPO actor outputs to a discrete index (0..action_dim-1) ---
def _to_discrete_idx_from_actor(actor, st, action_dim: int) -> int:
    """
    Robustly turn any actor output (logits/tensor/tuple) into a valid discrete action index.
    """
    with torch.no_grad():
        a = None
        if hasattr(actor, "get_action"):
            a = actor.get_action(st)
            if isinstance(a, (tuple, list)):
                a = a[0]
        if a is None:
            out = actor(st)
            if isinstance(out, (tuple, list)):
                out = out[0]
            a = out
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        a = np.asarray(a).squeeze()
        if a.ndim == 0:
            try:
                idx = int(a.item())
            except Exception:
                idx = int(a)
        else:
            idx = int(np.argmax(a))  # logits / vector → argmax
        if idx < 0: idx = 0
        if idx >= action_dim: idx = action_dim - 1
        return idx

# --- Helper G) Greedy rollout with logging to CSV (saves into out_dir) ---
def rollout_ppo(data, time_data, env_kwargs_base: dict, cfg, csv_name="rollout.csv", out_dir="./result"):
    """
    Run greedy actions (argmax over logits) using the loaded PPO actor on the given dataset.
    - Uses _env_init_kw() to pass only valid kwargs into env
    - Flattens obs to 1D if needed
    - (Light) fallback for missing info fields
    """
    Env, env_mod, env_src = _get_env_class()

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = dict(env_kwargs_base)
    base.update({
        "numeric_data": data,
        "time_data": time_data,
        "max_steps": min(base.get("max_steps", data.shape[0]-1), data.shape[0]-1),
    })
    env_kwargs = _env_init_kw(base)
    env = Env(**env_kwargs)

    agent, ckpt = load_ppo_actor(cfg)
    try:
        shutil.copy2(ckpt, out_dir / os.path.basename(ckpt))
    except Exception:
        pass

    s = env.reset()
    if isinstance(s, tuple):
        s = s[0]
    s = np.asarray(s, dtype=np.float32).reshape(-1)

    # Default action values for fallback allocation ratio
    default_action_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    try:
        if hasattr(env, "action_values"):
            av = np.asarray(getattr(env, "action_values"), dtype=np.float32)
            if av.ndim == 1 and av.size >= 2:
                default_action_values = av
    except Exception:
        pass

    records, done, total, cum_max = [], False, 0.0, 0.0
    while not done:
        st = torch.as_tensor(s, dtype=torch.float32, device=getattr(agent, "device", torch.device("cpu"))).view(1, -1)
        idx = _to_discrete_idx_from_actor(agent.act, st, int(getattr(cfg, "action_dim", 5)))

        out = env.step(idx)
        if isinstance(out, tuple) and len(out) == 5:
            s_next, r, term, trunc, info = out
            done = bool(term or trunc)
        else:
            s_next, r, done, info = out

        rr = float(0.0 if (r is None or not np.isfinite(r)) else r)
        total += rr
        if total > cum_max:
            cum_max = total

        s_next = np.asarray(s_next, dtype=np.float32).reshape(-1)
        s = s_next

        info = info or {}

        # light fallback
        alloc_ratio = info.get("allocation_ratio")
        if alloc_ratio is None or not np.isfinite(alloc_ratio):
            try:
                alloc_ratio = float(default_action_values[idx])
            except Exception:
                alloc_ratio = float("nan")

        records.append({
            "t": info.get("t", len(records)),
            "time": info.get("time", np.nan),
            "action_idx": info.get("allocation_idx", idx),
            "allocation_ratio": alloc_ratio,
            "position_notional": info.get("position_notional", np.nan),
            "liquidity": info.get("liquidity", np.nan),
            "price_in": info.get("price_in", np.nan),
            "price_out": info.get("price_out", np.nan),
            "fee": info.get("fee", np.nan),
            "lvr": info.get("lvr", np.nan),
            "gas": info.get("gas_applied", np.nan),
            "step_reward": rr,
            "cum_reward": total,
            "cum_reward_max": cum_max,
            "equity": info.get("equity", np.nan),
            "terminated_reason": info.get("terminated_reason", None),
        })

    df = pd.DataFrame(records)
    out_path = out_dir / csv_name
    df.to_csv(out_path, index=False)
    print(f"✅ PPO Rollout saved: {out_path}\nTotal return: {total:.6f}\nCheckpoint: {ckpt}")
    return df, total, str(out_path)

# --- Main: one-call pipeline (find best trial → build cfg → run train/test rollouts) ---
def best_ppo_rollout(
    train_data,
    train_time_data,
    test_data,
    test_time_data,
    TRAIN_ENV_KW: dict,
    TEST_ENV_KW: dict,
    *,
    result_root: str = "./result",
    experiment_dir: str | Path,   # ← REQUIRED
    action_dim: int = 5,
):
    """
    One-call pipeline for PPO:
      1) Locate the best Ray Tune trial by `cum_return_max`
      2) Build an ElegantRL Config for inference (discrete actor)
      3) Run greedy rollouts on train & test sets
      4) Save CSVs and a copy of the used checkpoint to result/<trial_name>/

    Returns a dict with file paths, metrics, and the cfg used for inference.
    """
    Env, _, _ = _get_env_class()

    # --- Probe true dims from the actual env (dynamic-filtered kwargs) ---
    probe_kw = {
        "numeric_data": train_data,
        "time_data": train_time_data,
        **TRAIN_ENV_KW,
        "max_steps": min(int(TRAIN_ENV_KW.get("max_steps", train_data.shape[0]-1)), int(train_data.shape[0]-1)),
    }
    state_dim, action_dim_probe = _probe_env_dims(probe_kw)
    action_dim_used = int(action_dim_probe if action_dim_probe and action_dim_probe > 0 else action_dim)

    # Locate experiment directory and pick the best trial
    exp_dir = _find_experiment_dir_ppo(experiment_dir)
    best_trial_dir, best_score = _pick_best_trial_ppo(exp_dir)
    trial_tag = best_trial_dir.name

    # Read params (net_dims) and build cfg for inference
    params = _read_trial_params_ppo(best_trial_dir)
    net_dims = tuple(params["net_dims"])
    cfg = _build_cfg_for_inference_ppo(
        best_trial_dir=best_trial_dir,
        env_class=Env,
        state_dim=int(state_dim),
        action_dim=int(action_dim_used),
        net_dims=net_dims,
    )

    # Prepare output directory: result/<trial_name>/
    result_root_path = Path(result_root).resolve()
    out_dir = result_root_path / trial_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run rollouts (train / test) and write CSVs into the result folder
    df_train, train_return, train_csv_path = rollout_ppo(
        train_data, train_time_data, TRAIN_ENV_KW, cfg,
        csv_name="train_result.csv", out_dir=str(out_dir)
    )
    df_test, test_return, test_csv_path = rollout_ppo(
        test_data, test_time_data, TEST_ENV_KW, cfg,
        csv_name="test_result.csv", out_dir=str(out_dir)
    )

    # Summary
    print("🏆 PPO Best trial:", best_trial_dir)
    print("   best cum_return_max:", best_score)
    print("   net_dims from params:", net_dims)
    print("📂 Output dir:", out_dir)
    print(f"📈 Train return: {train_return:.6f} | Test return: {test_return:.6f}")
    print(f"📝 Files: {train_csv_path} , {test_csv_path}")

    return {
        "trial_dir": str(best_trial_dir),
        "trial_tag": trial_tag,
        "best_cum_return_max": float(best_score),
        "net_dims": net_dims,
        "out_dir": str(out_dir),
        "train": {
            "csv": str(train_csv_path),
            "total_return": float(train_return),
            "dataframe": df_train,
        },
        "test": {
            "csv": str(test_csv_path),
            "total_return": float(test_return),
            "dataframe": df_test,
        },
        "cfg": cfg,  # you can reuse this cfg and its checkpoint later
    }
