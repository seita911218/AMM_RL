# ==== Best-trial loader + single-run rollout (saves to result/<trial_name>/) ====
# Requirements: numpy, pandas, torch, and your utils.env.UniswapV3LiquidityEnv.
# This version is robust to state_dim changes (e.g., you appended equity to the obs).
# It infers the checkpoint's expected input dim and will clip/pad current obs accordingly.
# IMPORTANT: Your env must keep numeric_data[:, 0:3] = [price, x_scaled_vol, y_scaled_vol] in that exact order.

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


# =========================
# Env loader & kwargs filter
# =========================

def _get_env_class():
    """
    Fixed import from utils.env (no env2 fallback).
    Returns (EnvClass, env_module, module_name).
    """
    from utils.env import UniswapV3LiquidityEnv as Env
    import utils.env as env_mod
    return Env, env_mod, "utils.env"


# Whitelist (used only if signature inspection fails)
ENV_INIT_KEYS_FALLBACK = {
    "numeric_data", "time_data",
    "init_value", "liquidation_value",
    "total_liquidity",
    "gas_cost", "fee_tier",
    "max_steps", "max_step",
    "start_index",
}

def _env_init_kw(d: dict) -> dict:
    """
    Keep only keys accepted by Env.__init__. Map a few legacy names if needed.
    - init_value -> total_liquidity
    - max_steps  -> max_step
    """
    try:
        Env, _, _ = _get_env_class()
        params = set(inspect.signature(Env.__init__).parameters.keys())
        params.discard("self")

        out = {k: d[k] for k in d.keys() if k in params}

        # Remap: init_value -> total_liquidity (older envs)
        if "init_value" in d and "init_value" not in params and "total_liquidity" in params and "total_liquidity" not in out:
            out["total_liquidity"] = d["init_value"]

        # Remap: max_steps -> max_step (older envs)
        if "max_steps" in d and "max_steps" not in params and "max_step" in params and "max_step" not in out:
            out["max_step"] = d["max_steps"]

        return out
    except Exception:
        # Fallback to static whitelist (best-effort)
        return {k: d[k] for k in d.keys() if k in ENV_INIT_KEYS_FALLBACK}

def _probe_env_dims(env_kw: dict) -> tuple[int, int]:
    """
    Instantiate env once to get the true flattened obs_dim and action_dim.
    """
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


# =========================
# Common helpers
# =========================

def _find_experiment_dir(explicit: str | Path | None) -> Path:
    """
    Require a Ray Tune experiment directory (or a file under it).
    If a file is provided (progress.csv/result.json), use its parent directory.
    """
    if explicit is None:
        raise ValueError("Please pass experiment_dir=... (absolute or relative path).")
    p = Path(explicit).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {p}")
    if p.is_file():
        if p.name in ("progress.csv", "result.json"):
            return p.parent
        raise FileNotFoundError(f"Expected a directory or a Ray log file (progress.csv/result.json), got file: {p}")
    return p

def _pick_best_trial_generic(exp_dir: Path) -> tuple[Path, float, str]:
    """
    Scan all trials; prefer max(final_equity), fallback to max(cum_return_max).
    Supports both progress.csv and result.json (JSONL).
    Returns (best_trial_dir, best_score, best_metric_name).
    """
    best_dir, best_score, best_metric = None, -np.inf, "final_equity"

    # 1) result.json: final_equity
    for js in exp_dir.rglob("result.json"):
        try:
            with open(js, "r") as f:
                local_best = -np.inf
                for line in f:
                    rec = json.loads(line)
                    v = rec.get("final_equity", None)
                    if v is not None and np.isfinite(v) and v > local_best:
                        local_best = float(v)
                if local_best > best_score:
                    best_score = local_best
                    best_dir = js.parent
                    best_metric = "final_equity"
        except Exception:
            pass

    # 2) progress.csv: final_equity → cum_return_max
    for csv_path in exp_dir.rglob("progress.csv"):
        try:
            df = pd.read_csv(csv_path)
            cand, metric_name = None, None
            if "final_equity" in df.columns:
                cand = pd.to_numeric(df["final_equity"], errors="coerce").max()
                metric_name = "final_equity"
            elif "cum_return_max" in df.columns:
                cand = pd.to_numeric(df["cum_return_max"], errors="coerce").max()
                metric_name = "cum_return_max"
            if cand is not None and pd.notna(cand) and float(cand) > best_score:
                best_score = float(cand)
                best_dir = csv_path.parent
                best_metric = metric_name
        except Exception:
            pass

    # 3) fallback result.json: cum_return_max
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
                        best_metric = "cum_return_max"
            except Exception:
                pass

    if best_dir is None:
        raise RuntimeError("No `final_equity` (or `cum_return_max`) found in any trial logs.")
    return best_dir, best_score, best_metric

def _read_trial_params(trial_dir: Path) -> dict:
    """
    Read hyperparameters (currently net_dims) in a robust way.
    Supports filenames: params.json / configuration.json / param.json.
    Returns {"net_dims": (h1, h2, ...)} with sensible defaults if missing.
    """
    def _coerce_net_dims(val):
        if val is None:
            return (64, 64)
        if isinstance(val, (list, tuple)):
            try:
                return tuple(int(x) for x in val) if len(val) > 0 else (64, 64)
            except Exception:
                return (64, 64)
        if isinstance(val, (int, float)):
            i = int(val)
            return (i, i)
        # Sometimes stored as a string like "[128, 128]"
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (list, tuple)):
                return tuple(int(x) for x in parsed)
        except Exception:
            pass
        return (64, 64)

    for name in ("params.json", "configuration.json", "param.json"):
        p = trial_dir / name
        if p.exists():
            try:
                with open(p, "r") as f:
                    params = json.load(f)
                net_dims = params.get("net_dims", None)
                if net_dims is None and isinstance(params.get("params"), dict):
                    net_dims = params["params"].get("net_dims", None)
                if net_dims is None and isinstance(params.get("config"), dict):
                    net_dims = params["config"].get("net_dims", None)
                return {"net_dims": _coerce_net_dims(net_dims)}
            except Exception:
                pass

    # Default fallback
    return {"net_dims": (64, 64)}


# =========================
# DQN inference
# =========================

def _build_cfg_for_inference_dqn(best_trial_dir: Path, env_class, state_dim: int, action_dim: int, net_dims: tuple):
    """
    Build a minimal ElegantRL Config for DQN inference and point cwd to <trial_dir>/erl if it exists.
    """
    from elegantrl.train.config import Config
    from elegantrl.agents.AgentDQN import AgentDQN  # noqa: F401 (ensure presence)

    erl_dir = best_trial_dir / "erl"
    cwd_dir = erl_dir if erl_dir.exists() else best_trial_dir

    cfg = Config(agent_class=None, env_class=env_class, env_args=None)
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

def _infer_required_state_dim_from_sd(sd: dict) -> int | None:
    """
    Try to infer the checkpoint's expected input dim from a state_dict-like mapping.
    Heuristics:
      - 'state_avg' 1D tensor length
      - 'net.0.weight' in_features (shape[1])
      - else: first 2D weight tensor's in_features
    """
    try:
        if "state_avg" in sd and torch.is_tensor(sd["state_avg"]) and sd["state_avg"].ndim == 1:
            return int(sd["state_avg"].numel())
    except Exception:
        pass
    try:
        if "net.0.weight" in sd and torch.is_tensor(sd["net.0.weight"]) and sd["net.0.weight"].ndim == 2:
            return int(sd["net.0.weight"].shape[1])
    except Exception:
        pass
    # Fallback: first 2D weight
    try:
        for k, v in sd.items():
            if torch.is_tensor(v) and v.ndim == 2:
                return int(v.shape[1])
    except Exception:
        pass
    return None

def load_dqn_actor(cfg, checkpoint_path: str | None = None):
    """
    Load DQN actor from checkpoint AND infer the required input state_dim.
    Returns (agent, ckpt_path, required_state_dim).
    """
    from elegantrl.agents.AgentDQN import AgentDQN
    device = torch.device('cpu' if getattr(cfg, "gpu_id", -1) < 0 else 'cuda')

    def _find_ckpt(root):
        for name in ("actor_latest.pth", "actor.pth", "actor.pt"):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
        cands = sorted(glob.glob(os.path.join(root, "actor__*.pt")))
        if cands:
            return cands[-1]
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d):
                p = _find_ckpt(d)
                if p:
                    return p
        return None

    ckpt = checkpoint_path or _find_ckpt(cfg.cwd)
    if not ckpt:
        raise FileNotFoundError(f"No actor checkpoint found under {cfg.cwd}")

    obj = torch.load(ckpt, map_location=device)

    # Infer required input dim from the checkpoint object
    req_dim = None
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        if isinstance(sd, dict):
            req_dim = _infer_required_state_dim_from_sd(sd)

    # If we couldn't infer, fall back to cfg.state_dim (current env dim)
    if req_dim is None:
        req_dim = int(getattr(cfg, "state_dim", 0)) or 1

    # Build agent with the required state_dim (so layer shapes match the checkpoint)
    cfg.state_dim = int(req_dim)
    agent = AgentDQN(cfg.net_dims, cfg.state_dim, cfg.action_dim, gpu_id=getattr(cfg, "gpu_id", -1), args=cfg)
    agent.device = device

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
            agent.act.load_state_dict(obj); agent.act.eval()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    return agent, ckpt, int(req_dim)

def _project_obs(obs_1d: np.ndarray, want: int) -> np.ndarray:
    """
    Clip or right-pad zeros to make obs length == want.
    IMPORTANT: By truncating at the END, we preserve the first features
    (price/x_vol/y_vol ...), and if equity was appended at the end, it is the
    first to be dropped when shrinking dims (safe for old checkpoints).
    """
    x = np.asarray(obs_1d, dtype=np.float32).reshape(-1)
    have = x.shape[0]
    if have == want:
        return x
    if have > want:
        return x[:want]
    # have < want → pad zeros at the end
    out = np.zeros((want,), dtype=np.float32)
    out[:have] = x
    return out

def rollout_dqn(
    data: np.ndarray,
    time_data: np.ndarray,
    env_kwargs_base: dict,
    cfg,
    csv_name: str = "dqn_result.csv",
    out_dir: str = "./result",
):
    """
    Run greedy actions using the loaded DQN actor on a single dataset.
    Records fields aligned with env.info: after_equity, prev_equity, return, gas_applied, etc.
    If checkpoint expects a different state_dim, observations are clipped/padded to match.
    """
    # Optional fallback liquidity calc (not required)
    try:
        from utils.uniswap import liquidity_multiplier as _liq_mult
    except Exception:
        _liq_mult = None

    Env, env_mod, _ = _get_env_class()
    decimal_0 = getattr(env_mod, "decimal_0", None)
    decimal_1 = getattr(env_mod, "decimal_1", None)

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

    # Load actor and infer required input dim
    agent, ckpt, req_state_dim = load_dqn_actor(cfg)
    try:
        shutil.copy2(ckpt, out_dir / os.path.basename(ckpt))
    except Exception:
        pass

    s = env.reset()
    if isinstance(s, tuple):
        s = s[0]
    s = np.asarray(s, dtype=np.float32).reshape(-1)

    # Default action values (fallback for allocation_ratio)
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
        s_proj = _project_obs(s, req_state_dim)
        st = torch.as_tensor(s_proj, dtype=torch.float32, device=agent.device).view(1, -1)
        with torch.no_grad():
            q = agent.act(st)
            a = int(q.argmax(dim=1).item())

        out = env.step(a)
        if isinstance(out, tuple) and len(out) == 5:
            s_next, r, term, trunc, info = out
            done = bool(term or trunc)
        else:
            s_next, r, done, info = out

        rr = float(0.0 if (r is None or not np.isfinite(r)) else r)
        total += rr
        cum_max = max(cum_max, total)

        s = np.asarray(s_next, dtype=np.float32).reshape(-1)
        info = info or {}

        # allocation_ratio
        alloc_ratio = info.get("allocation_ratio")
        if alloc_ratio is None or not np.isfinite(alloc_ratio):
            try:
                alloc_ratio = float(default_action_values[a])
            except Exception:
                alloc_ratio = float("nan")

        # equity fields
        after_equity = info.get("after_equity", np.nan)
        prev_equity  = info.get("prev_equity", np.nan)

        # position_notional fallback
        pos_notional = info.get("position_notional")
        if pos_notional is None or not np.isfinite(pos_notional):
            if np.isfinite(after_equity) and np.isfinite(alloc_ratio):
                pos_notional = float(after_equity * alloc_ratio)
            else:
                pos_notional = float("nan")

        # optional liquidity fallback
        liquidity = info.get("liquidity", None)
        if (liquidity is None or not np.isfinite(liquidity)) and _liq_mult and (decimal_0 is not None) and (decimal_1 is not None):
            try:
                price_in = float(info.get("price_in", np.nan))
                if np.isfinite(price_in) and np.isfinite(pos_notional):
                    tick = int(round(math.log(price_in * (10 ** (decimal_1 - decimal_0)), 1.0001)))
                    mult = float(_liq_mult(tick, 500))
                    liquidity = float(mult * pos_notional * (10 ** decimal_1))
                else:
                    liquidity = float("nan")
            except Exception:
                liquidity = float("nan")

        # expected gas (first step free; only when action changes)
        expected_gas = 0.0 if (last_action is None or a == last_action) else float(getattr(env, "gas_cost", 0.0))
        last_action = a

        debug_note = None
        if not printed_debug_once:
            printed_debug_once = True
            try:
                EnvSig = str(inspect.signature(_get_env_class()[0].__init__))
            except Exception:
                EnvSig = "n/a"
            debug_note = f"env_sig={EnvSig}; keys_in_info={list(info.keys())}; ckpt_state_dim={req_state_dim}"

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
            "gas_applied": info.get("gas_applied", np.nan),
            "expected_gas": expected_gas,
            "step_reward": rr,
            "cum_reward": total,
            "cum_reward_max": cum_max,
            "prev_equity": prev_equity,
            "after_equity": after_equity,
            "return": info.get("return", np.nan),
            "total_reward_env": info.get("total_reward", np.nan),
            "terminated_reason": info.get("terminated_reason", None),
            "debug_note": debug_note,
        })

    df = pd.DataFrame(records)
    out_path = Path(out_dir) / csv_name
    df.to_csv(out_path, index=False)
    print(f"✅ DQN Rollout saved: {out_path}\nTotal return (sum rewards): {total:.6f}\nCheckpoint: {ckpt}")
    return df, total, str(out_path)

def best_dqn_rollout(
    data: np.ndarray,
    time_data: np.ndarray,
    ENV_KW: dict,
    *,
    result_root: str = "./result",
    experiment_dir: str | Path,   # REQUIRED
    action_dim: int = 5,
    csv_name: str = "dqn_result.csv",
):
    """
    Find best DQN trial (final_equity → cum_return_max), build cfg, run ONE rollout, save CSV.
    This will adapt obs to the checkpoint's expected input dim (clip/pad).
    """
    Env, _, _ = _get_env_class()

    # Probe real dims from the current env (e.g., feature_dim + 1 if you appended equity)
    probe_kw = {"numeric_data": data, "time_data": time_data, **ENV_KW,
                "max_steps": min(int(ENV_KW.get("max_steps", data.shape[0]-1)), int(data.shape[0]-1))}
    state_dim, action_dim_probe = _probe_env_dims(probe_kw)
    action_dim_used = int(action_dim_probe if action_dim_probe and action_dim_probe > 0 else action_dim)

    # Pick best trial
    exp_dir = _find_experiment_dir(experiment_dir)
    best_trial_dir, best_score, best_metric = _pick_best_trial_generic(exp_dir)
    trial_tag = best_trial_dir.name

    # Read params & build cfg (we'll overwrite cfg.state_dim later if checkpoint says otherwise)
    params = _read_trial_params(best_trial_dir)
    net_dims = tuple(params["net_dims"])
    cfg = _build_cfg_for_inference_dqn(best_trial_dir, Env, int(state_dim), int(action_dim_used), net_dims)

    # Output dir
    out_dir = Path(result_root).resolve() / trial_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rollout once
    df, total_return, csv_path = rollout_dqn(data, time_data, ENV_KW, cfg, csv_name=csv_name, out_dir=str(out_dir))

    # Summary
    print("🏆 DQN Best trial:", best_trial_dir)
    print(f"   best {best_metric}:", best_score)
    print("   net_dims:", net_dims)
    print("📂 Output dir:", out_dir)
    print(f"📈 Total return (sum rewards): {total_return:.6f}")
    print(f"📝 File:", csv_path)

    return {
        "trial_dir": str(best_trial_dir),
        "trial_tag": trial_tag,
        "best_metric": best_metric,
        "best_metric_value": float(best_score),
        "net_dims": net_dims,
        "out_dir": str(out_dir),
        "result": {"csv": str(csv_path), "total_return": float(total_return), "dataframe": df},
        "cfg": cfg,
    }


# =========================
# PPO inference
# =========================

def _build_cfg_for_inference_ppo(best_trial_dir: Path, env_class, state_dim: int, action_dim: int, net_dims: tuple):
    """
    Build a minimal ElegantRL Config for PPO inference (discrete).
    """
    from elegantrl.train.config import Config
    try:
        from elegantrl.agents.AgentPPO import AgentDiscretePPO as PPOAgent  # noqa
    except Exception:
        from elegantrl.agents.AgentPPO import AgentPPO as PPOAgent          # noqa

    erl_ppo_dir = best_trial_dir / "erl_ppo"
    erl_dir     = best_trial_dir / "erl"
    cwd_dir = erl_ppo_dir if erl_ppo_dir.exists() else (erl_dir if erl_dir.exists() else best_trial_dir)

    cfg = Config(agent_class=None, env_class=env_class, env_args=None)
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

def _infer_required_state_dim_from_sd_generic(sd: dict) -> int | None:
    """
    Infer input dim from arbitrary PPO actor state_dict (best-effort).
    Try keys like 'net.0.weight', else any 2D weight's in_features.
    """
    try:
        if "net.0.weight" in sd and torch.is_tensor(sd["net.0.weight"]) and sd["net.0.weight"].ndim == 2:
            return int(sd["net.0.weight"].shape[1])
    except Exception:
        pass
    try:
        for k, v in sd.items():
            if torch.is_tensor(v) and v.ndim == 2:
                return int(v.shape[1])
    except Exception:
        pass
    return None

def load_ppo_actor(cfg, checkpoint_path: str | None = None):
    """
    Robust PPO actor loader (prefers actor_latest/actor_best) AND infer required input dim.
    Returns (agent, ckpt_path, required_state_dim).
    """
    try:
        from elegantrl.agents.AgentPPO import AgentDiscretePPO as PPOAgent
    except Exception:
        from elegantrl.agents.AgentPPO import AgentPPO as PPOAgent

    device = torch.device('cpu' if getattr(cfg, "gpu_id", -1) < 0 else 'cuda')

    def _find_ckpt(root):
        for name in ("actor_latest.pth", "actor_best.pth", "actor.pth", "actor.pt"):
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
        cands = sorted(glob.glob(os.path.join(root, "actor__*.pt")))
        if cands:
            return cands[-1]
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d):
                p = _find_ckpt(d)
                if p:
                    return p
        return None

    ckpt = checkpoint_path or _find_ckpt(cfg.cwd)
    if not ckpt:
        raise FileNotFoundError(f"No PPO actor checkpoint found under {cfg.cwd}")

    obj = torch.load(ckpt, map_location=device)

    req_dim = None
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        if isinstance(sd, dict):
            req_dim = _infer_required_state_dim_from_sd_generic(sd)
    if req_dim is None:
        req_dim = int(getattr(cfg, "state_dim", 0)) or 1

    # Build agent with required dim
    cfg.state_dim = int(req_dim)
    agent  = PPOAgent(cfg.net_dims, cfg.state_dim, cfg.action_dim, gpu_id=getattr(cfg, "gpu_id", -1), args=cfg)
    agent.device = device

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
            agent.act.load_state_dict(obj); agent.act.eval()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    return agent, ckpt, int(req_dim)

def _to_discrete_idx_from_actor(actor, st, action_dim: int) -> int:
    """
    Turn actor output into a valid discrete action index [0..action_dim-1].
    Supports logits tensor or Agent actor outputs.
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
            idx = int(np.argmax(a))
        if idx < 0: idx = 0
        if idx >= action_dim: idx = action_dim - 1
        return idx

def rollout_ppo(
    data: np.ndarray,
    time_data: np.ndarray,
    env_kwargs_base: dict,
    cfg,
    csv_name: str = "ppo_result.csv",
    out_dir: str = "./result",
):
    """
    Run greedy actions (argmax over logits) using the loaded PPO actor on a single dataset.
    Records fields aligned with env.info.
    If checkpoint expects a different state_dim, observations are clipped/padded to match.
    """
    Env, _, _ = _get_env_class()

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

    agent, ckpt, req_state_dim = load_ppo_actor(cfg)
    try:
        shutil.copy2(ckpt, out_dir / os.path.basename(ckpt))
    except Exception:
        pass

    s = env.reset()
    if isinstance(s, tuple):
        s = s[0]
    s = np.asarray(s, dtype=np.float32).reshape(-1)

    # Default action values (fallback for allocation_ratio)
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
        s_proj = _project_obs(s, req_state_dim)
        st = torch.as_tensor(s_proj, dtype=torch.float32, device=getattr(agent, "device", torch.device("cpu"))).view(1, -1)
        idx = _to_discrete_idx_from_actor(agent.act, st, int(getattr(cfg, "action_dim", 5)))

        out = env.step(idx)
        if isinstance(out, tuple) and len(out) == 5:
            s_next, r, term, trunc, info = out
            done = bool(term or trunc)
        else:
            s_next, r, done, info = out

        rr = float(0.0 if (r is None or not np.isfinite(r)) else r)
        total += rr
        cum_max = max(cum_max, total)

        s = np.asarray(s_next, dtype=np.float32).reshape(-1)
        info = info or {}

        # allocation_ratio fallback
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
            "gas_applied": info.get("gas_applied", np.nan),
            "step_reward": rr,
            "cum_reward": total,
            "cum_reward_max": cum_max,
            "prev_equity": info.get("prev_equity", np.nan),
            "after_equity": info.get("after_equity", np.nan),
            "return": info.get("return", np.nan),
            "total_reward_env": info.get("total_reward", np.nan),
            "terminated_reason": info.get("terminated_reason", None),
            "ckpt_state_dim": req_state_dim,
        })

    df = pd.DataFrame(records)
    out_path = Path(out_dir) / csv_name
    df.to_csv(out_path, index=False)
    print(f"✅ PPO Rollout saved: {out_path}\nTotal return (sum rewards): {total:.6f}\nCheckpoint: {ckpt}")
    return df, total, str(out_path)

def best_ppo_rollout(
    data: np.ndarray,
    time_data: np.ndarray,
    ENV_KW: dict,
    *,
    result_root: str = "./result",
    experiment_dir: str | Path,   # REQUIRED
    action_dim: int = 5,
    csv_name: str = "ppo_result.csv",
):
    """
    Find best PPO trial (final_equity → cum_return_max), build cfg, run ONE rollout, save CSV.
    This will adapt obs to the checkpoint's expected input dim (clip/pad).
    """
    Env, _, _ = _get_env_class()

    probe_kw = {"numeric_data": data, "time_data": time_data, **ENV_KW,
                "max_steps": min(int(ENV_KW.get("max_steps", data.shape[0]-1)), int(data.shape[0]-1))}
    state_dim, action_dim_probe = _probe_env_dims(probe_kw)
    action_dim_used = int(action_dim_probe if action_dim_probe and action_dim_probe > 0 else action_dim)

    exp_dir = _find_experiment_dir(experiment_dir)
    best_trial_dir, best_score, best_metric = _pick_best_trial_generic(exp_dir)
    trial_tag = best_trial_dir.name

    params = _read_trial_params(best_trial_dir)
    net_dims = tuple(params["net_dims"])
    cfg = _build_cfg_for_inference_ppo(best_trial_dir, Env, int(state_dim), int(action_dim_used), net_dims)

    out_dir = Path(result_root).resolve() / trial_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df, total_return, csv_path = rollout_ppo(data, time_data, ENV_KW, cfg, csv_name=csv_name, out_dir=str(out_dir))

    print("🏆 PPO Best trial:", best_trial_dir)
    print(f"   best {best_metric}:", best_score)
    print("   net_dims:", net_dims)
    print("📂 Output dir:", out_dir)
    print(f"📈 Total return (sum rewards): {total_return:.6f}")
    print(f"📝 File:", csv_path)

    return {
        "trial_dir": str(best_trial_dir),
        "trial_tag": trial_tag,
        "best_metric": best_metric,
        "best_metric_value": float(best_score),
        "net_dims": net_dims,
        "out_dir": str(out_dir),
        "result": {"csv": str(csv_path), "total_return": float(total_return), "dataframe": df},
        "cfg": cfg,
    }
