# ==== Best-trial loader + rollout (self-contained; saves to result/<trial_name>/) ====
# Requirements in your session: numpy, pandas, torch, and your utils.env.UniswapV3LiquidityEnv.

import os, json, glob, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#####################################################################################################################################
# DQN
#####################################################################################################################################

# --- Helper A) Locate experiment directory (Ray Tune results) ---
def _find_experiment_dir(explicit: str | Path | None = None) -> Path:
    """
    Return the path to the Ray Tune experiment directory containing trial folders.
    Tries (1) explicit, (2) common local paths.
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Experiment dir not found: {p}")

    candidates = [
        Path("./ray_results/dqn_univ3_search").resolve(),
        Path("./experiments/ray_results/dqn_univ3_search").resolve(),
        Path("/Users/seitahuang/Desktop/AMM_RL/experiments/ray_results/dqn_univ3_search").resolve(),  # adjust if needed
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Experiment directory not found. Set an explicit path or adjust candidates.")

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
    device = torch.device('cpu' if cfg.gpu_id < 0 else 'cuda')
    agent  = AgentDQN(cfg.net_dims, cfg.state_dim, cfg.action_dim, gpu_id=cfg.gpu_id, args=cfg)
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
    - Flattens observation to 1D for DQN.
    - Saves a CSV with step-level records to out_dir/csv_name.
    - Copies the used checkpoint into out_dir for auditing.
    """
    from utils.env import UniswapV3LiquidityEnv

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose environment kwargs
    env_kwargs = dict(env_kwargs_base)
    env_kwargs["numeric_data"] = data
    env_kwargs["time_data"] = time_data
    env_kwargs["max_steps"] = min(env_kwargs.get("max_steps", data.shape[0]-1), data.shape[0]-1)
    env = UniswapV3LiquidityEnv(**env_kwargs)

    # Load actor
    agent, ckpt = load_dqn_actor(cfg)

    # Copy checkpoint for reference
    try:
        shutil.copy2(ckpt, out_dir / os.path.basename(ckpt))
    except Exception:
        pass

    # Reset (Gymnasium-compatible)
    s = env.reset()
    if isinstance(s, tuple):  # (obs, info)
        s = s[0]
    s = np.asarray(s, dtype=np.float32)
    if s.ndim > 1:
        s = s.reshape(-1)

    records, done, total = [], False, 0.0
    while not done:
        st = torch.as_tensor(s, dtype=torch.float32, device=agent.device).view(1, -1)  # [1, state_dim]
        with torch.no_grad():
            q = agent.act(st)                # [1, action_dim]
            a = int(q.argmax(dim=1).item())  # greedy action

        out = env.step(a)
        if isinstance(out, tuple) and len(out) == 5:
            s_next, r, term, trunc, info = out
            done = bool(term or trunc)
        else:
            s_next, r, done, info = out

        total += float(r)

        s_next = np.asarray(s_next, dtype=np.float32)
        if s_next.ndim > 1:
            s_next = s_next.reshape(-1)
        s = s_next

        info = info or {}
        records.append({
            "t": info.get("t", len(records)),
            "time": info.get("time", np.nan),
            "action_idx": info.get("allocation_idx", a),
            "allocation_ratio": info.get("allocation_ratio", np.nan),
            "price_in": info.get("price_in", np.nan),
            "price_out": info.get("price_out", np.nan),
            "fee": info.get("fee", np.nan),
            "lvr": info.get("lvr", np.nan),
            "gas": info.get("gas_applied", np.nan),
            "step_reward": float(r),
            "cum_reward": total,
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
    experiment_dir: str | Path | None = None,
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
    from utils.env import UniswapV3LiquidityEnv  # needed for cfg construction

    # Infer state_dim from windowed observations (flatten to 1D)
    win = int(TRAIN_ENV_KW.get("window_size", 1))
    feat = int(getattr(train_data, "shape", (0, 0))[1])
    state_dim = win * feat if win > 1 else feat

    # Locate experiment directory and pick the best trial
    exp_dir = _find_experiment_dir(experiment_dir)
    best_trial_dir, best_score = _pick_best_trial(exp_dir)
    trial_tag = best_trial_dir.name

    # Read params (net_dims) and build cfg for inference
    params = _read_trial_params(best_trial_dir)
    net_dims = tuple(params["net_dims"])
    cfg = _build_cfg_for_inference(
        best_trial_dir=best_trial_dir,
        env_class=UniswapV3LiquidityEnv,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
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
        test_data, test_time_data, TEST_ENV_KW, cfg
        , csv_name="test_result.csv", out_dir=str(out_dir)
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
# Requirements in your session: numpy, pandas, torch, and your utils.env.UniswapV3LiquidityEnv.

# --- Helper A) Locate experiment directory (Ray Tune results) ---
def _find_experiment_dir_ppo(explicit: str | Path | None = None) -> Path:
    """
    Return the path to the Ray Tune PPO experiment directory containing trial folders.
    Tries (1) explicit, (2) common local paths.
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Experiment dir not found: {p}")

    candidates = [
        Path("./ray_results/ppo_univ3_search").resolve(),
        Path("./experiments/ray_results/ppo_univ3_search").resolve(),
        Path("/Users/seitahuang/Desktop/AMM_RL/experiments/ray_results/ppo_univ3_search").resolve(),  # adjust if needed
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("PPO experiment directory not found. Set an explicit path or adjust candidates.")

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
    - Flattens observation to 1D if needed.
    - Saves a CSV with step-level records to out_dir/csv_name.
    - Copies the used checkpoint into out_dir for auditing.
    """
    from utils.env import UniswapV3LiquidityEnv

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose environment kwargs
    env_kwargs = dict(env_kwargs_base)
    env_kwargs["numeric_data"] = data
    env_kwargs["time_data"] = time_data
    env_kwargs["max_steps"] = min(env_kwargs.get("max_steps", data.shape[0]-1), data.shape[0]-1)
    env = UniswapV3LiquidityEnv(**env_kwargs)

    # Load actor
    agent, ckpt = load_ppo_actor(cfg)

    # Copy checkpoint for reference
    try:
        shutil.copy2(ckpt, out_dir / os.path.basename(ckpt))
    except Exception:
        pass

    # Reset (Gymnasium-compatible)
    s = env.reset()
    if isinstance(s, tuple):  # (obs, info)
        s = s[0]
    s = np.asarray(s, dtype=np.float32)
    if s.ndim > 1:
        s = s.reshape(-1)

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

        # sanitize
        rr = float(0.0 if (r is None or not np.isfinite(r)) else r)
        total += rr
        if total > cum_max: cum_max = total

        s_next = np.asarray(s_next, dtype=np.float32)
        if s_next.ndim > 1:
            s_next = s_next.reshape(-1)
        s = s_next

        info = info or {}
        records.append({
            "t": info.get("t", len(records)),
            "time": info.get("time", np.nan),
            "action_idx": info.get("allocation_idx", idx),
            "allocation_ratio": info.get("allocation_ratio", np.nan),
            "price_in": info.get("price_in", np.nan),
            "price_out": info.get("price_out", np.nan),
            "fee": info.get("fee", np.nan),
            "lvr": info.get("lvr", np.nan),
            "gas": info.get("gas_applied", np.nan),
            "step_reward": rr,
            "cum_reward": total,
            "cum_reward_max": cum_max,
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
    experiment_dir: str | Path | None = None,
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
    from utils.env import UniswapV3LiquidityEnv  # needed for cfg construction

    # Infer state_dim from windowed observations (flatten to 1D)
    win = int(TRAIN_ENV_KW.get("window_size", 1))
    feat = int(getattr(train_data, "shape", (0, 0))[1])
    state_dim = win * feat if win > 1 else feat

    # Locate experiment directory and pick the best trial
    exp_dir = _find_experiment_dir_ppo(experiment_dir)
    best_trial_dir, best_score = _pick_best_trial_ppo(exp_dir)
    trial_tag = best_trial_dir.name

    # Read params (net_dims) and build cfg for inference
    params = _read_trial_params_ppo(best_trial_dir)
    net_dims = tuple(params["net_dims"])
    cfg = _build_cfg_for_inference_ppo(
        best_trial_dir=best_trial_dir,
        env_class=UniswapV3LiquidityEnv,
        state_dim=int(state_dim),
        action_dim=int(action_dim),
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
