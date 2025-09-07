#####################################################################################################################################
# DQN
#####################################################################################################################################


def tune_dqn_with_ray(
    train_data,
    train_time_data,
    TRAIN_ENV_KW: dict,
    *,
    num_samples: int = 20,
    experiment_name: str = "dqn_univ3_search",
    storage_root: str = "./ray_results",
    param_space: dict | None = None,
):
    """
    Run Ray Tune for ElegantRL DQN with robust logging/checkpointing and return (results, best).

    - Requires: train_data, train_time_data, TRAIN_ENV_KW
    - Automatically sets up staging, initializes Ray, and runs ASHA search.
      Uses exp_r_ema as the intermediate metric and cum_return_max as the final metric.
    """

    # ---------------- Imports (local to function) ----------------
    from pathlib import Path
    import os, sys, shutil, glob, json
    import numpy as np
    import pandas as pd
    import torch, ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.air import session
    from elegantrl.agents.AgentDQN import AgentDQN
    from elegantrl.train.config import Config
    from elegantrl.train.run import train_agent

    # ---------------- A) Prepare staging (copy only small/needed files) ----------------
    ROOT = Path.cwd().resolve()
    tmp = ROOT
    while tmp != tmp.parent and not (tmp / "utils").exists():
        tmp = tmp.parent
    ROOT = tmp

    STAGING = ROOT / "ray_staging"
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    def _copy_tree_safe(src: Path, dst: Path, exts_exclude=(".parquet", ".pt", ".pth", ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg", ".ipynb")):
        """Shallow-copy src → dst, excluding common large file extensions."""
        if not src.exists():
            return
        if src.is_file():
            if not src.suffix.lower() in exts_exclude:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return
        for p in src.rglob("*"):
            if p.is_dir():
                continue
            if p.suffix.lower() in exts_exclude:
                continue
            rel = p.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, out)
            except Exception:
                pass

    _copy_tree_safe(ROOT / "utils", STAGING / "utils")
    (STAGING / "utils" / "__init__.py").touch(exist_ok=True)
    _copy_tree_safe(ROOT / "config", STAGING / "config")
    DATA_SRC = ROOT / "data"
    if DATA_SRC.exists():
        for p in DATA_SRC.rglob("*.csv"):
            try:
                if p.stat().st_size <= 10 * 1024 * 1024:
                    out = (STAGING / "data" / p.relative_to(DATA_SRC))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, out)
            except Exception:
                pass

    print("✅ staged working_dir:", STAGING)

    # ---------------- B) Ray init with staging ----------------
    ray.shutdown()
    ray.init(
        runtime_env={
            "working_dir": str(STAGING),
            "env_vars": {
                "PYTHONPATH": str(STAGING),
                "PROJECT_ROOT": str(STAGING),
            },
        },
        ignore_reinit_error=True,
    )
    print("✅ Ray initialized with working_dir =", STAGING)

    # ---------------- C) Dimensions ----------------
    WIN = int(TRAIN_ENV_KW.get("window_size", 1))
    FEAT = int(getattr(train_data, "shape", (0, 0))[1]) if hasattr(train_data, "shape") else 0
    STATE_DIM  = int(WIN * FEAT) if WIN > 1 else int(FEAT)
    ACTION_DIM = 5
    IF_DISCRETE = True
    T = int(min(TRAIN_ENV_KW["max_steps"], train_data.shape[0] - 1))
    print(f"[Tune] STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, IF_DISCRETE={IF_DISCRETE}, T={T}")

    # ---------------- D) Logger (report exp_r_ema + save actor_latest.pth) ----------------
    def install_tune_logger():
        import time, csv
        import numpy as _np
        import elegantrl.train.evaluator as erl_eval
        from ray.tune import report as tune_report
        import torch as _torch

        # Local states inside each worker
        LATEST_METRICS = {"step": 0, "exp_r": -1e30, "exp_r_ema": -1e30, "objC": float("nan"), "objA": float("nan")}
        FINAL_METRICS  = {"cum_return_max": -1e30, "val_return": -1e30}

        def _meanify(x):
            try:
                if x is None: return _np.nan
                try:
                    if isinstance(x, _torch.Tensor):
                        return float(x.detach().mean().cpu().item())
                except Exception:
                    pass
                if isinstance(x, (int, float)): return float(x)
                arr = _np.asarray(x, dtype=float)
                return float(_np.nanmean(arr)) if arr.size else _np.nan
            except Exception:
                return _np.nan

        def _csv_and_report(self, actor=None, steps=0, exp_r=None, logging_tuple=None):
            nonlocal LATEST_METRICS, FINAL_METRICS
            self.total_step = int(getattr(self, "total_step", 0)) + int(steps or 0)
            if not hasattr(self, "start_time"): self.start_time = time.time()
            elapsed_min = (time.time() - self.start_time) / 60.0

            expR = _meanify(exp_r)
            objC = objA = _np.nan
            if logging_tuple is not None:
                if isinstance(logging_tuple, (list, tuple)):
                    if len(logging_tuple) > 0: objC = _meanify(logging_tuple[0])
                    if len(logging_tuple) > 1: objA = _meanify(logging_tuple[1])
                else:
                    objC = _meanify(logging_tuple)

            s = getattr(self, "_ema_state", {})
            ema = s.get("ema_expR", expR if not _np.isnan(expR) else 0.0)
            if not _np.isnan(expR): ema = 0.9 * ema + 0.1 * expR
            s["ema_expR"] = ema; self._ema_state = s

            LATEST_METRICS = {
                "step": int(self.total_step),
                "exp_r": float(expR if _np.isfinite(expR) else -1e30),
                "exp_r_ema": float(ema if _np.isfinite(ema) else -1e30),
                "objC": float(objC) if _np.isfinite(objC) else float("nan"),
                "objA": float(objA) if _np.isfinite(objA) else float("nan"),
            }

            # Always overwrite latest weights to ensure a usable checkpoint
            try:
                if actor is not None:
                    os.makedirs(self.cwd, exist_ok=True)
                    ckpt_latest = os.path.join(self.cwd, "actor_latest.pth")
                    try:
                        _torch.save(actor.state_dict(), ckpt_latest)
                    except Exception:
                        _torch.save(actor, ckpt_latest)
            except Exception:
                pass

            # Append CSV log
            try:
                os.makedirs(self.cwd, exist_ok=True)
                path = os.path.join(self.cwd, "training_metrics.csv")
                write_header = not os.path.exists(path)
                with open(path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["step", "expR", "expR_ema", "objC", "objA", "elapsed_min"])
                    w.writerow([self.total_step, expR, float(ema), objC, objA, round(elapsed_min, 3)])
            except Exception:
                pass

            # Always report the required metrics for ASHA and final selection
            tune_report({
                "step": LATEST_METRICS["step"],
                "exp_r": LATEST_METRICS["exp_r"],
                "exp_r_ema": LATEST_METRICS["exp_r_ema"],
                "objC": LATEST_METRICS["objC"],
                "objA": LATEST_METRICS["objA"],
                "cum_return_max": float(FINAL_METRICS["cum_return_max"]),
                "val_return": float(FINAL_METRICS["val_return"]),
            })

        erl_eval.Evaluator.evaluate_and_save = _csv_and_report
        erl_eval.Evaluator.save_training_curve_jpg = lambda self: None

        # Provide a getter so the trainable can read the latest and final metrics
        def _get_latest_and_final():
            return LATEST_METRICS, FINAL_METRICS
        return _get_latest_and_final

    # ---------------- E) Greedy evaluator (cum_return_max) ----------------
    def _greedy_eval_cummax(actor, env_kwargs):
        pr = os.environ.get("PROJECT_ROOT")
        if pr:
            try: os.chdir(pr)
            except Exception: pass
        from utils.env import UniswapV3LiquidityEnv

        env = UniswapV3LiquidityEnv(**env_kwargs)
        s = env.reset();  s = s[0] if isinstance(s, tuple) else s
        device = next(actor.parameters()).device
        s = np.asarray(s, dtype=np.float32)
        if s.ndim > 1:
            s = s.reshape(-1)

        cum, cum_max = 0.0, -float("inf"); done = False
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = actor(st); a = int(q.argmax(dim=1).item())
            out = env.step(a)
            if isinstance(out, tuple) and len(out) == 5:
                s, r, term, trunc, _ = out; done = bool(term or trunc)
            else:
                s, r, done, _ = out
            s = np.asarray(s, dtype=np.float32)
            if s.ndim > 1: s = s.reshape(-1)
            cum += float(r)
            if cum > cum_max: cum_max = cum
        return float(cum), float(cum_max)

    # ---------------- F) Trainable (Ray worker entry) ----------------
    def make_trainable():
        def trainable_ray(hp: dict):
            from ray.tune import report as tune_report
            # Initial warmup report (ensures ASHA sees required metrics immediately)
            tune_report({"step": 0, "exp_r": -1e30, "exp_r_ema": -1e30,
                         "objC": float("nan"), "objA": float("nan"),
                         "cum_return_max": -1e30, "val_return": -1e30})

            pr = os.environ.get("PROJECT_ROOT")
            if pr and pr not in sys.path:
                sys.path.insert(0, pr)
            if pr:
                try: os.chdir(pr)
                except Exception: pass

            get_latest_and_final = install_tune_logger()
            from utils.env import UniswapV3LiquidityEnv

            env_args = {
                "env_name": "UniswapV3LiquidityEnv",
                "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
                "if_discrete": IF_DISCRETE, "max_step": T, "num_envs": 1,
                "numeric_data": train_data, "time_data": train_time_data,
                **{**TRAIN_ENV_KW, "max_steps": T},
            }

            cfg = Config(agent_class=AgentDQN, env_class=UniswapV3LiquidityEnv, env_args=env_args)
            try:
                trial_dir = session.get_trial_dir()
                cfg.cwd = os.path.join(trial_dir, "erl")
            except Exception:
                trial_id = os.getenv("RAY_TRIAL_ID", "trial")
                cfg.cwd = os.path.join("./ray_runs_dqn", trial_id)

            # Do not auto-remove outputs; keep checkpoints and logs
            cfg.if_remove = False
            try: cfg.if_keep_save = True
            except Exception: pass

            cfg.random_seed = int(hp.get("seed", 0))
            cfg.env_num = 1
            cfg.gpu_id = 0 if torch.cuda.is_available() else -1

            cfg.net_dims        = tuple(hp["net_dims"])
            cfg.batch_size      = int(hp["batch_size"])
            cfg.learning_rate   = float(hp["lr"])
            cfg.gamma           = float(hp["gamma"])
            cfg.soft_update_tau = float(hp["tau"])
            cfg.horizon_len     = int(hp["horizon_len"])
            passes              = int(hp.get("passes", 1))
            cfg.break_step      = T * passes * cfg.env_num
            cfg.eval_times      = 0
            cfg.eval_per_step   = 10**12
            cfg.explore_rate    = float(hp["eps"])
            cfg.buffer_size     = int(hp["buffer"])
            cfg.repeat_times    = int(hp["repeat"])
            cfg.if_use_per      = bool(hp["use_per"])

            try:
                train_agent(cfg)

                # Locate checkpoint
                ckpt = None
                for name in ("actor_latest.pth", "actor.pth", "actor.pt"):
                    p = os.path.join(cfg.cwd, name)
                    if os.path.isfile(p): ckpt = p; break
                if ckpt is None:
                    cands = sorted(glob.glob(os.path.join(cfg.cwd, "actor__*.pt")))
                    ckpt = cands[-1] if cands else None
                if ckpt is None:
                    L, F = get_latest_and_final()
                    tune_report({**L, **{"cum_return_max": -1e30, "val_return": -1e30}})
                    return

                # Load and perform final greedy evaluation
                device = torch.device('cpu' if cfg.gpu_id < 0 else 'cuda')
                agent  = AgentDQN(cfg.net_dims, cfg.state_dim, cfg.action_dim, gpu_id=cfg.gpu_id, args=cfg)
                obj    = torch.load(ckpt, map_location=device)
                try:
                    agent.act.load_state_dict(obj)
                except Exception:
                    try:
                        agent.act.load_state_dict(obj["state_dict"])
                    except Exception:
                        agent.act = obj.to(device)
                actor = agent.act.eval()

                env_eval_kw = {"numeric_data": train_data, "time_data": train_time_data, **{**TRAIN_ENV_KW, "max_steps": T}}
                total_return, cum_return_max = _greedy_eval_cummax(actor, env_eval_kw)
                if not np.isfinite(cum_return_max): cum_return_max = -1e30
                if not np.isfinite(total_return):  total_return  = -1e30

                L, F = get_latest_and_final()
                F.update({"cum_return_max": float(cum_return_max), "val_return": float(total_return)})

                tune_report({
                    "step": L["step"],
                    "exp_r": L["exp_r"],
                    "exp_r_ema": L["exp_r_ema"],
                    "objC": L["objC"],
                    "objA": L["objA"],
                    "cum_return_max": F["cum_return_max"],
                    "val_return": F["val_return"],
                    "ckpt": ckpt
                })
                return
            except Exception as e:
                L, _ = get_latest_and_final()
                tune_report({**L, **{"cum_return_max": -1e30, "val_return": -1e30, "error": str(e)[:200]}})
                return
        return trainable_ray

    trainable_ray = make_trainable()

    # ---------------- G) Param space & ASHA ----------------
    if param_space is None:
        param_space = {
            "net_dims":   tune.choice([(32,32), (64,64), (128,128)]),
            "lr":         tune.loguniform(1e-5, 5e-4),
            "batch_size": tune.choice([32, 64, 128]),
            "gamma":      tune.uniform(0.95, 0.999),
            "tau":        tune.loguniform(1e-4, 5e-3),
            "horizon_len":tune.choice([512, 1024, 2048]),
            "eps":        tune.uniform(0.05, 0.3),
            "buffer":     tune.qlograndint(int(5e4), int(5e5), int(1e4)),
            "repeat":     tune.qrandint(1, 4, 1),
            "use_per":    tune.choice([False, True]),
            "passes":     tune.choice([1, 2]),
            "seed":       tune.randint(0, 1_000_000),
        }

    storage_root = Path(storage_root).expanduser().resolve()
    try:
        rc = tune.RunConfig(name=experiment_name, storage_path=storage_root.as_uri(), verbose=1)
    except TypeError:
        rc = tune.RunConfig(name=experiment_name, local_dir=str(storage_root), verbose=1)

    scheduler = ASHAScheduler(metric="exp_r_ema", mode="max", grace_period=2, reduction_factor=3)

    tuner = tune.Tuner(
        trainable_ray,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, scheduler=scheduler),
        run_config=rc,
    )

    results = tuner.fit()

    # --- Obtain best result with a robust fallback ---
    try:
        best = results.get_best_result(metric="cum_return_max", mode="max", filter_nan_and_inf=False)
        m = (best.metrics or {})
        print("Best cum_return_max:", m.get("cum_return_max"))
        print("Best (total) val_return:", m.get("val_return"))
        print("Best config:", best.config)
        print("Best logdir:", best.path)
    except Exception:
        df = results.get_dataframe()
        cand_cols = [c for c in df.columns if c.split("/")[-1] == "cum_return_max"]
        if cand_cols:
            col = cand_cols[0]
            dff = df[np.isfinite(df[col])]
            if len(dff):
                row = dff.loc[dff[col].idxmax()]
                print("Best cum_return_max:", row[col])
                best = type(
                    "BestLike",
                    (),
                    {
                        "metrics": {"cum_return_max": row[col]},
                        "config": {k.split("/",1)[1]: row[k] for k in df.columns if k.startswith("config/")},
                        "path": None
                    }
                )
        else:
            print("No `cum_return_max` found in results; did any trial reach final evaluation?")
            best = None

    return results, best

#####################################################################################################################################
# PPO
#####################################################################################################################################
def tune_ppo_with_ray(
    train_data,
    train_time_data,
    TRAIN_ENV_KW: dict,
    *,
    num_samples: int = 20,
    experiment_name: str = "ppo_univ3_search",
    storage_root: str = "./ray_results",
    param_space: dict | None = None,
):
    """
    Ray Tune for ElegantRL PPO (discrete). Robust setup:
    - FIFO (no early stopping)
    - eval_per_step=1 to force an initial checkpoint
    - Inline (logger) quick greedy eval so cum_return_max is reported mid-training
    - Final full greedy eval + emergency checkpoint if none exists
    - NaN/Inf safety everywhere
    - Uses AgentDiscretePPO (direct import)
    - Preflight environment: auto-fix STATE_DIM/ACTION_DIM and run a sanity check
    - Dynamically clamp batch_size to guarantee PPO update_times ≥ 1
    """

    # ---------------- Imports ----------------
    from pathlib import Path
    import os, sys, shutil, glob, math, traceback
    import numpy as np
    import pandas as pd
    import torch, ray
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler
    from ray.air import session

    # ---- Use discrete PPO agent directly (no fallback) ----
    from elegantrl.agents.AgentPPO import AgentDiscretePPO as PPOAgent

    # Optionally force discrete network classes if available
    try:
        from elegantrl.agents.net import ActorDiscretePPO, CriticPPO
        _FORCE_DISCRETE_NETS = True
    except Exception:
        _FORCE_DISCRETE_NETS = False

    from elegantrl.train.config import Config
    from elegantrl.train.run import train_agent

    # --- Bridge Gym → Gymnasium (recommended to also change your env file to import gymnasium as gym) ---
    try:
        import gymnasium as _gym
        sys.modules.setdefault("gym", _gym)
    except Exception:
        pass

    # ---------------- A) Staging (copy only small/needed files) ----------------
    ROOT = Path.cwd().resolve()
    tmp = ROOT
    while tmp != tmp.parent and not (tmp / "utils").exists():
        tmp = tmp.parent
    ROOT = tmp

    STAGING = ROOT / "ray_staging"
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    def _copy_tree_safe(src: Path, dst: Path, exts_exclude=(".parquet", ".pt", ".pth", ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg", ".ipynb")):
        if not src.exists():
            return
        if src.is_file():
            if not src.suffix.lower() in exts_exclude:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return
        for p in src.rglob("*"):
            if p.is_dir():
                continue
            if p.suffix.lower() in exts_exclude:
                continue
            rel = p.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, out)
            except Exception:
                pass

    _copy_tree_safe(ROOT / "utils", STAGING / "utils")
    (STAGING / "utils" / "__init__.py").touch(exist_ok=True)
    _copy_tree_safe(ROOT / "config", STAGING / "config")
    DATA_SRC = ROOT / "data"
    if DATA_SRC.exists():
        for p in DATA_SRC.rglob("*.csv"):
            try:
                if p.stat().st_size <= 10 * 1024 * 1024:
                    out = (STAGING / "data" / p.relative_to(DATA_SRC))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, out)
            except Exception:
                pass

    print("✅ Staged working_dir:", STAGING)

    # ---------------- B) Ray init ----------------
    ray.shutdown()
    ray.init(
        runtime_env={
            "working_dir": str(STAGING),
            "env_vars": {
                "PYTHONPATH": str(STAGING),
                "PROJECT_ROOT": str(STAGING),
            },
        },
        ignore_reinit_error=True,
    )
    print("✅ Ray initialized with working_dir =", STAGING)

    # ---------------- C) Dimensions (initial guess) ----------------
    WIN = int(TRAIN_ENV_KW.get("window_size", 1))
    FEAT = int(getattr(train_data, "shape", (0, 0))[1]) if hasattr(train_data, "shape") else 0
    STATE_DIM  = int(WIN * FEAT) if WIN > 1 else int(FEAT)
    ACTION_DIM = int(TRAIN_ENV_KW.get("action_dim", 5))  # can be overridden by TRAIN_ENV_KW
    IF_DISCRETE = True
    T = int(min(int(TRAIN_ENV_KW["max_steps"]), int(train_data.shape[0] - 1)))
    print(f"[Tune] (guess) STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, IF_DISCRETE={IF_DISCRETE}, T={T}")

    ENV_EVAL_KW = {"numeric_data": train_data, "time_data": train_time_data, **{**TRAIN_ENV_KW, "max_steps": T}}

    # ---------------- Helpers ----------------
    def _sanitize_arr(x):
        x = np.asarray(x, dtype=np.float32)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def _to_discrete_idx_from_actor(actor, st, action_dim: int):
        """Convert actor output to a discrete action index in [0, action_dim-1]."""
        import numpy as _np, torch as _torch
        with _torch.no_grad():
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
            if isinstance(a, _torch.Tensor):
                a = a.detach().cpu().numpy()
            a = _np.asarray(a).squeeze()
            if a.ndim == 0:
                try:
                    idx = int(a.item())
                except Exception:
                    idx = int(a)
            else:
                idx = int(_np.argmax(a))  # logits/continuous vector → argmax
            if idx < 0: idx = 0
            if idx >= action_dim: idx = action_dim - 1
            return idx

    # ---------------- Preflight: probe env for true dims & sanity ----------------
    def _preflight_env(env_kw, action_dim_guess):
        """Return (obs_dim, action_dim_real, ok_reward, msg)."""
        msg = []
        try:
            pr = os.environ.get("PROJECT_ROOT")
            if pr:
                try: os.chdir(pr)
                except Exception: pass
            from utils.env import UniswapV3LiquidityEnv
            env = UniswapV3LiquidityEnv(**env_kw)
            obs0 = env.reset()
            obs0 = obs0[0] if isinstance(obs0, tuple) else obs0
            obs0 = _sanitize_arr(obs0)
            if obs0.ndim > 1:
                obs_dim = int(np.prod(obs0.shape))
            else:
                obs_dim = int(obs0.shape[0])
            act_dim = int(action_dim_guess)
            if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
                act_dim = int(getattr(env.action_space, "n"))
                msg.append(f"action_space.n detected = {act_dim}")
            ok_reward = True
            steps = min(16, int(env_kw.get("max_steps", 128)))
            s = obs0
            for _ in range(max(4, steps // 4)):
                a = int(np.random.randint(0, max(1, act_dim)))
                out = env.step(a)
                if isinstance(out, tuple) and len(out) == 5:
                    s, r, term, trunc, _ = out
                else:
                    s, r, done, _ = out
                s = _sanitize_arr(s)
                if not np.isfinite(r):
                    ok_reward = False
            return obs_dim, act_dim, ok_reward, "; ".join(msg)
        except Exception as e:
            return None, None, False, f"preflight error: {type(e).__name__}: {str(e)[:200]}"

    true_obs_dim, true_act_dim, ok_reward, pf_msg = _preflight_env(ENV_EVAL_KW, ACTION_DIM)
    if pf_msg:
        print("[Preflight]", pf_msg)
    if isinstance(true_obs_dim, int) and true_obs_dim > 0 and true_obs_dim != STATE_DIM:
        print(f"[Preflight] override STATE_DIM: {STATE_DIM} -> {true_obs_dim}")
        STATE_DIM = true_obs_dim
    if isinstance(true_act_dim, int) and true_act_dim >= 2 and true_act_dim != ACTION_DIM:
        print(f"[Preflight] override ACTION_DIM: {ACTION_DIM} -> {true_act_dim}")
        ACTION_DIM = true_act_dim
    if not ok_reward:
        print("[Preflight] WARNING: reward has non-finite values in random probe; will sanitize during eval.")

    ENV_EVAL_KW = {**ENV_EVAL_KW}  # copy

    # ---------------- quick / full eval ----------------
    def _quick_eval(actor):
        """Short inline greedy eval for logger (≤ max_steps/4, at least 256 steps); NaN/Inf-safe."""
        if actor is None:
            return None
        try:
            pr = os.environ.get("PROJECT_ROOT")
            if pr:
                try: os.chdir(pr)
                except Exception: pass
            from utils.env import UniswapV3LiquidityEnv

            env = UniswapV3LiquidityEnv(**ENV_EVAL_KW)
            s = env.reset(); s = s[0] if isinstance(s, tuple) else s
            s = _sanitize_arr(s)
            if s.ndim > 1: s = s.reshape(-1)
            device = next(actor.parameters()).device

            cum, cum_max, done = 0.0, 0.0, False
            max_probe = int(min(max(256, ENV_EVAL_KW.get("max_steps", 1024)//4), ENV_EVAL_KW.get("max_steps", 1024)))
            steps = 0
            while not done and steps < max_probe:
                steps += 1
                st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                idx = _to_discrete_idx_from_actor(actor, st, ACTION_DIM)
                out = env.step(idx)
                if isinstance(out, tuple) and len(out) == 5:
                    s, r, term, trunc, _ = out; done = bool(term or trunc)
                else:
                    s, r, done, _ = out
                s = _sanitize_arr(s)
                if s.ndim > 1: s = s.reshape(-1)
                r = float(0.0 if (r is None or not np.isfinite(r)) else r)
                cum += r;  cum_max = max(cum_max, cum)
            return float(cum), float(cum_max)
        except Exception:
            return None

    def _full_eval(actor):
        """Final full greedy eval; NaN/Inf-safe."""
        pr = os.environ.get("PROJECT_ROOT")
        if pr:
            try: os.chdir(pr)
            except Exception: pass
        from utils.env import UniswapV3LiquidityEnv

        env = UniswapV3LiquidityEnv(**ENV_EVAL_KW)
        s = env.reset(); s = s[0] if isinstance(s, tuple) else s
        s = _sanitize_arr(s)
        if s.ndim > 1: s = s.reshape(-1)
        device = next(actor.parameters()).device

        cum, cum_max, done = 0.0, 0.0, False
        safety_steps = int(ENV_EVAL_KW.get("max_steps", 10_000)) + 5
        steps = 0
        while not done and steps < safety_steps:
            steps += 1
            st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            idx = _to_discrete_idx_from_actor(actor, st, ACTION_DIM)
            out = env.step(idx)
            if isinstance(out, tuple) and len(out) == 5:
                s, r, term, trunc, _ = out; done = bool(term or trunc)
            else:
                s, r, done, _ = out
            s = _sanitize_arr(s)
            if s.ndim > 1: s = s.reshape(-1)
            r = float(0.0 if (r is None or not np.isfinite(r)) else r)
            cum += r;  cum_max = max(cum_max, cum)
        return float(cum), float(cum_max)

    # ---------------- D) Logger (override Evaluator methods + inline eval) ----------------
    def install_tune_logger():
        import time, csv
        import numpy as _np
        import elegantrl.train.evaluator as erl_eval
        from ray.tune import report as tune_report
        import torch as _torch

        LATEST_METRICS = {"step": 0, "exp_r": -1e30, "exp_r_ema": -1e30, "objC": float("nan"), "objA": float("nan")}
        FINAL_METRICS  = {"cum_return_max": -1e30, "val_return": -1e30}

        def _meanify(x):
            try:
                if x is None: return _np.nan
                if isinstance(x, _torch.Tensor): return float(x.detach().mean().cpu().item())
                if isinstance(x, (int, float)): return float(x)
                arr = _np.asarray(x, dtype=float)
                return float(_np.nanmean(arr)) if arr.size else _np.nan
            except Exception:
                return _np.nan

        def _csv_and_report(self, actor=None, steps=0, exp_r=None, logging_tuple=None):
            nonlocal LATEST_METRICS, FINAL_METRICS
            self.total_step = int(getattr(self, "total_step", 0)) + int(steps or 0)
            if not hasattr(self, "start_time"): self.start_time = time.time()
            elapsed_min = (time.time() - self.start_time) / 60.0

            expR = _meanify(exp_r)
            objC = objA = _np.nan
            if logging_tuple is not None:
                if isinstance(logging_tuple, (list, tuple)):
                    if len(logging_tuple) > 0: objC = _meanify(logging_tuple[0])
                    if len(logging_tuple) > 1: objA = _meanify(logging_tuple[1])
                else:
                    objC = _meanify(logging_tuple)

            ema = getattr(self, "_ema_state", {}).get("ema_expR", expR if not _np.isnan(expR) else 0.0)
            if not _np.isnan(expR): ema = 0.9 * ema + 0.1 * expR
            self._ema_state = {"ema_expR": ema}

            LATEST_METRICS = {
                "step": int(self.total_step),
                "exp_r": float(expR if _np.isfinite(expR) else -1e30),
                "exp_r_ema": float(ema if _np.isfinite(ema) else -1e30),
                "objC": float(objC) if _np.isfinite(objC) else float("nan"),
                "objA": float(objA) if _np.isfinite(objA) else float("nan"),
            }

            # Always write the latest checkpoint
            try:
                if actor is not None:
                    os.makedirs(self.cwd, exist_ok=True)
                    ckpt_latest = os.path.join(self.cwd, "actor_latest.pth")
                    try:
                        _torch.save(actor.state_dict(), ckpt_latest)
                    except Exception:
                        _torch.save(actor, ckpt_latest)
            except Exception:
                pass

            # Inline quick eval → update final metrics (avoid -1e+30)
            try:
                if actor is not None:
                    probe = _quick_eval(actor)
                    if probe is not None:
                        total_ret, cum_max = probe
                        if not _np.isfinite(total_ret): total_ret = 0.0
                        if not _np.isfinite(cum_max):   cum_max   = 0.0
                        prev = FINAL_METRICS.get("cum_return_max", -1e30)
                        FINAL_METRICS["cum_return_max"] = float(max(prev, cum_max))
                        FINAL_METRICS["val_return"] = float(total_ret)
            except Exception:
                pass

            # CSV append
            try:
                os.makedirs(self.cwd, exist_ok=True)
                path = os.path.join(self.cwd, "training_metrics.csv")
                write_header = not os.path.exists(path)
                with open(path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["step", "expR", "expR_ema", "objC", "objA", "elapsed_min"])
                    w.writerow([self.total_step, expR, float(ema), objC, objA, round(elapsed_min, 3)])
            except Exception:
                pass

            tune_report({
                "step": LATEST_METRICS["step"],
                "exp_r": LATEST_METRICS["exp_r"],
                "exp_r_ema": LATEST_METRICS["exp_r_ema"],
                "objC": LATEST_METRICS["objC"],
                "objA": LATEST_METRICS["objA"],
                "cum_return_max": float(FINAL_METRICS["cum_return_max"]),
                "val_return": float(FINAL_METRICS["val_return"]),
            })

        # Override multiple potential evaluator method names (version differences)
        try:
            import elegantrl.train.evaluator as erl_eval_mod
            erl_eval_mod.Evaluator.evaluate_and_save = _csv_and_report
            if hasattr(erl_eval_mod.Evaluator, "evaluate_save_and_plot"):
                erl_eval_mod.Evaluator.evaluate_save_and_plot = _csv_and_report
            if hasattr(erl_eval_mod.Evaluator, "evaluate_and_save_mp"):
                erl_eval_mod.Evaluator.evaluate_and_save_mp = _csv_and_report
            erl_eval_mod.Evaluator.save_training_curve_jpg = lambda self: None
        except Exception:
            import elegantrl.train.evaluator as erl_eval
            erl_eval.Evaluator.evaluate_and_save = _csv_and_report
            try:
                if hasattr(erl_eval.Evaluator, "evaluate_save_and_plot"):
                    erl_eval.Evaluator.evaluate_save_and_plot = _csv_and_report
                if hasattr(erl_eval.Evaluator, "evaluate_and_save_mp"):
                    erl_eval.Evaluator.evaluate_and_save_mp = _csv_and_report
                erl_eval.Evaluator.save_training_curve_jpg = lambda self: None
            except Exception:
                pass

        def _get_latest_and_final():
            return LATEST_METRICS, FINAL_METRICS
        return _get_latest_and_final

    # ---------------- F) Trainable ----------------
    def make_trainable():
        def trainable_ray_ppo(hp: dict):
            from ray.tune import report as tune_report
            tune_report({"step": 0, "exp_r": -1e30, "exp_r_ema": -1e30,
                         "objC": float("nan"), "objA": float("nan"),
                         "cum_return_max": -1e+30, "val_return": -1e+30})

            pr = os.environ.get("PROJECT_ROOT")
            if pr and pr not in sys.path:
                sys.path.insert(0, pr)
            if pr:
                try: os.chdir(pr)
                except Exception: pass

            get_latest_and_final = install_tune_logger()
            from utils.env import UniswapV3LiquidityEnv

            env_args = {
                "env_name": "UniswapV3LiquidityEnv",
                "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
                "if_discrete": True, "max_step": T, "num_envs": 1,
                "numeric_data": train_data, "time_data": train_time_data,
                **{**TRAIN_ENV_KW, "max_steps": T},
            }

            cfg = Config(agent_class=PPOAgent, env_class=UniswapV3LiquidityEnv, env_args=env_args)
            try:
                trial_dir = session.get_trial_dir()
                cfg.cwd = os.path.join(trial_dir, "erl_ppo")
            except Exception:
                trial_id = os.getenv("RAY_TRIAL_ID", "trial_ppo")
                cfg.cwd = os.path.join("./ray_runs_ppo", trial_id)

            cfg.if_remove = False
            try: cfg.if_keep_save = True
            except Exception: pass

            cfg.random_seed = int(hp.get("seed", 0))
            cfg.env_num = 1
            cfg.gpu_id = 0 if torch.cuda.is_available() else -1

            cfg.net_dims        = tuple(hp["net_dims"])
            cfg.horizon_len     = int(hp["horizon_len"])
            cfg.batch_size      = int(hp["batch_size"])
            cfg.learning_rate   = float(hp["lr"])
            cfg.gamma           = float(hp["gamma"])
            passes              = int(hp.get("passes", 1))
            cfg.repeat_times    = max(1, int(hp["repeat"]))

            # --- update_times guard: ensure int(repeat_times * buf_len / batch_size) ≥ 1 ---
            buf_len   = int(cfg.horizon_len) * max(1, int(getattr(cfg, "env_num", 1)))
            max_batch = buf_len * max(1, int(cfg.repeat_times))
            desired   = min(int(cfg.batch_size), int(max_batch))
            aligned_dn = (desired // buf_len) * buf_len
            if aligned_dn < buf_len:
                aligned_dn = buf_len
            if aligned_dn != cfg.batch_size:
                print(f"[PPO] adjust batch_size {cfg.batch_size} -> {aligned_dn} "
                      f"(buf_len={buf_len}, repeat={cfg.repeat_times}, max_batch={max_batch})")
            cfg.batch_size = aligned_dn

            cfg.break_step      = max(int(T * passes * cfg.env_num), int(cfg.horizon_len), 1024)
            cfg.eval_times      = 1
            cfg.eval_per_step   = 1
            try:
                cfg.save_gap = max(200, cfg.horizon_len // 2)
            except Exception:
                pass

            # Force discrete setup
            try: cfg.if_discrete = True
            except Exception: pass
            if _FORCE_DISCRETE_NETS:
                try:
                    cfg.act_class = ActorDiscretePPO
                    cfg.cri_class = CriticPPO
                except Exception:
                    pass

            # Additional PPO hyperparams (best-effort, tolerant to version differences)
            for _setter in [
                ("if_use_gae", True),
                ("lambda_gae", float(hp.get("gae_lambda", 0.95))),
                ("ratio_clip", float(hp.get("clip_ratio", 0.2))),
                ("lambda_entropy", float(hp.get("entropy_coef", 0.01))),
                ("lambda_value", float(hp.get("vf_coef", 0.5))),
                ("if_off_policy", False),
            ]:
                try:
                    setattr(cfg, _setter[0], _setter[1])
                except Exception:
                    pass
            try:
                delattr(cfg, "explore_rate")
            except Exception:
                pass

            print(f"[PPO] break_step={cfg.break_step}, eval_per_step={cfg.eval_per_step}, "
                  f"horizon_len={cfg.horizon_len}, batch_size={cfg.batch_size}, repeat={cfg.repeat_times}")

            try:
                train_agent(cfg)

                # Find checkpoint
                ckpt = None
                for name in ("actor_latest.pth", "actor_best.pth", "actor.pth", "actor.pt"):
                    p = os.path.join(cfg.cwd, name)
                    if os.path.isfile(p): ckpt = p; break
                if ckpt is None:
                    cands = sorted(glob.glob(os.path.join(cfg.cwd, "actor__*.pt")))
                    ckpt = cands[-1] if cands else None

                # Emergency save if none
                if ckpt is None:
                    device = torch.device('cpu' if (not torch.cuda.is_available() or cfg.gpu_id < 0) else 'cuda')
                    emergency_agent = PPOAgent(cfg.net_dims, STATE_DIM, ACTION_DIM, gpu_id=cfg.gpu_id, args=cfg)
                    emergency_actor = emergency_agent.act.eval().to(device)
                    os.makedirs(cfg.cwd, exist_ok=True)
                    ckpt = os.path.join(cfg.cwd, "actor_latest.pth")
                    try:
                        torch.save(emergency_actor.state_dict(), ckpt)
                    except Exception:
                        torch.save(emergency_actor, ckpt)

                print(f"[PPO] using ckpt: {ckpt}")
                try:
                    print(f"[PPO] cwd listing: {os.listdir(cfg.cwd)[:10]}")
                except Exception:
                    pass

                # Final full eval
                device = torch.device('cpu' if cfg.gpu_id < 0 else 'cuda')
                agent  = PPOAgent(cfg.net_dims, STATE_DIM, ACTION_DIM, gpu_id=cfg.gpu_id, args=cfg)
                obj    = torch.load(ckpt, map_location=device)
                try:
                    agent.act.load_state_dict(obj)
                except Exception:
                    try:
                        agent.act.load_state_dict(obj["state_dict"])
                    except Exception:
                        agent.act = obj.to(device) if hasattr(obj, "to") else agent.act
                actor = agent.act.eval()

                total_return, cum_return_max = _full_eval(actor)
                if not np.isfinite(cum_return_max): cum_return_max = 0.0
                if not np.isfinite(total_return):  total_return  = 0.0

                print(f"[PPO] final eval → total_return={total_return:.6f}, cum_return_max={cum_return_max:.6f}")

                L, F = get_latest_and_final()
                F.update({"cum_return_max": float(cum_return_max), "val_return": float(total_return)})

                tune_report({
                    "step": L["step"],
                    "exp_r": L["exp_r"],
                    "exp_r_ema": L["exp_r_ema"],
                    "objC": L["objC"],
                    "objA": L["objA"],
                    "cum_return_max": F["cum_return_max"],
                    "val_return": F["val_return"],
                    "ckpt": ckpt
                })
                return
            except Exception as e:
                tb = traceback.format_exc()
                L, _ = get_latest_and_final()
                print(f"[PPO][ERROR] {type(e).__name__}: {str(e)[:200]}\n{tb}")
                tune_report({**L, **{
                    "cum_return_max": -1e30, "val_return": -1e30,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "traceback": tb[-9000:],
                }})
                return
        return trainable_ray_ppo

    trainable_ray_ppo = make_trainable()

    # ---------------- G) Param space & FIFO ----------------
    if param_space is None:
        param_space = {
            "net_dims":     tune.choice([(64,64), (128,128), (256,128)]),
            "lr":           tune.loguniform(1e-5, 3e-4),
            "batch_size":   tune.choice([2048, 4096]),
            "gamma":        tune.uniform(0.95, 0.999),
            "horizon_len":  tune.choice([512, 1024, 2048]),
            "repeat":       tune.qrandint(2, 6, 1),
            "gae_lambda":   tune.uniform(0.90, 0.98),
            "clip_ratio":   tune.uniform(0.10, 0.30),
            "entropy_coef": tune.loguniform(5e-4, 2e-2),
            "vf_coef":      tune.uniform(0.3, 0.7),
            "passes":       tune.choice([2, 3]),
            "seed":         tune.randint(0, 1_000_000),
        }

    storage_root = Path(storage_root).expanduser().resolve()
    try:
        rc = tune.RunConfig(name=experiment_name, storage_path=storage_root.as_uri(), verbose=1)
    except TypeError:
        rc = tune.RunConfig(name=experiment_name, local_dir=str(storage_root), verbose=1)

    scheduler = FIFOScheduler()  # no early stopping

    tuner = tune.Tuner(
        trainable_ray_ppo,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, scheduler=scheduler),
        run_config=rc,
    )

    results = tuner.fit()

    # --- best by cum_return_max ---
    try:
        best = results.get_best_result(metric="cum_return_max", mode="max", filter_nan_and_inf=False)
        m = (best.metrics or {})
        print("Best cum_return_max:", m.get("cum_return_max"))
        print("Best (total) val_return:", m.get("val_return"))
        print("Best config:", best.config)
        print("Best logdir:", best.path)
    except Exception:
        df = results.get_dataframe()
        cand_cols = [c for c in df.columns if c.split("/")[-1] == "cum_return_max"]
        if cand_cols:
            col = cand_cols[0]
            dff = df[np.isfinite(df[col])]
            if len(dff):
                row = dff.loc[dff[col].idxmax()]
                print("Best cum_return_max:", row[col])
                best = type(
                    "BestLike",
                    (),
                    {
                        "metrics": {"cum_return_max": row[col]},
                        "config": {k.split("/",1)[1]: row[k] for k in df.columns if k.startswith("config/")},
                        "path": None
                    }
                )
        else:
            print("No `cum_return_max` found in results; did any trial reach final evaluation?")
            best = None

    return results, best
