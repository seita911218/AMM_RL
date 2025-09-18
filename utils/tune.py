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
    Ray Tune for ElegantRL DQN (FIFO, no early stopping).
    Final selection uses env.info['after_equity'] as final_equity.
    """

    # ---------------- Imports (local to function) ----------------
    from pathlib import Path
    import os, sys, shutil, glob
    import numpy as np
    import torch, ray
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler
    from ray.air import session
    from elegantrl.agents.AgentDQN import AgentDQN
    from elegantrl.train.config import Config
    from elegantrl.train.run import train_agent

    # ---------------- A) Staging ----------------
    ROOT = Path.cwd().resolve()
    tmp = ROOT
    while tmp != tmp.parent and not (tmp / "utils").exists():
        tmp = tmp.parent
    ROOT = tmp
    STAGING = ROOT / "ray_staging"
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    def _copy_tree_safe(src: Path, dst: Path, exts_exclude=(
        ".parquet", ".pt", ".pth", ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg", ".ipynb"
    )):
        if not src.exists(): return
        if src.is_file():
            if src.suffix.lower() not in exts_exclude:
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
            return
        for p in src.rglob("*"):
            if p.is_dir(): continue
            if p.suffix.lower() in exts_exclude: continue
            rel = p.relative_to(src); out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try: shutil.copy2(p, out)
            except Exception: pass

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
            except Exception: pass

    print("✅ staged working_dir:", STAGING)

    # ---------------- B) Ray init ----------------
    ray.shutdown()
    ray.init(
        runtime_env={"working_dir": str(STAGING),
                     "env_vars": {"PYTHONPATH": str(STAGING), "PROJECT_ROOT": str(STAGING)}},
        ignore_reinit_error=True,
    )
    print("✅ Ray initialized with working_dir =", STAGING)

    # ---------------- C) Dimensions (probe from the actual env) ----------------
    def _probe_dims_for_training(train_data, train_time_data, TRAIN_ENV_KW):
        from utils.env import UniswapV3LiquidityEnv
        env = UniswapV3LiquidityEnv(
            numeric_data=train_data,
            time_data=train_time_data,
            **{**TRAIN_ENV_KW, "max_steps": 1},
        )
        s0 = env.reset()
        if isinstance(s0, tuple):
            s0 = s0[0]
        s0 = np.asarray(s0, dtype=np.float32)
        state_dim = int(np.prod(s0.shape))
        action_dim = int(getattr(env.action_space, "n", 1))
        try: env.close()
        except Exception: pass
        return state_dim, action_dim

    STATE_DIM, ACTION_DIM = _probe_dims_for_training(train_data, train_time_data, TRAIN_ENV_KW)
    IF_DISCRETE = True
    T_BASE = int(min(int(TRAIN_ENV_KW["max_steps"]), int(train_data.shape[0] - 1)))
    print(f"[Tune] PROBED STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, IF_DISCRETE={IF_DISCRETE}, T_BASE={T_BASE}")

    # ---------------- D) Logger ----------------
    def install_tune_logger():
        import numpy as _np
        import elegantrl.train.evaluator as erl_eval
        from ray.tune import report as tune_report
        import torch as _torch

        LATEST = {"step": 0, "exp_r": -1e30, "exp_r_ema": -1e30, "objC": float("nan"), "objA": float("nan")}
        FINAL  = {"final_equity": -1e30}

        def _meanify(x):
            try:
                if x is None: return _np.nan
                if isinstance(x, _torch.Tensor): return float(x.detach().mean().cpu().item())
                if isinstance(x, (int, float)): return float(x)
                return float(_np.nanmean(_np.asarray(x, dtype=float)))
            except Exception:
                return _np.nan

        def _csv_and_report(self, actor=None, steps=0, exp_r=None, logging_tuple=None):
            nonlocal LATEST, FINAL
            self.total_step = int(getattr(self, "total_step", 0)) + int(steps or 0)
            expR = _meanify(exp_r)
            objC = objA = _meanify(logging_tuple[0]) if (isinstance(logging_tuple, (list,tuple)) and len(logging_tuple)>0) else _np.nan
            ema_prev = getattr(self, "_ema_state", {}).get("ema_expR", expR if not _np.isnan(expR) else 0.0)
            if not _np.isnan(expR): ema_prev = 0.9 * ema_prev + 0.1 * expR
            self._ema_state = {"ema_expR": ema_prev}

            LATEST = {
                "step": int(self.total_step),
                "exp_r": float(expR if _np.isfinite(expR) else -1e30),
                "exp_r_ema": float(ema_prev if _np.isfinite(ema_prev) else -1e30),
                "objC": float(objC) if _np.isfinite(objC) else float("nan"),
                "objA": float(objA) if _np.isfinite(objA) else float("nan"),
            }

            try:
                if actor is not None:
                    os.makedirs(self.cwd, exist_ok=True)
                    _torch.save(getattr(actor, "state_dict", lambda: actor)(), os.path.join(self.cwd, "actor_latest.pth"))
            except Exception: pass

            tune_report({
                "step": LATEST["step"], "exp_r": LATEST["exp_r"], "exp_r_ema": LATEST["exp_r_ema"],
                "objC": LATEST["objC"], "objA": LATEST["objA"],
                "final_equity": float(FINAL["final_equity"]),
            })

        erl_eval.Evaluator.evaluate_and_save = _csv_and_report
        erl_eval.Evaluator.save_training_curve_jpg = lambda self: None

        def _get_latest_and_final(): return LATEST, FINAL
        return _get_latest_and_final

    # ---------------- E) Full eval: final_equity ----------------
    def _eval_final_equity(actor, env_kwargs):
        pr = os.environ.get("PROJECT_ROOT")
        if pr:
            try: os.chdir(pr)
            except Exception: pass
        from utils.env import UniswapV3LiquidityEnv
        import numpy as _np, torch as _torch
        env = UniswapV3LiquidityEnv(**env_kwargs)
        s = env.reset(); s = s[0] if isinstance(s, tuple) else s
        device = next(actor.parameters()).device
        final_equity = None; done = False
        while not done:
            st = _torch.as_tensor(_np.asarray(s, _np.float32).reshape(1, -1), dtype=_torch.float32, device=device)
            with _torch.no_grad():
                q = actor(st); a = int(q.argmax(dim=1).item())
            out = env.step(a)
            if isinstance(out, tuple) and len(out) == 5:
                s, r, term, trunc, info = out; done = bool(term or trunc)
            else:
                s, r, done, info = out
            try:
                final_equity = float((info or {}).get("after_equity"))
            except Exception: pass
        try: env.close()
        except Exception: pass
        return float(final_equity if final_equity is not None else -1e30)

    # ---------------- F) Trainable ----------------
    def make_trainable():
        def trainable_ray(hp: dict):
            from ray.tune import report as tune_report
            tune_report({"step": 0, "exp_r": -1e30, "exp_r_ema": -1e30,
                         "objC": float("nan"), "objA": float("nan"),
                         "final_equity": -1e30, "terminate_reason": "init"})

            pr = os.environ.get("PROJECT_ROOT")
            if pr and pr not in sys.path: sys.path.insert(0, pr)
            if pr:
                try: os.chdir(pr)
                except Exception: pass

            get_latest_and_final = install_tune_logger()
            from utils.env import UniswapV3LiquidityEnv

            episode_len = int(hp.get("episode_len", T_BASE))
            episode_len = max(1, min(episode_len, int(train_data.shape[0] - 1)))
            T_local = episode_len

            env_args = {
                "env_name": "UniswapV3LiquidityEnv",
                "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
                "if_discrete": IF_DISCRETE, "max_step": T_local, "num_envs": 1,
                "numeric_data": train_data, "time_data": train_time_data,
                **{**TRAIN_ENV_KW, "max_steps": T_local},
            }

            cfg = Config(agent_class=AgentDQN, env_class=UniswapV3LiquidityEnv, env_args=env_args)
            try:
                trial_dir = session.get_trial_dir(); cfg.cwd = os.path.join(trial_dir, "erl")
            except Exception:
                trial_id = os.getenv("RAY_TRIAL_ID", "trial"); cfg.cwd = os.path.join("./ray_runs_dqn", trial_id)

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
            if int(cfg.horizon_len) > int(T_local):
                print(f"[Guard] clamp horizon_len {cfg.horizon_len} -> {T_local}")
                cfg.horizon_len = int(T_local)
            passes              = int(hp.get("passes", 1))
            cfg.repeat_times    = int(hp["repeat"])
            cfg.buffer_size     = int(hp["buffer"])
            cfg.break_step      = T_local * passes * cfg.env_num
            cfg.eval_times      = 0
            cfg.eval_per_step   = T_local
            cfg.explore_rate    = float(hp["eps"])
            cfg.if_use_per      = bool(hp["use_per"])

            # ====== DQN buffer/batch guards ======
            def _guard_dqn_batch_and_buffer(cfg, T_local):
                env_num = max(1, int(getattr(cfg, "env_num", 1)))
                buf_len = int(getattr(cfg, "horizon_len", T_local)) * env_num
                repeat  = max(1, int(getattr(cfg, "repeat_times", 1)))

                max_batch_for_updates = buf_len * repeat
                if int(cfg.batch_size) > int(max_batch_for_updates):
                    print(f"[DQN Guard] batch {cfg.batch_size}->{max_batch_for_updates} (buf_len={buf_len}, repeat={repeat})")
                    cfg.batch_size = int(max_batch_for_updates)

                if int(cfg.batch_size) > int(cfg.buffer_size):
                    print(f"[DQN Guard] batch {cfg.batch_size} > buffer {cfg.buffer_size} -> shrink batch to buffer")
                    cfg.batch_size = int(cfg.buffer_size)

                min_buffer = max(2 * buf_len, 4 * int(cfg.batch_size))
                if int(cfg.buffer_size) < int(min_buffer):
                    print(f"[DQN Guard] buffer {cfg.buffer_size}->{min_buffer} (>=2*buf_len & >=4*batch)")
                    cfg.buffer_size = int(min_buffer)

            _guard_dqn_batch_and_buffer(cfg, T_local)

            print(f"[DQN] ep_len={T_local}, passes={passes}, buf={cfg.buffer_size}, batch={cfg.batch_size}, "
                  f"repeat={cfg.repeat_times}, break_step={cfg.break_step}")

            try:
                train_agent(cfg)

                # checkpoint
                ckpt = None
                for name in ("actor_latest.pth", "actor.pth", "actor.pt"):
                    p = os.path.join(cfg.cwd, name)
                    if os.path.isfile(p): ckpt = p; break
                if ckpt is None:
                    cands = sorted(glob.glob(os.path.join(cfg.cwd, "actor__*.pt"))); ckpt = cands[-1] if cands else None
                if ckpt is None:
                    L, F = get_latest_and_final()
                    tune_report({**L, "final_equity": -1e30, "terminate_reason": "no_checkpoint"})
                    return

                # final eval
                device = torch.device('cpu' if cfg.gpu_id < 0 else 'cuda')
                agent  = AgentDQN(cfg.net_dims, cfg.state_dim, cfg.action_dim, gpu_id=cfg.gpu_id, args=cfg)
                obj    = torch.load(ckpt, map_location=device)
                try:
                    agent.act.load_state_dict(obj)
                except Exception:
                    try: agent.act.load_state_dict(obj["state_dict"])
                    except Exception: agent.act = obj.to(device)
                actor = agent.act.eval()

                env_eval_kw = {"numeric_data": train_data, "time_data": train_time_data, **{**TRAIN_ENV_KW, "max_steps": T_local}}
                final_equity = _eval_final_equity(actor, env_eval_kw)

                L, F = get_latest_and_final()
                F.update({"final_equity": float(final_equity)})
                tune_report({
                    "step": L["step"], "exp_r": L["exp_r"], "exp_r_ema": L["exp_r_ema"],
                    "objC": L["objC"], "objA": L["objA"],
                    "final_equity": float(final_equity), "terminate_reason": "completed_eval",
                    "ckpt": ckpt
                })
                return

            except Exception as e:
                L, _ = get_latest_and_final()
                tune_report({**L, "final_equity": -1e30, "terminate_reason": f"error:{type(e).__name__}",
                             "error": str(e)[:200]})
                return
        return trainable_ray

    trainable_ray = make_trainable()

    # ---------------- G) Param space (default: exploration-boosted) ----------------
    if param_space is None:
        from ray import tune as _t
        param_space = {
            "net_dims":   _t.choice([(64,64), (128,128), (256,256)]),
            "lr":         _t.loguniform(1e-5, 5e-4),
            "batch_size": _t.choice([64, 128, 256]),
            "gamma":      _t.uniform(0.96, 0.995),
            "tau":        _t.loguniform(1e-4, 1e-2),
            "horizon_len":_t.choice([128, 256, 512, 1024]),
            "eps":        _t.uniform(0.30, 0.80),
            "buffer":     _t.qlograndint(int(2e5), int(1e6), int(1e4)),
            "repeat":     _t.qrandint(1, 2, 1),
            "use_per":    _t.choice([False, True]),
            "passes":     _t.choice([1, 2]),
            "episode_len":_t.choice([int(train_data.shape[0])]),
            "seed":       _t.randint(0, 1_000_000),
        }

    # ---------------- H) Reporter + RunConfig ----------------
    try:
        from ray.tune import CLIReporter
    except ImportError:
        from ray.tune.progress_reporter import CLIReporter
    reporter = CLIReporter(
        parameter_columns=["batch_size","buffer","episode_len","passes","repeat","lr","net_dims","use_per"],
        metric_columns=["iter","total time (s)","step","exp_r_ema","final_equity","terminate_reason"],
        max_progress_rows=50,
    )

    storage_root = Path(storage_root).expanduser().resolve()
    try:
        rc = tune.RunConfig(name=experiment_name, storage_path=storage_root.as_uri(),
                            verbose=1, progress_reporter=reporter)
    except TypeError:
        rc = tune.RunConfig(name=experiment_name, local_dir=str(storage_root),
                            verbose=1, progress_reporter=reporter)

    scheduler = FIFOScheduler()  # no early stopping

    tuner = tune.Tuner(
        trainable_ray,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, scheduler=scheduler),
        run_config=rc,
    )

    results = tuner.fit()

    # --- Best by final_equity ---
    try:
        best = results.get_best_result(metric="final_equity", mode="max", filter_nan_and_inf=False)
        m = (best.metrics or {})
        print("Best final_equity:", m.get("final_equity"))
        print("Best config:", best.config)
        print("Best logdir:", best.path)
    except Exception:
        df = results.get_dataframe()
        cand_cols = [c for c in df.columns if c.split("/")[-1] == "final_equity"]
        if cand_cols:
            col = cand_cols[0]
            dff = df[np.isfinite(df[col])]
            if len(dff):
                row = dff.loc[dff[col].idxmax()]
                print("Best final_equity:", row[col])
                best = type("BestLike", (), {
                    "metrics": {"final_equity": row[col]},
                    "config": {k.split("/",1)[1]: row[k] for k in df.columns if k.startswith("config/")},
                    "path": None
                })
        else:
            print("No `final_equity` found in results; did any trial finish?")
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
    Ray Tune for ElegantRL PPO (FIFO). Final selection uses final_equity from env.info['after_equity'].
    """

    # ---------------- Imports ----------------
    from pathlib import Path
    import os, sys, shutil, glob, traceback
    import numpy as np
    import torch, ray
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler
    from ray.air import session

    # Safe import (Discrete → fallback)
    try:
        from elegantrl.agents.AgentPPO import AgentDiscretePPO as PPOAgent
        _HAS_DISCRETE = True
    except Exception:
        from elegantrl.agents.AgentPPO import AgentPPO as PPOAgent
        _HAS_DISCRETE = False

    try:
        from elegantrl.agents.net import ActorDiscretePPO, CriticPPO
        _FORCE_DISCRETE_NETS = True
    except Exception:
        _FORCE_DISCRETE_NETS = False

    from elegantrl.train.config import Config
    from elegantrl.train.run import train_agent

    # Bridge gymnasium -> gym
    try:
        import gymnasium as _gym
        sys.modules.setdefault("gym", _gym)
    except Exception:
        pass

    # ---------------- A) Staging ----------------
    ROOT = Path.cwd().resolve()
    tmp = ROOT
    while tmp != tmp.parent and not (tmp / "utils").exists():
        tmp = tmp.parent
    ROOT = tmp
    STAGING = ROOT / "ray_staging"
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    def _copy_tree_safe(src: Path, dst: Path, exts_exclude=(
        ".parquet", ".pt", ".pth", ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg", ".ipynb"
    )):
        if not src.exists(): return
        if src.is_file():
            if src.suffix.lower() not in exts_exclude:
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
            return
        for p in src.rglob("*"):
            if p.is_dir(): continue
            if p.suffix.lower() in exts_exclude: continue
            rel = p.relative_to(src); out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try: shutil.copy2(p, out)
            except Exception: pass

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
            except Exception: pass

    print("✅ Staged working_dir:", STAGING)

    # ---------------- B) Ray init ----------------
    ray.shutdown()
    ray.init(
        runtime_env={"working_dir": str(STAGING),
                     "env_vars": {"PYTHONPATH": str(STAGING), "PROJECT_ROOT": str(STAGING)}},
        ignore_reinit_error=True,
    )
    print("✅ Ray initialized with working_dir =", STAGING)

    # ---------------- C) Dimensions (probe from the actual env) ----------------
    def _probe_dims_for_training(train_data, train_time_data, TRAIN_ENV_KW):
        from utils.env import UniswapV3LiquidityEnv
        env = UniswapV3LiquidityEnv(
            numeric_data=train_data,
            time_data=train_time_data,
            **{**TRAIN_ENV_KW, "max_steps": 1},
        )
        s0 = env.reset()
        if isinstance(s0, tuple):
            s0 = s0[0]
        s0 = np.asarray(s0, dtype=np.float32)
        state_dim = int(np.prod(s0.shape))
        action_dim = int(getattr(env.action_space, "n", 1))
        try: env.close()
        except Exception: pass
        return state_dim, action_dim

    STATE_DIM, ACTION_DIM = _probe_dims_for_training(train_data, train_time_data, TRAIN_ENV_KW)
    IF_DISCRETE = True
    T_BASE = int(min(int(TRAIN_ENV_KW["max_steps"]), int(train_data.shape[0] - 1)))
    print(f"[Tune] PROBED STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, IF_DISCRETE={IF_DISCRETE}, T_BASE={T_BASE}")

    ENV_EVAL_KW_BASE = {"numeric_data": train_data, "time_data": train_time_data, **{**TRAIN_ENV_KW, "max_steps": T_BASE}}

    # ---------------- Helpers ----------------
    def _sanitize_arr(x):
        x = np.asarray(x, dtype=np.float32)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def _to_discrete_idx_from_actor(actor, st, action_dim: int):
        import numpy as _np, torch as _torch
        with _torch.no_grad():
            a = None
            if hasattr(actor, "get_action"):
                a = actor.get_action(st);  a = a[0] if isinstance(a, (tuple,list)) else a
            if a is None:
                out = actor(st); out = out[0] if isinstance(out, (tuple,list)) else out
                a = out
            if isinstance(a, _torch.Tensor): a = a.detach().cpu().numpy()
            a = _np.asarray(a).squeeze()
            if a.ndim == 0:
                try: idx = int(a.item())
                except Exception: idx = int(a)
            else:
                idx = int(_np.argmax(a))
            if idx < 0: idx = 0
            if idx >= action_dim: idx = action_dim - 1
            return idx

    # ---------------- Final full eval (equity-based) ----------------
    def _full_eval_final_equity(actor, env_kw):
        pr = os.environ.get("PROJECT_ROOT")
        if pr:
            try: os.chdir(pr)
            except Exception: pass
        from utils.env import UniswapV3LiquidityEnv
        import numpy as _np, torch as _torch
        env = UniswapV3LiquidityEnv(**env_kw)
        s = env.reset(); s = s[0] if isinstance(s, tuple) else s
        s = _sanitize_arr(s);  s = s.reshape(-1) if s.ndim > 1 else s
        device = next(actor.parameters()).device
        final_equity = None; done = False
        safety_steps = int(env_kw.get("max_steps", 10_000)) + 5; c = 0
        while not done and c < safety_steps:
            c += 1
            st = _torch.as_tensor(s, dtype=_torch.float32, device=device).unsqueeze(0)
            idx = _to_discrete_idx_from_actor(actor, st, ACTION_DIM)
            out = env.step(idx)
            if isinstance(out, tuple) and len(out) == 5:
                s, r, term, trunc, info = out; done = bool(term or trunc)
            else:
                s, r, done, info = out
            s = _sanitize_arr(s); s = s.reshape(-1) if s.ndim > 1 else s
            try: final_equity = float((info or {}).get("after_equity"))
            except Exception: pass
        try: env.close()
        except Exception: pass
        return float(final_equity if final_equity is not None else -1e30)

    # ---------------- D) Logger ----------------
    def install_tune_logger():
        import numpy as _np
        import elegantrl.train.evaluator as erl_eval
        from ray.tune import report as tune_report
        import torch as _torch

        LATEST = {"step": 0, "exp_r": -1e30, "exp_r_ema": -1e30, "objC": float("nan"), "objA": float("nan")}
        FINAL  = {"final_equity": -1e30}

        def _meanify(x):
            try:
                if x is None: return _np.nan
                if isinstance(x, _torch.Tensor): return float(x.detach().mean().cpu().item())
                if isinstance(x, (int, float)): return float(x)
                return float(_np.nanmean(_np.asarray(x, dtype=float)))
            except Exception:
                return _np.nan

        def _csv_and_report(self, actor=None, steps=0, exp_r=None, logging_tuple=None):
            nonlocal LATEST, FINAL
            self.total_step = int(getattr(self, "total_step", 0)) + int(steps or 0)
            expR = _meanify(exp_r)
            objC = objA = _meanify(logging_tuple[0]) if (isinstance(logging_tuple, (list,tuple)) and len(logging_tuple)>0) else _np.nan
            ema_prev = getattr(self, "_ema_state", {}).get("ema_expR", expR if not _np.isnan(expR) else 0.0)
            if not _np.isnan(expR): ema_prev = 0.9 * ema_prev + 0.1 * expR
            self._ema_state = {"ema_expR": ema_prev}

            LATEST = {
                "step": int(self.total_step),
                "exp_r": float(expR if _np.isfinite(expR) else -1e30),
                "exp_r_ema": float(ema_prev if _np.isfinite(ema_prev) else -1e30),
                "objC": float(objC) if _np.isfinite(objC) else float("nan"),
                "objA": float(objA) if _np.isfinite(objA) else float("nan"),
            }

            try:
                if actor is not None:
                    os.makedirs(self.cwd, exist_ok=True)
                    _torch.save(getattr(actor, "state_dict", lambda: actor)(), os.path.join(self.cwd, "actor_latest.pth"))
            except Exception: pass

            tune_report({
                "step": LATEST["step"], "exp_r": LATEST["exp_r"], "exp_r_ema": LATEST["exp_r_ema"],
                "objC": LATEST["objC"], "objA": LATEST["objA"],
                "final_equity": float(FINAL["final_equity"]),
            })

        erl_eval.Evaluator.evaluate_and_save = _csv_and_report
        erl_eval.Evaluator.save_training_curve_jpg = lambda self: None

        def _get_latest_and_final(): return LATEST, FINAL
        return _get_latest_and_final

    # ---------------- F) Trainable ----------------
    def make_trainable():
        def trainable_ray_ppo(hp: dict):
            from ray.tune import report as tune_report
            tune_report({"step": 0, "exp_r": -1e30, "exp_r_ema": -1e30,
                         "objC": float("nan"), "objA": float("nan"),
                         "final_equity": -1e30, "terminate_reason": "init"})

            pr = os.environ.get("PROJECT_ROOT")
            if pr and pr not in sys.path: sys.path.insert(0, pr)
            if pr:
                try: os.chdir(pr)
                except Exception: pass

            get_latest_and_final = install_tune_logger()
            from utils.env import UniswapV3LiquidityEnv

            episode_len = int(hp.get("episode_len", T_BASE))
            episode_len = max(1, min(episode_len, int(train_data.shape[0] - 1)))
            T_local = episode_len

            env_args = {
                "env_name": "UniswapV3LiquidityEnv",
                "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
                "if_discrete": True, "max_step": T_local, "num_envs": 1,
                "numeric_data": train_data, "time_data": train_time_data,
                **{**TRAIN_ENV_KW, "max_steps": T_local},
            }

            cfg = Config(agent_class=PPOAgent, env_class=UniswapV3LiquidityEnv, env_args=env_args)
            try:
                trial_dir = session.get_trial_dir(); cfg.cwd = os.path.join(trial_dir, "erl_ppo")
            except Exception:
                trial_id = os.getenv("RAY_TRIAL_ID", "trial_ppo"); cfg.cwd = os.path.join("./ray_runs_ppo", trial_id)

            cfg.if_remove = False
            try: cfg.if_keep_save = True
            except Exception: pass

            cfg.random_seed = int(hp.get("seed", 0))
            cfg.env_num = 1
            cfg.gpu_id = 0 if torch.cuda.is_available() else -1

            cfg.net_dims      = tuple(hp["net_dims"])
            cfg.horizon_len   = int(hp["horizon_len"])
            if int(cfg.horizon_len) > int(T_local):
                print(f"[Guard] clamp horizon_len {cfg.horizon_len} -> {T_local}")
                cfg.horizon_len = int(T_local)
            cfg.batch_size    = int(hp["batch_size"])
            cfg.learning_rate = float(hp["lr"])
            cfg.gamma         = float(hp["gamma"])
            passes            = int(hp.get("passes", 1))
            cfg.repeat_times  = max(1, int(hp["repeat"]))

            # --- batch alignment guard ---
            buf_len   = int(cfg.horizon_len) * max(1, int(getattr(cfg, "env_num", 1)))
            max_batch = buf_len * max(1, int(cfg.repeat_times))
            desired   = min(int(cfg.batch_size), int(max_batch))
            aligned_dn = (desired // buf_len) * buf_len
            if aligned_dn < buf_len: aligned_dn = buf_len
            if aligned_dn != cfg.batch_size:
                print(f"[PPO] adjust batch_size {cfg.batch_size} -> {aligned_dn} "
                      f"(buf_len={buf_len}, repeat={cfg.repeat_times}, max_batch={max_batch})")
            cfg.batch_size = aligned_dn

            cfg.break_step    = max(int(T_local * passes * cfg.env_num), int(cfg.horizon_len), 1024)
            cfg.eval_times    = 1
            cfg.eval_per_step = T_local
            try: cfg.save_gap = max(200, cfg.horizon_len // 2)
            except Exception: pass

            try: cfg.if_discrete = True
            except Exception: pass
            if _HAS_DISCRETE and _FORCE_DISCRETE_NETS:
                try:
                    cfg.act_class = ActorDiscretePPO; cfg.cri_class = CriticPPO
                except Exception: pass

            for _setter in [
                ("if_use_gae", True),
                ("lambda_gae", float(hp.get("gae_lambda", 0.95))),
                ("ratio_clip", float(hp.get("clip_ratio", 0.2))),
                ("lambda_entropy", float(hp.get("entropy_coef", 0.01))),
                ("lambda_value", float(hp.get("vf_coef", 0.5))),
                ("if_off_policy", False),
            ]:
                try: setattr(cfg, _setter[0], _setter[1])
                except Exception: pass
            try: delattr(cfg, "explore_rate")
            except Exception: pass

            print(f"[PPO] ep_len={T_local}, horizon_len={cfg.horizon_len}, passes={passes}, "
                  f"batch={cfg.batch_size}, repeat={cfg.repeat_times}, break_step={cfg.break_step}")

            try:
                train_agent(cfg)

                # checkpoint
                ckpt = None
                for name in ("actor_latest.pth", "actor_best.pth", "actor.pth", "actor.pt"):
                    p = os.path.join(cfg.cwd, name)
                    if os.path.isfile(p): ckpt = p; break
                if ckpt is None:
                    cands = sorted(glob.glob(os.path.join(cfg.cwd, "actor__*.pt"))); ckpt = cands[-1] if cands else None
                if ckpt is None:
                    L, F = get_latest_and_final()
                    tune_report({**L, "final_equity": -1e30, "terminate_reason": "no_checkpoint"})
                    return

                print(f"[PPO] using ckpt: {ckpt}")

                # final eval
                device = torch.device('cpu' if cfg.gpu_id < 0 else 'cuda')
                agent  = PPOAgent(cfg.net_dims, STATE_DIM, ACTION_DIM, gpu_id=cfg.gpu_id, args=cfg)
                obj    = torch.load(ckpt, map_location=device)
                try:
                    agent.act.load_state_dict(obj)
                except Exception:
                    try: agent.act.load_state_dict(obj["state_dict"])
                    except Exception: agent.act = obj.to(device) if hasattr(obj, "to") else agent.act
                actor = agent.act.eval()

                env_eval_kw = {"numeric_data": train_data, "time_data": train_time_data, **{**TRAIN_ENV_KW, "max_steps": T_local}}
                final_equity = _full_eval_final_equity(actor, env_eval_kw)

                L, F = get_latest_and_final()
                F.update({"final_equity": float(final_equity)})
                tune_report({
                    "step": L["step"], "exp_r": L["exp_r"], "exp_r_ema": L["exp_r_ema"],
                    "objC": L["objC"], "objA": L["objA"],
                    "final_equity": float(final_equity), "terminate_reason": "completed_eval",
                    "ckpt": ckpt
                })
                return

            except Exception as e:
                tb = traceback.format_exc()
                L, _ = get_latest_and_final()
                print(f"[PPO][ERROR] {type(e).__name__}: {str(e)[:200]}\n{tb}")
                tune_report({**L, "final_equity": -1e30, "terminate_reason": f"error:{type(e).__name__}",
                             "error": f"{type(e).__name__}: {str(e)[:200]}"})
                return
        return trainable_ray_ppo

    trainable_ray_ppo = make_trainable()

    # ---------------- G) Param space (default: exploration-boosted) ----------------
    if param_space is None:
        from ray import tune as _t
        param_space = {
            "net_dims":     _t.choice([(64, 64), (128, 128), (256, 128)]),
            "lr":           _t.loguniform(1e-5, 3e-4),
            "batch_size":   _t.choice([1024, 2048, 4096]),
            "gamma":        _t.uniform(0.96, 0.999),
            "episode_len":  _t.choice([int(train_data.shape[0])]),
            "horizon_len":  _t.sample_from(lambda spec: spec.config["episode_len"]),
            "repeat":       _t.qrandint(2, 6, 1),
            "gae_lambda":   _t.uniform(0.90, 0.98),
            "clip_ratio":   _t.uniform(0.15, 0.35),
            "entropy_coef": _t.loguniform(1e-3, 5e-2),
            "vf_coef":      _t.uniform(0.3, 0.7),
            "passes":       _t.choice([1, 2, 4, 8]),
            "seed":         _t.randint(0, 1_000_000),
        }

    # ---------------- H) Reporter + RunConfig ----------------
    try:
        from ray.tune import CLIReporter
    except ImportError:
        from ray.tune.progress_reporter import CLIReporter
    reporter = CLIReporter(
        parameter_columns=["batch_size","episode_len","passes","repeat","lr","net_dims"],
        metric_columns=["iter","total time (s)","step","exp_r_ema","final_equity","terminate_reason"],
        max_progress_rows=50,
    )

    storage_root = Path(storage_root).expanduser().resolve()
    try:
        rc = tune.RunConfig(name=experiment_name, storage_path=storage_root.as_uri(),
                            verbose=1, progress_reporter=reporter)
    except TypeError:
        rc = tune.RunConfig(name=experiment_name, local_dir=str(storage_root),
                            verbose=1, progress_reporter=reporter)

    scheduler = FIFOScheduler()

    tuner = tune.Tuner(
        trainable_ray_ppo,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, scheduler=scheduler),
        run_config=rc,
    )

    results = tuner.fit()

    # --- Best by final_equity ---
    try:
        best = results.get_best_result(metric="final_equity", mode="max", filter_nan_and_inf=False)
        m = (best.metrics or {})
        print("Best final_equity:", m.get("final_equity"))
        print("Best config:", best.config)
        print("Best logdir:", best.path)
    except Exception:
        df = results.get_dataframe()
        cand_cols = [c for c in df.columns if c.split("/")[-1] == "final_equity"]
        if cand_cols:
            col = cand_cols[0]
            dff = df[np.isfinite(df[col])]
            if len(dff):
                row = dff.loc[dff[col].idxmax()]
                print("Best final_equity:", row[col])
                best = type("BestLike", (), {
                    "metrics": {"final_equity": row[col]},
                    "config": {k.split("/",1)[1]: row[k] for k in df.columns if k.startswith("config/")},
                    "path": None
                })
        else:
            print("No `final_equity` found in results; did any trial finish?")
            best = None

    return results, best
