from __future__ import annotations
import os, json, math, time, shutil, inspect
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
import optuna
from optuna.exceptions import TrialPruned

from stable_baselines3 import PPO, DQN, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure as sb3_configure_logger

# Suppress gym deprecation warning (SB3 compatibility)
import warnings
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')


# Net architecture mapping for string identifiers
NET_ARCH_MAP = {
    "128x2": [128, 128],
    "256x2": [256, 256],
    "256x2x128": [256, 256, 128],
    "256x3": [256, 256, 256],
    "512x2": [512, 512],
}

def run_optuna(
    env_kw: Dict[str, Any],
    param_space: Dict[str, Any],
    n_trials: int,
    direction: str = "maximize",
    algo_name: str = "ppo",
    total_timesteps: int = 200_000,
    eval_episodes: int = 3,
    eval_freq: Optional[int] = None,
    out_dir: str | Path = "./result_optuna",
    seed: Optional[int] = 42,
    study_name: Optional[str] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Improved Optuna + SB3 Hyperparameter Tuning Tool
    
    Main improvements:
    1. Fixed DQN learning_starts (reduced from 30k to 5-15k)
    2. Fixed DQN exploration_fraction (reduced from 0.5-0.7 to 0.2-0.35)
    3. Fixed PPO ent_coef (increased from 1e-2~8e-2 to 5e-2~1e-1)
    4. Fixed PPO clip_range (increased from 0.2-0.3 to 0.3-0.4)
    5. Added PPO vf_coef parameter tuning
    6. Environment level: max_episode_steps and reward_scale
    
    Args:
        env_kw: Environment parameters (recommended: max_episode_steps=2000 and reward_scale=100.0)
        param_space: Hyperparameter search space (recommended: use IMPROVED_PARAM_SPACE_* series)
        n_trials: Number of Optuna trials
        direction: "maximize" or "minimize"
        algo_name: Algorithm name - "ppo", "dqn", "sac", or "td3"
        total_timesteps: Total training steps per trial
        eval_episodes: Number of recent episodes for evaluation
        eval_freq: Evaluation frequency (None = auto: total_timesteps // 20)
        out_dir: Output directory for results
        seed: Random seed
        study_name: Optional study name
        verbose: Verbosity level (0=quiet, 1=info, 2=debug)
    
    Returns:
        Dictionary containing best parameters, value, model path, and study info
    """
    assert direction in {"maximize", "minimize"}

    # Algo map and environment
    ALGOS = {"ppo": PPO, "dqn": DQN, "sac": SAC, "td3": TD3}
    algo_name_l = algo_name.lower()
    if algo_name_l not in ALGOS:
        raise ValueError(f"algo_name must be one of {list(ALGOS)}")
    Algo = ALGOS[algo_name_l]

    # Auto-switch action_mode for SAC/TD3 (continuous Box actions)
    env_kw = dict(env_kw)  # shallow copy
    if algo_name_l in {"sac", "td3"} and env_kw.get("action_mode", "discrete") != "continuous":
        env_kw["action_mode"] = "continuous"

    # ★ Environment configuration recommendations
    if "max_episode_steps" not in env_kw:
        print("⚠️  Recommendation: max_episode_steps not set in env_kw, suggest adding max_episode_steps=2000")
    if "reward_scale" not in env_kw:
        print("⚠️  Recommendation: reward_scale not set in env_kw, suggest adding reward_scale=100.0")

    # Output dirs
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    study_dir = out_root / f"study-{study_name or ts}"
    study_dir.mkdir(parents=True, exist_ok=True)
    eval_every = int(eval_freq) if eval_freq is not None else max(1, total_timesteps // 20)

    # Optuna
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=min(5, max(1, n_trials // 10)))
    study   = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, study_name=study_name)

    # Suggest helpers
    def _suggest_new(trial: optuna.Trial, spec: Dict[str, Any], name: str):
        """New dict schema: {type: float/int/categorical/fixed, ...}"""
        t = str(spec["type"]).lower()
        if t == "fixed":
            return spec.get("value")
        if t == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        if t == "int":
            low, high = int(spec["low"]), int(spec["high"])
            step = spec.get("step")
            log  = bool(spec.get("log", False))
            return trial.suggest_int(name, low, high, step=None if step is None else int(step), log=log)
        if t == "float":
            low, high = float(spec["low"]), float(spec["high"])
            step = spec.get("step")
            log  = bool(spec.get("log", False))
            return trial.suggest_float(name, low, high, step=None if step is None else float(step), log=log)
        raise ValueError(f"unknown type {t} for {name}")

    def _suggest_legacy(trial: optuna.Trial, tpl: tuple, name: str):
        """Legacy tuple schema: ('loguniform'| 'uniform'| 'choice', ...)."""
        kind = str(tpl[0]).lower()
        if kind == "loguniform":
            lo, hi = float(tpl[1]), float(tpl[2])
            return trial.suggest_float(name, lo, hi, log=True)
        if kind == "uniform":
            lo, hi = float(tpl[1]), float(tpl[2])
            return trial.suggest_float(name, lo, hi)
        if kind == "choice":
            choices = tpl[1]
            return trial.suggest_categorical(name, choices)
        raise ValueError(f"Unknown legacy spec for {name}: {tpl}")

    def _suggest_params(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
        """Support both new dict and legacy tuple specs."""
        out: Dict[str, Any] = {}
        for k, spec in space.items():
            if isinstance(spec, dict) and "type" in spec:
                out[k] = _suggest_new(trial, spec, k)
            elif isinstance(spec, tuple):
                out[k] = _suggest_legacy(trial, spec, k)
            else:
                # fixed literal
                out[k] = spec
        return out

    def _pack_policy_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
        """Pack policy kwargs and handle net_arch mapping."""
        params = dict(params)
        pk = dict(params.pop("policy_kwargs", {}) or {})
        
        # Handle net_arch mapping and other policy kwargs
        for key in ("net_arch", "log_std_init", "noisy_net"):
            if key in params and key not in pk:
                val = params.pop(key)
                # Convert net_arch string identifier to list
                if key == "net_arch" and isinstance(val, str) and val in NET_ARCH_MAP:
                    val = NET_ARCH_MAP[val]
                # Convert tuple to list (fallback for legacy usage)
                elif key == "net_arch" and isinstance(val, tuple):
                    val = list(val)
                pk[key] = val
        
        if pk:
            params["policy_kwargs"] = pk
        return params

    def _logr(info, last_eq):
        """Extract log return from info dict."""
        if isinstance(info, list) and info:
            info = info[0]
        if not isinstance(info, dict):
            return 0.0, last_eq
        if "log_return" in info and np.isfinite(info["log_return"]):
            return float(info["log_return"]), info.get("equity", last_eq)
        if "step_return" in info and np.isfinite(info["step_return"]):
            try:
                return math.log1p(float(info["step_return"])), info.get("equity", last_eq)
            except Exception:
                pass
        if "equity" in info and last_eq not in (None, 0):
            try:
                eq = float(info["equity"]) if np.isfinite(info["equity"]) else last_eq
                if eq and last_eq:
                    return math.log(max(eq, 1e-12) / max(float(last_eq), 1e-12)), eq
            except Exception:
                pass
        return 0.0, info.get("equity", last_eq) if isinstance(info, dict) else last_eq

    # In-memory history cache
    history_cache: Dict[int, list] = {}

    # Optuna objective
    def objective(trial: optuna.Trial) -> float:
        # 1) Sample hyperparameters
        params = _suggest_params(trial, param_space)
        params = _pack_policy_kwargs(params)

        # 2) Per-trial directory
        tdir = study_dir / f"trial_{trial.number:03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        print(f"[Trial {trial.number}] Log directory: {tdir}")
        print(f"[Trial {trial.number}] Hyperparameters: {json.dumps(params, indent=2, ensure_ascii=False)}")

        # 3) Build environment (single DummyVecEnv with one env instance)
        from utils.env import UniswapV3LiquidityEnv as Env

        def _make_env():
            e = Env(**env_kw)
            try:
                e.reset(seed=seed)
            except TypeError:
                pass
            return e

        train_env = DummyVecEnv([_make_env])

        # 4) Fix PPO batch divisibility if needed
        if Algo is PPO:
            n_envs = 1  # DummyVecEnv with single env
            n_steps = params.get("n_steps")
            bs = params.get("batch_size")
            if n_steps and bs and (n_steps * n_envs) % bs != 0:
                # Pick the nearest feasible from a small set
                for cand in [64, 128, 256, 512, 1024]:
                    if (n_steps * n_envs) % cand == 0:
                        params["batch_size"] = cand
                        print(f"[Trial {trial.number}] Adjusted batch_size to {cand} for divisibility")
                        break

        # 5) Instantiate algorithm
        model = Algo(
            policy="MlpPolicy",
            env=train_env,
            seed=seed,
            verbose=verbose,
            **params,
        )

        # 6) Logger
        logger = sb3_configure_logger(str(tdir), ["stdout", "csv"])
        model.set_logger(logger)

        # 7) Training loop with inline callback
        steps = 0
        best_metric = -np.inf if direction == "maximize" else np.inf
        history = []
        ep_sum_lr = 0.0
        last_eq = None
        recent_episode_sums: list[float] = []
        episode_rewards: list[float] = []
        current_episode_reward = 0.0
        cumulative_reward = 0.0
        cumulative_log_return = 0.0

        def _callback(locals_, globals_):
            nonlocal steps, ep_sum_lr, last_eq, best_metric, current_episode_reward
            nonlocal cumulative_reward, cumulative_log_return
            steps += 1
            infos = locals_.get("infos") or []
            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else infos
            
            # Get reward from this step
            rewards = locals_.get("rewards")
            if rewards is not None:
                step_reward = 0.0
                if isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) > 0:
                    step_reward = float(rewards[0])
                elif np.isscalar(rewards):
                    step_reward = float(rewards)
                current_episode_reward += step_reward
                cumulative_reward += step_reward

            if info is not None:
                lr, last_eq = _logr(info, last_eq)
                ep_sum_lr += lr
                cumulative_log_return += lr
                
                # Detect done
                done_flags = locals_.get("dones")
                done = False
                if isinstance(done_flags, (list, tuple)) and len(done_flags) > 0:
                    done = bool(done_flags[0])
                elif isinstance(done_flags, np.ndarray):
                    done = bool(done_flags[0])
                elif isinstance(done_flags, (bool, np.bool_)):
                    done = bool(done_flags)
                if done:
                    recent_episode_sums.append(ep_sum_lr)
                    episode_rewards.append(current_episode_reward)
                    
                    ep_sum_lr = 0.0
                    current_episode_reward = 0.0
                    last_eq = None

            # Record custom metrics every step - SB3 will dump them when it logs training metrics
            try:
                # Cumulative metrics (always available)
                model.logger.record("custom/cumulative_reward", cumulative_reward)
                model.logger.record("custom/cumulative_log_return", cumulative_log_return)
                model.logger.record("custom/total_steps", steps)
                
                # Episode-based metrics (when available)
                if len(episode_rewards) > 0:
                    model.logger.record("custom/total_episodes", len(episode_rewards))
                    model.logger.record("custom/mean_episode_reward", np.mean(episode_rewards[-10:]))
                    model.logger.record("custom/latest_episode_reward", episode_rewards[-1])
                if len(recent_episode_sums) > 0:
                    model.logger.record("custom/mean_log_return", np.mean(recent_episode_sums[-10:]))
                    model.logger.record("custom/latest_log_return", recent_episode_sums[-1])
            except Exception:
                pass

            # Force dump every 100 steps for off-policy algorithms (DQN/SAC/TD3)
            # PPO will dump automatically based on its rollout schedule
            if Algo in {SAC, DQN, TD3} and steps % 100 == 0:
                try:
                    model.logger.dump(step=steps)
                except Exception:
                    pass

            if steps % eval_every != 0 or len(recent_episode_sums) < max(1, int(eval_episodes)):
                return True

            k = int(eval_episodes)
            window = recent_episode_sums[-k:]
            metric = float(np.mean(window))
            loss   = -metric if direction == "maximize" else metric
            value  = metric if direction == "maximize" else -metric

            trial.report(value, step=steps)
            history.append({
                "step": int(steps),
                "sum_log_return": metric,
                "loss": loss,
                "mean_episode_reward": float(np.mean(episode_rewards[-k:])) if len(episode_rewards) >= k else None,
            })

            improved = (metric > best_metric) if direction == "maximize" else (metric < best_metric)
            if improved:
                best_metric = metric
                try:
                    model.save(str(tdir / "best_model.zip"))
                    print(f"[Trial {trial.number}] Step {steps}: New best metric = {metric:.6f}")
                except Exception:
                    pass

            if trial.should_prune():
                print(f"[Trial {trial.number}] Pruned at step {steps}")
                raise TrialPruned()
            return True

        try:
            model.learn(total_timesteps=total_timesteps, callback=_callback, progress_bar=False)
        except TrialPruned:
            with open(tdir / "params.json", "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            with open(tdir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"history": history, "pruned": True}, f, indent=2, ensure_ascii=False)
            history_cache[trial.number] = history
            raise
        finally:
            try:
                train_env.close()
            except Exception:
                pass

        # 8) Final metric
        if len(recent_episode_sums) >= max(1, int(eval_episodes)):
            k = int(eval_episodes)
            final_metric = float(np.mean(recent_episode_sums[-k:]))
        elif len(recent_episode_sums) > 0:
            final_metric = float(np.mean(recent_episode_sums))
        else:
            final_metric = float(best_metric) if np.isfinite(best_metric) else 0.0

        loss  = -final_metric if direction == "maximize" else final_metric
        value  =  final_metric if direction == "maximize" else -final_metric

        with open(tdir / "params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        with open(tdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "history": history,
                "final": {
                    "sum_log_return": float(final_metric),
                    "loss": float(loss),
                    "objective_value": float(value),
                    "mean_episode_reward": float(np.mean(episode_rewards[-eval_episodes:])) if len(episode_rewards) >= eval_episodes else None,
                    "total_episodes": len(episode_rewards),
                    "cumulative_reward": float(cumulative_reward),
                    "cumulative_log_return": float(cumulative_log_return),
                },
            }, f, indent=2, ensure_ascii=False)
        try:
            model.save(str(tdir / "final_model.zip"))
        except Exception:
            pass

        print(f"[Trial {trial.number}] Completed! Final metric = {final_metric:.6f}")
        history_cache[trial.number] = history
        return value

    # Run study
    print(f"\n{'='*60}")
    print(f"Starting Optuna Hyperparameter Tuning: {algo_name.upper()}")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"Total timesteps per trial: {total_timesteps:,}")
    print(f"Evaluation frequency: every {eval_every:,} steps")
    print(f"Direction: {direction}")
    print(f"Output directory: {study_dir}")
    print(f"{'='*60}\n")
    
    study.optimize(objective, n_trials=int(n_trials), gc_after_trial=True, show_progress_bar=False)

    # Post-study summary
    import pandas as pd
    df = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "datetime_start", "datetime_complete")
    )

    extra_rows = []
    for t in study.trials:
        tdir = study_dir / f"trial_{t.number:03d}"
        hist = history_cache.get(t.number, [])
        extra_rows.append({
            "number": t.number,
            "trial_dir": str(tdir),
            "last_sum_log_return": hist[-1]["sum_log_return"] if hist else None,
            "last_loss": hist[-1]["loss"] if hist else None,
        })
    try:
        df = df.merge(pd.DataFrame(extra_rows), on="number", how="left")
    except Exception:
        pass

    summary_csv = study_dir / "study_summary.csv"
    df.to_csv(summary_csv, index=False)

    # Copy best model to study root
    best_t  = study.best_trial
    bestdir = study_dir / f"trial_{best_t.number:03d}"
    src = bestdir / "best_model.zip"
    if not src.exists():
        alt = bestdir / "final_model.zip"
        if alt.exists():
            src = alt
    dst = study_dir / "best_model.zip"
    try:
        shutil.copyfile(src, dst)
    except Exception:
        pass

    result = {
        "best_params": best_t.params,
        "best_value": float(best_t.value),
        "best_trial": int(best_t.number),
        "best_model_path": str(dst if dst.exists() else src),
        "study_dir": str(study_dir),
        "summary_csv": str(summary_csv),
    }

    print("\n" + "="*60)
    print("==== Optuna Hyperparameter Tuning Completed ====")
    print("="*60)
    print(f"Study directory: {study_dir}")
    print(f"Best trial: {result['best_trial']} | value={result['best_value']:.6f}")
    print(f"Best parameters:\n{json.dumps(result['best_params'], indent=2, ensure_ascii=False)}")
    print(f"Best model: {result['best_model_path']}")
    print(f"Summary CSV: {result['summary_csv']}")
    print("="*60 + "\n")
    
    return result

