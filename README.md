# Introduction
---
This repository contains the source code and configurations for our submission—**"Hybrid DRL and Market Regime Identification for Optimal Uniswap V3 Liquidity Provision."** to the  [**FinAI Contest 2025 Task 3 — FinRL-DeFi**](https://github.com/Open-Finance-Lab/FinAI_Contest_2025/tree/main/Task_3_FinRL_DeFi#task-3--finrl-defi).

We propose a hybrid method that combines market regime detection via K-Means clustering and deep reinforcement learning (DRL) to optimize LP strategies in Uniswap V3. Our method dynamically adjusts liquidity positions based on latent market regimes derived from historical data.

# Key Features
---
- Market regime detection using unsupervised clustering
- DRL agent implementation using PPO and DQN (via ElegantRL)
- Reward function based on hedged PnL (accounting for fees, LVR, gas)
- Regime-aware state space design

# Repository Structure and Usage
---
- `config/` – Pool related parameters
- `data/` – Sample preprocessed features and regime labels
- `data_request/` – Scripts and notebooks (`.py`, `.ipynb`) for downloading, preprocessing, transforming to features and some analysis.
- `experiments/` – Jupyter notebooks for training and evaluation experiments.
- `utilis/` –miscellaneous utility modules that support environment construction, evaluation, hyperparameters tuning and Uniswap-related computation.
