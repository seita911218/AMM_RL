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

# Repository Structure
---

- `config/` – Pool related parameters
- `data/` – Sample preprocessed features and regime labels
- `data_request/` – Scripts and notebooks (`.py`, `.ipynb`) for downloading, preprocessing, transforming to features and some analysis.
- `experiments/` – Jupyter notebooks for training and evaluation experiments.
- `utilis/` –miscellaneous utility modules that support environment construction, evaluation, hyperparameters tuning and Uniswap-related computation.

# Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/seita911218/AMM_RL.git
    ```

2. **(Recommended) Create a virtual environment**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

# Usage

## 1. Data Downloading and Processing

- Use scripts in the `data_request/` directory for data acquisition and feature engineering:
    - `data_downloading.py`: Download raw data
    - `data_processing.py`: Clean and preprocess data
    - `features_clustering.py`: Feature engineering and clustering
    - `organize_data.py`: Organize data format

- You can also use the Jupyter Notebooks (`data.ipynb`, `feature.ipynb`) for interactive processing.

### 2. Configuration

- Edit `config/config.json` to set pool parameters, tokens, fee tier, etc.

### 3. Reinforcement Learning Environment

- The main environment is implemented in `utils/env.py` and can be imported as follows:
    ```python
    from utils.env import UniswapV3LiquidityEnv
    ```

- Supporting functions are in `utils/uniswap.py`, `utils/pnl.py`, `utils/eval.py`, and `utils/visualize.py`.

### 4. Experiments and Backtesting

- Use `experiments/experiments.ipynb` as a workflow example, including environment setup, strategy training, backtesting and visualization.  At the end of `data_request/feature.ipynb`, we provide a simple analysis of a sample RL result as a demonstration. You can refer to this notebook for an example of how to interpret and visualize RL agent performance.


# Requirements

See `requirements.txt` for all dependencies. 


