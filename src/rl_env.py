"""
Gymnasium-based cryptocurrency trading environment and PPO training helpers.

The environment simulates a simple spot-trading agent that can Buy, Hold,
or Sell on each time-step.  Reward is the change in portfolio value,
accounting for transaction costs and random slippage.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    """Discrete-action crypto trading environment.

    Actions
    -------
    0 - Hold
    1 - Buy  (spend all available cash)
    2 - Sell (liquidate entire position)

    Observation
    -----------
    Current row of feature columns from the supplied DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (+ optional indicators).  Must contain a ``close`` column.
    initial_balance : float
        Starting cash.
    transaction_cost : float
        Proportional fee per trade (e.g. 0.001 = 0.1 %).
    max_slippage : float
        Maximum random slippage applied to execution price.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.001,
        max_slippage: float = 0.005,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_slippage = max_slippage

        self.action_space = spaces.Discrete(3)
        n_features = len(self.df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance

    def _get_obs(self) -> np.ndarray:
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def _current_price(self) -> float:
        return float(self.df.iloc[self.current_step]["close"])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        price = self._current_price()
        slippage = self.np_random.uniform(-self.max_slippage, self.max_slippage)

        if action == 1 and self.balance > 0:
            effective_price = price * (1 + slippage)
            cost_per_share = effective_price * (1 + self.transaction_cost)
            shares_bought = self.balance / cost_per_share
            self.balance -= shares_bought * effective_price
            self.shares_held += shares_bought

        elif action == 2 and self.shares_held > 0:
            effective_price = price * (1 - slippage)
            revenue = self.shares_held * effective_price * (1 - self.transaction_cost)
            self.balance += revenue
            self.shares_held = 0.0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        new_price = self._current_price()
        self.portfolio_value = self.balance + self.shares_held * new_price
        reward = self.portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = self.portfolio_value

        obs = self._get_obs()
        info = {
            "balance": self.balance,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value,
        }

        return obs, reward, done, False, info

    def render(self) -> None:
        print(
            f"Step {self.current_step:>6d} | "
            f"Portfolio ${self.portfolio_value:,.2f} | "
            f"Cash ${self.balance:,.2f} | "
            f"Shares {self.shares_held:.6f}"
        )


def train_ppo(
    df: pd.DataFrame,
    total_timesteps: int = 100_000,
    initial_balance: float = 10_000.0,
    verbose: int = 1,
):
    """Train a PPO agent on the trading environment.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched OHLCV data.
    total_timesteps : int
    initial_balance : float
    verbose : int

    Returns
    -------
    model : stable_baselines3.PPO
        Trained PPO model.
    env : gymnasium.Env
    """
    from stable_baselines3 import PPO

    env = CryptoTradingEnv(df, initial_balance=initial_balance)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    model.learn(total_timesteps=total_timesteps)
    return model, env


def backtest(model, env: CryptoTradingEnv) -> pd.DataFrame:
    """Run the trained agent through the full dataset and collect results.

    Returns a DataFrame with columns: ``step``, ``action``,
    ``portfolio_value``, ``balance``, ``shares_held``.
    """
    obs, _ = env.reset()
    records = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        records.append({
            "step": env.current_step,
            "action": int(action),
            "portfolio_value": info["portfolio_value"],
            "balance": info["balance"],
            "shares_held": info["shares_held"],
        })

    return pd.DataFrame(records)
