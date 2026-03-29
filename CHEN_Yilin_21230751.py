"""
Compare three portfolio strategies:
  - solve_portfolio: marginal-utility allocation (optimal_adjustment_with_budget)
  - solve_portfolio_simple: simple proportional scaling toward Merton target
  - MultiAssetDDPG: defined in this file (requires PyTorch when run_ddpg=True)

Usage:
  from compare_strategies_three import compare_strategies_three, covariance_matrix
  res = compare_strategies_three(
      r, mu, cov, p0, W0, a, delta, T,
      ddpg_reward_mode="potential",  # or 'sparse' (default when omitted)
      ddpg_options={"num_episodes": 500},
  )
  res["best_path"]  # path of the strategy with highest terminal wealth; tables print if print_paths=True

Aliases:
  compare_three_strategies = compare_strategies_three
  compare_three_pick_best(...)  # optional extra summary block; defaults to plot=False

If you paste this file into a Jupyter cell and imports fail, prepend:
  from pathlib import Path
  sys.path.insert(0, str(Path.home() / "Desktop" / "MAFS5370"))

------------------------------------------------------------------------------
compare_strategies_three — parameters
------------------------------------------------------------------------------
r : float
    Risk-free rate (per period).
mu : array_like
    Expected excess or gross returns per risky asset (shape matches cov).
cov : ndarray
    Covariance matrix of risky assets.
p0 : array_like
    Initial portfolio weights including cash: [cash, asset1, asset2, ...]; must sum to 1.
W0 : float
    Initial wealth.
a : float
    CARA risk aversion (>0).
delta : float
    Max fraction of wealth adjustable per asset per period (constraint scale).
T : int
    Number of periods.
verbose : bool, default False
    If True: print progress lines (Running marginal / simple / DDPG), DDPG training logs, and
    when print_paths is True or \"best\"/\"all\": also print the setup block and unconstrained
    Merton holdings before the simulated path(s). If False: suppress that chatter and DDPG
    stdout during train/simulate; period-by-period path still prints whenever print_paths
    requests it (except print_paths=False). Ignored for print_paths=\"compact\".
plot : bool, default True
    If True, show matplotlib wealth paths at the end (two or three lines).
run_ddpg : bool, default True
    If False, skip DDPG (no PyTorch); only marginal vs simple Merton.
ddpg_options : dict or None
    Hyperparameters passed to MultiAssetDDPG (e.g. num_episodes, learning_rate_actor,
    learning_rate_critic, reward_mode, gamma). None uses built-in defaults.
    Aliases: ``learning_rate`` sets both actor and critic LRs; ``lr_actor`` / ``actor_lr`` and
    ``lr_critic`` / ``critic_lr`` map to the two optimizers. Unknown keys are dropped with a
    warning (they are not forwarded), so typos do not become TypeError.
ddpg_solver : MultiAssetDDPG or None
    Pre-trained solver: if given, training is skipped and only simulate() is run.
ddpg_module_path : str or None
    Optional absolute path to an alternate multi_asset_ddpg.py (legacy override); the
    default implementation uses MultiAssetDDPG in this file.
ddpg_reward_mode : str or None, default None
    DDPG reward: ``\"sparse\"`` (terminal CARA only) or ``\"potential\"`` (potential-based
    shaping). If None, use ``ddpg_options[\"reward_mode\"]`` when given, else ``\"sparse\"``.
    If not None, this value overrides ``ddpg_options[\"reward_mode\"]``.
print_paths : bool or str, default True
    Console output style. True or \"best\": period-by-period \"Final simulated path summary\"
    for the best terminal-wealth strategy; if verbose is also True, prepend setup + unconstrained
    Merton block. \"all\": detailed path for each strategy; if verbose, prepend setup once.
    \"compact\": legacy one-line-per-period table for the best path only. False: suppress path
    printing (paths still in res).
    print_comparison_table : bool or None, default None
    If True: print the three-way CARA / return summary block.
    If False: never print it.
    If None (default): omit that block whenever print_paths is truthy (best-path workflow stays clean);
    show it when print_paths is False. Override with True/False as needed.
    Note: print_paths=\"compact\" is truthy for this rule; use print_comparison_table=True to
    force the three-way table when using compact path output.

------------------------------------------------------------------------------
compare_three_pick_best — extra parameters (forwards the rest to compare_strategies_three)
------------------------------------------------------------------------------
print_summary : bool, default True
    If True, print a short block listing terminal wealth per strategy and the winner.
print_paths : bool or str
    Same as compare_strategies_three (default True there).
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
import warnings
from collections import deque
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]


# -------------------------------------------------------------
# 1. Helpers
# -------------------------------------------------------------
def covariance_matrix(var, covar=None):
    """Build n x n covariance from variances and optional {(i,j): cov} for i<j."""
    n = len(var)
    cov = np.zeros((n, n))
    for i in range(n):
        cov[i, i] = var[i]
    if covar is not None:
        for (i, j), val in covar.items():
            cov[i, j] = val
            cov[j, i] = val
    return cov


def compute_marginal_utility(mu, cov, pi, W, a, r):
    n = len(mu)
    marginal = np.zeros(n)
    for i in range(n):
        cov_part = np.dot(cov[i, :], pi)
        marginal[i] = (mu[i] - r) - a * cov_part
    return marginal


def cara_terminal_utility(W, a):
    """Terminal CARA utility U(W) = -exp(-a*W)/a for a > 0."""
    W = float(W)
    if a == 0:
        return W
    return -np.exp(-a * W) / a


def potential_shaping_identity():
    """势函数整形：在 r 上加 F=γΦ(s')-Φ(s)。对任意 Φ，最优策略与仅稀疏终端奖励时一致（Ng et al., ICML 1999）。"""
    return (
        "在 r 上增加 F(s,s')=γΦ(s')-Φ(s) 时，若 Φ 为状态势函数，则最优策略不变；"
        "总折扣回报与原来相差与 Φ(s0)、γ^TΦ(sT) 相关的边界项。"
    )


MultiAssetDDPG = None  # type: ignore[misc, assignment]

if torch is not None:

    class ReplayBuffer:
        """经验回放池；可选终端转移加权采样。"""

        def __init__(self, capacity=10000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size, terminal_boost=0.0):
            buf = list(self.buffer)
            n = len(buf)
            if n < batch_size:
                raise ValueError("Replay buffer smaller than batch size")
            if terminal_boost <= 0:
                batch = random.sample(buf, batch_size)
            else:
                term_idx = [i for i in range(n) if buf[i][4]]
                n_term_want = min(len(term_idx), int(batch_size * terminal_boost))
                idx = set()
                if n_term_want > 0 and term_idx:
                    idx.update(random.sample(term_idx, n_term_want))
                need = batch_size - len(idx)
                if need > 0:
                    pool = [i for i in range(n) if i not in idx]
                    idx.update(random.sample(pool, min(need, len(pool))))
                if len(idx) < batch_size:
                    pool = [i for i in range(n) if i not in idx]
                    if pool:
                        idx.update(random.sample(pool, batch_size - len(idx)))
                batch = [buf[i] for i in idx]
            state, action, reward, next_state, done = zip(*batch)
            return (
                np.array(state),
                np.array(action),
                np.array(reward),
                np.array(next_state),
                np.array(done),
            )

        def __len__(self):
            return len(self.buffer)

    class Actor(nn.Module):
        """策略网络：输入状态 (wealth, old_alloc)，输出调整向量 Δπ (N维)"""

        def __init__(self, state_dim, action_dim, hidden_layers=(256, 256)):
            super().__init__()
            layers = []
            layers.append(nn.Linear(state_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_layers[-1], action_dim))
            layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)

        def forward(self, state):
            return self.net(state)

    class Critic(nn.Module):
        """Q网络：输入 (state, action)，输出 Q值"""

        def __init__(self, state_dim, action_dim, hidden_layers=(256, 256)):
            super().__init__()
            layers = []
            layers.append(nn.Linear(state_dim + action_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_layers[-1], 1))
            self.net = nn.Sequential(*layers)

        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            return self.net(x)

    class MultiAssetDDPG:
        """
        DDPG 多资产配置；调整约束与 solve_portfolio_simple 一致：
        每资产 |Δπ_i|≤δW，且 sum_i|Δπ_i|+|∑Δπ_i|≤2δW，再截断非负与风险资产合计≤W。
        reward_mode="sparse"：仅期末 CARA；"potential"：势函数整形 r+γΦ(s')-Φ(s)（Ng et al.）。
        """

        def __init__(
            self,
            r,
            mu,
            Sigma,
            a,
            delta,
            T,
            p0=None,
            W_min=0.5,
            W_max=5.0,
            hidden_layers=(256, 256),
            learning_rate_actor=1e-4,
            learning_rate_critic=1e-3,
            gamma=1.0,
            tau=0.005,
            buffer_capacity=100000,
            batch_size=256,
            num_episodes=1000,
            eval_interval=50,
            eval_episodes=10,
            max_steps_per_episode=100,
            updates_per_episode=10,
            lr_plateau_factor=0.5,
            lr_plateau_min_lr=1e-6,
            lr_plateau_patience=None,
            early_stop_k=None,
            early_stop_epsilon=None,
            terminal_batch_fraction=0.25,
            reward_mode="sparse",
        ):
            self.r = r
            self.mu = np.asarray(mu)
            self.Sigma = np.asarray(Sigma)
            self.a = a
            self.delta = delta
            self.T = T
            self.N = len(mu)
            self.p0 = np.asarray(p0) if p0 is not None else None
            self.W_min = W_min
            self.W_max = W_max

            risk_premium = self.mu - r
            try:
                Sigma_inv = np.linalg.inv(self.Sigma)
            except np.linalg.LinAlgError:
                Sigma_inv = np.linalg.pinv(self.Sigma)
            self.myopic = (1.0 / a) * Sigma_inv @ risk_premium

            n_quad = 5
            x, w = np.polynomial.hermite.hermgauss(n_quad)
            grids = np.meshgrid(*[x] * self.N, indexing="ij")
            self.z = np.column_stack([g.ravel() for g in grids])
            self.quad_weights = np.prod(
                np.meshgrid(*[w] * self.N, indexing="ij"), axis=0
            ).ravel()
            self.L = np.linalg.cholesky(self.Sigma)
            self.R_minus_r = (self.mu - r) + self.z @ self.L.T

            self.state_dim = 1 + self.N
            self.action_dim = self.N
            self.gamma = gamma
            self.tau = tau
            self.batch_size = batch_size
            self.num_episodes = num_episodes
            self.eval_interval = eval_interval
            self.eval_episodes = eval_episodes
            self.max_steps_per_episode = max_steps_per_episode
            self.updates_per_episode = updates_per_episode
            self.lr_plateau_factor = lr_plateau_factor
            self.lr_plateau_min_lr = lr_plateau_min_lr
            self.lr_plateau_patience = (
                lr_plateau_patience
                if lr_plateau_patience is not None
                else max(2, num_episodes // 10)
            )
            self.early_stop_k = early_stop_k
            self.early_stop_epsilon = early_stop_epsilon
            self.terminal_batch_fraction = terminal_batch_fraction
            self.reward_mode = reward_mode

            self.actor = Actor(self.state_dim, self.action_dim, hidden_layers)
            self.actor_target = Actor(self.state_dim, self.action_dim, hidden_layers)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic = Critic(self.state_dim, self.action_dim, hidden_layers)
            self.critic_target = Critic(self.state_dim, self.action_dim, hidden_layers)
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.actor_optimizer = optim.Adam(
                self.actor.parameters(), lr=learning_rate_actor
            )
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=learning_rate_critic
            )

            self.replay_buffer = ReplayBuffer(buffer_capacity)

            self.noise_std = 0.2
            self.noise_decay = 0.995

        @staticmethod
        def _actor_state_cpu_clone(actor):
            """Best actor weights in RAM only (no torch.save / disk path)."""
            return {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}

        def _state_to_tensor(self, wealth, old_alloc):
            state = np.concatenate([[wealth], old_alloc])
            return torch.FloatTensor(state).unsqueeze(0)

        def _project_action(self, action, wealth, old_alloc):
            """
            与 solve_portfolio_simple 一致：
            每资产 |Δπ_i|≤δW；总度量 sum|Δπ_i|+|∑Δπ_i|≤2δW；再非负且风险资产合计≤W。
            """
            W = float(wealth)
            max_pa = self.delta * W
            max_tot = 2.0 * self.delta * W
            a = np.asarray(action, dtype=float).reshape(-1)
            if a.size < self.N:
                a = np.pad(a, (0, self.N - a.size))
            delta_pi = a[: self.N] * max_pa
            measure = np.sum(np.abs(delta_pi)) + np.abs(np.sum(delta_pi))
            if measure > max_tot + 1e-8:
                delta_pi = delta_pi * (max_tot / measure)
            oa = np.asarray(old_alloc, dtype=float).reshape(-1)
            if oa.size < self.N:
                oa = np.pad(oa, (0, self.N - oa.size))
            new_alloc = oa[: self.N] + delta_pi
            new_alloc = np.clip(new_alloc, 0.0, None)
            delta_pi = new_alloc - oa[: self.N]
            s = float(np.sum(new_alloc))
            if s > W + 1e-8:
                new_alloc = new_alloc * (W / s)
                delta_pi = new_alloc - oa[: self.N]
            return delta_pi, new_alloc

        def _project_delta_torch(self, raw, wealth, old_alloc):
            """与 _project_action 相同规则（批处理）。"""
            max_pa = self.delta * wealth
            max_tot = 2.0 * self.delta * wealth
            delta_pi = raw * max_pa
            measure = torch.sum(torch.abs(delta_pi), dim=1, keepdim=True) + torch.abs(
                torch.sum(delta_pi, dim=1, keepdim=True)
            )
            scale_tot = torch.where(
                measure > max_tot + 1e-8,
                max_tot / (measure + 1e-12),
                torch.ones_like(measure),
            )
            delta_pi = delta_pi * scale_tot
            new_alloc = old_alloc + delta_pi
            new_alloc = torch.clamp(new_alloc, min=0.0)
            delta_pi = new_alloc - old_alloc
            s = torch.sum(new_alloc, dim=1, keepdim=True)
            scale_w = torch.where(
                s > wealth + 1e-8,
                wealth / (s + 1e-12),
                torch.ones_like(s),
            )
            new_alloc = new_alloc * scale_w
            delta_pi = new_alloc - old_alloc
            return delta_pi

        def _get_action(self, state, add_noise=False):
            wealth, old_alloc = state[0], state[1:]
            state_tensor = self._state_to_tensor(wealth, old_alloc)
            with torch.no_grad():
                raw_action = self.actor(state_tensor).cpu().numpy().flatten()
            if add_noise:
                raw_action += np.random.normal(0, self.noise_std, size=self.action_dim)
                raw_action = np.clip(raw_action, -1, 1)
            delta_pi, new_alloc = self._project_action(raw_action, wealth, old_alloc)
            return delta_pi, new_alloc

        def _utility_wealth(self, W):
            W = float(W)
            if self.a == 0:
                return W
            return -np.exp(-self.a * W) / self.a

        def _potential_state(self, state):
            return self._utility_wealth(state[0])

        def _compute_reward_sparse_terminal(self, next_wealth, t):
            if t == self.T - 1:
                return self._utility_wealth(next_wealth)
            return 0.0

        def _compute_reward_potential_shaping(self, state, next_state, next_wealth, t):
            r_sparse = self._compute_reward_sparse_terminal(next_wealth, t)
            phi_s = self._potential_state(state)
            phi_sp = self._potential_state(next_state)
            return r_sparse + self.gamma * phi_sp - phi_s

        def _compute_reward(self, state, next_state, wealth, new_alloc, next_wealth, t):
            if self.reward_mode == "potential":
                return self._compute_reward_potential_shaping(
                    state, next_state, next_wealth, t
                )
            return self._compute_reward_sparse_terminal(next_wealth, t)

        def _sample_transition(self):
            if self.p0 is not None:
                initial_wealth = 1.0
                asset_shares = self.p0[1:]
                old_alloc = asset_shares * initial_wealth
            else:
                initial_wealth = np.random.uniform(self.W_min, self.W_max)
                old_alloc = np.random.uniform(
                    0.0, max(1e-6, initial_wealth), size=self.N
                )
            wealth = initial_wealth

            for t in range(self.T):
                state = np.concatenate([[wealth], old_alloc])
                delta_pi, new_alloc = self._get_action(state, add_noise=True)
                W_next = wealth * (1 + self.r) + new_alloc @ (self.mu - self.r)
                next_state = np.concatenate([[W_next], new_alloc])
                reward = self._compute_reward(
                    state, next_state, wealth, new_alloc, W_next, t
                )
                done = t == self.T - 1
                self.replay_buffer.push(state, delta_pi, reward, next_state, done)
                wealth = W_next
                old_alloc = new_alloc

        def _update_networks(self):
            if len(self.replay_buffer) < self.batch_size:
                return None
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size, terminal_boost=self.terminal_batch_fraction
            )

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            with torch.no_grad():
                raw_next = self.actor_target(next_states)
                next_delta = self._project_delta_torch(
                    raw_next, next_states[:, 0:1], next_states[:, 1:]
                )
                next_q = self.critic_target(next_states, next_delta)
                target_q = rewards + self.gamma * (1 - dones) * next_q

            current_q = self.critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            raw_act = self.actor(states)
            delta_pol = self._project_delta_torch(
                raw_act, states[:, 0:1], states[:, 1:]
            )
            actor_loss = -self.critic(states, delta_pol).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(
                self.actor_target.parameters(), self.actor.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for target_param, param in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            return critic_loss.item(), actor_loss.item()

        def _evaluate_policy(self):
            if self.p0 is None:
                return 0.0
            total_return = 0.0
            for _ in range(self.eval_episodes):
                wealth = 1.0
                asset_shares = self.p0[1:]
                old_alloc = asset_shares * wealth
                for t in range(self.T):
                    state = np.concatenate([[wealth], old_alloc])
                    _, new_alloc = self._get_action(state, add_noise=False)
                    W_next = wealth * (1 + self.r) + new_alloc @ (self.mu - self.r)
                    if t == self.T - 1:
                        total_return += self._utility_wealth(W_next)
                    wealth = W_next
                    old_alloc = new_alloc
            return total_return / self.eval_episodes

        def solve(self):
            """ReduceLROnPlateau（Actor/Critic）；可选 loss 早停。"""
            print("Training DDPG ...")
            sched_c = optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer,
                mode="min",
                factor=self.lr_plateau_factor,
                patience=self.lr_plateau_patience,
                min_lr=self.lr_plateau_min_lr,
            )
            sched_a = optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer,
                mode="min",
                factor=self.lr_plateau_factor,
                patience=self.lr_plateau_patience,
                min_lr=self.lr_plateau_min_lr,
            )
            best_return = -np.inf
            best_actor_state = self._actor_state_cpu_clone(self.actor)

            use_loss_es = (
                self.early_stop_k is not None
                and self.early_stop_epsilon is not None
                and self.early_stop_k > 0
            )
            prev_ep_loss = None
            loss_stable = 0
            last_mean_c = None
            last_mean_a = None

            for episode in range(self.num_episodes):
                self._sample_transition()
                sum_c, sum_a, n_u = 0.0, 0.0, 0
                for _ in range(self.updates_per_episode):
                    out = self._update_networks()
                    if out is not None:
                        c, a = out
                        sum_c += c
                        sum_a += a
                        n_u += 1
                if n_u > 0:
                    mean_c = sum_c / n_u
                    mean_a = sum_a / n_u
                    last_mean_c = mean_c
                    last_mean_a = mean_a
                    sched_c.step(mean_c)
                    sched_a.step(mean_a)
                    ep_loss = mean_c + abs(mean_a)
                    if use_loss_es and prev_ep_loss is not None:
                        if abs(ep_loss - prev_ep_loss) < self.early_stop_epsilon:
                            loss_stable += 1
                            if loss_stable >= self.early_stop_k:
                                print(
                                    f"Early stop at episode {episode}: "
                                    f"{self.early_stop_k} consecutive |Δepisode_loss| < "
                                    f"{self.early_stop_epsilon}"
                                )
                                break
                        else:
                            loss_stable = 0
                    prev_ep_loss = ep_loss

                if episode % self.eval_interval == 0:
                    avg_return = self._evaluate_policy()
                    lr_a = self.actor_optimizer.param_groups[0]["lr"]
                    lr_c = self.critic_optimizer.param_groups[0]["lr"]
                    print(
                        f"Episode {episode}, utility: {avg_return:.4f}, "
                        f"lr_actor: {lr_a:.2e}, lr_critic: {lr_c:.2e}"
                    )
                    if last_mean_c is not None:
                        ep_loss = last_mean_c + abs(last_mean_a)
                        print(
                            f"  loss: critic={last_mean_c:.6f}, "
                            f"actor={last_mean_a:.6f}, ep_loss={ep_loss:.6f}"
                        )
                    else:
                        print("  loss: (skipped, replay buffer < batch_size)")
                    if avg_return > best_return:
                        best_return = avg_return
                        best_actor_state = self._actor_state_cpu_clone(self.actor)
                self.noise_std *= self.noise_decay

            print("Training finished.")
            self.actor.load_state_dict(best_actor_state)

        def get_allocation(self, wealth, old_alloc):
            state = np.concatenate([[wealth], old_alloc])
            _, new_alloc = self._get_action(state, add_noise=False)
            return new_alloc

        def simulate(self, W0, p0, returns=None):
            W = W0
            asset_shares = p0[1:]
            alloc_old = asset_shares * W
            cash_old = W - np.sum(alloc_old)

            path = []
            for t in range(self.T):
                path.append(
                    {
                        "t": t,
                        "wealth": W,
                        "cash_dollar": cash_old,
                        "cash_share": cash_old / W if W > 0 else 0.0,
                        "holdings_dollar": alloc_old.copy(),
                        "holdings_share": alloc_old / W
                        if W > 0
                        else np.zeros_like(alloc_old),
                    }
                )

                alloc_new = self.get_allocation(W, alloc_old)

                path[t]["new_holdings_dollar"] = alloc_new.copy()
                path[t]["new_holdings_share"] = (
                    alloc_new / W if W > 0 else np.zeros_like(alloc_new)
                )

                if returns is None:
                    ret_risky = self.mu
                else:
                    ret_risky = returns[t]
                W_next = W * (1 + self.r) + alloc_new @ (ret_risky - self.r)

                W = W_next
                alloc_old = alloc_new
                cash_old = W - np.sum(alloc_old)

            return path


def _exec_ddpg_module(path: Path):
    """Load multi_asset_ddpg from file; raise ImportError with hints if torch etc. is missing."""
    spec = importlib.util.spec_from_file_location("multi_asset_ddpg", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        hint = ""
        name = getattr(e, "name", None)
        msg_l = str(e).lower()
        if name == "torch" or "torch" in msg_l:
            hint = (
                "\nDDPG requires PyTorch. In this environment run:\n"
                "  pip install torch\n"
                "(conda: conda install pytorch -c pytorch)\n"
                "To skip DDPG: compare_strategies_three(..., run_ddpg=False).\n"
            )
        raise ImportError(
            f"Failed to load {path}. {hint}Original error: {e}"
        ) from e
    return mod.MultiAssetDDPG


def import_multi_asset_ddpg(module_path=None):
    """
    Return MultiAssetDDPG: use the class defined in this file when PyTorch is available;
    otherwise load from multi_asset_ddpg.py (legacy) or raise with install hints.

    Parameters
    ----------
    module_path : str | None
        If set, load exactly this file (recommended in notebooks as an absolute path).
    """
    if module_path is not None:
        p = Path(module_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"multi_asset_ddpg not found at: {p}")
        return _exec_ddpg_module(p)

    if torch is not None and MultiAssetDDPG is not None:
        return MultiAssetDDPG

    try:
        from multi_asset_ddpg import MultiAssetDDPG as _Ext

        if _Ext is not None:
            return _Ext
    except ModuleNotFoundError:
        pass
    except ImportError as e:
        if getattr(e, "name", None) == "torch" or "torch" in str(e).lower():
            raise ImportError(
                "multi_asset_ddpg requires PyTorch. Run: pip install torch\n"
                "Or set run_ddpg=False to compare only the first two strategies.\n"
                f"Original error: {e}"
            ) from e
        raise

    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent / "multi_asset_ddpg.py")
    except NameError:
        pass
    candidates.append(Path.cwd() / "multi_asset_ddpg.py")
    env = os.environ.get("MAFS5370_ROOT")
    if env:
        candidates.append(Path(env).expanduser().resolve() / "multi_asset_ddpg.py")
    home = Path.home()
    candidates.append(home / "Desktop" / "MAFS5370" / "multi_asset_ddpg.py")

    for p in candidates:
        if p.is_file():
            return _exec_ddpg_module(p)

    raise ImportError(
        "PyTorch is not installed (DDPG classes unavailable) and no usable "
        "multi_asset_ddpg.py was found. Install torch, or set run_ddpg=False.\n"
        "Optional: place multi_asset_ddpg.py next to this file or set MAFS5370_ROOT / "
        "ddpg_module_path."
    )


# -------------------------------------------------------------
# 2. Marginal-utility allocation
# -------------------------------------------------------------
def optimal_adjustment_with_budget(mu, cov, pi_cur, W_cur, a, r,
                                   adjustment_needed,
                                   max_adjust_per_asset,
                                   max_total_abs_adjust,
                                   is_zero_var=None):
    n = len(pi_cur)
    if is_zero_var is None:
        is_zero_var = np.zeros(n, dtype=bool)

    marginal_util = compute_marginal_utility(mu, cov, pi_cur, W_cur, a, r)

    urgency = np.zeros(n)
    for i in range(n):
        if adjustment_needed[i] > 0:
            urgency[i] = max(0, marginal_util[i])
        elif adjustment_needed[i] < 0:
            urgency[i] = max(0, -marginal_util[i])

    total_urgency = np.sum(urgency)
    if total_urgency == 0 or np.sum(np.abs(adjustment_needed)) == 0:
        return pi_cur.copy()

    allocation_ratio = urgency / total_urgency
    raw_allocation = allocation_ratio * max_total_abs_adjust

    final_adjustment = np.zeros(n)
    remaining_budget = max_total_abs_adjust

    for i in range(n):
        if urgency[i] == 0:
            continue
        direction = 1 if adjustment_needed[i] > 0 else -1
        max_allowed = min(abs(adjustment_needed[i]), max_adjust_per_asset)
        adjust_amount = min(raw_allocation[i], max_allowed, remaining_budget)
        if adjust_amount > 0:
            final_adjustment[i] = direction * adjust_amount
            remaining_budget -= adjust_amount

    if remaining_budget > 1e-10:
        remaining_urgency = urgency.copy()
        for i in range(n):
            if abs(final_adjustment[i]) > 0:
                remaining_urgency[i] = 0
        total_rem_urgency = np.sum(remaining_urgency)
        if total_rem_urgency > 0:
            rem_ratio = remaining_urgency / total_rem_urgency
            second_allocation = rem_ratio * remaining_budget
            for i in range(n):
                if remaining_urgency[i] == 0:
                    continue
                direction = 1 if adjustment_needed[i] > 0 else -1
                remaining_capacity = min(
                    abs(adjustment_needed[i]) - abs(final_adjustment[i]),
                    max_adjust_per_asset - abs(final_adjustment[i])
                )
                additional = min(second_allocation[i], remaining_capacity, remaining_budget)
                if additional > 0:
                    final_adjustment[i] += direction * additional
                    remaining_budget -= additional

    pi_new = pi_cur + final_adjustment
    cash_cur = W_cur - np.sum(pi_cur)
    cash_new = W_cur - np.sum(pi_new)
    cash_adjustment = cash_new - cash_cur
    total_abs_used = np.sum(np.abs(final_adjustment)) + abs(cash_adjustment)

    if total_abs_used > max_total_abs_adjust + 1e-8:
        scale = max_total_abs_adjust / total_abs_used
        final_adjustment_scaled = final_adjustment * scale
        pi_new = pi_cur + final_adjustment_scaled

    return pi_new


def solve_portfolio(r, mu, cov, p0, W0, a, delta, T, verbose=False):
    """Marginal-utility constrained rebalancing path."""
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    mu = np.asarray(mu)
    cov = np.asarray(cov)
    p0 = np.asarray(p0)
    n_assets = len(mu)

    epsilon = 1e-8
    is_zero_var = np.array([cov[i, i] <= epsilon for i in range(n_assets)])
    has_variance = ~is_zero_var
    zero_var_indices = np.where(is_zero_var)[0]

    if not np.any(has_variance):
        path = []
        W = W0
        pi_cur = p0[1:] * W0
        for t in range(T):
            cash_cur = W - np.sum(pi_cur)
            entry = {'t': t, 'wealth': W, 'holdings_dollar': pi_cur.copy(),
                     'holdings_share': pi_cur / W if W > 0 else np.zeros_like(pi_cur),
                     'cash_dollar': cash_cur, 'cash_share': cash_cur / W if W > 0 else 0.0}
            max_adjust_per_asset = delta * W
            max_total_abs_adjust = 2 * delta * W
            best_asset_idx = np.argmax(mu)
            best_return = mu[best_asset_idx]
            if best_return > r:
                pi_target = np.zeros(n_assets)
                pi_target[best_asset_idx] = W
            else:
                pi_target = np.zeros(n_assets)
            adjustment_needed = pi_target - pi_cur
            pi_new = optimal_adjustment_with_budget(
                mu, cov, pi_cur, W, a, r, adjustment_needed,
                max_adjust_per_asset, max_total_abs_adjust, is_zero_var)
            pi_new = np.maximum(pi_new, 0)
            total_invested = np.sum(pi_new)
            if total_invested > W:
                pi_new *= W / total_invested
            entry['new_holdings_dollar'] = pi_new.copy()
            entry['new_holdings_share'] = pi_new / W if W > 0 else np.zeros_like(pi_new)
            path.append(entry)
            portfolio_return = (np.sum(pi_new * mu) + cash_cur * r) / W
            W *= (1 + portfolio_return)
            pi_cur = pi_new.copy()
        if not verbose:
            sys.stdout = old_stdout
        return path

    risky_indices = np.where(has_variance)[0]
    mu_risky = mu[risky_indices]
    cov_risky = cov[np.ix_(risky_indices, risky_indices)]
    initial_pi = p0[1:] * W0

    if len(zero_var_indices) > 0:
        max_zero_return = np.max(mu[zero_var_indices])
        if max_zero_return > r:
            r = max_zero_return

    risk_premium = mu_risky - r
    try:
        Sigma_inv = np.linalg.inv(cov_risky)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(cov_risky)
    optimal_weights_risky = (1.0 / a) * Sigma_inv @ risk_premium

    path = []
    W_cur = W0
    pi_cur = initial_pi.copy()

    for t in range(T):
        optimal_pi_full = np.zeros(n_assets)
        optimal_pi_full[risky_indices] = optimal_weights_risky * W_cur
        for idx in zero_var_indices:
            optimal_pi_full[idx] = 0

        cash_cur = W_cur - np.sum(pi_cur)
        entry = {'t': t, 'wealth': W_cur, 'holdings_dollar': pi_cur.copy(),
                 'holdings_share': pi_cur / W_cur if W_cur > 0 else np.zeros_like(pi_cur),
                 'cash_dollar': cash_cur, 'cash_share': cash_cur / W_cur if W_cur > 0 else 0.0}
        max_adjust_per_asset = delta * W_cur
        max_total_abs_adjust = 2 * delta * W_cur

        adjustment_needed = optimal_pi_full - pi_cur
        pi_new = optimal_adjustment_with_budget(
            mu, cov, pi_cur, W_cur, a, r, adjustment_needed,
            max_adjust_per_asset, max_total_abs_adjust, is_zero_var)

        pi_new = np.maximum(pi_new, 0)
        total_invested = np.sum(pi_new)
        if total_invested > W_cur:
            pi_new *= W_cur / total_invested

        entry['new_holdings_dollar'] = pi_new.copy()
        entry['new_holdings_share'] = pi_new / W_cur if W_cur > 0 else np.zeros_like(pi_new)
        path.append(entry)

        cash_new = W_cur - np.sum(pi_new)
        cash_next = cash_new * (1 + r)
        pi_next = np.zeros(n_assets)
        pi_next[risky_indices] = pi_new[risky_indices] * (1 + mu_risky)
        if len(zero_var_indices) > 0:
            pi_next[zero_var_indices] = pi_new[zero_var_indices] * (1 + mu[zero_var_indices])
        W_next = cash_next + np.sum(pi_next)
        pi_cur = pi_next
        W_cur = W_next

    if not verbose:
        sys.stdout = old_stdout
    return path


# -------------------------------------------------------------
# 3. Simple Merton (proportional scaling)
# -------------------------------------------------------------
def solve_portfolio_simple(r, mu, cov, p0, W0, a, delta, T, verbose=False):
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    mu = np.asarray(mu)
    cov = np.asarray(cov)
    p0 = np.asarray(p0)
    n_assets = len(mu)

    epsilon = 1e-8
    is_zero_var = np.array([cov[i, i] <= epsilon for i in range(n_assets)])
    has_variance = ~is_zero_var
    zero_var_indices = np.where(is_zero_var)[0]

    if not np.any(has_variance):
        path = []
        W = W0
        pi_cur = p0[1:] * W0
        for t in range(T):
            cash_cur = W - np.sum(pi_cur)
            entry = {'t': t, 'wealth': W, 'holdings_dollar': pi_cur.copy(),
                    'holdings_share': pi_cur / W if W > 0 else np.zeros_like(pi_cur),
                    'cash_dollar': cash_cur, 'cash_share': cash_cur / W if W > 0 else 0.0}
            max_adjust_per_asset = delta * W
            max_total_abs_adjust = 2 * delta * W
            best_asset_idx = np.argmax(mu)
            best_return = mu[best_asset_idx]
            if best_return > r:
                pi_target = np.zeros(n_assets)
                pi_target[best_asset_idx] = W
            else:
                pi_target = np.zeros(n_assets)
            adjustment_needed = pi_target - pi_cur
            abs_adj_needed = np.abs(adjustment_needed)
            max_single_adj = np.max(abs_adj_needed) if n_assets > 0 else 0
            total_abs_adj = np.sum(abs_adj_needed) + abs(-np.sum(adjustment_needed))

            single_ok = max_single_adj <= max_adjust_per_asset + 1e-10
            total_ok = total_abs_adj <= max_total_abs_adjust + 1e-10
            if single_ok and total_ok:
                pi_new = pi_target.copy()
            else:
                scale_single = max_adjust_per_asset / max_single_adj if not single_ok else 1.0
                scale_total = max_total_abs_adjust / total_abs_adj if not total_ok else 1.0
                scale = min(scale_single, scale_total)
                pi_new = pi_cur + adjustment_needed * scale

            pi_new = np.maximum(pi_new, 0)
            total_invested = np.sum(pi_new)
            if total_invested > W:
                pi_new *= W / total_invested

            entry['new_holdings_dollar'] = pi_new.copy()
            entry['new_holdings_share'] = pi_new / W if W > 0 else np.zeros_like(pi_new)
            path.append(entry)
            cash_new = W - np.sum(pi_new)
            portfolio_return = (np.sum(pi_new * mu) + cash_new * r) / W
            W *= (1 + portfolio_return)
            pi_cur = pi_new.copy()
        if not verbose:
            sys.stdout = old_stdout
        return path

    risky_indices = np.where(has_variance)[0]
    mu_risky = mu[risky_indices]
    cov_risky = cov[np.ix_(risky_indices, risky_indices)]
    initial_pi = p0[1:] * W0

    if len(zero_var_indices) > 0:
        max_zero_return = np.max(mu[zero_var_indices])
        if max_zero_return > r:
            r = max_zero_return

    risk_premium = mu_risky - r
    try:
        Sigma_inv = np.linalg.inv(cov_risky)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(cov_risky)
    optimal_weights_risky = (1.0 / a) * Sigma_inv @ risk_premium

    path = []
    W_cur = W0
    pi_cur = initial_pi.copy()
    for t in range(T):
        optimal_pi_full = np.zeros(n_assets)
        optimal_pi_full[risky_indices] = optimal_weights_risky * W_cur
        for idx in zero_var_indices:
            optimal_pi_full[idx] = 0

        cash_cur = W_cur - np.sum(pi_cur)
        entry = {'t': t, 'wealth': W_cur, 'holdings_dollar': pi_cur.copy(),
                 'holdings_share': pi_cur / W_cur if W_cur > 0 else np.zeros_like(pi_cur),
                 'cash_dollar': cash_cur, 'cash_share': cash_cur / W_cur if W_cur > 0 else 0.0}

        max_adjust_per_asset = delta * W_cur
        max_total_abs_adjust = 2 * delta * W_cur

        adjustment_needed = optimal_pi_full - pi_cur
        abs_adj_needed = np.abs(adjustment_needed)
        max_single_adj = np.max(abs_adj_needed) if n_assets > 0 else 0
        total_abs_adj = np.sum(abs_adj_needed) + abs(-np.sum(adjustment_needed))

        single_ok = max_single_adj <= max_adjust_per_asset + 1e-10
        total_ok = total_abs_adj <= max_total_abs_adjust + 1e-10
        if single_ok and total_ok:
            pi_new = optimal_pi_full.copy()
        else:
            scale_single = max_adjust_per_asset / max_single_adj if not single_ok else 1.0
            scale_total = max_total_abs_adjust / total_abs_adj if not total_ok else 1.0
            scale = min(scale_single, scale_total)
            pi_new = pi_cur + adjustment_needed * scale

        pi_new = np.maximum(pi_new, 0)
        total_invested = np.sum(pi_new)
        if total_invested > W_cur:
            pi_new *= W_cur / total_invested

        entry['new_holdings_dollar'] = pi_new.copy()
        entry['new_holdings_share'] = pi_new / W_cur if W_cur > 0 else np.zeros_like(pi_new)
        path.append(entry)

        cash_new = W_cur - np.sum(pi_new)
        cash_next = cash_new * (1 + r)
        pi_next = np.zeros(n_assets)
        pi_next[risky_indices] = pi_new[risky_indices] * (1 + mu_risky)
        if len(zero_var_indices) > 0:
            pi_next[zero_var_indices] = pi_new[zero_var_indices] * (1 + mu[zero_var_indices])
        W_next = cash_next + np.sum(pi_next)
        pi_cur = pi_next
        W_cur = W_next

    if not verbose:
        sys.stdout = old_stdout
    return path


# -------------------------------------------------------------
# 4. Two-strategy comparison (legacy)
# -------------------------------------------------------------
def compare_strategies(r, mu, cov, p0, W0, a, delta, T, verbose=False, plot=True):
    print("\n" + "="*80)
    print("Running marginal-utility allocation (solve_portfolio)...")
    path_marginal = solve_portfolio(r, mu, cov, p0, W0, a, delta, T, verbose=verbose)

    print("\nRunning simple Merton (solve_portfolio_simple)...")
    path_simple = solve_portfolio_simple(r, mu, cov, p0, W0, a, delta, T, verbose=verbose)

    wealth_marginal = [entry['wealth'] for entry in path_marginal]
    wealth_simple = [entry['wealth'] for entry in path_simple]

    final_m = wealth_marginal[-1]
    final_s = wealth_simple[-1]

    ret_m = (final_m / W0 - 1) * 100
    ret_s = (final_s / W0 - 1) * 100

    ann_m = (final_m / W0) ** (1/T) - 1
    ann_s = (final_s / W0) ** (1/T) - 1

    excess = final_m - final_s
    excess_pct = (final_m / final_s - 1) * 100

    print("\n" + "="*80)
    print("Comparison")
    print("="*80)
    print(f"{'Strategy':<25} {'Final W':<12} {'Total ret %':<12} {'Ann. ret %':<12}")
    print("-"*80)
    print(f"{'Marginal utility':<25} {final_m:<12.6f} {ret_m:<12.2f}% {ann_m*100:<12.2f}%")
    print(f"{'Simple Merton':<25} {final_s:<12.6f} {ret_s:<12.2f}% {ann_s*100:<12.2f}%")
    print("-"*80)
    print(f"Excess (marginal - simple): {excess:.6f}  ({excess_pct:+.2f}%)")
    print("="*80)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(T), wealth_marginal, marker='o', label='Marginal Allocation Strategy')
        plt.plot(range(T), wealth_simple, marker='s', label='Simple Merton Strategy')
        plt.xlabel('Time')
        plt.ylabel('Wealth')
        plt.title('Wealth Path Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    return path_marginal, path_simple, wealth_marginal, wealth_simple


# -------------------------------------------------------------
# 5. Three strategies (+ optional DDPG)
# -------------------------------------------------------------
def _rank_terminal_wealth_paths(out):
    """
    Given dict with path_*, final_*, optional ddpg: add
    best_key, best_label, best_terminal_wealth, best_path, paths_by_key, terminal_wealth_by_key.
    """
    labels = {
        "marginal": "Marginal utility (solve_portfolio)",
        "simple": "Simple Merton",
        "ddpg": "DDPG",
    }
    terminal_wealth_by_key = {
        "marginal": float(out["final_marginal"]),
        "simple": float(out["final_simple"]),
    }
    if out.get("wealth_ddpg") is not None and "final_ddpg" in out:
        terminal_wealth_by_key["ddpg"] = float(out["final_ddpg"])

    paths_by_key = {
        "marginal": out["path_marginal"],
        "simple": out["path_simple"],
    }
    if out.get("path_ddpg") is not None:
        paths_by_key["ddpg"] = out["path_ddpg"]

    best_key = max(terminal_wealth_by_key, key=lambda k: terminal_wealth_by_key[k])
    return {
        "best_key": best_key,
        "best_label": labels[best_key],
        "best_terminal_wealth": terminal_wealth_by_key[best_key],
        "best_path": paths_by_key[best_key],
        "paths_by_key": paths_by_key,
        "terminal_wealth_by_key": terminal_wealth_by_key,
    }


def _is_zero_variance_flags(cov, n_assets, epsilon=1e-8):
    cov = np.asarray(cov)
    return np.array([cov[i, i] <= epsilon for i in range(n_assets)], dtype=bool)


def _unconstrained_merton_holdings_dollars(r, mu, cov, W0, a):
    """Same unconstrained target as solve_portfolio (Merton weights × W0 on risky subset)."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n_assets = len(mu)
    is_zv = _is_zero_variance_flags(cov, n_assets)
    has_risky = np.any(~is_zv)
    zero_var_indices = np.where(is_zv)[0]

    if not has_risky:
        pi = np.zeros(n_assets)
        best_idx = int(np.argmax(mu))
        if mu[best_idx] > r:
            pi[best_idx] = W0
        cash = float(W0 - np.sum(pi))
        return pi, cash, float(r), is_zv

    r_eff = float(r)
    if len(zero_var_indices) > 0:
        mz = float(np.max(mu[zero_var_indices]))
        if mz > r_eff:
            r_eff = mz

    risky_idx = np.where(~is_zv)[0]
    mu_r = mu[risky_idx]
    cov_r = cov[np.ix_(risky_idx, risky_idx)]
    rp = mu_r - r_eff
    try:
        Sigma_inv = np.linalg.inv(cov_r)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(cov_r)
    w_risky = (1.0 / a) * Sigma_inv @ rp
    pi = np.zeros(n_assets)
    pi[risky_idx] = w_risky * W0
    cash = float(W0 - np.sum(pi))
    return pi, cash, r_eff, is_zv


def _print_setup_block(r, mu, cov, p0, W0, a, delta, T, title="Optimization Strategy Setup"):
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    p0 = np.asarray(p0)
    n = len(mu)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Risk-free rate: {100 * r:.2f}%")
    print(f"Asset expected returns: {np.array2string(mu, precision=2, suppress_small=True)}")
    print("Covariance matrix:")
    print(np.array2string(cov, precision=2, suppress_small=False))
    parts = [f"Cash={p0[0] * 100:.1f}%"]
    for i in range(n):
        parts.append(f"Asset{i + 1}={p0[i + 1] * 100:.1f}%")
    print("Initial allocation: " + ", ".join(parts))
    print(f"Initial wealth: {W0:.2f}")
    print(f"Risk aversion: {a:.1f}")
    print(f"Per-asset adjustment limit (fraction of wealth): {100 * delta:.1f}%")
    print(f"Total adjustment limit (sum of absolute values): {100 * 2 * delta:.1f}%")
    print(f"Investment horizon: {T} periods")
    print("=" * 70)


def _print_unconstrained_optimal(r, mu, cov, W0, a):
    pi, cash, _r_eff, is_zv = _unconstrained_merton_holdings_dollars(r, mu, cov, W0, a)
    n = len(mu)
    print("\nUnconstrained optimal holdings:")
    for i in range(n):
        sh = 100.0 * pi[i] / W0 if W0 > 1e-15 else 0.0
        if is_zv[i]:
            print(f"  Quasi-riskless asset {i + 1}: ${pi[i]:.4f} ({sh:.1f}%)")
        else:
            print(f"  Risky asset {i + 1}: ${pi[i]:.4f} ({sh:.1f}%)")
    csh = 100.0 * cash / W0 if W0 > 1e-15 else 0.0
    print(f"  Cash: ${cash:.4f} ({csh:.1f}%)\n")


def _print_path_entries_detailed(path, mu, cov, *, path_label=None):
    """Period-by-period holdings (beginning-of-period wealth, pre-decision and post-decision)."""
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = len(mu)
    is_zv = _is_zero_variance_flags(cov, n)
    print("\n" + "=" * 70)
    if path_label:
        print(f"Final simulated path summary — {path_label}")
    else:
        print("Final simulated path summary")
    print("=" * 70)
    for e in path:
        t = int(e["t"])
        W = float(e["wealth"])
        hd = np.asarray(e.get("holdings_dollar", np.zeros(n)), dtype=float).ravel()
        if hd.size < n:
            hd = np.pad(hd, (0, max(0, n - hd.size)))
        hd = hd[:n]
        cash = float(e.get("cash_dollar", W - float(np.sum(hd))))
        print(f"\nPeriod {t} (beginning):")
        print(f"  Wealth: {W:.4f}")
        print(f"  Cash: ${cash:.4f} ({100.0 * cash / W if W > 1e-15 else 0.0:.1f}%)")
        for i in range(n):
            d = float(hd[i])
            sh = 100.0 * d / W if W > 1e-15 else 0.0
            if is_zv[i]:
                print(f"  Asset{i + 1} (zero variance): ${d:.4f} ({sh:.1f}%)")
            else:
                print(f"  Asset{i + 1}: ${d:.4f} ({sh:.1f}%)")

        nhd = np.asarray(e.get("new_holdings_dollar", hd), dtype=float).ravel()
        if nhd.size < n:
            nhd = np.pad(nhd, (0, max(0, n - nhd.size)))
        nhd = nhd[:n]
        new_cash = float(W - np.sum(nhd))
        print("  New holdings after decision:")
        print(f"    Cash: ${new_cash:.4f} ({100.0 * new_cash / W if W > 1e-15 else 0.0:.1f}%)")
        for i in range(n):
            d = float(nhd[i])
            sh = 100.0 * d / W if W > 1e-15 else 0.0
            if is_zv[i]:
                print(f"    Asset{i + 1} (zero variance): ${d:.4f} ({sh:.1f}%)")
            else:
                print(f"    Asset{i + 1}: ${d:.4f} ({sh:.1f}%)")
    print("=" * 70)


def _print_single_path_table(path, title):
    """Compact one-line-per-period table (used when print_paths='compact')."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'t':>4} {'W (start)':>14} {'Cash share':>12} {'Risky $ sum':>16}")
    print("-" * 80)
    for e in path:
        t = int(e["t"])
        W = float(e["wealth"])
        hd = np.asarray(e.get("holdings_dollar", 0.0))
        hsum = float(np.sum(hd))
        cash = float(e.get("cash_dollar", W - hsum))
        cash_share = cash / W if W > 1e-15 else 0.0
        print(f"{t:4d} {W:14.6f} {cash_share:12.4f} {hsum:16.6f}")
    print("=" * 80)


def _print_paths_console(out, print_paths, *, r, mu, cov, p0, W0, a, delta, T, verbose=False):
    """
    print_paths:
        False — print nothing
        True or \"best\" — detailed path for best (title includes strategy + best_key); if verbose,
            prepend setup + unconstrained optimal
        \"all\" — detailed path per strategy (each titled); if verbose, prepend setup once
        \"compact\" — legacy compact table for best path only (unchanged by verbose)
    """
    if not print_paths:
        return
    labels = {
        "marginal": "Marginal utility (solve_portfolio)",
        "simple": "Simple Merton",
        "ddpg": "DDPG",
    }
    if print_paths == "compact":
        _print_single_path_table(
            out["best_path"],
            f"Path detail — best terminal wealth: {out['best_label']} (best_key={out['best_key']})",
        )
        return

    def _is_best_mode(pp):
        if pp is True or pp == "best":
            return True
        try:
            return isinstance(pp, np.bool_) and bool(pp)
        except Exception:
            return False

    if _is_best_mode(print_paths):
        if verbose:
            _print_setup_block(r, mu, cov, p0, W0, a, delta, T)
            _print_unconstrained_optimal(r, mu, cov, W0, a)
        _print_path_entries_detailed(
            out["best_path"],
            mu,
            cov,
            path_label=f"{out['best_label']} (best_key={out['best_key']})",
        )
        return

    if print_paths == "all":
        if verbose:
            _print_setup_block(r, mu, cov, p0, W0, a, delta, T)
            _print_unconstrained_optimal(r, mu, cov, W0, a)
        for key in ("marginal", "simple", "ddpg"):
            if key not in out.get("paths_by_key", {}):
                continue
            _print_path_entries_detailed(
                out["paths_by_key"][key], mu, cov, path_label=labels[key]
            )


_DDPG_OPTION_KEYS = frozenset(
    {
        "hidden_layers",
        "learning_rate_actor",
        "learning_rate_critic",
        "gamma",
        "tau",
        "buffer_capacity",
        "batch_size",
        "num_episodes",
        "eval_interval",
        "eval_episodes",
        "max_steps_per_episode",
        "updates_per_episode",
        "lr_plateau_factor",
        "lr_plateau_min_lr",
        "lr_plateau_patience",
        "early_stop_k",
        "early_stop_epsilon",
        "terminal_batch_fraction",
        "reward_mode",
    }
)

_DDPG_REWARD_MODES = frozenset({"sparse", "potential"})


def _normalize_ddpg_options(opts: dict) -> None:
    """
    Map common LR aliases onto MultiAssetDDPG __init__ names; remove keys that are not
    constructor kwargs so ``learning_rate`` etc. do not raise TypeError.
    """
    if "learning_rate" in opts:
        lr = opts.pop("learning_rate")
        opts["learning_rate_actor"] = lr
        opts["learning_rate_critic"] = lr
    for alias, canonical in (
        ("lr_actor", "learning_rate_actor"),
        ("actor_lr", "learning_rate_actor"),
        ("lr_critic", "learning_rate_critic"),
        ("critic_lr", "learning_rate_critic"),
    ):
        if alias in opts:
            opts[canonical] = opts.pop(alias)
    unknown = set(opts) - _DDPG_OPTION_KEYS
    if unknown:
        warnings.warn(
            "ddpg_options: ignoring unknown keys (not MultiAssetDDPG parameters): "
            + ", ".join(sorted(unknown)),
            UserWarning,
            stacklevel=3,
        )
        for k in unknown:
            del opts[k]


def compare_strategies_three(
    r, mu, cov, p0, W0, a, delta, T,
    verbose=False,
    plot=True,
    run_ddpg=True,
    ddpg_options=None,
    ddpg_reward_mode=None,
    ddpg_solver=None,
    ddpg_module_path=None,
    print_paths=True,
    print_comparison_table=None,
):
    """
    Compare marginal utility, simple Merton, and optionally DDPG.

    Full parameter descriptions are in the module docstring at the top of this file.
    ``ddpg_reward_mode``: ``\"sparse\"`` or ``\"potential\"``; if set, overrides
    ``ddpg_options[\"reward_mode\"]``.

    Returns
    -------
    dict
        path_marginal, path_simple, path_ddpg (maybe None): list of per-period dicts
        (t, wealth, holdings_dollar, new_holdings_dollar, ...).
        wealth_*, final_*, utility_*; ddpg_solver if DDPG ran.
        Ranking by last start-of-period wealth: best_key, best_label, best_terminal_wealth,
        best_path, paths_by_key, terminal_wealth_by_key.
    """
    if verbose:
        print("\n" + "="*80)
        print("Running marginal-utility allocation (solve_portfolio)...")
    path_marginal = solve_portfolio(r, mu, cov, p0, W0, a, delta, T, verbose=verbose)

    if verbose:
        print("\nRunning simple Merton (solve_portfolio_simple)...")
    path_simple = solve_portfolio_simple(r, mu, cov, p0, W0, a, delta, T, verbose=verbose)

    wealth_m = [e['wealth'] for e in path_marginal]
    wealth_s = [e['wealth'] for e in path_simple]

    path_ddpg = None
    wealth_d = None
    solver = None
    if ddpg_solver is not None:
        solver = ddpg_solver

    if run_ddpg:
        if solver is None:
            MultiAssetDDPG = import_multi_asset_ddpg(ddpg_module_path)

            opts = {
                "hidden_layers": [256, 256],
                "learning_rate_actor": 1e-2,
                "learning_rate_critic": 1e-1,
                "num_episodes": 2000,
                "eval_interval": 50,
                "updates_per_episode": 20,
                "lr_plateau_factor": 0.5,
                "lr_plateau_min_lr": 1e-6,
                "lr_plateau_patience": max(2, 2000 // 100),
                "early_stop_k": None,
                "early_stop_epsilon": None,
                "gamma": 1.0,
                "terminal_batch_fraction": 0.25,
                "reward_mode": "sparse",
            }
            if ddpg_options:
                opts.update(ddpg_options)
            _normalize_ddpg_options(opts)
            if ddpg_reward_mode is not None:
                if ddpg_reward_mode not in _DDPG_REWARD_MODES:
                    raise ValueError(
                        "ddpg_reward_mode must be 'sparse' or 'potential', "
                        f"got {ddpg_reward_mode!r}"
                    )
                opts["reward_mode"] = ddpg_reward_mode
            ne = opts.get("num_episodes", 2000)
            opts["lr_plateau_patience"] = opts.get(
                "lr_plateau_patience", max(2, ne // 100)
            )

            if verbose:
                print("\nTraining DDPG (MultiAssetDDPG)...")
            solver = MultiAssetDDPG(
                r, mu, cov, a, delta, T, p0=p0,
                W_min=0.5, W_max=5.0,
                **opts,
            )
            if verbose:
                solver.solve()
            else:
                _ddpg_out = StringIO()
                _old_out = sys.stdout
                try:
                    sys.stdout = _ddpg_out
                    solver.solve()
                finally:
                    sys.stdout = _old_out
        else:
            if verbose:
                print("\nUsing provided DDPG solver; skipping training...")

        if verbose:
            print("\nSimulating DDPG path...")
            path_ddpg = solver.simulate(W0, p0)
        else:
            _ddpg_out = StringIO()
            _old_out = sys.stdout
            try:
                sys.stdout = _ddpg_out
                path_ddpg = solver.simulate(W0, p0)
            finally:
                sys.stdout = _old_out
        wealth_d = [e['wealth'] for e in path_ddpg]

    # Metrics: last start-of-period wealth; CARA utility for comparison with DDPG objective
    final_m, final_s = wealth_m[-1], wealth_s[-1]
    u_m, u_s = cara_terminal_utility(final_m, a), cara_terminal_utility(final_s, a)

    rows = [
        ("Marginal (solve_portfolio)", final_m, u_m),
        ("Simple Merton", final_s, u_s),
    ]
    if wealth_d is not None:
        final_d = wealth_d[-1]
        u_d = cara_terminal_utility(final_d, a)
        rows.append(("DDPG", final_d, u_d))

    if print_comparison_table is None:
        show_three_way = not bool(print_paths)
    else:
        show_three_way = print_comparison_table
    if show_three_way:
        print("\n" + "="*80)
        print("Three-way comparison (last start-of-period wealth & CARA U(W))")
        print("="*80)
        print(f"{'Strategy':<28} {'W (last period)':<16} {'CARA U(W)':<14} {'Total ret %':<12} {'Ann. ret %':<12}")
        print("-"*80)
        for name, fw, uw in rows:
            ret = (fw / W0 - 1) * 100
            ann = (fw / W0) ** (1 / T) - 1 if T > 0 else 0.0
            print(f"{name:<28} {fw:<16.6f} {uw:<14.6f} {ret:<12.2f} {ann*100:<12.2f}")
        print("="*80)
        print("Note: wealth is start-of-period W_t; CARA U is evaluated at the last such W_T.")

    out = {
        "path_marginal": path_marginal,
        "path_simple": path_simple,
        "path_ddpg": path_ddpg,
        "wealth_marginal": wealth_m,
        "wealth_simple": wealth_s,
        "wealth_ddpg": wealth_d,
        "final_marginal": final_m,
        "final_simple": final_s,
        "utility_marginal": u_m,
        "utility_simple": u_s,
        "ddpg_solver": solver,
    }
    if wealth_d is not None:
        out["final_ddpg"] = wealth_d[-1]
        out["utility_ddpg"] = cara_terminal_utility(wealth_d[-1], a)
    out.update(_rank_terminal_wealth_paths(out))
    _print_paths_console(
        out,
        print_paths,
        r=r,
        mu=mu,
        cov=cov,
        p0=p0,
        W0=W0,
        a=a,
        delta=delta,
        T=T,
        verbose=verbose,
    )

    if plot and wealth_d is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(T), wealth_m, marker='o', label='Marginal (solve_portfolio)')
        plt.plot(range(T), wealth_s, marker='s', label='Simple Merton')
        plt.plot(range(T), wealth_d, marker='^', label='DDPG')
        plt.xlabel('Period (start-of-period)')
        plt.ylabel('Wealth')
        plt.title('Wealth Path: Three Strategies')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(T), wealth_m, marker='o', label='Marginal (solve_portfolio)')
        plt.plot(range(T), wealth_s, marker='s', label='Simple Merton')
        plt.xlabel('Period (start-of-period)')
        plt.ylabel('Wealth')
        plt.title('Wealth Path: Two Strategies')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return out


def compare_three_pick_best(
    r,
    mu,
    cov,
    p0,
    W0,
    a,
    delta,
    T,
    *,
    verbose=False,
    plot=False,
    run_ddpg=True,
    ddpg_options=None,
    ddpg_reward_mode=None,
    ddpg_solver=None,
    ddpg_module_path=None,
    print_summary=True,
    print_paths=True,
    print_comparison_table=None,
):
    """
    Run the three-way comparison and expose the winner by last start-of-period wealth.

    Defaults: plot=False (no figure), print_paths forwarded to compare_strategies_three (default True).
    print_comparison_table forwarded (default None: hide three-way CARA block when printing paths).
    Extra parameter print_summary: short winner block at the end.

    Full parameter list: see module docstring at the top of this file.

    Returns
    -------
    dict
        best_key, best_label, best_terminal_wealth, best_path, paths_by_key,
        terminal_wealth_by_key, compare_result (full output of compare_strategies_three, includes best_*).
    """
    out = compare_strategies_three(
        r,
        mu,
        cov,
        p0,
        W0,
        a,
        delta,
        T,
        verbose=verbose,
        plot=plot,
        run_ddpg=run_ddpg,
        ddpg_options=ddpg_options,
        ddpg_reward_mode=ddpg_reward_mode,
        ddpg_solver=ddpg_solver,
        ddpg_module_path=ddpg_module_path,
        print_paths=print_paths,
        print_comparison_table=print_comparison_table,
    )

    labels = {
        "marginal": "Marginal utility (solve_portfolio)",
        "simple": "Simple Merton",
        "ddpg": "DDPG",
    }
    terminal_wealth_by_key = out["terminal_wealth_by_key"]
    best_label = out["best_label"]
    best_w = out["best_terminal_wealth"]

    if print_summary:
        print("\n" + "=" * 72)
        print("Terminal wealth (last start-of-period W; same convention as table above)")
        print("=" * 72)
        for k in ("marginal", "simple", "ddpg"):
            if k in terminal_wealth_by_key:
                print(f"  {labels[k]:<32} {terminal_wealth_by_key[k]:.6f}")
        print("-" * 72)
        print(f"Highest terminal wealth: {best_label}  (W = {best_w:.6f})")
        print("=" * 72)

    return {
        "best_key": out["best_key"],
        "best_label": best_label,
        "best_terminal_wealth": best_w,
        "best_path": out["best_path"],
        "paths_by_key": out["paths_by_key"],
        "terminal_wealth_by_key": terminal_wealth_by_key,
        "compare_result": out,
    }


compare_three_strategies = compare_strategies_three


if __name__ == "__main__":
    r = 0.05
    mu = np.array([0.04, 0.05, 0.09])
    var = [0.3, 0.1, 0.2]
    cov = covariance_matrix(var)
    p0 = np.array([0.3, 0.2, 0.4, 0.1])
    W0 = 1.0
    a = 2.0
    delta = 0.1
    T = 9

    # DDPG reward: "sparse" (terminal CARA only) or "potential" (Ng-style shaping).
    # Alternatively set ddpg_options={"reward_mode": "potential"} — but if you pass
    # ddpg_reward_mode below, it overrides reward_mode inside ddpg_options.

    # Quick test: lower num_episodes, e.g. ddpg_options={"num_episodes": 200}
    res = compare_strategies_three(
        r, mu, cov, p0, W0, a, delta, T,
        verbose=False,
        plot=True,
        run_ddpg=True,
        ddpg_reward_mode="potential",
        ddpg_options={"num_episodes": 300},
    )
    _ = res["best_path"]
