"""
Compare three portfolio strategies:
  - solve_portfolio: marginal-utility allocation (optimal_adjustment_with_budget)
  - solve_portfolio_simple: simple proportional scaling toward Merton target
  - MultiAssetDDPG: imported from multi_asset_ddpg (same as in the DDPG notebook cell)

Usage:
  from compare_strategies_three import compare_strategies_three, covariance_matrix
  res = compare_strategies_three(
      r, mu, cov, p0, W0, a, delta, T,
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
n_quad : int, default 5
    Quadrature order for Gaussian integration in solve_portfolio (risky case).
verbose : bool, default False
    If True, print internal solver chatter from solve_portfolio / solve_portfolio_simple.
plot : bool, default True
    If True, show matplotlib wealth paths at the end (two or three lines).
run_ddpg : bool, default True
    If False, skip DDPG (no PyTorch); only marginal vs simple Merton.
ddpg_options : dict or None
    Hyperparameters passed to MultiAssetDDPG (e.g. num_episodes, learning_rate_actor,
    learning_rate_critic, reward_mode, gamma). None uses built-in defaults.
ddpg_solver : MultiAssetDDPG or None
    Pre-trained solver: if given, training is skipped and only simulate() is run.
ddpg_module_path : str or None
    Absolute path to multi_asset_ddpg.py when the module is not on sys.path.
print_paths : bool or str, default True
    Console output style. True or \"best\": full report — setup block, unconstrained Merton
    holdings, then period-by-period path for the best terminal-wealth strategy (similar to
    course handout format). \"all\": same setup and unconstrained block once, then a detailed
    path for each of the three strategies. \"compact\": legacy one-line-per-period table for
    the best path only. False: suppress path printing (paths still in res).
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
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss


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
    Load MultiAssetDDPG: try normal import first, else locate multi_asset_ddpg.py and load dynamically.

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

    try:
        from multi_asset_ddpg import MultiAssetDDPG

        return MultiAssetDDPG
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
        "Could not find multi_asset_ddpg.py. Try one of:\n"
        "  1) Place compare_strategies_three.py next to multi_asset_ddpg.py and run from that folder;\n"
        "  2) At the start of the notebook: import sys; sys.path.insert(0, r'/path/to/MAFS5370');\n"
        "  3) Set env MAFS5370_ROOT to the folder containing multi_asset_ddpg.py;\n"
        "  4) Call compare_strategies_three(..., ddpg_module_path=r'/.../multi_asset_ddpg.py')."
    )


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


def solve_portfolio(r, mu, cov, p0, W0, a, delta, T, n_quad=5, verbose=False):
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
    N_risky = len(mu_risky)
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

    x, w = hermgauss(n_quad)
    grids = np.meshgrid(*[x] * N_risky, indexing='ij')
    z = np.column_stack([g.ravel() for g in grids])
    _ = np.prod(np.meshgrid(*[w] * N_risky, indexing='ij'), axis=0).ravel()
    L = np.linalg.cholesky(cov_risky)
    _ = mu_risky + z @ L.T
    _ = _ - r

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
def solve_portfolio_simple(r, mu, cov, p0, W0, a, delta, T, n_quad=5, verbose=False):
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
    N_risky = len(mu_risky)
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
def compare_strategies(r, mu, cov, p0, W0, a, delta, T, n_quad=5, verbose=False, plot=True):
    print("\n" + "="*80)
    print("Running marginal-utility allocation (solve_portfolio)...")
    path_marginal = solve_portfolio(r, mu, cov, p0, W0, a, delta, T, n_quad, verbose=verbose)

    print("\nRunning simple Merton (solve_portfolio_simple)...")
    path_simple = solve_portfolio_simple(r, mu, cov, p0, W0, a, delta, T, n_quad, verbose=verbose)

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


def _print_path_entries_detailed(path, mu, cov):
    """Period-by-period holdings (beginning-of-period wealth, pre-decision and post-decision)."""
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = len(mu)
    is_zv = _is_zero_variance_flags(cov, n)
    print("\n" + "=" * 70)
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


def _print_paths_console(out, print_paths, *, r, mu, cov, p0, W0, a, delta, T):
    """
    print_paths:
        False — print nothing
        True or \"best\" — setup + unconstrained optimal + detailed path for best strategy
        \"all\" — setup + unconstrained once, then detailed path for each strategy
        \"compact\" — legacy compact table for best path only
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

    if print_paths is True or print_paths == "best":
        _print_setup_block(r, mu, cov, p0, W0, a, delta, T)
        _print_unconstrained_optimal(r, mu, cov, W0, a)
        print(f"\n--- Simulated path: {out['best_label']} (best_key={out['best_key']}) ---")
        _print_path_entries_detailed(out["best_path"], mu, cov)
        return

    if print_paths == "all":
        _print_setup_block(r, mu, cov, p0, W0, a, delta, T)
        _print_unconstrained_optimal(r, mu, cov, W0, a)
        for key in ("marginal", "simple", "ddpg"):
            if key not in out.get("paths_by_key", {}):
                continue
            print(f"\n--- Simulated path: {labels[key]} ---")
            _print_path_entries_detailed(out["paths_by_key"][key], mu, cov)


def compare_strategies_three(
    r, mu, cov, p0, W0, a, delta, T,
    n_quad=5,
    verbose=False,
    plot=True,
    run_ddpg=True,
    ddpg_options=None,
    ddpg_solver=None,
    ddpg_module_path=None,
    print_paths=True,
    print_comparison_table=None,
):
    """
    Compare marginal utility, simple Merton, and optionally DDPG.

    Full parameter descriptions are in the module docstring at the top of this file.

    Returns
    -------
    dict
        path_marginal, path_simple, path_ddpg (maybe None): list of per-period dicts
        (t, wealth, holdings_dollar, new_holdings_dollar, ...).
        wealth_*, final_*, utility_*; ddpg_solver if DDPG ran.
        Ranking by last start-of-period wealth: best_key, best_label, best_terminal_wealth,
        best_path, paths_by_key, terminal_wealth_by_key.
    """
    print("\n" + "="*80)
    print("Running marginal-utility allocation (solve_portfolio)...")
    path_marginal = solve_portfolio(r, mu, cov, p0, W0, a, delta, T, n_quad, verbose=verbose)

    print("\nRunning simple Merton (solve_portfolio_simple)...")
    path_simple = solve_portfolio_simple(r, mu, cov, p0, W0, a, delta, T, n_quad, verbose=verbose)

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
            ne = opts.get("num_episodes", 2000)
            opts["lr_plateau_patience"] = opts.get(
                "lr_plateau_patience", max(2, ne // 100)
            )

            print("\nTraining DDPG (MultiAssetDDPG)...")
            solver = MultiAssetDDPG(
                r, mu, cov, a, delta, T, p0=p0,
                W_min=0.5, W_max=5.0,
                **opts,
            )
            solver.solve()
        else:
            print("\nUsing provided DDPG solver; skipping training...")

        print("\nSimulating DDPG path...")
        path_ddpg = solver.simulate(W0, p0)
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
    n_quad=5,
    verbose=False,
    plot=False,
    run_ddpg=True,
    ddpg_options=None,
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
        n_quad=n_quad,
        verbose=verbose,
        plot=plot,
        run_ddpg=run_ddpg,
        ddpg_options=ddpg_options,
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
    T = 3
    n_quad = 5

    # Quick test: lower num_episodes, e.g. ddpg_options={"num_episodes": 200}
    res = compare_strategies_three(
        r, mu, cov, p0, W0, a, delta, T,
        n_quad=n_quad,
        verbose=False,
        plot=False,
        run_ddpg=True,
        ddpg_options={"num_episodes": 300},
    )
    _ = res["best_path"]
