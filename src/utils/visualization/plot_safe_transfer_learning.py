# -----------------------------------------------------------------------------
#  Project: Spark Self-Tuning Framework (STL-PARN-ILS-TS-BO)
#  File: yoro_perf_model_runner.py
#  Copyright (c) 2025 Mariano Garralda Barrio
#  Affiliation: Universidade da Coruña
#  SPDX-License-Identifier: CC-BY-NC-4.0 OR LicenseRef-Commercial
#
#  Associated publication:
#    "A hybrid metaheuristics–Bayesian optimization framework with safe transfer learning for continuous Spark tuning"
#    Mariano Garralda Barrio, Verónica Bolón Canedo, Carlos Eiras Franco
#    Universidade da Coruña, 2025.
#
#  Academic & research use: CC BY-NC 4.0
#    https://creativecommons.org/licenses/by-nc/4.0/
#  Commercial use: requires prior written consent.
#    Contact: mariano.garralda@udc.es
#
#  Distributed on an "AS IS" basis, without warranties or conditions of any kind.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def plot_knee_curve(d_pat_sorted: np.ndarray, k_knee: int) -> None:
    """
    Plot the pattern-distance curve d_pat(k) and highlight the knee τ2.
    Parameters
    ----------
    d_pat_sorted : array-like, shape (m,)
        Ascending pattern distances for the m kept neighbours ([:m_take]).
    k_knee : int
        Knee index τ2 returned by STL (1..m-1). We mark the vertical line at k_knee.
    """
    m = len(d_pat_sorted)
    if m == 0:
        print("plot_knee_curve: empty distance array.")
        return

    ks = np.arange(1, m + 1)
    plt.figure()
    plt.plot(ks, d_pat_sorted, marker='o', linewidth=1)
    if 1 <= k_knee <= m:
        plt.axvline(k_knee, linestyle='--')
        plt.scatter([k_knee], [d_pat_sorted[k_knee-1]])
        plt.title(f"Pattern-distance curve and knee τ2 = {k_knee}")
    else:
        plt.title("Pattern-distance curve (no knee highlighted)")
    plt.xlabel("k (neighbour rank)")
    plt.ylabel("block-normalized L1 pattern distance")
    plt.tight_layout()
    plt.show()


def plot_SL_rbeta_distribution(r_beta: np.ndarray, idx_SL: np.ndarray) -> None:
    """
    Plot the distribution of r_beta within S_L (pre-knee) to check centering and spread.
    Parameters
    ----------
    r_beta : array-like, shape (n,)
        Residuals r_beta for all historical points (same order as inputs to STL).
    idx_SL : array-like of ints
        Indices of neighbours inside S_L (pre-knee), ideally already sorted by r_beta.
    """
    if len(idx_SL) == 0:
        print("plot_SL_rbeta_distribution: empty S_L.")
        return

    r_sl = np.asarray(r_beta)[idx_SL]
    med = float(np.median(r_sl))
    q1, q3 = np.percentile(r_sl, [25, 75])

    plt.figure()
    plt.hist(r_sl, bins=min(25, max(5, len(r_sl)//3)))
    plt.axvline(med, linestyle='--')
    plt.title("Distribution of r_beta in S_L")
    plt.xlabel("r_beta (size-normalized residual)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    print(f"r_beta@S_L: median={med:.4f}, IQR=[{q1:.4f}, {q3:.4f}], n={len(r_sl)}")


def plot_size_vs_objective_with_fit(
        data_sizes_bytes: np.ndarray,
        objective_values: np.ndarray,
        *,
        size_unit: float = 1024.0**3,
        winsor_q=(0.025, 0.975)
) -> tuple[float, float]:
    """
    Scatter log(1+size) vs log(1+J_beta) with the robust linear fit used for r_beta.
    Returns (a_hat, b_hat) so you can reuse if needed.

    Parameters
    ----------
    data_sizes_bytes : array-like, shape (n,)
        Input sizes per run (bytes by default).
    objective_values : array-like, shape (n,)
        J_beta per run (T if beta=1, T_R if beta=0.5).
    size_unit : float
        Unit to convert bytes -> GB (1.0 if already GB).
    winsor_q : tuple(float, float)
        Quantiles for light winsorization before fitting.

    Notes
    -----
    Fit is: logJ ~ a_hat + b_hat * s, where s = log(1+size_GB).
    """
    size_gb = (np.asarray(data_sizes_bytes, float).ravel() / float(size_unit))
    s = np.log1p(size_gb)
    J = np.asarray(objective_values, float).ravel()
    logJ = np.log1p(J)

    # winsorize for robustness
    lo_s, hi_s = np.quantile(s, winsor_q)
    lo_j, hi_j = np.quantile(logJ, winsor_q)
    s_w = np.clip(s, lo_s, hi_s)
    j_w = np.clip(logJ, lo_j, hi_j)

    # closed-form OLS for speed (same as LinearRegression on 1D)
    s1 = np.vstack([np.ones_like(s_w), s_w]).T
    # theta = (X^T X)^-1 X^T y
    XtX = s1.T @ s1
    Xty = s1.T @ j_w
    theta = np.linalg.solve(XtX, Xty)
    a_hat, b_hat = float(theta[0]), float(theta[1])

    # line for full x-range
    s_line = np.linspace(s.min(), s.max(), 100)
    j_line = a_hat + b_hat * s_line

    # plot
    plt.figure()
    plt.scatter(s, logJ, s=12)
    plt.plot(s_line, j_line)
    plt.title("Size–objective relation with robust log fit")
    plt.xlabel("s = log(1 + size_GB)")
    plt.ylabel("log(1 + J_beta)")
    plt.tight_layout()
    plt.show()

    print(f"Fit: log(1+J) = {a_hat:.4f} + {b_hat:.4f} * s")
    return a_hat, b_hat
