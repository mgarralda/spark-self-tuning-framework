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


from typing import List, Tuple, Optional
import numpy as np
from math import erf, sqrt, exp, pi
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

from framework.experimental.metrics.optimization_model.evaluation import (
    EvaluationOptimizationMetrics,
    TargetWorkloadOptimization,
)
from framework.experimental.metrics.optimization_model.metrics import RunTimeOptimizationMetrics
from framework.proposed.bayesian_optimization import OptimizationObjective
from framework.proposed.parameters import SparkParameters
from framework.proposed.workload_characterization.workload import (
    WorkloadCharacterized,
    WorkloadRepository,
)
from utils.spark.hibench import run_once_workload_hibench


# =========================================================================================
# Acquisition (GP + EI) over a FIXED, SHARED candidate pool
# - Matches TurBO’s surrogate: Matern(ν=2.5) + Constant + White
# - Standardize inputs (X) with StandardScaler
# - Manually standardize targets (y) → GP is fit on standardized y
# - EI is computed in ORIGINAL time units; we return -EI so we can argmin
# =========================================================================================
class NaiveAcquisitionEI:
    def __init__(self,
                 bounds: List[Tuple[int, int, int]],
                 candidate_pool: np.ndarray,
                 seed: int = 42) -> None:
        self.d = len(bounds)
        C_pool = np.asarray(candidate_pool)
        if C_pool.ndim != 2 or C_pool.shape[1] != self.d:
            raise ValueError("candidate_pool must be (N, d) with d=len(bounds).")
        self.pool = C_pool.astype(int)
        self._unused_mask = np.ones(len(self.pool), dtype=bool)

        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(self.d),
            nu=2.5,
            length_scale_bounds=(1e-2, 1e2)
        ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))
        self.gp = GaussianProcessRegressor(kernel=kernel, random_state=seed)

        self.x_scaler = StandardScaler()
        self._y_mean = 0.0
        self._y_std = 1.0
        self._incumbent = float("+inf")

        # Train data cache
        self.X = None  # (n,d) float
        self.y = None  # (n,) float

    def mark_used_vector(self, x_vec: np.ndarray) -> None:
        """Mask a specific vector as already used (e.g., default & warm-start seeds)."""
        if self.pool.size == 0:
            return
        eq = np.all(self.pool == np.asarray(x_vec, dtype=int), axis=1)
        self._unused_mask[eq] = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP on standardized X and standardized y; store y stats and incumbent in ORIGINAL units."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.X, self.y = X, y

        self.x_scaler.fit(X)
        Xs = self.x_scaler.transform(X)

        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y) + 1e-12)
        ys = (y - self._y_mean) / self._y_std

        self._incumbent = float(np.min(y))
        self.gp.fit(Xs, ys)

    def _predict_mu_sigma(self, x_vec: np.ndarray) -> Tuple[float, float]:
        """Predict μ, σ in ORIGINAL time units for a single x."""
        xs = self.x_scaler.transform(x_vec.reshape(1, -1))
        yhat_s, std_s = self.gp.predict(xs, return_std=True)
        mu = self._y_mean + yhat_s[0] * self._y_std
        sigma = max(1e-9, std_s[0] * self._y_std)
        return float(mu), float(sigma)

    @staticmethod
    def _ei_min(mu: float, sigma: float, best: float) -> float:
        """One-point Expected Improvement for minimization; return -EI so we can argmin."""
        if sigma <= 1e-12:
            return 0.0 if mu < best else float("+inf")
        z = (best - mu) / sigma
        cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        pdf = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * z * z)
        ei = (best - mu) * cdf + sigma * pdf
        return -ei  # minimize(-EI) == maximize(EI)

    def suggest(self) -> Tuple[np.ndarray, float, float]:
        """
        Pick x* = argmin(-EI) over UNUSED candidates.
        Returns:
            x_best  : selected configuration (np.ndarray, int)
            acq_val : -EI value at x_best (lower is better)
            mu_pred : GP posterior mean (predicted runtime) at x_best (float, seconds)
        """
        cand_pool = self.pool[self._unused_mask]
        if cand_pool.size == 0:
            raise RuntimeError("Candidate pool is exhausted.")

        best_idx = None
        best_acq = float("+inf")
        best_mu = float("+inf")

        for j, x in enumerate(cand_pool):
            mu, sigma = self._predict_mu_sigma(x)
            val = self._ei_min(mu, sigma, self._incumbent)
            if val < best_acq:
                best_acq = val
                best_idx = j
                best_mu = mu

        x_best = cand_pool[int(best_idx)]
        # Mark used in the original mask
        original_indices = np.flatnonzero(self._unused_mask)
        self._unused_mask[original_indices[int(best_idx)]] = False

        return x_best, float(best_acq), float(best_mu)


# =========================================================================================
# Naïve BO loop (strict LOGO): GP + EI, initialized with 3 Sobol samples (no target used)
# - Iters 1–3: execute three Sobol seeds from the SAME fixed candidate pool
# - Then run BO+EI for the remaining budget, masking used candidates
# - Report the same metrics, capped at 10 realized evaluations
# =========================================================================================
async def naive_bo_loop(
        target_workload: "WorkloadCharacterized",
        bounds: List[Tuple[int, int, int]],
        budget: int,
        best_lhs_time: int,
        config: dict,
        *,
        candidate_pool: np.ndarray
):
    """
    Naïve BO (GP + EI) under strict LOGO:
      • Initialization: 3 Sobol seeds from the shared pool (no target data used).
      • BO phase: EI over the remaining unused candidates until reaching the budget.

    Notes:
      - Fixed Sobol pool is REQUIRED (fairness across methods).
      - With β=1, OptimizationObjective reduces to runtime-only.
    """
    pool = np.asarray(candidate_pool, dtype=int)
    assert budget >= 1, "Budget must be >= 1."

    per_iter_times: List[float] = []
    X_arr = np.empty((0, len(bounds)), dtype=float)
    y_arr = np.empty((0,), dtype=float)

    # -------------------- Iters 1–3: Sobol seeds --------------------
    n_ws = min(3, budget)
    sobol_seeds: List[np.ndarray] = []
    seen = set()
    for row in pool:
        key = tuple(row.tolist())
        if key in seen:
            continue
        seen.add(key)
        sobol_seeds.append(row)
        if len(sobol_seeds) >= n_ws:
            break

    acq = NaiveAcquisitionEI(bounds=bounds, candidate_pool=pool, seed=42)

    it = 0
    current_failures = 0
    max_allowed_failures = max(1, int(budget * 0.3))

    for x0 in sobol_seeds:
        it += 1
        cfg0 = SparkParameters.from_vector(x0)
        print(f"[NaiveBO][Sobol] Iter {it}: executing seed → cfg={x0}")

        wc0: Optional["WorkloadCharacterized"] = await run_once_workload_hibench(
            data_scale=target_workload.app_benchmark_data_size.value,
            framework=config.get("framework"),
            parameters=cfg0,
            config=config
        )
        if not wc0:
            print("[NaiveBO][Sobol] Warning: run FAILED. Skipping.")
            it -= 1
            current_failures += 1
            if current_failures >= max_allowed_failures:
                print("[NaiveBO] Too many failures during seeds. Stopping.")
                return
            continue

        T_real0 = float(wc0.time_execution)
        per_iter_times.append(T_real0)

        X_arr = np.vstack([X_arr, x0.astype(float).reshape(1, -1)])
        y_arr = np.append(y_arr, T_real0)

        # β=1 → runtime-only objective for logging
        R0 = OptimizationObjective.calculate_resource_usage(cfg0)
        of_real0 = OptimizationObjective.objective_function(T=T_real0, R=R0, beta=config.get("beta", 1))
        wc0.time_resources = of_real0

        eval_ws = EvaluationOptimizationMetrics(
            id=wc0.id,
            experiment_id=config.get("experiment_id"),
            experiment_iteration=it,
            target_workload=TargetWorkloadOptimization(
                id=target_workload.id,
                execution_time=target_workload.time_execution,
                name=target_workload.app_benchmark_workload,
                input_data_size=target_workload.app_benchmark_data_size.value,
                configuration=target_workload.environment
            ),
            acquisition_function_score=float("nan"),
            configuration=cfg0,
            execution_time=T_real0,
            execution_time_error=0.0,
            resource_usage_value=R0,
            objective_function_real=float(of_real0),
            objective_function_predict=float("nan"),
            repeated_config=(cfg0 == target_workload.environment)
        )
        WorkloadRepository(collection=config.get("collection_save_results")).save_optimized_workload_into_mongo(eval_ws)
        acq.mark_used_vector(x0)

        print(f"[NaiveBO][Sobol] T_real={T_real0:.2f} | cfg={x0}")

        if len(per_iter_times) >= min(10, budget):
            # If we already hit the reporting cap, compute metrics and exit
            times10 = per_iter_times[:10]
            T_best = float(min(times10))
            i_best = 1 + int(np.argmin(times10))
            T_first = float(times10[0])
            SU = RunTimeOptimizationMetrics.speedup(float(target_workload.time_execution), T_best)
            TC = RunTimeOptimizationMetrics.tuning_cost(times10)
            # Hit@0.10 is optional; not used in the main runtime table
            nAOCC = RunTimeOptimizationMetrics.naocc(times10)

            print("\n=== Metrics (≤10 iterations) — Naïve BO (GP + EI) ===")
            print(f"T best ↓   : {T_best:.2f} (found at i={i_best})")
            print(f"T first ↓  : {T_first:.2f}")
            print(f"SU (%) ↑   : {SU:.2f}")
            print(f"TC ↓       : {TC:.2f}")
            print(f"nAOCC ↓    : {nAOCC:.4f}")
            return

    # -------------------- BO phase (remaining budget) --------------------
    remaining = max(0, int(budget) - len(per_iter_times))
    if remaining <= 0:
        times10 = per_iter_times[:10]
        T_best = float(min(times10))
        i_best = 1 + int(np.argmin(times10))
        T_first = float(times10[0])
        SU = RunTimeOptimizationMetrics.speedup(float(target_workload.time_execution), T_best)
        TC = RunTimeOptimizationMetrics.tuning_cost(times10)
        nAOCC = RunTimeOptimizationMetrics.naocc(times10)

        print("\n=== Metrics (≤10 iterations) — Naïve BO (GP + EI) ===")
        print(f"T best ↓   : {T_best:.2f} (found at i={i_best})")
        print(f"T first ↓  : {T_first:.2f}")
        print(f"SU (%) ↑   : {SU:.2f}")
        print(f"TC ↓       : {TC:.2f}")
        print(f"nAOCC ↓    : {nAOCC:.4f}")
        return

    acq.fit(X_arr, y_arr)
    print(f"[NaiveBO] Remaining BO iterations: {remaining}")

    while remaining > 0:
        remaining -= 1
        it += 1

        try:
            x_next, acq_val, mu_pred = acq.suggest()
        except RuntimeError as e:
            print(f"[NaiveBO] {e} Ending early.")
            break

        cfg = SparkParameters.from_vector(x_next)
        print(f"[NaiveBO] Iter {it} | -EI={acq_val:.3f} | mu_pred(T)={mu_pred:.2f} | cfg={x_next}")

        wc: Optional["WorkloadCharacterized"] = await run_once_workload_hibench(
            data_scale=target_workload.app_benchmark_data_size.value,
            framework=config.get("framework"),
            parameters=cfg,
            config=config
        )
        if not wc:
            print("[NaiveBO] Warning: run FAILED. Skipping iteration.")
            it -= 1
            current_failures += 1
            if current_failures >= max_allowed_failures:
                print("[NaiveBO] Too many failures. Stopping optimization.")
                break
            continue

        T_real = float(wc.time_execution)
        per_iter_times.append(T_real)

        R = OptimizationObjective.calculate_resource_usage(cfg)
        T_pred = float(mu_pred)
        of_pred = OptimizationObjective.objective_function(T=T_pred, R=R, beta=config.get("beta", 1))
        of_real = OptimizationObjective.objective_function(T=T_real, R=R, beta=config.get("beta", 1))
        wc.time_resources = of_real

        eval_opt_metrics = EvaluationOptimizationMetrics(
            id=wc.id,
            experiment_id=config.get("experiment_id"),
            experiment_iteration=it,
            target_workload=TargetWorkloadOptimization(
                id=target_workload.id,
                execution_time=target_workload.time_execution,
                name=target_workload.app_benchmark_workload,
                input_data_size=target_workload.app_benchmark_data_size.value,
                configuration=target_workload.environment
            ),
            acquisition_function_score=float(acq_val),
            configuration=cfg,
            execution_time=T_real,
            execution_time_error=abs(int(T_real) - int(T_pred)),
            resource_usage_value=R,
            objective_function_real=float(of_real),
            objective_function_predict=float(of_pred),
            repeated_config=(cfg == target_workload.environment)
        )
        WorkloadRepository(collection=config.get("collection_save_results")).save_optimized_workload_into_mongo(eval_opt_metrics)

        print(f"[NaiveBO] Iter {it}: T_real={T_real:.2f} | cfg={x_next}")

        X_arr = np.vstack([X_arr, x_next.astype(float).reshape(1, -1)])
        y_arr = np.append(y_arr, T_real)
        acq.fit(X_arr, y_arr)

        if len(per_iter_times) >= 10:
            break

    # -------------------- Final metrics (≤10) --------------------
    if len(per_iter_times) == 0:
        print("[NaiveBO] No successful evaluations; cannot compute metrics.")
        return

    times10 = per_iter_times[:10]
    T_best = float(min(times10))
    i_best = 1 + int(np.argmin(times10))
    T_first = float(times10[0])  # first Sobol run
    Hit10 = RunTimeOptimizationMetrics.hit_at_epsilon(
        per_iter_times, best_time=best_lhs_time, epsilon=0.10, as_percentage=True
    )
    SU = RunTimeOptimizationMetrics.speedup(float(target_workload.time_execution), T_best)
    TC = RunTimeOptimizationMetrics.tuning_cost(times10)
    nAOCC = RunTimeOptimizationMetrics.naocc(times10)

    print("\n=== Metrics (≤10 iterations) — Naïve BO (GP + EI) ===")
    print(f"T best ↓   : {T_best:.2f} (found at i={i_best})")
    print(f"T first ↓  : {T_first:.2f}")
    print(f"SU (%) ↑   : {SU:.2f}")
    print(f"TC ↓       : {TC:.2f}")
    print(f"Hit@0.10 ↑ : {Hit10:.2f}")
    print(f"nAOCC ↓    : {nAOCC:.4f}")

