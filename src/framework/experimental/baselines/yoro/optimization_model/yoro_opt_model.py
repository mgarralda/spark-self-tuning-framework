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

from typing import List, Tuple, Optional, Literal
import numpy as np
from math import erf, sqrt, exp, pi
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from framework.experimental.metrics.optimization_model.metrics import RunTimeOptimizationMetrics
from framework.proposed.bayesian_optimization import OptimizationObjective
from framework.proposed.workload_characterization.workload import WorkloadCharacterized, WorkloadRepository
from framework.proposed.parameters import SparkParameters
from framework.experimental.metrics.optimization_model.evaluation import (
    EvaluationOptimizationMetrics, TargetWorkloadOptimization
)
from utils.spark.hibench import run_once_workload_hibench


# -----------------------------------------------------------------------------------------
# FAIRNESS:
#   • Use the SAME fixed Sobol candidate pool across methods (TurBO, NaïveBO, YORO, Ours).
#   • Oracle (RF) calls are cheap and NOT counted in TC; only real runs count.
#   • SBO mode below is a simulated-BO loop over the oracle, as in YORO.
# -----------------------------------------------------------------------------------------


# ================== RF Oracle with scaler selection (paper-faithful) ==================
def get_random_forest_regressor_model(
        X: List,
        y: List,
) -> Tuple[GridSearchCV, Optional[StandardScaler]]:
    """
    Random-Forest oracle as in YORO/SBO (Buchaca et al., 2020).
    CV chooses whether to scale features (StandardScaler) or not (passthrough).
    RF grid keeps the model-selection spirit of the original approach.
    Returns: (fitted GridSearchCV pipeline, chosen StandardScaler or None)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    pipe = Pipeline([
        ("feature_scaler", StandardScaler()),  # may be replaced by 'passthrough' via grid
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Optional log-target; disabled by default for robustness.
    log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
    rf_ttr = TransformedTargetRegressor(
        regressor=pipe,
        transformer=None,  # set to log_transformer if your target distribution benefits
        check_inverse=True
    )

    param_grid = {
        "regressor__feature_scaler": ["passthrough", StandardScaler()],
        "regressor__rf__n_estimators": [50, 100, 200],
        "regressor__rf__max_depth": [10, 20, None],
        "regressor__rf__max_features": ["sqrt", "log2", None],
    }

    rfr = GridSearchCV(
        estimator=rf_ttr,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        refit=True,
        verbose=0
    )
    rfr.fit(X, y)

    scaler = None
    best_est = rfr.best_estimator_
    if hasattr(best_est, "regressor") and hasattr(best_est.regressor, "named_steps"):
        scaler = best_est.regressor.named_steps.get("feature_scaler", None)

    return rfr, scaler


# ===================== SBO Optimizer (GP + EI) over s using RF oracle =====================
class AcquisitionFunction:
    """
    Simulated Bayesian Optimization (YORO/SBO) over the Spark config subvector s.

    • Surrogate GP is trained on (s, ŷ_RF(s)) pairs; “observations” are oracle predictions.
    • Acquisition = Expected Improvement (EI) for minimization, identical policy to TurBO.
    • Candidates = FIXED Sobol pool shared by all methods (fairness).

    Implementation mirrors TurBO details for the surrogate:
      - Standardize inputs s via StandardScaler.
      - Manually standardize targets (time) to zero mean / unit variance before GP.fit.
      - Convert predictions back to original time units to compute EI.
    """

    def __init__(
            self,
            model: GridSearchCV,
            bounds: List[Tuple[float, float, float]],
            workload_ref: List,                 # full vector (x||s0) in your layout
            seed: int = 24,
            candidate_pool: np.ndarray | None = None
    ) -> None:
        self.model = model
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.d = len(bounds)

        # Reference full vector (x||s0) in your layout (last d = s)
        self._x_ref_full = np.array(workload_ref[0], dtype=float)
        self._s0 = self._x_ref_full[-self.d:].astype(int)

        # Fixed Sobol pool (FAIRNESS)
        if candidate_pool is None:
            raise ValueError("AcquisitionFunction requires a FIXED Sobol candidate_pool shared across methods.")
        C_pool = np.asarray(candidate_pool)
        if C_pool.ndim != 2 or C_pool.shape[1] != self.d:
            raise ValueError("candidate_pool must be (N, d) with d=len(bounds).")
        self.pool = C_pool.astype(int)
        self._unused_mask = np.ones(len(self.pool), dtype=bool)

        # GP over s -> ŷ_RF(s) with the same kernel as TurBO
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(self.d), nu=2.5, length_scale_bounds=(1e-2, 1e2)
        ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))
        self.gp = GaussianProcessRegressor(kernel=kernel, random_state=seed)

        # Input scaler for s (like TurBO); y standardization handled manually
        self._x_scaler = StandardScaler()
        self._y_mean = 0.0
        self._y_std = 1.0
        self._incumbent = float("+inf")  # best observed (in original units)

        # SBO memory
        self.S_hist: List[np.ndarray] = []
        self.Y_hist: List[float] = []

    # ----- helpers -----
    def _build_z(self, s_vec: np.ndarray) -> np.ndarray:
        """Build z = (x||s) preserving current layout (last d = s)."""
        z = self._x_ref_full.copy()
        z[-self.d:] = s_vec
        return z

    def _oracle_predict(self, s_vec: np.ndarray) -> float:
        """RF oracle prediction in seconds (pipeline handles scaling internally if selected by CV)."""
        z = self._build_z(s_vec).reshape(1, -1)
        y_hat = self.model.predict(z)
        return float(y_hat[0])

    def _fit_gp(self) -> None:
        """Fit GP on standardized inputs and standardized targets (TurBO-style)."""
        X = np.vstack(self.S_hist)                         # (n, d)
        y = np.asarray(self.Y_hist, dtype=float)          # (n,)
        self._x_scaler.fit(X)
        Xs = self._x_scaler.transform(X)

        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y) + 1e-12)
        ys = (y - self._y_mean) / self._y_std

        self._incumbent = float(np.min(y))                # best in ORIGINAL units
        self.gp.fit(Xs, ys)

    def _predict_mu_sigma(self, s_vec: np.ndarray) -> Tuple[float, float]:
        """Predict μ, σ in ORIGINAL time units for a single s."""
        xs = self._x_scaler.transform(s_vec.reshape(1, -1))
        yhat_s, std_s = self.gp.predict(xs, return_std=True)
        mu = self._y_mean + yhat_s[0] * self._y_std
        sigma = max(1e-9, std_s[0] * self._y_std)
        return float(mu), float(sigma)

    @staticmethod
    def _ei_min(mu: float, sigma: float, best: float) -> float:
        """
        One-point Expected Improvement for minimization.
        We return -EI so we can argmin over the acquisition values.
        """
        if sigma <= 1e-12:
            return 0.0 if mu < best else float("+inf")
        z = (best - mu) / sigma
        cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        pdf = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * z * z)
        ei = (best - mu) * cdf + sigma * pdf
        return -ei  # minimize(-EI) == maximize(EI)

    # ----- SBO steps -----
    def _gp_update_with(self, s_vec: np.ndarray, y_hat: float) -> None:
        """Append (s, ŷ) and refit the GP."""
        self.S_hist.append(np.asarray(s_vec, dtype=float))
        self.Y_hist.append(float(y_hat))
        self._fit_gp()

    def _suggest_one(self) -> Tuple[float, np.ndarray]:
        """One EI step over the UNUSED subset of the pool."""
        cand_pool = self.pool[self._unused_mask]
        # Compute -EI for each candidate and pick the minimum
        best_idx = None
        best_acq = float("+inf")
        for j, s in enumerate(cand_pool):
            mu, sigma = self._predict_mu_sigma(s)
            val = self._ei_min(mu, sigma, self._incumbent)
            if val < best_acq:
                best_acq = val
                best_idx = j

        s_next = cand_pool[int(best_idx)]

        # mark as used in the original pool indexing
        original_indices = np.flatnonzero(self._unused_mask)
        self._unused_mask[original_indices[int(best_idx)]] = False

        # query oracle and update GP
        y_hat = self._oracle_predict(s_next)
        self._gp_update_with(s_next, y_hat)
        return y_hat, s_next

    def suggest(self, n_best_x: int) -> List[Tuple[float, np.ndarray]]:
        """
        Return n_best_x suggestions as (ŷ_RF(s), s) by running a simulated BO loop (EI).
        Seed with s0 to stabilize the first step.
        """
        if n_best_x <= 0:
            return []

        # seed GP with (s0, ŷ_RF(s0))
        y0 = self._oracle_predict(self._s0)
        self.S_hist = [self._s0.astype(float)]
        self.Y_hist = [float(y0)]
        self._fit_gp()

        selected: List[Tuple[float, np.ndarray]] = []
        for _ in range(n_best_x):
            y_hat, s_next = self._suggest_one()
            selected.append((float(y_hat), s_next))
            if not self._unused_mask.any():
                break

        # Optional: sort by predicted time
        selected.sort(key=lambda t: t[0])
        return selected


# ===================== Greedy selection (no BO): rank full pool by oracle =====================
def yoro_greedy_topk(
        model: GridSearchCV,
        workload_ref: List,               # full (x||s0)
        candidate_pool: np.ndarray,
        k: int
) -> List[Tuple[float, np.ndarray]]:
    """
    Greedy YORO (no BO): score the ENTIRE shared Sobol pool with the RF oracle and
    select the top-k configurations with the smallest predicted runtime.
    """
    if k <= 0:
        return []

    x_ref = np.array(workload_ref[0], dtype=float)
    d = candidate_pool.shape[1]

    # Build Z efficiently: replicate x_ref and splice s at the tail
    S = np.asarray(candidate_pool, dtype=float)
    Z = np.tile(x_ref, (S.shape[0], 1))
    Z[:, -d:] = S

    y_hat = model.predict(Z).astype(float)  # vectorized oracle predictions
    order = np.argsort(y_hat)               # ascending predicted time
    top_idx = order[:k]
    top_pairs = [(float(y_hat[i]), candidate_pool[i].astype(int)) for i in top_idx]
    return top_pairs


# =============================== YORO (Greedy or SBO) with shared metrics ===============================
async def yoro_loop(
        target_workload: WorkloadCharacterized,
        target_full_vector: List,                 # (x||s0) in your layout
        X_init: List,                             # historical + target-default
        y_init: List,
        bounds: List[Tuple[float, float, float]],
        budget: int,
        best_lhs_time: int,
        config: dict,
        candidate_pool: np.ndarray,               # FIXED Sobol pool (shared)
        mode: Literal["greedy", "sbo"] = "sbo"
):
    """
    Unified YORO baseline:
      - mode="greedy": rank the ENTIRE pool by oracle prediction and run the top-K (K=budget).
      - mode="sbo": run a simulated BO (GP+EI over ŷ_RF) to produce K suggestions.

    In both cases we execute exactly K real runs and report the SAME metrics
    over up to 10 successful evaluations (T_best, T_first, SU, TC, Hit@0.10, nAOCC).
    """
    # 1) Train the RF oracle (paper-faithful; scaler choice via CV)
    rfr, _ = get_random_forest_regressor_model(X=X_init, y=y_init)

    # 2) Select K=budget configs according to mode, always using the SAME pool
    if mode == "sbo":
        acq = AcquisitionFunction(
            model=rfr,
            bounds=bounds,
            workload_ref=target_full_vector,
            seed=42,
            candidate_pool=candidate_pool
        )
        selection = acq.suggest(n_best_x=budget)
        print(f"[YORO/SBO] Selected {len(selection)} candidates via GP+EI over oracle.")
        for _, s_next in selection:
            print(f"cfg={s_next}")
    else:
        selection = yoro_greedy_topk(
            model=rfr,
            workload_ref=target_full_vector,
            candidate_pool=candidate_pool,
            k=budget
        )
        print(f"[YORO/Greedy] Selected {len(selection)} candidates by oracle ranking.")
        for _, s_next in selection:
            print(f"cfg={s_next}")

    # 3) Execute the selected K configs (same logging/metrics as other baselines)
    per_iter_times: List[float] = []
    successes = 0
    it = 0
    current_failures = 0
    max_allowed_failures = int(max(1, budget * 1.0))  # conservative cap

    for pred_time, s_next in selection:
        if successes >= budget:
            break

        cfg = SparkParameters.from_vector(s_next)
        it += 1
        print(f"[YORO/{mode.upper()}] Iter {it}: T_pred={float(pred_time):.2f} | cfg={s_next}")

        wc: WorkloadCharacterized = await run_once_workload_hibench(
            data_scale=target_workload.app_benchmark_data_size.value,
            framework=config.get("framework"),
            parameters=cfg,
            config=config
        )

        if wc:
            T_real = float(wc.time_execution)
            per_iter_times.append(T_real)
            successes += 1

            resources = OptimizationObjective.calculate_resource_usage(cfg)
            of_pred = OptimizationObjective.objective_function(T=float(pred_time), R=resources, beta=config.get("beta", 1))
            of_real = OptimizationObjective.objective_function(T=T_real, R=resources, beta=config.get("beta", 1))
            wc.time_resources = of_real

            eval_opt_metrics = EvaluationOptimizationMetrics(
                id=wc.id,
                experiment_id=config.get("experiment_id"),
                experiment_iteration=successes,  # successful eval index
                target_workload=TargetWorkloadOptimization(
                    id=target_workload.id,
                    execution_time=target_workload.time_execution,
                    name=target_workload.app_benchmark_workload,
                    input_data_size=target_workload.app_benchmark_data_size.value,
                    configuration=target_workload.environment
                ),
                acquisition_function_score=float(pred_time),  # log oracle’s predicted T
                configuration=cfg,
                execution_time=T_real,
                execution_time_error=abs(T_real - int(pred_time)),
                resource_usage_value=OptimizationObjective.calculate_resource_usage(cfg),
                objective_function_real=float(of_real),
                objective_function_predict=float(of_pred),
            )
            WorkloadRepository(collection=config.get("collection_save_results")).save_optimized_workload_into_mongo(eval_opt_metrics)

            print(
                f"\n{'=' * 150}\n"
                f"[YORO/{mode.upper()}] Evaluated (success #{successes}):\n"
                f"    T_real={T_real:.2f} | T_pred={float(pred_time):.2f} | OF_real={float(of_real):.2f}\n"
                f"    cfg={s_next}\n"
                f"{'=' * 150}\n"
            )

            # Keep strict comparability at 10 realized evals when budget allows
            if len(per_iter_times) >= 10:
                break

        else:
            print(
                f"\n{'=' * 150}\n"
                f"[YORO/{mode.upper()}] Warning: this run FAILED. Will not count towards the 10 evals.\n"
                f"{'=' * 150}\n"
            )
            current_failures += 1
            if current_failures >= max_allowed_failures:
                print(f"[YORO/{mode.upper()}] Maximum allowed failures ({max_allowed_failures}) reached. Stopping.")
                break

    # ---- Metrics (same printout used elsewhere) ----
    if len(per_iter_times) == 0:
        print(f"[YORO/{mode.upper()}] No successful evaluations; cannot compute metrics.")
        return

    times10 = per_iter_times[:10]
    T_best = float(min(times10))
    i_best = 1 + int(np.argmin(times10))
    T_first = float(times10[0])

    SU = RunTimeOptimizationMetrics.speedup(float(target_workload.time_execution), T_best)
    TC = RunTimeOptimizationMetrics.tuning_cost(times10)
    Hit10 = RunTimeOptimizationMetrics.hit_at_epsilon(
        times10, best_time=best_lhs_time, epsilon=0.10, as_percentage=True
    )
    nAOCC = RunTimeOptimizationMetrics.naocc(times10)

    print(f"\n=== Metrics (10 iterations) — YORO/{mode.upper()} ===")
    print(f"T best ↓   : {T_best:.2f}  (found at i={i_best})")
    print(f"T first ↓  : {T_first:.2f}")
    print(f"SU (%) ↑   : {SU:.2f}")
    print(f"TC ↓       : {TC:.2f}")
    print(f"Hit@0.10 ↑ : {Hit10:.2f}")
    print(f"nAOCC ↓    : {nAOCC:.4f}")
