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

from typing import Optional, Tuple, List
import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from framework.experimental.metrics.optimization_model.evaluation import (
    EvaluationOptimizationMetrics, TargetWorkloadOptimization
)
from framework.experimental.metrics.optimization_model.metrics import RunTimeOptimizationMetrics
from framework.proposed.bayesian_optimization import OptimizationObjective
from framework.proposed.parameters import SparkParameters
from framework.proposed.workload_characterization.workload import WorkloadCharacterized, WorkloadRepository
from utils.spark.hibench import run_once_workload_hibench


# ============================================================================
# TurBO similarity (Eq. 6)  →  Dist(cur,j)=Σ_k[(a_k-b_k)/(a_k+b_k)]^2
# Convert to similarity s=1/(1+d); min-max to [ε,1] for numerical stability.
# ============================================================================
def _ratio_distance(cur: np.ndarray, base: np.ndarray, eps: float = 1e-12) -> float:
    r = (cur - base) / (cur + base + eps)
    return float(np.sum(r * r))

def _similarity_weights(w_meta: np.ndarray, w_ref: np.ndarray) -> np.ndarray:
    W = np.asarray(w_meta, float)
    v = np.asarray(w_ref, float).reshape(-1)
    if W.ndim != 2:
        raise ValueError("w_meta must be 2D [N_hist, D].")
    if v.ndim != 1 or v.shape[0] != W.shape[1]:
        raise ValueError("w_ref must be 1D with same dimensionality as w_meta columns.")
    d = np.array([_ratio_distance(v, Wi) for Wi in W], float)
    s = 1.0 / (1.0 + d)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return np.maximum(s, 1e-6)


# ============================================================================
# Meta-learning model (CASampling) — ALWAYS similarity-weighted (paper-faithful)
# ============================================================================
class MetaLearningFilter:
    """Historical meta-model for CASampling: predicts runtime from configuration."""

    def __init__(self, model_path: str = "meta_rf_model.joblib"):
        self.model_path = model_path
        self.model: Optional[GridSearchCV] = None
        self.ready = False

    def _build_rf_grid(self) -> GridSearchCV:
        pipe = Pipeline([
            ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        # (Optional) log-target; disabled by default.
        log_t = FunctionTransformer(np.log1p, np.expm1, validate=True)

        rf_ttr = TransformedTargetRegressor(
            regressor=pipe,
            transformer=None,
            check_inverse=True
        )

        param_grid = {
            "regressor__rf__n_estimators": [50, 100, 200],
            "regressor__rf__max_depth": [10, 20, None],
            "regressor__rf__max_features": ["sqrt", "log2", None],
        }

        gs = GridSearchCV(
            rf_ttr,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            n_jobs=-1,
            refit=True,
            verbose=0
        )
        return gs

    def load_or_train(
            self,
            X_meta: np.ndarray,
            y_meta: np.ndarray,
            *,
            w_meta: np.ndarray,
            w_ref: np.ndarray,
            force_retrain: bool = True
    ) -> None:
        """
        Train on historical-only data with similarity-based sample weights (TurBO Eq. 6).
        ALWAYS weighted (no unweighted fallback). If a cached model exists and is not
        marked as WEIGHTED, we ignore it and retrain to guarantee paper faithfulness.
        """
        if X_meta is None or y_meta is None or len(X_meta) == 0:
            raise ValueError("[MetaLearning] Missing historical data for CASampling.")
        if w_meta is None or w_ref is None:
            raise ValueError("[MetaLearning] w_meta and w_ref are required for weighted training.")

        # Only reuse cache if it was trained WEIGHTED and retrain is not forced
        if (not force_retrain) and os.path.exists(self.model_path):
            cached = joblib.load(self.model_path)
            if getattr(cached, "trained_with_weights_", False):
                self.model = cached
                self.ready = True
                print(f"[MetaLearning] Loaded WEIGHTED model from {self.model_path}")
                return
            else:
                print("[MetaLearning] Cached model is not WEIGHTED → retraining.")

        # Build estimator
        gs = self._build_rf_grid()
        X_meta = np.asarray(X_meta, float)
        y_meta = np.asarray(y_meta, float).ravel()

        # Compute similarity weights and route them to inner RF in TTR(Pipeline)
        sw = _similarity_weights(w_meta=w_meta, w_ref=w_ref)
        if sw.shape[0] != X_meta.shape[0]:
            raise ValueError("sample_weight length must match X_meta rows.")
        fit_params = {"rf__sample_weight": sw}

        # Fit (weighted) and annotate audit flags for persistence
        gs.fit(X_meta, y_meta, **fit_params)
        gs.trained_with_weights_ = True
        gs.training_info_ = {
            "weights_min": float(np.min(sw)),
            "weights_max": float(np.max(sw)),
            "weights_mean": float(np.mean(sw)),
        }
        print(f"[MetaLearning] Trained RF (WEIGHTED). Best params: {gs.best_params_}")
        print(f"[MetaLearning] Weight stats: {gs.training_info_}")

        joblib.dump(gs, self.model_path)
        self.model = gs
        self.ready = True
        print(f"[MetaLearning] Saved WEIGHTED model to {self.model_path}")

    def predict_score(self, candidates: np.ndarray) -> np.ndarray:
        """Predict expected runtime (lower is better)."""
        if not self.ready or self.model is None:
            raise RuntimeError("[MetaLearning] Model is not ready. Call load_or_train first.")
        C = np.asarray(candidates, float)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        return self.model.predict(C)


# ============================================================================
# TurBO baseline:
#   GP (Matérn ν=2.5) + EI, fixed Sobol pool, CASampling (meta-RF) + BO-AdaPP.
# ============================================================================
class TurBO:
    def __init__(self,
                 bounds: List[Tuple[float, float, float]],
                 *,
                 candidate_pool: np.ndarray,
                 enable_pseudopts: bool = True,
                 tau: float = 0.01,
                 lmax: int = 10):
        self.bounds = bounds

        C_pool = np.asarray(candidate_pool)
        if C_pool.ndim != 2 or C_pool.shape[1] != len(bounds):
            raise ValueError("candidate_pool must be (N, d) with d=len(bounds).")
        self.C_pool = C_pool.astype(int)

        self.scaler = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=C(1.0, (1e-3, 1e3)) *
                   Matern(length_scale=np.ones(len(bounds)),
                          nu=2.5, length_scale_bounds=(1e-2, 1e2)) +
                   WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2)),
            random_state=42
        )
        self._y_mean = 0.0
        self._y_std = 1.0
        self.current_best = float("+inf")

        self.enable_pseudopts = bool(enable_pseudopts)
        self.tau = float(tau)
        self.lmax = int(lmax)

    def _augment_with_pseudo_points(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enable_pseudopts or len(X) == 0:
            return X, y
        sX = StandardScaler().fit(X)
        Xs = sX.transform(X)
        Cs = sX.transform(self.C_pool)

        X_aug, y_aug = [X], [y]
        for i, xi in enumerate(Xs):
            dist = np.abs(Cs - xi).mean(axis=1)   # L1 (scaled)
            idx = np.where(dist <= self.tau)[0]
            if idx.size == 0:
                continue
            choose = np.random.choice(idx, size=min(self.lmax, idx.size), replace=False)
            X_aug.append(self.C_pool[choose])
            y_aug.append(np.full(len(choose), y[i], float))
        return np.vstack(X_aug), np.concatenate(y_aug)

    def fit_gp(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_fit, y_fit = self._augment_with_pseudo_points(X_train, y_train)
        self.scaler.fit(X_fit)
        Xs = self.scaler.transform(X_fit)
        self._y_mean = float(np.mean(y_fit))
        self._y_std = float(np.std(y_fit) + 1e-12)
        ys = (y_fit - self._y_mean) / self._y_std
        self.current_best = float(np.min(y_fit))
        self.gp.fit(Xs, ys)

    def predict_mu_sigma(self, x_vec: np.ndarray) -> Tuple[float, float]:
        xs = self.scaler.transform(x_vec.reshape(1, -1))
        yhat_s, std_s = self.gp.predict(xs, return_std=True)
        mu = self._y_mean + yhat_s[0] * self._y_std
        sigma = max(1e-9, std_s[0] * self._y_std)
        return float(mu), float(sigma)

    def _ei_min(self, mu: float, sigma: float, best: float) -> float:
        if sigma <= 1e-12:
            return 0.0 if mu < best else float("+inf")
        from math import erf, sqrt, exp, pi
        z = (best - mu) / sigma
        cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        pdf = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * z * z)
        ei = (best - mu) * cdf + sigma * pdf
        return -ei

    def suggest(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float, float]:
        self.fit_gp(X_train, y_train)
        best_x, best_acq = None, float("inf")
        for x in self.C_pool:
            mu, sigma = self.predict_mu_sigma(x)
            val = self._ei_min(mu, sigma, self.current_best)
            if val < best_acq:
                best_acq = val
                best_x = x
        mu_pred, _ = self.predict_mu_sigma(best_x)
        return best_x, float(best_acq), float(mu_pred)


# ============================================================================
# BO loop with CASampling (WEIGHTED) + BO-AdaPP. Counts ONLY real evals.
# ============================================================================
async def turbo_bo_loop(
        target_workload: "WorkloadCharacterized",
        X_init: np.ndarray,
        y_init: np.ndarray,
        bounds: List[Tuple[float, float, float]],
        budget: int,
        best_lhs_time: int,
        config: dict,
        *,
        X_meta: np.ndarray,
        y_meta: np.ndarray,
        candidate_pool: np.ndarray
):
    # 1) Meta-RF (CASampling) — ALWAYS weighted by similarity (Eq. 6)
    meta_filter = MetaLearningFilter()
    w_meta = np.asarray(config["w_meta"], float)   # required
    w_ref  = np.asarray(config["w_ref"], float).reshape(-1)  # required
    meta_filter.load_or_train(
        X_meta=X_meta, y_meta=y_meta,
        w_meta=w_meta, w_ref=w_ref,
        force_retrain=True  # guarantees WEIGHTED training
    )

    # 2) TurBO (GP+EI) over FIXED pool
    bo = TurBO(
        bounds=bounds,
        candidate_pool=candidate_pool,
        enable_pseudopts=bool(config.get("enable_pseudopts", True)),
        tau=float(config.get("tau", 0.01)),
        lmax=int(config.get("lmax", 10))
    )

    X = np.asarray(X_init, float).copy()
    y = np.asarray(y_init, float).copy()

    real_evals = 0
    safety_steps = 0
    safety_cap = max(budget * 50, 200)

    per_iter_times: List[float] = []
    while real_evals < budget and safety_steps < safety_cap:
        safety_steps += 1

        x_next, acq_val, mu_pred = bo.suggest(X, y)
        cfg = SparkParameters.from_vector(x_next)
        print(f"[TurBO] Step {safety_steps} | -EI={acq_val:.3f} | mu_pred(T)={mu_pred:.2f} | cfg={x_next}")

        # CASampling decision: dynamic threshold = mean of REAL times so far (or target's T if none)
        f_pred = float(meta_filter.predict_score(np.asarray(x_next).reshape(1, -1))[0])
        f_th = float(np.mean(per_iter_times)) if per_iter_times else float(target_workload.time_execution)
        suboptimal = (f_pred >= f_th)
        print(f"[TurBO][CAS] f_pred={f_pred:.2f} vs f_th={f_th:.2f} -> {'SKIP' if suboptimal else 'RUN'}")

        if suboptimal:
            X = np.vstack([X, x_next.reshape(1, -1)])
            y = np.append(y, f_pred)
            continue

        wc: WorkloadCharacterized = await run_once_workload_hibench(
            data_scale=target_workload.app_benchmark_data_size.value,
            framework=config.get("framework"),
            parameters=cfg,
            config=config
        )
        if not wc:
            print("[TurBO] Warning: failed run; retrying this step.")
            continue

        T_real = float(wc.time_execution)
        per_iter_times.append(T_real)
        real_evals += 1

        resources = OptimizationObjective.calculate_resource_usage(cfg)
        T_pred = float(mu_pred)
        of_pred = OptimizationObjective.objective_function(T=int(T_pred), R=resources, beta=config.get("beta", 1))
        of_real = OptimizationObjective.objective_function(T=int(T_real), R=resources, beta=config.get("beta", 1))
        wc.time_resources = of_real

        eval_opt_metrics = EvaluationOptimizationMetrics(
            id=wc.id,
            experiment_id=config.get("experiment_id"),
            experiment_iteration=real_evals,
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
            execution_time_error=abs(T_real - int(T_pred)),
            resource_usage_value=OptimizationObjective.calculate_resource_usage(cfg),
            objective_function_real=float(of_real),
            objective_function_predict=float(of_pred),
            repeated_config=(cfg == wc.environment)
        )
        WorkloadRepository(collection=config.get("collection_save_results")).save_optimized_workload_into_mongo(eval_opt_metrics)

        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.append(y, T_real)

    if len(per_iter_times) == 0:
        print("[TurBO] No successful evaluations; cannot compute metrics.")
        return

    T_best = float(min(per_iter_times))
    i_best = 1 + int(np.argmin(per_iter_times))
    T_first = float(per_iter_times[0])

    SU = RunTimeOptimizationMetrics.speedup(float(target_workload.time_execution), T_best)
    TC = RunTimeOptimizationMetrics.tuning_cost(per_iter_times)
    Hit10 = RunTimeOptimizationMetrics.hit_at_epsilon(
        per_iter_times, best_time=best_lhs_time, epsilon=0.10, as_percentage=True
    )
    nAOCC = RunTimeOptimizationMetrics.naocc(per_iter_times)

    print("\n=== Metrics (10 iterations / real evals) — TurBO baseline ===")
    print(f"T best ↓ : {T_best:.2f}  (found at i={i_best})")
    print(f"T first ↓: {T_first:.2f}")
    print(f"SU (%) ↑ : {SU:.2f}")
    print(f"TC ↓     : {TC:.2f}")
    print(f"Hit@0.10 ↑ : {Hit10:.2f}")
    print(f"nAOCC ↓  : {nAOCC:.4f}")
