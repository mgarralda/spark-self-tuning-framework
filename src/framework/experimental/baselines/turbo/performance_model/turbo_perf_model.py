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

import traceback
import warnings
from typing import List
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from framework.experimental.metrics.performance_model.evaluation import EvaluationMetrics


# ======================================================================
# Similarity stage (TurBO Eq.(6): ratio-based distance; no k, no thresholds)
# ======================================================================
class SimilarityStage:
    """
    Workload similarity as in TurBO (Dou et al., 2023, Eq.(6)):
        Dist(cur, j) = sum_k [ (a_k - b_k) / (a_k + b_k) ]^2

    We compute distances between the target descriptor (w_ref) and each TRAIN
    SAMPLE descriptor (rows of w_train), then convert to similarities s = 1/(1+d)
    and normalize to [0,1] for numerical stability (with a small floor).
    """

    @staticmethod
    def _ratio_distance(cur: np.ndarray, base: np.ndarray, eps: float = 1e-12) -> float:
        num = cur - base
        den = cur + base + eps
        r = num / den
        return float(np.sum(r * r))

    def sample_similarities(self, w_train: np.ndarray, w_ref: np.ndarray) -> np.ndarray:
        """
        Args:
            w_train: (N_train, D) descriptors per TRAIN sample.
            w_ref:   (D,) descriptor of the TARGET sample.
        Returns:
            sim: (N_train,) similarities in [~1e-6, 1], higher = closer.
        """
        W = np.asarray(w_train, dtype=float)
        v = np.asarray(w_ref, dtype=float).reshape(-1)
        if W.ndim != 2:
            raise ValueError("w_train must be 2D [N_train, D].")
        if v.ndim != 1 or v.shape[0] != W.shape[1]:
            raise ValueError("w_ref must be 1D with same dimensionality as w_train columns.")

        d = np.array([self._ratio_distance(v, Wi) for Wi in W], dtype=float)
        s = 1.0 / (1.0 + d)
        # Normalize to [0,1] (preserves order). Use tiny epsilon to avoid div-by-zero.
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        # Small floor to avoid all-zero weights in degenerate folds
        s = np.maximum(s, 1e-6)
        return s


# ======================================================================
# Supervised stage (single RF with sample weights; no k)
# ======================================================================
class SupervisedStage:
    """
    Single Random Forest surrogate trained with per-sample weights derived
    from similarity (CASampling principle, without ensembling/thresholds).
    """

    def get_weighted_rf(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: np.ndarray,
    ) -> GridSearchCV:
        """
        RF inside TransformedTargetRegressor. GridSearchCV chooses
        {StandardScaler | passthrough} and RF hyperparameters.

        IMPORTANT: fit params must be routed to the RF step as
                   'regressor__rf__sample_weight' so that Pipeline and CV
                   index them per-fold correctamente.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        sw = np.asarray(sample_weight, dtype=float).ravel()
        if sw.shape[0] != X.shape[0]:
            raise ValueError("sample_weight must have the same length as X.")

        pipe = Pipeline([
            ("feature_scaler", StandardScaler()),            # may be swapped by grid
            ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])

        # Optional log-target (kept off by default; enable if heavy-tailed)
        log_t = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
        rf_ttr = TransformedTargetRegressor(
            regressor=pipe,
            transformer=None,  # set to log_t if needed
            check_inverse=True,
        )

        param_grid = {
            "regressor__feature_scaler": ["passthrough", StandardScaler()],
            "regressor__rf__n_estimators": [50, 100, 200],
            "regressor__rf__max_depth": [10, 20, None],
            "regressor__rf__max_features": ["sqrt", "log2", None],
        }

        gs = GridSearchCV(
            estimator=rf_ttr,
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

        gs.fit(X, y, **{"rf__sample_weight": sw})
        return gs


# ======================================================================
# DouPerformanceModel (drop-in name; CASampling-aligned; no k)
# ======================================================================
class DouPerformanceModel:
    """
    TurBO-aligned performance model (prediction-only), without k-nearest selection:

      1) Compute similarity between the target descriptor and each TRAIN sample
         descriptor using Eq.(6) (ratio distance).
      2) Convert distances to similarities and use them as per-sample weights.
      3) Train a single RF (with GridSearchCV) using those weights.
      4) Predict runtime for the given configuration(s).

    Notes:
      • No numeric assumptions on group labels; inputs are per-sample.
      • No ensembles, no thresholds, no hard neighbor counts.
    """

    def __init__(self) -> None:
        self.sim_stage = SimilarityStage()
        self.supervised_stage = SupervisedStage()

    def predict(
            self,
            *,
            w_train: np.ndarray,     # (N_train, D)
            cs_train: np.ndarray,    # (N_train, C)
            t_train: np.ndarray,     # (N_train,)
            w_ref: np.ndarray,       # (D,)
            cs_query: np.ndarray,    # (C,) or (M, C)
    ) -> np.ndarray:
        # 1) Similarity weights per TRAIN sample
        sw = self.sim_stage.sample_similarities(
            w_train=np.asarray(w_train, float),
            w_ref=np.asarray(w_ref, float),
        )
        # 2) Fit weighted RF
        regressor = self.supervised_stage.get_weighted_rf(
            X=np.asarray(cs_train, float),
            y=np.asarray(t_train, float),
            sample_weight=sw,
        )
        # 3) Predict
        Xq = np.asarray(cs_query, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        return regressor.predict(Xq)


# ======================================================================
# CrossValidation (same public API; works with non-numeric group labels)
# ======================================================================
class CrossValidation:
    """
    Cross-validation harness for the similarity-weighted performance model.
    Reporting: pooled (micro) metrics + macro summaries across folds.
    Inputs remain per SAMPLE (w, w_groups, cs, t), so existing pipelines work unchanged.
    """

    def __init__(
            self,
            w: np.ndarray,          # descriptors per SAMPLE (N × D)
            w_groups: np.ndarray,   # group labels per SAMPLE (may be strings)
            cs: np.ndarray,         # configs per SAMPLE (N × C)
            t: np.ndarray,          # times per SAMPLE (N,)
    ) -> None:
        self.w = np.asarray(w, float)
        self.w_groups = np.asarray(w_groups)  # can be strings; never cast to int
        self.cs = np.asarray(cs, float)
        self.t = np.asarray(t, float)
        self.model = DouPerformanceModel()

    def leave_one_out(self) -> EvaluationMetrics:
        """
        LOO over SAMPLES: hold out one sample; train on the rest.
        Each fold predicts the left-out sample using similarity-weighted RF.
        """
        y_true_all, y_pred_all = [], []
        fold_metrics: List[EvaluationMetrics] = []

        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(X=self.w):
            try:
                w_train, w_test = self.w[train_idx], self.w[test_idx]
                cs_train, cs_test = self.cs[train_idx], self.cs[test_idx]
                t_train, t_test = self.t[train_idx], self.t[test_idx]

                # Single left-out sample
                w_ref = w_test[0]
                cs_ref = cs_test[0]
                t_ref = float(np.asarray(t_test).ravel()[0])

                pred = self.model.predict(
                    w_train=w_train,
                    cs_train=cs_train,
                    t_train=t_train,
                    w_ref=w_ref,
                    cs_query=cs_ref,
                )
                pred_val = float(np.asarray(pred).ravel()[0])

                y_true_all.append(t_ref)
                y_pred_all.append(pred_val)
                fold_metrics.append(EvaluationMetrics([t_ref], [pred_val]))

            except Exception as e:
                traceback.print_exc()
                warnings.warn(f"leave_one_out: {e}", UserWarning)

        em = EvaluationMetrics(y_true=y_true_all, y_pred=y_pred_all)
        EvaluationMetrics.summarize_loocv(fold_metrics, print_summary=True)
        return em

    def leave_one_group_out(self) -> EvaluationMetrics:
        """
        LOGO over WORKLOAD groups: in each fold, hold out all samples whose
        group label equals the test label; train on the rest. Works with
        non-numeric labels (e.g., 'aggregation', 'lr', ...).
        """
        y_true_all, y_pred_all = [], []
        per_group_metrics: List[EvaluationMetrics] = []

        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(X=self.w, groups=self.w_groups):
            try:
                w_train, w_test = self.w[train_idx], self.w[test_idx]
                cs_train, cs_test = self.cs[train_idx], self.cs[test_idx]
                t_train, t_test = self.t[train_idx], self.t[test_idx]

                # Predict each sample in the held-out group independently
                y_true_fold, y_pred_fold = [], []
                for i in range(len(w_test)):
                    w_ref = w_test[i]          # (D,)
                    cs_ref = cs_test[i]        # (C,)
                    t_ref = float(np.asarray(t_test[i]).ravel()[0])

                    pred = self.model.predict(
                        w_train=w_train,
                        cs_train=cs_train,
                        t_train=t_train,
                        w_ref=w_ref,
                        cs_query=cs_ref,
                    )
                    pred_val = float(np.asarray(pred).ravel()[0])

                    y_true_all.append(t_ref)
                    y_pred_all.append(pred_val)
                    y_true_fold.append(t_ref)
                    y_pred_fold.append(pred_val)

                per_group_metrics.append(EvaluationMetrics(y_true=y_true_fold, y_pred=y_pred_fold))

            except Exception as e:
                traceback.print_exc()
                warnings.warn(f"leave_one_group_out: {e}", UserWarning)

        em = EvaluationMetrics(y_true=y_true_all, y_pred=y_pred_all)
        EvaluationMetrics.summarize_logocv(per_group_metrics, print_summary=True)
        return em
