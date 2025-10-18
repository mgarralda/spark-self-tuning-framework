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
from typing import List
import warnings
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, LeaveOneGroupOut, GroupKFold, KFold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from framework.experimental.metrics.performance_model.evaluation import EvaluationMetrics


# ========================== Performance model (RF oracle, paper-faithful) ==========================
class PratsPerformanceModel:
    """
    Random-Forest performance model aligned with oracle-style predictors (e.g., YORO/SBO):
      - Pipeline: (StandardScaler | passthrough) + RandomForestRegressor
      - Hyperparameter search via GridSearchCV
      - Optional TransformedTargetRegressor hook (disabled by default)
    References:
      • Buchaca et al., “You Only Run Once” (YORO): RF oracle with model selection.
      • JPDC-2023 TurBO baseline uses runtime-centric evaluation; here we only fit the model.
    """

    def get_random_forest_regressor_model(
            self,
            X: List,
            y: List,
            *,
            inner_cv=None,          # Optional CV object (e.g., GroupKFold for LOGO)
            inner_groups=None       # Optional groups for group-aware inner CV
    ) -> GridSearchCV:
        """
        Fit a GridSearchCV over (StandardScaler | passthrough) + RF.
        If `inner_cv` is provided (e.g., GroupKFold), we pass it to GridSearchCV to avoid leakage.
        If `inner_groups` is provided, we pass it to .fit(..., groups=inner_groups).

        Returns:
            GridSearchCV: fitted model ready for .predict()
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        pipe = Pipeline([
            ("feature_scaler", StandardScaler()),  # may be replaced by 'passthrough' via grid
            ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
        ])

        # Optional target transform (disabled by default; enable if your y distribution demands it)
        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
        rf_ttr = TransformedTargetRegressor(
            regressor=pipe,
            transformer=None,  # switch to log_transformer if needed
            check_inverse=True
        )

        param_grid = {
            "regressor__feature_scaler": ["passthrough", StandardScaler()],
            "regressor__rf__n_estimators": [50, 100, 200],
            "regressor__rf__max_depth": [10, 20, None],
            "regressor__rf__max_features": ["sqrt", "log2", None],
        }

        # Default inner CV if none provided (simple KFold=5 is fine for LOOCV outer)
        cv_obj = inner_cv if inner_cv is not None else KFold(n_splits=5, shuffle=True, random_state=42)

        rfr = GridSearchCV(
            estimator=rf_ttr,
            param_grid=param_grid,
            cv=cv_obj,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
            refit=True,
            verbose=0
        )

        # If group-aware CV is requested, pass groups to fit()
        if inner_groups is not None:
            rfr.fit(X, y, groups=inner_groups)
        else:
            rfr.fit(X, y)

        return rfr


# =================================== Cross-validation wrappers ===================================
class CrossValidation:
    """
    Outer-CV wrappers:
      • LOOCV: Leave-One-Out on instances.
      • LOGOCV: Leave-One-Group-Out with group-aware inner model selection to avoid leakage.

    We report both pooled (micro) and average-across-folds (macro) using EvaluationMetrics.
    """

    def __init__(
            self,
            w: np.ndarray,          # features (workload descriptors, configs, sizes, etc.)
            w_groups: np.ndarray,   # group labels for LOGO (e.g., workload IDs)
            t: np.ndarray           # execution times
    ) -> None:
        self.w = np.asarray(w, dtype=float)
        self.w_groups = np.asarray(w_groups)
        self.t = np.asarray(t, dtype=float).ravel()
        self.perf_model = PratsPerformanceModel()

    def leave_one_out(self) -> EvaluationMetrics:
        """
        LOOCV outer loop:
          - Inner model selection: KFold(5) (default) inside GridSearchCV.
          - Reports pooled (micro) metrics and macro (avg. across folds).
        """
        y_true_all, y_pred_all = [], []
        fold_metrics = []

        loocv = LeaveOneOut()
        for train_index, test_index in loocv.split(X=self.w):
            try:
                X_tr, X_te = self.w[train_index], self.w[test_index]
                y_tr, y_te = self.t[train_index], self.t[test_index]

                # Inner CV: default KFold(5) (suficiente y consistente)
                model = self.perf_model.get_random_forest_regressor_model(X=X_tr, y=y_tr)

                preds = model.predict(X_te)  # (1,)
                preds = np.asarray(preds).ravel()
                y_te = np.asarray(y_te).ravel()

                y_pred_all.extend(preds.tolist())
                y_true_all.extend(y_te.tolist())

                fold_metrics.append(EvaluationMetrics(y_true=y_te, y_pred=preds))

            except Exception as e:
                traceback.print_exc()
                warnings.warn(f"leave_one_out: {e}", UserWarning)

        # Pooled (micro) + macro (mean across folds)
        em = EvaluationMetrics(y_true=y_true_all, y_pred=y_pred_all)
        EvaluationMetrics.summarize_loocv(fold_metrics, print_summary=True)
        return em

    def leave_one_group_out(self) -> EvaluationMetrics:
        """
        LOGOCV outer loop (paper-faithful grouping):
          - Inner model selection is also group-aware via GroupKFold on the TRAIN split only.
          - This avoids group leakage between inner folds and the left-out group.

        If there are < 2 groups in the training split, we gracefully fall back to KFold(5).
        """
        y_true_all, y_pred_all = [], []
        per_group_metrics = []

        logocv = LeaveOneGroupOut()
        for train_index, test_index in logocv.split(X=self.w, groups=self.w_groups):
            try:
                X_tr, X_te = self.w[train_index], self.w[test_index]
                y_tr, y_te = self.t[train_index], self.t[test_index]
                g_tr = self.w_groups[train_index]

                # Inner group-aware CV: use GroupKFold on TRAIN groups if feasible
                unique_groups = np.unique(g_tr)
                if unique_groups.size >= 3:
                    inner_cv = GroupKFold(n_splits=min(5, unique_groups.size))
                    model = self.perf_model.get_random_forest_regressor_model(
                        X=X_tr, y=y_tr, inner_cv=inner_cv, inner_groups=g_tr
                    )
                elif unique_groups.size == 2:
                    inner_cv = GroupKFold(n_splits=2)
                    model = self.perf_model.get_random_forest_regressor_model(
                        X=X_tr, y=y_tr, inner_cv=inner_cv, inner_groups=g_tr
                    )
                else:
                    # Too few groups for group-aware inner CV -> fall back to KFold(5)
                    model = self.perf_model.get_random_forest_regressor_model(X=X_tr, y=y_tr)

                preds = model.predict(X_te)  # (n_test_group,)
                preds = np.asarray(preds).ravel()
                y_te = np.asarray(y_te).ravel()

                y_pred_all.extend(preds.tolist())
                y_true_all.extend(y_te.tolist())

                per_group_metrics.append(EvaluationMetrics(y_true=y_te, y_pred=preds))

            except Exception as e:
                traceback.print_exc()
                warnings.warn(f"leave_one_group_out: {e}", UserWarning)

        # Pooled (micro) + macro (mean across groups)
        em = EvaluationMetrics(y_true=y_true_all, y_pred=y_pred_all)

        EvaluationMetrics.summarize_logocv(per_group_metrics, print_summary=True)

        return em
