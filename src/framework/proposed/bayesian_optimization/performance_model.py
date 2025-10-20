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

from typing import Optional, Tuple, Dict, Any, List, Literal
from numpy.linalg import pinv
import numpy as np
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, HuberRegressor, Ridge, Lasso, ElasticNet, RidgeCV, GammaRegressor, PoissonRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import LeaveOneOut, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C, DotProduct
from sklearn.pipeline import Pipeline
from framework.proposed.safe_transfer_learning import SafeTransferLearningStage
from framework.proposed.workload_characterization.workload import WorkloadCharacterized


class SupervisedStage:
    """ Incremental lazy predictive model """

    def get_non_negative_least_squares_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Weighted NNLS regression with log-transformed target and safe weight scaling.
        """

        # --- Feature pipeline ---
        pipeline = Pipeline([
            ('feature_scaler', MinMaxScaler()),
            ('lr', LinearRegression(positive=True)) # fit_intercept=True
        ])

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer= None,
        )

        if sample_weight is not None:
            regressor.fit(X, y, lr__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        regressor.fit(X, y)

        # Extract fitted feature scaler
        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']

        return regressor, feature_scaler, y.min(), y.max()

    def get_linear_regression_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            degree: int = 2,
            sample_weight: Optional[np.ndarray] = None
    ) -> tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Linear Regression para μ entrenado en log1p(y) (via TTR) y devuelve predicciones en T_R.
        Permite añadir términos polinómicos de grado 1 o 2.

        Args:
          X: matriz de características.
          y: objetivo en escala natural.
          degree: grado del polinomio (1=linear, 2=interacciones).
          sample_weight: pesos opcionales.

        Returns:
          (regressor_TTR, fitted_feature_scaler, y_min, y_max)
        """
        steps = [('feature_scaler', MinMaxScaler())]
        if degree > 1:
            steps.append(('poly', PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)))
        steps.append(('lr', LinearRegression()))
        pipeline = Pipeline(steps)

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=log_transformer,
            check_inverse=True
        )

        if sample_weight is not None:
            regressor.fit(X, y, lr__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_ridge_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            alpha: float = 2,
            random_state: int = 42,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Ridge regression para μ entrenado en log1p(y) (via TTR) y devuelve predicciones en T_R.
        """
        steps = [('feature_scaler', MinMaxScaler())]
        steps.append(('ridge', Ridge(alpha=alpha, random_state=random_state)))
        pipeline = Pipeline(steps)

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        # log_transformer = FunctionTransformer(func=fwd, inverse_func=inv, validate=True)

        regressor = TransformedTargetRegressor(
                    regressor=pipeline,
                    # transformer=log_transformer,
                    transformer=None,
                    check_inverse=False
                )

        if sample_weight is not None:
            regressor.fit(X, y, ridge__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_gamma_regressor_model_GridSearch(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            alpha: float = 1.0,          # regularización L2 suave
            max_iter: int = 1000,
            tol: float = 1e-6,
            sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[Pipeline, StandardScaler, float, float]:
        """
        GammaRegressor (log-link) para μ (T_R>0).
        Ventajas: positividad garantizada, buen ranking, simple y coherente estadísticamente.
        """
        # Gamma es GLM; no necesita transformar el target. Escalamos X para estabilidad numérica.
        steps = [
            ('feature_scaler', StandardScaler(with_mean=True, with_std=True)),
            ('gamma', GammaRegressor(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=True))
        ]
        pipeline = Pipeline(steps)

        if sample_weight is not None:
            pipeline.fit(X, y, gamma__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            pipeline.fit(X, y)

        feature_scaler = pipeline.named_steps['feature_scaler']
        return pipeline, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_gamma_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Gamma regression para μ, requiere y > 0.
        """

        steps = [('feature_scaler', StandardScaler())]
        steps.append(('gamma', GammaRegressor(alpha=1.5)))
        pipeline = Pipeline(steps)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=None,  # No log, GammaRegressor ya modela escala positiva
            check_inverse=True
        )

        if sample_weight is not None:
            regressor.fit(X, y, gamma__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_poisson_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            use_interactions: bool = False,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Poisson regression para μ, requiere y > 0.
        """
        steps = [('feature_scaler', MinMaxScaler())]
        if use_interactions:
            steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
        steps.append(('poisson', PoissonRegressor(max_iter=1000)))
        pipeline = Pipeline(steps)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=None,  # No log, PoissonRegressor ya modela escala positiva
            check_inverse=True
        )

        if sample_weight is not None:
            regressor.fit(X, y, poisson__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_huber_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            epsilon: float = 1.35,
            use_interactions: bool = False,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Huber regression para μ entrenado en log1p(y) (via TTR) y devuelve predicciones en T_R.

        Args:
          X: matriz de características.
          y: objetivo en escala natural.
          epsilon: parámetro de robustez de Huber.
          use_interactions: si True, añade términos de interacción de grado 2.
          sample_weight: pesos opcionales.

        Returns:
          (regressor_TTR, fitted_feature_scaler, y_min, y_max)
        """
        steps = [('feature_scaler', MinMaxScaler())]
        if use_interactions:
            steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
        steps.append(('huber', HuberRegressor(epsilon=epsilon)))
        pipeline = Pipeline(steps)

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=log_transformer,
            check_inverse=True
        )

        if sample_weight is not None:
            regressor.fit(X, y, huber__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_lasso_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            alpha: float = 1.0,
            use_interactions: bool = False,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Lasso regression para μ entrenado en log1p(y) (via TTR) y devuelve predicciones en T_R.
        """
        steps = [('feature_scaler', MinMaxScaler())]
        if use_interactions:
            steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
        steps.append(('lasso', Lasso(alpha=alpha, random_state=0, max_iter=10000)))
        pipeline = Pipeline(steps)

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            # transformer=log_transformer,
            transformer=None,
            check_inverse=True
        )

        if sample_weight is not None:
            regressor.fit(X, y, lasso__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_elasticnet_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            alpha: float = 1.0,
            l1_ratio: float = 0.5,
            use_interactions: bool = False,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        ElasticNet regression para μ entrenado en log1p(y) (via TTR) y devuelve predicciones en T_R.
        """
        steps = [('feature_scaler', MinMaxScaler())]
        if use_interactions:
            steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
        steps.append(('elasticnet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0, max_iter=10000)))
        pipeline = Pipeline(steps)

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=log_transformer,
            check_inverse=True
        )

        if sample_weight is not None:
            regressor.fit(X, y, elasticnet__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_extra_trees_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, None]:
        """
        ExtraTrees for μ, trained in log1p(y) and returning predictions in T_R (expm1).

        Rationale:
          - Robust with discrete/categorical-like features and few samples.
          - Diversity from random thresholds (no bootstrap), reducing plateaus vs. RF.
          - No feature scaling required for tree-based models.

        Notes:
          - Keep leaves slightly larger to avoid overly sharp fits with tiny datasets.
          - n_estimators scaled mildly with n for stability.

        Returns:
          (regressor_TTR, None)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n = len(X)
        n_estimators = max(200, min(800, int(6 * (n ** 0.5)) * 10))

        base_et = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=42,
            min_samples_leaf=1, 
            min_samples_split=2,
            max_depth=None,
            n_jobs=-1,
        )

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)

        et_log = TransformedTargetRegressor(
            regressor=base_et,
            # transformer=log_transformer,
            transformer=None,
            check_inverse=False
        )

        if sample_weight is not None:
            et_log.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            et_log.fit(X, y)

        return et_log, None

    def get_random_forest_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: Optional[np.ndarray] = None,
            random_state: int = 42
    ) -> tuple[TransformedTargetRegressor, None]:
        pipeline = Pipeline([
            ('rf', RandomForestRegressor(random_state=random_state, n_jobs=-1))
        ])

        no_transform = FunctionTransformer(func=None, inverse_func=None, validate=False)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=no_transform,
            check_inverse=False
        )

        fit_params = {}
        if sample_weight is not None:
            fit_params['rf__sample_weight'] = np.asarray(sample_weight, dtype=float).ravel()

        regressor.fit(X, y, **fit_params)

        return regressor, None

    def get_extra_trees_regressor_model_gridsearch(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            random_state: int = 42
    ) -> Tuple[TransformedTargetRegressor, None]:
        """
        ExtraTrees con GridSearchCV para los hiperparámetros principales.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        base_et = ExtraTreesRegressor(
            random_state=random_state,
            n_jobs=-1
        )

        param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3, 5],
            "min_samples_split": [2, 5, 10]
        }

        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = np.asarray(sample_weight, dtype=float).ravel()

        grid = GridSearchCV(
            estimator=base_et,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(X, y, **fit_params)

        best_et = grid.best_estimator_

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
        et_log = TransformedTargetRegressor(
            regressor=best_et,
            # transformer=log_transformer,
            transformer=None,
            check_inverse=True
        )
        et_log.fit(X, y, **fit_params)

        return et_log, None

    def get_catboost_regressor_model_(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            cat_features: Optional[list] = None,
            random_state: int = 42
    ) -> Tuple[TransformedTargetRegressor, None]:
        """
        CatBoost regression para μ, entrenado en log1p(y) y devuelve predicciones en T_R (expm1).
        - No requiere escalado de X.
        - Soporta sample_weight.
        - cat_features: lista de índices de columnas categóricas (opcional).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        base_cb = CatBoostRegressor(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            random_state=random_state,
            verbose=0
        )

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        cb_log = TransformedTargetRegressor(
            regressor=base_cb,
            transformer=log_transformer,
            check_inverse=True
        )

        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = np.asarray(sample_weight, dtype=float).ravel()
        if cat_features is not None:
            fit_params["cat_features"] = cat_features

        cb_log.fit(X, y, **fit_params)

        return cb_log, None

    def get_catboost_regressor_model_gridsearch(
            self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            cat_features: Optional[list] = None,
            random_state: int = 42
    ) -> Tuple[TransformedTargetRegressor, None]:
        """
        CatBoost con GridSearchCV para los hiperparámetros principales.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        base_cb = CatBoostRegressor(
            random_state=random_state,
            verbose=0
        )

        param_grid = {
            "learning_rate": [0.03, 0.1, 0.3],
            "depth": [4, 6, 8],
            "iterations": [100, 300, 500]
        }

        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = np.asarray(sample_weight, dtype=float).ravel()
        if cat_features is not None:
            fit_params["cat_features"] = cat_features

        grid = GridSearchCV(
            estimator=base_cb,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(X, y, **fit_params)

        best_cb = grid.best_estimator_

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
        cb_log = TransformedTargetRegressor(
            regressor=best_cb,
            transformer=log_transformer,
            check_inverse=True
        )
        cb_log.fit(X, y, **fit_params)

        return cb_log, None

    def get_knn_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_neighbors: int = 5
    ) -> Tuple[TransformedTargetRegressor, StandardScaler]:
        """
        KNN regression para μ, entrenado en log1p(y) y devuelve predicciones en T_R (expm1).

        Buenas prácticas:
          - Normaliza X (StandardScaler).
          - Aplica log1p/expm1 al target.
          - No soporta sample_weight.

        Returns:
          (regressor_TTR, fitted_feature_scaler)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        pipeline = Pipeline([
            ('feature_scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', n_jobs=-1))
        ])

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=log_transformer,
            check_inverse=True
        )

        regressor.fit(X, y)

        feature_scaler: StandardScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler

    def get_gradientboosting_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: Optional[np.ndarray] = None
    ) -> tuple[TransformedTargetRegressor, StandardScaler]:
        """
        Gradient Boosting regression para μ entrenado en log1p(y) (via TTR) y devuelve predicciones en T_R.
        """
        pipeline = Pipeline([
            ('feature_scaler', StandardScaler()),
            ('gr', GradientBoostingRegressor(random_state=42))
        ])

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            # transformer=log_transformer,
            transformer=None,
            check_inverse=True
        )

        if sample_weight is not None:
            regressor.fit(X, y, gr__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: StandardScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler

    def get_adaboost_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: Optional[np.ndarray] = None
    ) -> tuple[TransformedTargetRegressor, StandardScaler, float, float]:
        """
        AdaBoost regression para μ entrenado en log1p(y) (via TTR) y devuelve predicciones en T_R.
        """
        pipeline = Pipeline([
            ('feature_scaler', StandardScaler()),
            ('AdaBoost', AdaBoostRegressor())
        ])

        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=log_transformer,
            check_inverse=True
        )

        if sample_weight is not None:
            regressor.fit(X, y, AdaBoost__sample_weight=np.asarray(sample_weight, dtype=float).ravel())
        else:
            regressor.fit(X, y)

        feature_scaler: StandardScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_bayesian_ridge_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Bayesian Ridge en escala natural (sin log).
        Parámetros fijos conservadores, sin gridsearch.
        Devuelve un TransformedTargetRegressor con transformer=None para
        conservar el interfaz de las otras fábricas.
        """
        steps = [
            ('feature_scaler', MinMaxScaler()),
            ('br', BayesianRidge(
                # priors Jeffreys-suaves (evitan sobre/infra-regularización)
                alpha_1=1e-6, alpha_2=1e-6,
                lambda_1=1e-6, lambda_2=1e-6,
                # velocidad/estabilidad
                tol=1e-4,
                compute_score=False,  # más rápido
                fit_intercept=True
            ))
        ]
        pipeline = Pipeline(steps)

        # Sin transformación (escala natural)
        no_transform = FunctionTransformer(func=None, inverse_func=None, validate=False)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=no_transform,   # sin log
            check_inverse=False
        )

        fit_params = {}
        if sample_weight is not None:
            fit_params['br__sample_weight'] = np.asarray(sample_weight, dtype=float).ravel()

        regressor.fit(X, y, **fit_params)

        feature_scaler: MinMaxScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_gaussian_process_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: Optional[np.ndarray] = None  # presente, pero no usado en la versión mínima
    ) -> tuple[TransformedTargetRegressor, StandardScaler, float, float]:
        """
        Versión mínima y estable para CV (LOGO):
        - Escala de X con StandardScaler
        - y con log1p/expm1
        - Kernel Matern + WhiteKernel
        - normalize_y y algunos reinicios
        """
        import numpy as np
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, FunctionTransformer
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

        # Comprobaciones mínimas
        if not np.isfinite(X).all():
            raise ValueError("X contiene NaN/Inf.")
        if not np.isfinite(y).all():
            raise ValueError("y contiene NaN/Inf.")
        if np.any(y < -1.0):
            raise ValueError("y tiene valores < -1; log1p no es válido.")

        n_features = X.shape[1]

        # Kernel sencillo y robusto (ARD opcional via length_scale por dimensión)
        kernel = (
                C(1.0, (1e-3, 1e3)) *
                Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-3, 1e3), nu=2.5)
                # init más realista y límite más alto para el ruido
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )

        pipeline = Pipeline([
            ('feature_scaler', StandardScaler()),
            ('gp', gp)
        ])

        # Transformación mínima de y (solo log1p/expm1)
        y_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)

        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=y_transformer,
            check_inverse=True
        )

        # Entrena
        regressor.fit(X, y)

        feature_scaler: StandardScaler = regressor.regressor_.named_steps['feature_scaler']
        return regressor, feature_scaler, float(np.min(y)), float(np.max(y))

    def get_bayesian_ridge_regressor_model_GridSearch(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: Optional[np.ndarray] = None,
            choose_variant: str = "auto",   # "auto" | "log" | "linear"
            random_state: int = 42
    ) -> Tuple[TransformedTargetRegressor, MinMaxScaler, float, float]:
        """
        Bayesian Ridge para μ(T). Prueba dos variantes y elige por MAE CV (3-fold):

          • "log":   TTR[log1p↔expm1] + BR  → no negativos por construcción.
          • "linear": TTR[identidad] + BR envuelto en clip ≥ 0.

        Args:
          X, y: datos (y en escala natural, y >= 0).
          sample_weight: pesos opcionales.
          choose_variant: "auto" (default), "log" o "linear".
          random_state: solo para reproducibilidad del split.

        Returns:
          (regressor_TTR_ajustado, feature_scaler_ajustado, y_min, y_max)

        Atributos útiles añadidos al regresor devuelto:
          .best_variant_ -> "log" o "linear"
          .cv_mae_       -> dict con MAE por variante (si auto)
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError("y debe ser >= 0 para usar log1p de forma segura.")

        def _fit_full(variant: str):
            if variant == "log":
                pipe = self._build_pipeline(clip_output=False)
                ttr = TransformedTargetRegressor(
                    regressor=pipe,
                    transformer=FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True),
                    check_inverse=True
                )
            elif variant == "linear":
                pipe = self._build_pipeline(clip_output=True)  # clip a >= 0 en predicción
                ttr = TransformedTargetRegressor(
                    regressor=pipe,
                    transformer=None,
                    check_inverse=False
                )
            else:
                raise ValueError("variant debe ser 'log' o 'linear'.")

            fit_params = {}
            if sample_weight is not None:
                fit_params['br__sample_weight'] = np.asarray(sample_weight, dtype=float).ravel()

            ttr.fit(X, y, **fit_params)
            scaler: MinMaxScaler = ttr.regressor_.named_steps['feature_scaler']
            return ttr, scaler

        def _cv_mae(variant: str, n_splits: int = 3) -> float:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            maes = []
            for tr, va in kf.split(X):
                ttr, _ = _fit_full(variant)
                # Reajustamos SOLO con el train del fold
                fit_params = {}
                if sample_weight is not None:
                    # fit_params['regressor__br__sample_weight'] = np.asarray(sample_weight, dtype=float).ravel()[tr]
                    fit_params['br__sample_weight'] = np.asarray(sample_weight, dtype=float).ravel()[tr]
                ttr.fit(X[tr], y[tr], **fit_params)

                y_pred = ttr.predict(X[va])
                sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()[va]
                if sw is None:
                    mae = float(np.mean(np.abs(y_pred - y[va])))
                else:
                    sw = np.asarray(sw, dtype=float)
                    sw = np.where(np.isfinite(sw), sw, 0.0)
                    w = sw / (sw.sum() + 1e-12)
                    mae = float(np.sum(w * np.abs(y_pred - y[va])))
                maes.append(mae)
            return float(np.mean(maes))

        if choose_variant == "auto":
            mae_log = _cv_mae("log")
            mae_lin = _cv_mae("linear")
            best_variant = "log" if mae_log <= mae_lin else "linear"
            ttr, scaler = _fit_full(best_variant)
            # Añadimos metadatos útiles
            ttr.best_variant_ = best_variant
            ttr.cv_mae_ = {"log": mae_log, "linear": mae_lin}
        else:
            best_variant = choose_variant.lower()
            ttr, scaler = _fit_full(best_variant)
            ttr.best_variant_ = best_variant
            ttr.cv_mae_ = None

        return ttr, scaler, float(np.min(y)), float(np.max(y))


    def mahalanobis_batch(X, x0, VI):
        # Distancia Mahalanobis por filas: sqrt((x - x0) VI (x - x0)^T)
        D = X - x0
        return np.sqrt(np.sum(D @ VI * D, axis=1))

    def compute_sample_weights_latent(self, W_LE, w_target, idx_S_L, idx_S_E, eps=1e-12):
        """
        W_LE: (n, p) latentes de S_{L+E}
        w_target: (p,) latente del workload objetivo
        idx_S_L, idx_S_E: arrays de índices (disjuntos) para S_L y S_E (pueden venir de STL)
        """
        # 1) Estandariza geometría latente
        scaler = StandardScaler().fit(W_LE)
        Wz = scaler.transform(W_LE)
        wz = scaler.transform(w_target.reshape(1, -1)).ravel()

        # 2) Matriz de precisión robusta (pinv de cov por estabilidad)
        VI = pinv(np.cov(Wz, rowvar=False) + eps*np.eye(Wz.shape[1]))

        # 3) Distancias a w_target
        d = self.mahalanobis_batch(Wz, wz, VI)

        # 4) Escala automática lambda
        lam = np.median(d) + eps

        # 5) Peso base
        w_base = np.exp(-d / lam)

        # 6) Multiplicador por región (corrige desbalance S_L vs S_E)
        nL, nE = len(idx_S_L), len(idx_S_E)
        eta = min(1.0, nL / max(nE, 1))  # <=1

        mult = np.ones_like(w_base)
        mult[idx_S_E] *= eta  # S_E reducido si es muy grande

        w = w_base * mult

        # 7) Normalización opcional a media 1 (estético)
        w *= (len(w) / (w.sum() + eps))
        return w


class PerformanceModel:
    """ Incremental transfer learning adaptation for efficient performance modeling of big data workloads """

    def __init__(self) -> None:
        self.safe_transfer_learning_stage = SafeTransferLearningStage()
        self.supervised_stage = SupervisedStage()
        # self.perf_utils = PerformanceModelUtils()

        self.weights_diff_threshold: int = 0

    def fit_predict(
            self,
            workload_descriptors: np.ndarray,
            configuration_settings: np.ndarray,
            execution_times: np.ndarray,
            input_data_sizes: np.ndarray,
            workload_ref: np.ndarray,
            configuration_settings_ref: np.ndarray,
            execution_time_ref: int,
            input_data_size_ref: int,
            workload_names:np.ndarray,
            worload_config_shapes:np.ndarray,
            workload_config_shape_ref:np.ndarray,
            workload_resources:np.ndarray,
            execution_resources_ref:float,
            # k_min: int,
            k_limit: int,
            cv: Literal["LOOCV", "LOGOCV"] = "LOOCV"
    ) -> np.ndarray:
        """
        Instance of the performance model to use in inferencing

        :param k_limit:
        :param workload_descriptors:
        :param configuration_settings:
        :param execution_times:
        :param workload_ref:
        :param configuration_settings_ref: new configuration settings to fit_predict
        :param k_min: minimum number of neighbors
        :param k_max: maximum number of neighbors
        :param eebc_weight: Proportion for the quality and distance
        :return: prediction
        """

        (
            idx_SL,            # N
            idx_S_LE,          # N+C
        ) = self.safe_transfer_learning_stage.build_zones(
            workload_descriptors,
            execution_times,
            input_data_sizes,
            workload_ref,
            execution_time_ref,
            input_data_size_ref,
        )

        idx_N, w_N = self.safe_transfer_learning_stage.last_weights["N"]  
        idx_C_w, w_C = self.safe_transfer_learning_stage.last_weights["C"] 

        X_train = configuration_settings[idx_N]
        y_train = execution_times[idx_N]
        ids_sl = input_data_sizes[idx_N]
        wn_sl = workload_names[idx_N]

        X_train_k_knee = np.vstack([X_train, configuration_settings[idx_S_LE]])
        y_train_k_knee = np.hstack([y_train, execution_times[idx_S_LE]])
        w_TC = np.hstack([w_N, np.ones(len(idx_S_LE))])  # pesos duros 1.0 para S_E
        # print(f"\n{len(idx_SL)=} | {len(idx_S_LE)=}")
        # assert len(X_train_k_knee) == len(idx_SL) + len(idx_S_LE)

        X_train_k_opt_k_knee = configuration_settings[idx_S_LE]
        y_train_k_opt_k_knee = execution_times[idx_S_LE]
        ids_sle = input_data_sizes[idx_S_LE]
        wn_sle = workload_names[idx_S_LE]

        regressor, *_  = self.supervised_stage.get_ridge_regressor_model(
            X=X_train,
            y=y_train,
            sample_weight=w_N,
        )

        # Check if the new_configuration_settings is a 1D array, thus only sent a single configuration setting to fit_predict
        if configuration_settings_ref.ndim == 1:
            configuration_settings_ref = configuration_settings_ref.reshape(1, -1)

        prediction = regressor.predict(configuration_settings_ref)

        return prediction

    @staticmethod
    def soft_size_weights(ids_N, ids_star, size_unit):
        """
        Compute soft weights w_i = 1/(1 + d_n) with d_n = |s_i - s*| / MAD(|s - s*|),
        s = log1p(ids/size_unit). Robust, no knobs.
        """
        s_N = np.log1p(ids_N.astype(float) / size_unit)
        s_star = np.log1p(float(ids_star) / size_unit)
        d = np.abs(s_N - s_star)
        med = np.median(d)
        mad = np.median(np.abs(d - med)) + 1e-12  # robust scale, avoid zero
        d_n = d / mad
        w = 1.0 / (1.0 + d_n) # linear
        # w = 1.0 / (1.0 + d_n**2) # quadratic
        return w

    @staticmethod
    def compute_training_residuals(
            regressor,
            X_train_k_opt_k_knee,
            y_train_k_opt_k_knee
    ):
        """
        The residuals reflect model disagreement in the extended zone S_LE from regessor in S_L[k_min: k_opt]

        """
        y_pred = regressor.predict(X_train_k_opt_k_knee)
        residuals = np.abs(y_train_k_opt_k_knee - y_pred)
        return residuals

    @staticmethod
    def audit_stl_selection(idx_SL, idx_S_LE, idx_SE,
                            settings, objectives, sizes, *,
                            name="STL@audit", verbose=True):
        n = len(settings)
        # 1) índices válidos
        assert np.all(idx_SL>=0) and np.all(idx_SL<n)
        assert np.all(idx_S_LE>=0) and np.all(idx_S_LE<n)
        assert np.all(idx_SE>=0) and np.all(idx_SE<n)
        # 2) relaciones de inclusión / disyunción
        is_subset = set(idx_SL).issubset(set(idx_S_LE))
        disjoint_SE = (len(set(idx_SL)&set(idx_SE))==0) and (len(set(idx_SE)&set(idx_S_LE))==len(set(idx_SE)))
        ok_sizes = len(idx_S_LE) >= len(idx_SL) and len(idx_SE) == (len(idx_S_LE)-len(idx_SL))
        # 3) mini-hash para detectar desalineaciones X↔y
        def h(a): return float(np.sum(np.asarray(a, float)) % 1_000_000)
        y_sl = np.array(objectives)[idx_SL]; x_sl = np.array(settings)[idx_SL]
        y_le = np.array(objectives)[idx_S_LE]; x_le = np.array(settings)[idx_S_LE]
        y_se = np.array(objectives)[idx_SE]; x_se = np.array(settings)[idx_SE]
        summary = {
            "n_total": n,
            "len_SL": len(idx_SL),
            "len_S_LE": len(idx_S_LE),
            "len_S_E": len(idx_SE),
            "SL_subset_of_S_LE": bool(is_subset),
            "S_E_equals_S_LE_minus_S_L": bool(disjoint_SE and ok_sizes),
            "hash_x_SL": h(x_sl), "hash_y_SL": h(y_sl),
            "hash_x_S_LE": h(x_le), "hash_y_S_LE": h(y_le),
            "hash_x_S_E": h(x_se), "hash_y_S_E": h(y_se),
        }
        if verbose:
            print(f"[{name}] {summary}")
        # asserts fuertes (comenta si prefieres solo logging)
        assert is_subset, "S_L no es subconjunto de S_LE"
        assert disjoint_SE and ok_sizes, "S_E debe ser S_LE \\ S_L (disjuntos y tamaños coherentes)"
        return summary


