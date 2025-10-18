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
from typing import Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, KFold


class UncertaintyModel:
    """
    σ_T_R(x) = 1.4826 * MAD_k( e_j ) + eps,
    con e_j = | μ^{-fold}(x_j) - y_j | (residuales OOS en N+C),
    y vecinos N_k(x) en espacio estandarizado (StandardScaler).

    Sencillo, eficaz para explotación y en la misma escala T_R que μ.
    """

    def __init__(
            self,
            model: Any,
            X_SL: np.ndarray, y_SL: np.ndarray,
            X_LE: np.ndarray, y_LE: np.ndarray,
            scaler: Optional[Any] = None,            # scaler del modelo de μ (si lo usas)
            *,
            k: Optional[int] = None,                 # si None -> ~sqrt(n) (clipped)
            k_min: int = 8, k_max: int = 64,
            winsor_q: Optional[float] = 0.95,        # cap global opcional de residuales
            eps: float = 1e-12,
            random_state: int = 0,
    ):


        self.model = model
        self.scaler_pm = scaler
        self.eps = float(eps)
        self.random_state = int(random_state)
        self.winsor_q = winsor_q

        # --- Datos: usa S_{L+E} si existe, si no S_L ---
        X_SL = np.asarray(X_SL, dtype=float); y_SL = np.asarray(y_SL, dtype=float).ravel()
        X_LE = np.asarray(X_LE, dtype=float); y_LE = np.asarray(y_LE, dtype=float).ravel()
        if X_LE.size:
            self.X_all = np.vstack([X_SL, X_LE])
            self.y_all = np.concatenate([y_SL, y_LE])
        else:
            self.X_all, self.y_all = X_SL, y_SL

        self.n = int(self.X_all.shape[0])
        if self.n == 0:
            # estado vacío
            self.abs_res = np.zeros(0, dtype=float)
            self.tau = 0.0
            self.scaler_geo = None
            self.k = 1
            return

        # --- Geometría para distancias (estable) ---
        self._StandardScaler = StandardScaler  # para evitar import arriba
        self.scaler_geo = self._StandardScaler().fit(self.X_all)
        self.X_std = self.scaler_geo.transform(self.X_all)

        # --- Residuales OOS en T_R (LOO<=200, si no OOF KFold) ---
        X, y = self.X_all, self.y_all
        mu_oof = np.empty_like(y, dtype=float)
        splitter = LeaveOneOut() if self.n <= 200 else KFold(n_splits=min(10, self.n), shuffle=True, random_state=self.random_state)

        for tr, te in splitter.split(X):
            est = clone(self.model)
            if self.scaler_pm is None:
                est.fit(X[tr], y[tr])
                mu_oof[te] = np.asarray(est.predict(X[te])).ravel()
            else:
                sc = clone(self.scaler_pm)
                Xtr = sc.fit_transform(X[tr]); est.fit(Xtr, y[tr])
                Xte = sc.transform(X[te]);    mu_oof[te] = np.asarray(est.predict(Xte)).ravel()

        abs_res = np.abs(mu_oof - y).astype(float)
        if self.winsor_q is not None and abs_res.size:
            cap = float(np.quantile(abs_res, self.winsor_q))
            abs_res = np.minimum(abs_res, cap)
        self.abs_res = abs_res

        # Escala robusta global (solo como suelo suave)
        med = float(np.median(abs_res)); mad = float(np.median(np.abs(abs_res - med)))
        self.tau = 1.4826 * mad

        # --- k automático ≈ √n (clipped) si no se especifica ---
        self.k = int(np.clip(round(np.sqrt(self.n)), k_min, min(k_max, self.n))) if k is None else int(np.clip(k, 1, self.n))

    def predict(self, cs_new: np.ndarray) -> float:
        """Devuelve σ(cs_new) en unidades T_R (misma escala que μ)."""
        if self.n == 0:
            return 0.0

        x = np.asarray(cs_new, dtype=float).reshape(1, -1)
        x_s = self.scaler_geo.transform(x)

        # k-NN uniformes en espacio estandarizado
        d = np.linalg.norm(self.X_std - x_s, axis=1)
        k = min(self.k, self.n)
        idx = np.argpartition(d, k - 1)[:k]

        # Dispersión robusta local (MAD)
        e_loc = self.abs_res[idx]
        med = float(np.median(e_loc))
        mad = float(np.median(np.abs(e_loc - med)))
        sigma_loc = 1.4826 * mad

        # Suelo suave para evitar colapso numérico en zonas muy lisas
        sigma = max(sigma_loc, 0.1 * self.tau) + self.eps

        # (Opcional, micro-inflador muy suave por lejanía; dejar comentado si quieres lo más mínimo)
        # d_min, d_med = float(np.min(d[idx])), float(np.median(d[idx]))
        # sigma *= (1.0 + min(1.0, d_min / (d_med + self.eps)) * 0.25)

        return float(sigma)

