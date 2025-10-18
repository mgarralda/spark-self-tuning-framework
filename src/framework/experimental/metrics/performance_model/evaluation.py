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

import warnings
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error

# --- RMSE (compat) ---
def root_mean_squared_error(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e**2)))

# --- Kendall tau helper (tie-aware if scipy is present) ---
def _kendall_tau_safe(a: np.ndarray, b: np.ndarray) -> float:
    """
    Kendall's tau-b if scipy is available; otherwise fallback to Spearman rho
    (highly correlated with tau) with a warning. Handles minimization targets.
    """
    try:
        from scipy.stats import kendalltau
        tau, _ = kendalltau(a, b)
        return float(tau)
    except Exception:
        warnings.warn(
            "scipy.stats.kendalltau not available; falling back to Spearman's rho.",
            UserWarning
        )
        # Spearman rho as fallback
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        # Pearson corr between ranks
        ra = ra.astype(float); rb = rb.astype(float)
        ra = (ra - ra.mean()) / (ra.std() + 1e-12)
        rb = (rb - rb.mean()) / (rb.std() + 1e-12)
        rho = float(np.mean(ra * rb))
        return rho

# --- Precision@k helper (minimization) ---
def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Precision@k where 'relevant' items are the k lowest y_true (minimization),
    and y_score gives the ranking proposed by the model (lower is better).
    """
    n = y_true.size
    if n == 0:
        return float("nan")
    k = int(min(max(1, k), n))
    true_top = np.argpartition(y_true, k-1)[:k]
    pred_top = np.argpartition(y_score, k-1)[:k]
    # intersection size
    inter = len(np.intersect1d(true_top, pred_top, assume_unique=False))
    return float(inter / k)

@dataclass
class CVSummary:
    # micro/pooled (sobre todas las predicciones concatenadas)
    micro_mae: float
    micro_rmse: float
    micro_resid_median: float
    micro_resid_p95: float
    micro_resid_p98: float
    micro_resid_max: float
    # BO-aware (micro)
    micro_kendall_tau: float
    micro_precision_at_k: float
    micro_esr_mu: float
    # macro (promedio simple por grupo/fold)
    macro_mae: float
    macro_rmse: float
    macro_resid_median: float
    macro_resid_p95: float
    macro_resid_p98: float
    macro_resid_max: float
    # BO-aware (macro)
    macro_kendall_tau: float
    macro_precision_at_k: float
    macro_esr_mu: float
    # opcional: dispersión entre grupos
    macro_mae_std: float
    macro_rmse_std: float

class EvaluationMetrics:
    """Evaluation metrics for a single split/fold (y_true,y_pred)."""

    def __init__(self, y_true, y_pred) -> None:
        # 1) normaliza a float y 1D
        self.y_true = np.asarray(y_true, dtype=float).ravel()
        self.y_pred = np.asarray(y_pred, dtype=float).ravel()

        # 2) chequea longitudes
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {self.y_true.shape} vs y_pred {self.y_pred.shape}")

        # 3) filtra no-finitos en ambos
        mask = np.isfinite(self.y_true) & np.isfinite(self.y_pred)
        dropped = int(np.count_nonzero(~mask))
        if dropped > 0:
            warnings.warn(
                f"{dropped} non-finite values dropped in metrics computation.",
                UserWarning
            )
            self.y_true = self.y_true[mask]
            self.y_pred = self.y_pred[mask]

        # 4) precompute residuos
        self._res = self.y_true - self.y_pred
        self._abs = np.abs(self._res)

    # --- básicos por fold ---
    def n(self) -> int:
        return self._abs.size

    def residuals(self) -> np.ndarray:
        return self._res

    def abs_residuals(self) -> np.ndarray:
        return self._abs

    def residual_median(self) -> float:
        return float(np.median(self._abs))

    def residual_p95(self) -> float:
        return float(np.percentile(self._abs, 95))

    def residual_p98(self) -> float:
        return float(np.percentile(self._abs, 98))

    def residual_max(self) -> float:
        return float(np.max(self._abs))

    def residual_std_dev(self) -> float:
        # desviación típica de los RESIDUOS (no de los absolutos)
        return float(np.std(self._res))

    def MAE(self) -> float:
        return float(mean_absolute_error(self.y_true, self.y_pred))

    def RMSE(self) -> float:
        return float(root_mean_squared_error(self.y_true, self.y_pred))

    # --- BO-aware por fold ---
    def kendall_tau(self) -> float:
        # menor y_true/ y_pred = mejor (minimización)
        return _kendall_tau_safe(self.y_pred, self.y_true)

    def precision_at_k(self, k: int = 10) -> float:
        return _precision_at_k(self.y_true, self.y_pred, k=k)

    def esr_mu(self) -> float:
        """Expected Simple Regret approximation on this set:
        y(x_mu) - min_x y(x), where x_mu = argmin mu(x)=y_pred(x)."""
        if self.y_true.size == 0:
            return float("nan")
        idx_mu = int(np.argmin(self.y_pred))
        best_true = float(np.min(self.y_true))
        regret = float(self.y_true[idx_mu] - best_true)
        return regret

    # acumuladores útiles para micro-averaging
    def SAE(self) -> float:
        return float(self._abs.sum())

    def SSE(self) -> float:
        return float((self._res**2).sum())

    # --- agregaciones LOOCV (con BO-aware) ---
    @staticmethod
    def summarize_loocv(
            folds: List["EvaluationMetrics"],
            print_summary: bool = False,
            k_top: int = 10
    ) -> CVSummary:
        # ---- micro/pooled (todas las observaciones juntas)
        all_true = np.concatenate([f.y_true for f in folds]) if folds else np.array([])
        all_pred = np.concatenate([f.y_pred for f in folds]) if folds else np.array([])
        all_abs  = np.abs(all_true - all_pred) if all_true.size else np.array([])

        N   = all_true.size
        SAE = float(np.sum(all_abs)) if all_abs.size else 0.0
        SSE = float(np.sum((all_true - all_pred)**2)) if all_true.size else 0.0

        micro_mae  = SAE / N if N else np.nan
        micro_rmse = np.sqrt(SSE / N) if N else np.nan
        micro_med  = float(np.median(all_abs)) if all_abs.size else np.nan
        micro_p95  = float(np.percentile(all_abs, 95)) if all_abs.size else np.nan
        micro_p98  = float(np.percentile(all_abs, 98)) if all_abs.size else np.nan
        micro_max  = float(np.max(all_abs)) if all_abs.size else np.nan

        # BO-aware (micro) sobre el conjunto concatenado
        micro_tau  = _kendall_tau_safe(all_pred, all_true) if N else np.nan
        micro_p_at_k = _precision_at_k(all_true, all_pred, k=k_top) if N else np.nan
        micro_esr = (float(all_true[np.argmin(all_pred)] - np.min(all_true)) if N else np.nan)

        # ---- macro (media de métricas por fold)
        mae_vals   = np.array([f.MAE() for f in folds], float)
        rmse_vals  = np.array([f.RMSE() for f in folds], float)
        med_vals   = np.array([f.residual_median() for f in folds], float)
        p95_vals   = np.array([f.residual_p95() for f in folds], float)
        p98_vals   = np.array([f.residual_p98() for f in folds], float)
        max_vals   = np.array([f.residual_max() for f in folds], float)
        tau_vals   = np.array([f.kendall_tau() for f in folds], float)
        p_at_k_vals= np.array([f.precision_at_k(k_top) for f in folds], float)
        esr_vals   = np.array([f.esr_mu() for f in folds], float)

        macro_mae  = float(np.mean(mae_vals))  if mae_vals.size  else np.nan
        macro_rmse = float(np.mean(rmse_vals)) if rmse_vals.size else np.nan
        macro_med  = float(np.mean(med_vals))  if med_vals.size  else np.nan
        macro_p95  = float(np.mean(p95_vals))  if p95_vals.size  else np.nan
        macro_p98  = float(np.mean(p98_vals))  if p98_vals.size  else np.nan
        macro_max  = float(np.mean(max_vals))  if max_vals.size  else np.nan

        macro_tau   = float(np.mean(tau_vals))    if tau_vals.size    else np.nan
        macro_p_at_k= float(np.mean(p_at_k_vals)) if p_at_k_vals.size else np.nan
        macro_esr   = float(np.mean(esr_vals))    if esr_vals.size    else np.nan

        macro_mae_std  = float(np.std(mae_vals))  if mae_vals.size  else np.nan
        macro_rmse_std = float(np.std(rmse_vals)) if rmse_vals.size else np.nan

        if print_summary:
            print("LOOCV (micro/pooled):")
            print(f"\tMAE: {micro_mae:.3f} | RMSE: {micro_rmse:.3f} | P95: {micro_p95:.3f} | Max: {micro_max:.3f}")
            print(f"\tKendall tau: {micro_tau:.3f} | Precision@{k_top}: {micro_p_at_k:.3f} | ESR_mu: {micro_esr:.3f}")
            print("LOOCV (macro mean ± std):")
            print(f"\tMAE: {macro_mae:.3f} ± {macro_mae_std:.3f} | RMSE: {macro_rmse:.3f} ± {macro_rmse_std:.3f}")
            print(f"\tP95(avg): {macro_p95:.3f} | Max(avg): {macro_max:.3f}")
            print(f"\tKendall tau(avg): {macro_tau:.3f} | Precision@{k_top}(avg): {macro_p_at_k:.3f} | ESR_mu(avg): {macro_esr:.3f}")

        return CVSummary(
            micro_mae, micro_rmse, micro_med, micro_p95, micro_p98, micro_max,
            micro_tau, micro_p_at_k, micro_esr,
            macro_mae, macro_rmse, macro_med, macro_p95, macro_p98, macro_max,
            macro_tau, macro_p_at_k, macro_esr,
            macro_mae_std, macro_rmse_std
        )

    # --- agregaciones LOGOCV (con BO-aware) ---
    @staticmethod
    def summarize_logocv_(
            folds: List["EvaluationMetrics"],
            print_summary: bool = False,
            k_top: int = 10
    ) -> CVSummary:
        """Devuelve micro (pooled) + macro (por fold) para LOGOCV, con BO-aware metrics."""
        # micro/pooled
        all_true = np.concatenate([f.y_true for f in folds]) if folds else np.array([])
        all_pred = np.concatenate([f.y_pred for f in folds]) if folds else np.array([])
        all_abs  = np.abs(all_true - all_pred) if all_true.size else np.array([])

        N = all_true.size
        micro_mae  = float(np.mean(all_abs)) if N else np.nan
        micro_rmse = float(np.sqrt(np.mean((all_true - all_pred)**2))) if N else np.nan
        micro_med  = float(np.median(all_abs)) if all_abs.size else np.nan
        micro_p95  = float(np.percentile(all_abs, 95)) if all_abs.size else np.nan
        micro_p98  = float(np.percentile(all_abs, 98)) if all_abs.size else np.nan
        micro_max  = float(np.max(all_abs)) if all_abs.size else np.nan

        # BO-aware (micro)
        micro_tau   = _kendall_tau_safe(all_pred, all_true) if N else np.nan
        micro_p_at_k= _precision_at_k(all_true, all_pred, k=k_top) if N else np.nan
        micro_esr   = (float(all_true[np.argmin(all_pred)] - np.min(all_true)) if N else np.nan)

        # macro (promedio simple entre folds)
        mae_vals   = np.array([f.MAE() for f in folds], float)
        rmse_vals  = np.array([f.RMSE() for f in folds], float)
        med_vals   = np.array([f.residual_median() for f in folds], float)
        p95_vals   = np.array([f.residual_p95() for f in folds], float)
        p98_vals   = np.array([f.residual_p98() for f in folds], float)
        max_vals   = np.array([f.residual_max() for f in folds], float)
        tau_vals   = np.array([f.kendall_tau() for f in folds], float)
        p_at_k_vals= np.array([f.precision_at_k(k_top) for f in folds], float)
        esr_vals   = np.array([f.esr_mu() for f in folds], float)

        macro_mae  = float(np.mean(mae_vals))  if mae_vals.size  else np.nan
        macro_rmse = float(np.mean(rmse_vals)) if rmse_vals.size else np.nan
        macro_med  = float(np.mean(med_vals))  if med_vals.size  else np.nan
        macro_p95  = float(np.mean(p95_vals))  if p95_vals.size  else np.nan
        macro_p98  = float(np.mean(p98_vals))  if p98_vals.size  else np.nan
        macro_max  = float(np.mean(max_vals))  if max_vals.size  else np.nan

        macro_tau    = float(np.mean(tau_vals))    if tau_vals.size    else np.nan
        macro_p_at_k = float(np.mean(p_at_k_vals)) if p_at_k_vals.size else np.nan
        macro_esr    = float(np.mean(esr_vals))    if esr_vals.size    else np.nan

        macro_mae_std  = float(np.std(mae_vals))  if mae_vals.size  else np.nan
        macro_rmse_std = float(np.std(rmse_vals)) if rmse_vals.size else np.nan

        if print_summary:
            print("LOGOCV (micro/pooled over all left-out groups):")
            print(f"\tMAE: {micro_mae:.3f} | RMSE: {micro_rmse:.3f} | P95: {micro_p95:.3f} | Max: {micro_max:.3f}")
            print(f"\tKendall tau: {micro_tau:.3f} | Precision@{k_top}: {micro_p_at_k:.3f} | ESR_mu: {micro_esr:.3f}")
            print("LOGOCV (macro mean ± std across groups):")
            print(f"\tMAE: {macro_mae:.3f} ± {macro_mae_std:.3f} | RMSE: {macro_rmse:.3f} ± {macro_rmse_std:.3f}")
            print(f"\tP95(avg): {macro_p95:.3f} | Max(avg): {macro_max:.3f}")
            print(f"\tKendall tau(avg): {macro_tau:.3f} | Precision@{k_top}(avg): {macro_p_at_k:.3f} | ESR_mu(avg): {macro_esr:.3f}")

        return CVSummary(
            micro_mae, micro_rmse, micro_med, micro_p95, micro_p98, micro_max,
            micro_tau, micro_p_at_k, micro_esr,
            macro_mae, macro_rmse, macro_med, macro_p95, macro_p98, macro_max,
            macro_tau, macro_p_at_k, macro_esr,
            macro_mae_std, macro_rmse_std
        )

    @staticmethod
    def summarize_logocv(
            folds: List["EvaluationMetrics"],
            print_summary: bool = True,
            k_top: int = 10
    ) -> CVSummary:
        """Aggregates LOGOCV metrics.
        - Classic micro/pooled (MAE/RMSE/tails) by concatenation (como antes).
        - BO-aware micro aggregated **by fold** (no ranking global):
            * Kendall's tau: weighted by #pairs per fold.
            * Precision@k: sum of hits / sum of effective k across folds.
            * ESR_mu: mean across folds.
        - Macro: simple mean across folds.
        """
        # ---------- Classic micro/pooled (concat) ----------
        if folds:
            all_true = np.concatenate([f.y_true for f in folds])
            all_pred = np.concatenate([f.y_pred for f in folds])
            all_abs  = np.abs(all_true - all_pred)
            N = all_true.size
        else:
            all_true = all_pred = all_abs = np.array([])
            N = 0

        micro_mae  = float(np.mean(all_abs)) if N else np.nan
        micro_rmse = float(np.sqrt(np.mean((all_true - all_pred)**2))) if N else np.nan
        micro_med  = float(np.median(all_abs)) if all_abs.size else np.nan
        micro_p95  = float(np.percentile(all_abs, 95)) if all_abs.size else np.nan
        micro_p98  = float(np.percentile(all_abs, 98)) if all_abs.size else np.nan
        micro_max  = float(np.max(all_abs)) if all_abs.size else np.nan

        # ---------- Per-fold arrays (para macro y micro BO-aware) ----------
        mae_vals   = np.array([f.MAE() for f in folds], float)
        rmse_vals  = np.array([f.RMSE() for f in folds], float)
        med_vals   = np.array([f.residual_median() for f in folds], float)
        p95_vals   = np.array([f.residual_p95() for f in folds], float)
        p98_vals   = np.array([f.residual_p98() for f in folds], float)
        max_vals   = np.array([f.residual_max() for f in folds], float)

        sizes      = np.array([f.n() for f in folds], int)
        # Evita tau en folds con n<2
        tau_vals   = np.array([f.kendall_tau() if f.n() >= 2 else np.nan for f in folds], float)
        prec_vals  = np.array([f.precision_at_k(k_top) if f.n() >= 1 else np.nan for f in folds], float)
        esr_vals   = np.array([f.esr_mu() if f.n() >= 1 else np.nan for f in folds], float)

        # ---------- Macro (mean across folds) ----------
        macro_mae  = float(np.mean(mae_vals))  if mae_vals.size  else np.nan
        macro_rmse = float(np.mean(rmse_vals)) if rmse_vals.size else np.nan
        macro_med  = float(np.mean(med_vals))  if med_vals.size  else np.nan
        macro_p95  = float(np.mean(p95_vals))  if p95_vals.size  else np.nan
        macro_p98  = float(np.mean(p98_vals))  if p98_vals.size  else np.nan
        macro_max  = float(np.mean(max_vals))  if max_vals.size  else np.nan

        macro_tau   = float(np.nanmean(tau_vals))  if tau_vals.size  else np.nan
        macro_prec  = float(np.nanmean(prec_vals)) if prec_vals.size else np.nan
        macro_esr   = float(np.nanmean(esr_vals))  if esr_vals.size  else np.nan

        macro_mae_std  = float(np.std(mae_vals))  if mae_vals.size  else np.nan
        macro_rmse_std = float(np.std(rmse_vals)) if rmse_vals.size else np.nan

        # ---------- Micro BO-aware (AGREGADO POR FOLD) ----------
        # 1) Kendall tau: media ponderada por nº de pares (n*(n-1)/2) por fold
        pair_weights = np.where(sizes >= 2, sizes * (sizes - 1) / 2.0, 0.0).astype(float)
        if np.nansum(pair_weights) > 0 and np.any(np.isfinite(tau_vals)):
            micro_tau = float(np.nansum(tau_vals * pair_weights) / np.nansum(pair_weights))
        else:
            micro_tau = float("nan")

        # 2) Precision@k: sum(hits) / sum(k_eff) sobre folds
        hits = 0.0
        den  = 0.0
        for f in folds:
            n = f.n()
            if n <= 0:
                continue
            k_eff = int(min(k_top, n))
            true_top = np.argpartition(f.y_true, k_eff-1)[:k_eff]
            pred_top = np.argpartition(f.y_pred, k_eff-1)[:k_eff]
            inter = len(np.intersect1d(true_top, pred_top, assume_unique=False))
            hits += inter
            den  += k_eff
        micro_prec = float(hits / den) if den > 0 else float("nan")

        # 3) ESR_mu: media simple de los ESR por fold (puedes cambiar a ponderada por n si lo prefieres)
        micro_esr = float(np.nanmean(esr_vals)) if np.any(np.isfinite(esr_vals)) else float("nan")

        if print_summary:
            print("LOGOCV (micro/pooled over all left-out groups):")
            print(f"\tMAE: {micro_mae:.3f} | RMSE: {micro_rmse:.3f} | P95: {micro_p95:.3f} | Max: {micro_max:.3f}")
            print(f"\tKendall tau (fold-agg): {micro_tau:.3f} | Precision@{k_top} (fold-agg): {micro_prec:.3f} | ESR_mu (fold-avg): {micro_esr:.3f}")
            print("LOGOCV (macro mean ± std across groups):")
            print(f"\tMAE: {macro_mae:.3f} ± {macro_mae_std:.3f} | RMSE: {macro_rmse:.3f} ± {macro_rmse_std:.3f}")
            print(f"\tP95(avg): {macro_p95:.3f} | Max(avg): {macro_max:.3f}")
            print(f"\tKendall tau(avg): {macro_tau:.3f} | Precision@{k_top}(avg): {macro_prec:.3f} | ESR_mu(avg): {macro_esr:.3f}")

        return CVSummary(
            micro_mae, micro_rmse, micro_med, micro_p95, micro_p98, micro_max,
            micro_tau, micro_prec, micro_esr,
            macro_mae, macro_rmse, macro_med, macro_p95, macro_p98, macro_max,
            macro_tau, macro_prec, macro_esr,
            macro_mae_std, macro_rmse_std
        )
















#
# import warnings
# from dataclasses import dataclass
# from typing import List
# import numpy as np
# from sklearn.metrics import (
#     mean_absolute_error,
#     root_mean_squared_error
# )
#
# def root_mean_squared_error(y_true, y_pred):
#     e = np.asarray(y_true) - np.asarray(y_pred)
#     return float(np.sqrt(np.mean(e**2)))
#
#
# @dataclass
# class CVSummary:
#     # micro/pooled (sobre todas las predicciones concatenadas)
#     micro_mae: float
#     micro_rmse: float
#     micro_resid_median: float
#     micro_resid_p95: float
#     micro_resid_p98: float
#     micro_resid_max: float
#     # macro (promedio simple por grupo/fold)
#     macro_mae: float
#     macro_rmse: float
#     macro_resid_median: float
#     macro_resid_p95: float
#     macro_resid_p98: float
#     macro_resid_max: float
#     # opcional: dispersión entre grupos
#     macro_mae_std: float
#     macro_rmse_std: float
#
#
# class EvaluationMetrics:
#     """Evaluation metrics for a single split/fold (y_true,y_pred)."""
#
#     def __init__(self, y_true, y_pred) -> None:
#         # 1) normaliza a float y 1D
#         self.y_true = np.asarray(y_true, dtype=float).ravel()
#         self.y_pred = np.asarray(y_pred, dtype=float).ravel()
#
#         # 2) chequea longitudes
#         if self.y_true.shape != self.y_pred.shape:
#             raise ValueError(f"Shape mismatch: y_true {self.y_true.shape} vs y_pred {self.y_pred.shape}")
#
#         # 3) filtra no-finitos en ambos
#         mask = np.isfinite(self.y_true) & np.isfinite(self.y_pred)
#         dropped = int(np.count_nonzero(~mask))
#         if dropped > 0:
#             warnings.warn(
#                 f"{dropped} non-finite values dropped in metrics computation.",
#                 UserWarning
#             )
#             self.y_true = self.y_true[mask]
#             self.y_pred = self.y_pred[mask]
#
#         # 4) precompute residuos
#         self._res = self.y_true - self.y_pred
#         self._abs = np.abs(self._res)
#
#
#     # --- básicos por fold ---
#     def n(self) -> int:
#         return self._abs.size
#
#     def residuals(self) -> np.ndarray:
#         return self._res
#
#     def abs_residuals(self) -> np.ndarray:
#         return self._abs
#
#     def residual_median(self) -> float:
#         return float(np.median(self._abs))
#
#     def residual_p95(self) -> float:
#         return float(np.percentile(self._abs, 95))
#
#     def residual_p98(self) -> float:
#         return float(np.percentile(self._abs, 98))
#
#     def residual_max(self) -> float:
#         return float(np.max(self._abs))
#
#     def residual_std_dev(self) -> float:
#         # desviación típica de los RESIDUOS (no de los absolutos)
#         return float(np.std(self._res))
#
#     def MAE(self) -> float:
#         return float(mean_absolute_error(self.y_true, self.y_pred))
#
#     def RMSE(self) -> float:
#         return float(root_mean_squared_error(self.y_true, self.y_pred))
#
#     # acumuladores útiles para micro-averaging
#     def SAE(self) -> float:
#         return float(self._abs.sum())
#
#     def SSE(self) -> float:
#         return float((self._res**2).sum())
#
#     @staticmethod
#     def summarize_loocv(folds: List["EvaluationMetrics"], print_summary: bool = False) -> CVSummary:
#         """
#         LOOCV summary:
#           - micro/pooled: concatena todos los residuos de los folds y computa métricas una sola vez.
#           - macro: promedio simple de las métricas por fold.
#         Devuelve un CVSummary (mismo tipo que summarize_logocv).
#         """
#         # ---- micro/pooled (todas las observaciones juntas)
#         all_abs = np.concatenate([f.abs_residuals() for f in folds]) if folds else np.array([])
#         all_res = np.concatenate([f.residuals() for f in folds])     if folds else np.array([])
#         N   = sum(f.n()   for f in folds)
#         SAE = sum(f.SAE() for f in folds)  # suma |e|
#         SSE = sum(f.SSE() for f in folds)  # suma e^2
#
#         micro_mae  = SAE / N if N else np.nan
#         micro_rmse = np.sqrt(SSE / N) if N else np.nan
#         micro_med  = float(np.median(all_abs))    if all_abs.size else np.nan
#         micro_p95  = float(np.percentile(all_abs, 95)) if all_abs.size else np.nan
#         micro_p98  = float(np.percentile(all_abs, 98)) if all_abs.size else np.nan
#         micro_max  = float(np.max(all_abs))       if all_abs.size else np.nan
#
#         # ---- macro (media de métricas por fold)
#         mae_vals  = np.array([f.MAE() for f in folds], float)
#         rmse_vals = np.array([f.RMSE() for f in folds], float)
#         med_vals  = np.array([f.residual_median() for f in folds], float)
#         p95_vals  = np.array([f.residual_p95() for f in folds], float)
#         p98_vals  = np.array([f.residual_p98() for f in folds], float)
#         max_vals  = np.array([f.residual_max() for f in folds], float)
#
#         macro_mae  = float(np.mean(mae_vals))  if mae_vals.size  else np.nan
#         macro_rmse = float(np.mean(rmse_vals)) if rmse_vals.size else np.nan
#         macro_med  = float(np.mean(med_vals))  if med_vals.size  else np.nan
#         macro_p95  = float(np.mean(p95_vals))  if p95_vals.size  else np.nan
#         macro_p98  = float(np.mean(p98_vals))  if p98_vals.size  else np.nan
#         macro_max  = float(np.mean(max_vals))  if max_vals.size  else np.nan
#
#         macro_mae_std  = float(np.std(mae_vals))  if mae_vals.size  else np.nan
#         macro_rmse_std = float(np.std(rmse_vals)) if rmse_vals.size else np.nan
#
#         if print_summary:
#             print("LOOCV (micro/pooled over all left-out samples):")
#             print(f"\tMAE: {micro_mae:.2f}")
#             print(f"\tRMSE: {micro_rmse:.2f}")
#             print(f"\tResidual Median: {micro_med:.2f}")
#             print(f"\tResidual P95: {micro_p95:.2f}")
#             print(f"\tResidual P98: {micro_p98:.2f}")
#             print(f"\tResidual Max: {micro_max:.2f}")
#
#             print("LOOCV (macro: mean across folds) [± std across folds]:")
#             print(f"\tMAE: {macro_mae:.2f} ± {macro_mae_std:.2f}")
#             print(f"\tRMSE: {macro_rmse:.2f} ± {macro_rmse_std:.2f}")
#             print(f"\tResidual Median (avg): {macro_med:.2f}")
#             print(f"\tResidual P95 (avg):   {macro_p95:.2f}")
#             print(f"\tResidual P98 (avg):   {macro_p98:.2f}")
#             print(f"\tResidual Max (avg):   {macro_max:.2f}")
#
#         return CVSummary(
#             micro_mae, micro_rmse, micro_med, micro_p95, micro_p98, micro_max,
#             macro_mae, macro_rmse, macro_med, macro_p95, macro_p98, macro_max,
#             macro_mae_std, macro_rmse_std
#         )
#
#
#     # --- agregaciones LOGOCV ---
#     @staticmethod
#     def summarize_logocv(folds: List["EvaluationMetrics"], print_summary: bool=False) -> CVSummary:
#         """Devuelve micro (pooled) + macro (por fold) para LOGOCV."""
#         # micro/pooled
#         all_abs = np.concatenate([f.abs_residuals() for f in folds]) if folds else np.array([])
#         all_res = np.concatenate([f.residuals() for f in folds]) if folds else np.array([])
#         N = sum(f.n() for f in folds)
#         SAE = sum(f.SAE() for f in folds)
#         SSE = sum(f.SSE() for f in folds)
#
#         micro_mae  = SAE / N if N else np.nan
#         micro_rmse = np.sqrt(SSE / N) if N else np.nan
#         micro_med  = float(np.median(all_abs)) if all_abs.size else np.nan
#         micro_p95  = float(np.percentile(all_abs, 95)) if all_abs.size else np.nan
#         micro_p98  = float(np.percentile(all_abs, 98)) if all_abs.size else np.nan
#         micro_max  = float(np.max(all_abs)) if all_abs.size else np.nan
#
#         # macro (promedio simple entre folds)
#         mae_vals  = np.array([f.MAE() for f in folds], float)
#         rmse_vals = np.array([f.RMSE() for f in folds], float)
#         med_vals  = np.array([f.residual_median() for f in folds], float)
#         p95_vals  = np.array([f.residual_p95() for f in folds], float)
#         p98_vals  = np.array([f.residual_p98() for f in folds], float)
#         max_vals  = np.array([f.residual_max() for f in folds], float)
#
#         macro_mae  = float(np.mean(mae_vals))  if mae_vals.size  else np.nan
#         macro_rmse = float(np.mean(rmse_vals)) if rmse_vals.size else np.nan
#         macro_med  = float(np.mean(med_vals))  if med_vals.size  else np.nan
#         macro_p95  = float(np.mean(p95_vals))  if p95_vals.size  else np.nan
#         macro_p98  = float(np.mean(p98_vals))  if p98_vals.size  else np.nan
#         macro_max  = float(np.mean(max_vals))  if max_vals.size  else np.nan
#
#         macro_mae_std  = float(np.std(mae_vals))  if mae_vals.size  else np.nan
#         macro_rmse_std = float(np.std(rmse_vals)) if rmse_vals.size else np.nan
#
#         if print_summary:
#             print("LOGOCV (micro/pooled over all left-out groups):")
#             print(f"\tMAE: {micro_mae:.2f}")
#             print(f"\tRMSE: {micro_rmse:.2f}")
#             print(f"\tResidual Median: {micro_med:.2f}")
#             print(f"\tResidual P95: {micro_p95:.2f}")
#             print(f"\tResidual P98: {micro_p98:.2f}")
#             print(f"\tResidual Max: {micro_max:.2f}")
#
#             print("LOGOCV (macro: mean across groups) [± std across groups]:")
#             print(f"\tMAE: {macro_mae:.2f} ± {macro_mae_std:.2f}")
#             print(f"\tRMSE: {macro_rmse:.2f} ± {macro_rmse_std:.2f}")
#             print(f"\tResidual Median (avg): {macro_med:.2f}")
#             print(f"\tResidual P95 (avg):   {macro_p95:.2f}")
#             print(f"\tResidual P98 (avg):   {macro_p98:.2f}")
#             print(f"\tResidual Max (avg):   {macro_max:.2f}")
#
#
#         return CVSummary(
#             micro_mae, micro_rmse, micro_med, micro_p95, micro_p98, micro_max,
#             macro_mae, macro_rmse, macro_med, macro_p95, macro_p98, macro_max,
#             macro_mae_std, macro_rmse_std
#         )
#
