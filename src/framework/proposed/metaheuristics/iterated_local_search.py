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

from typing import List, Optional, Tuple
import numpy as np
from framework.proposed.workload_characterization.workload import WorkloadCharacterized


class IteratedLocalSearch:
    """
    Iterated Local Search (ILS) minimalista y robusto para tuning ruidoso.

    Principios:
      - Banda de tolerancia ±τ% (zona gris) para robustez al ruido.
      - Decisión 'best-aware': comparar siempre contra el incumbente.
      - Perturbación alineada con STL (descriptor latente 100-D + ids + φ):
          * mejora ('improve'): blend parcial (evita frontera) + re-anclaje suave ids/φ
          * gris ('grey'): paso leve hacia centroide bueno + jitter ortogonal pequeño
          * peor ('worse'): paso más decidido hacia centroide bueno con pesos por estabilidad
    """

    def __init__(self, tau_perc: float = 4.0):
        self.best_workload: Optional[WorkloadCharacterized] = None
        self.last_is_suboptimal: bool = False
        self.tau_perc: float = float(tau_perc)

        # Contexto para smart_perturb (paper-friendly):
        self.last_decision: str = "init"                  # "improve" | "grey" | "worse" | "init"
        self.prev_best_vector: Optional[np.ndarray] = None  # 100-D del best previo (para blend en 'improve')

    # ---------------- Acceptance (best-aware) ----------------

    def update_best_to_T(
            self,
            workload: WorkloadCharacterized,
            tau_perc: Optional[float] = None
    ) -> bool:
        """
        Aceptación con banda de ruido ±τ%.
          - f_new < (1−τ)·f_best → improve  → actualizar incumbente
          - f_new > (1+τ)·f_best → worse    → sugerir tabú
          - en otro caso         → grey     → estabilidad (sin cambio de best)
        """
        tau = float(self.tau_perc if tau_perc is None else tau_perc)

        if self.best_workload is None:
            self.best_workload = workload
            self.last_is_suboptimal = False
            self.last_decision = "improve"
            self.prev_best_vector = np.asarray(workload.vector_metrics_garralda, float).copy()
            return False

        low  = (1.0 - tau / 100.0) * self.best_workload.time_execution
        high = (1.0 + tau / 100.0) * self.best_workload.time_execution

        if workload.time_execution < low:
            # guarda el latente previo para el blend posterior
            self.prev_best_vector = np.asarray(self.best_workload.vector_metrics_garralda, float).copy()
            print(f"[ILS] New best: {workload.time_execution:.3f} < {self.best_workload.time_execution:.3f}  (tau={tau:.1f}%)")
            self.best_workload = workload
            self.last_is_suboptimal = False
            self.last_decision = "improve"
            return False

        if workload.time_execution > high:
            self.last_is_suboptimal = True
            self.last_decision = "worse"
            print(f"[ILS] Clearly worse: {workload.time_execution:.3f} > {high:.3f}  (tau={tau:.1f}%) → suggest TABU")
            return True

        print(f"[ILS] Gray zone: {workload.time_execution:.3f} ∈ [{low:.3f}, {high:.3f}]  (tau={tau:.1f}%)")
        self.last_is_suboptimal = False
        self.last_decision = "grey"
        return False

    # Reemplaza en IteratedLocalSearch
    def update_best(self, workload: WorkloadCharacterized, tau_perc: Optional[float] = None) -> bool:
        tau = float(self.tau_perc if tau_perc is None else tau_perc)

        # ===== métrica objetivo: TR si existe, si no T =====
        def _target(wc):
            return getattr(wc, "time_resources", None) if getattr(wc, "time_resources", None) is not None else wc.time_execution

        if self.best_workload is None:
            self.best_workload = workload
            self.last_is_suboptimal = False
            self.last_decision = "improve"
            self.prev_best_vector = np.asarray(workload.vector_metrics_garralda, float).copy()
            return False

        f_new  = float(_target(workload))
        f_best = float(_target(self.best_workload))
        low  = (1.0 - tau / 100.0) * f_best
        high = (1.0 + tau / 100.0) * f_best

        if f_new < low:
            self.prev_best_vector = np.asarray(self.best_workload.vector_metrics_garralda, float).copy()
            print(f"[ILS] New best: {f_new:.3f} < {f_best:.3f}  (tau={tau:.1f}%)")
            self.best_workload = workload
            self.last_is_suboptimal = False
            self.last_decision = "improve"
            return False

        if f_new > high:
            self.last_is_suboptimal = True
            self.last_decision = "worse"
            print(f"[ILS] Clearly worse: {f_new:.3f} > {high:.3f}  (tau={tau:.1f}%) → suggest TABU")
            return True

        print(f"[ILS] Gray zone: {f_new:.3f} ∈ [{low:.3f}, {high:.3f}]  (tau={tau:.1f}%)")
        self.last_is_suboptimal = False
        self.last_decision = "grey"
        return False


    @staticmethod
    def is_tabu_list(last_value: float, new_value: float, tolerance_perc: int = 3) -> bool:
        """Compatibilidad legacy: True si new_value > last_value·(1+τ%)."""
        high = last_value * (1 + tolerance_perc / 100.0)
        return new_value > high

    # ---------------- Perturbation (descriptor-aligned) ----------------

    def smart_perturb(
            self,
            last_vector: List[float],                 # z_last (100-D, del wc recién evaluado)
            historical_latents: List[np.ndarray],     # H (n,100)
            historical_targets: List[float],          # y (n,) - T
            historical_ids: List[int],                # ids (n,)
            historical_phi: List[List[float]],        # φ (n,4)
            historical_suboptimal: List[bool],        # flags (n,)
            step_size: float = 0.20,
            eta_improve: float = 0.35,
            jitter_frac: float = 0.08
    ) -> Tuple[Optional[List[float]], Optional["WorkloadCharacterized.Environment"]]:

        if not self.best_workload:
            return None, None

        # ----- datos y guardarraíles -----
        z_last = np.asarray(last_vector, float)
        H  = np.asarray(historical_latents, float)   if len(historical_latents) else None
        y  = np.asarray(historical_targets, float)   if len(historical_targets) else None
        ids = np.asarray(historical_ids, float)      if len(historical_ids)     else None
        PHI = np.asarray(historical_phi, float)      if len(historical_phi)     else None
        sub = np.asarray(historical_suboptimal, bool) if len(historical_suboptimal) else None

        # --- LOG 0: ancla antes de cualquier cambio ---
        ids0 = float(self.best_workload.dataset_size)
        phi0 = np.asarray(self.best_workload.resource_shape, float).copy()

        # Normaliza y sanea targets para pesos de rendimiento
        def _finite(a, fill):
            a = np.asarray(a)
            m = ~(np.isfinite(a))
            if m.any():
                a = a.copy()
                a[m] = fill
            return a

        # ---------- Caso 'improve': blend parcial + re-anclaje suave ----------
        if self.last_decision == "improve":
            z_prev = self.prev_best_vector if self.prev_best_vector is not None else z_last
            z_curr = np.asarray(self.best_workload.vector_metrics_garralda, float)
            z_new  = (1.0 - eta_improve) * z_prev + eta_improve * z_curr

            if H is not None and y is not None and ids is not None and PHI is not None and H.shape[0] > 0:
                y_s = _finite(y, np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 1.0)
                y_s = np.clip(y_s, 1e-12, None)
                w_perf = 1.0 / (y_s + 1e-8)
                sw = float(w_perf.sum())
                if sw <= 1e-12 or (not np.isfinite(sw)):
                    w_perf = np.ones_like(y_s) / y_s.size
                else:
                    w_perf /= sw

                ids_anchor = float(np.average(ids, weights=w_perf))
                phi_anchor = np.average(PHI, axis=0, weights=w_perf)
                alpha = 0.4
                self.best_workload.dataset_size = int(alpha * ids_anchor + (1 - alpha) * float(self.best_workload.dataset_size))
                self.best_workload.resource_shape = (
                        alpha * phi_anchor + (1 - alpha) * np.asarray(self.best_workload.resource_shape, float)
                ).tolist()

            self.best_workload.vector_metrics_garralda = z_new

            # --- LOG A: ancla tras re-anclaje en 'improve' ---
            ids1 = float(self.best_workload.dataset_size)
            phi1 = np.asarray(self.best_workload.resource_shape, float)
            print(f"[ILS] dec=improve anchor ids:{ids0:.0f}->{ids1:.0f}  Δids={ids1-ids0:+.0f}  Δφ_L1={float(np.sum(np.abs(phi1-phi0))):.3f}")

            return z_new.tolist(), self.best_workload.environment

        # ---------- Fallback: poco histórico → paso mínimo hacia best ----------
        if H is None or y is None or ids is None or PHI is None or H.shape[0] == 0:
            z_best = np.asarray(self.best_workload.vector_metrics_garralda, float)
            d = z_best - z_last
            nrm = np.linalg.norm(d)
            if nrm == 0.0:
                return None, None
            z_new = z_last + float(np.clip(step_size, 0.05, 0.5)) * (d / nrm)
            self.best_workload.vector_metrics_garralda = z_new

            # --- LOG A (fallback): ancla no cambia aquí, pero dejamos constancia mínima ---
            print(f"[ILS] dec={self.last_decision} fallback(no-history) w_sum=NA nz=NA/NA")
            print(f"[ILS] dec={self.last_decision} anchor ids:{ids0:.0f}->{ids0:.0f}  Δids=+0  Δφ_L1=0.000")

            return z_new.tolist(), self.best_workload.environment

        # ---------- Pesos ids/φ/perf/subopt (centroide 'bueno') ----------
        # ids: similitud robusta en s=log1p(ids)
        s = np.log1p(ids)
        s_med = np.median(s)
        s_ref = np.log1p(float(self.best_workload.dataset_size))
        mad_s = 1.4826 * np.median(np.abs(s - s_med)) + 1e-12
        d_ids = np.abs(s - s_ref) / mad_s
        w_ids = np.exp(-d_ids)
        w_ids[~np.isfinite(w_ids)] = 0.0

        # φ: normalización robusta por columna + distancia L1
        phi_med = np.median(PHI, axis=0)
        phi_mad = 1.4826 * np.median(np.abs(PHI - phi_med), axis=0) + 1e-12
        PHI_std = (PHI - phi_med) / phi_mad
        phi_ref = (np.asarray(self.best_workload.resource_shape, float) - phi_med) / phi_mad
        d_phi = np.sum(np.abs(PHI_std - phi_ref), axis=1)
        mad_dphi = 1.4826 * np.median(np.abs(d_phi - np.median(d_phi))) + 1e-12
        w_phi = np.exp(-d_phi / mad_dphi)
        w_phi[~np.isfinite(w_phi)] = 0.0

        # rendimiento: mejor T → más peso
        y_s = _finite(y, np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 1.0)
        y_s = np.clip(y_s, 1e-12, None)
        w_perf = 1.0 / (y_s + 1e-8)
        swp = float(w_perf.sum())
        if swp <= 1e-12 or (not np.isfinite(swp)):
            w_perf = np.ones_like(y_s) / y_s.size
        else:
            w_perf /= swp

        # penalización por historial subóptimo (suave)
        if sub is not None and sub.size == H.shape[0]:
            pen_sub = np.where(sub, 0.6, 1.0)
        else:
            pen_sub = np.ones(H.shape[0])

        # Peso final y salvaguarda de suma
        w = w_ids * w_phi * w_perf * pen_sub
        w[~np.isfinite(w)] = 0.0
        sum_w = float(w.sum())
        if sum_w <= 1e-12:
            # Fallback robusto: pesos uniformes
            w = np.ones(H.shape[0], dtype=float) / float(H.shape[0])
            print(f"[ILS] dec={self.last_decision} WARNING w_sum≈0 → using uniform weights")
        else:
            w = w / sum_w

        # --- LOG 1: resumen de pesos (mínimo y suficiente) ---
        print(f"[ILS] dec={self.last_decision} w_sum={float(w.sum()):.2e} nz={int(np.count_nonzero(w))}/{w.size}")

        # ---------- Dirección y estabilidad por dimensión ----------
        try:
            z_centroid = np.average(H, axis=0, weights=w)
        except ZeroDivisionError:
            z_centroid = np.mean(H, axis=0)

        d = z_centroid - z_last
        nrm = np.linalg.norm(d)
        if nrm == 0.0:
            return None, None
        d /= nrm

        # estabilidad por dimensión (MAD ponderada): dims más estables → mayor peso
        abs_dev = np.abs(H - z_centroid)                 # (n,D)
        mad_w = 1.4826 * np.average(abs_dev, axis=0, weights=w)
        mad_w[~np.isfinite(mad_w)] = 0.0
        max_mad = float(np.max(mad_w) + 1e-12)
        dim_w = 1.0 - (mad_w / max_mad)
        if np.all(dim_w <= 1e-12):
            dim_w = np.ones_like(dim_w)

        # ---------- Intensidad según decisión ----------
        if self.last_decision == "grey":
            step = float(np.clip(0.5 * step_size, 0.03, 0.25))
            z_new = z_last + step * dim_w * d

            k = max(1, int(0.10 * dim_w.size))
            idxs = np.argsort(-dim_w)[:k]
            ortho = np.random.randn(dim_w.size)
            # quitar componente en d para mantener jitter ortogonal
            ortho -= ortho.dot(d) * d
            n_ortho = np.linalg.norm(ortho)
            if n_ortho > 0:
                ortho /= n_ortho
                jitter = jitter_frac * step * ortho
                mask = np.zeros_like(jitter); mask[idxs] = 1.0
                z_new = z_new + jitter * mask
        else:  # "worse"
            step = float(np.clip(1.0 * step_size, 0.08, 0.50))
            z_new = z_last + step * dim_w * d

        # ---------- Re-anclaje ids/φ (centra TR/Sobol donde rinde) ----------
        ids_anchor = float(np.average(ids, weights=w)) if np.isfinite(np.average(ids, weights=w)) else float(np.median(ids))
        phi_anchor = np.average(PHI, axis=0, weights=w)
        if not np.all(np.isfinite(phi_anchor)):
            phi_anchor = np.median(PHI, axis=0)

        alpha = 0.6
        self.best_workload.dataset_size = int(alpha * ids_anchor + (1 - alpha) * float(self.best_workload.dataset_size))
        self.best_workload.resource_shape = (
                alpha * phi_anchor + (1 - alpha) * np.asarray(self.best_workload.resource_shape, float)
        ).tolist()

        self.best_workload.vector_metrics_garralda = z_new

        # --- LOG 2: ancla tras re-anclaje ---
        ids1 = float(self.best_workload.dataset_size)
        phi1 = np.asarray(self.best_workload.resource_shape, float)
        print(f"[ILS] dec={self.last_decision} anchor ids:{ids0:.0f}->{ids1:.0f}  Δids={ids1-ids0:+.0f}  Δφ_L1={float(np.sum(np.abs(phi1-phi0))):.3f}")

        print(f"[ILS] smart_perturb({self.last_decision}): step={step:.3f}, "
              f"ids→{self.best_workload.dataset_size}, φ_avg={np.round(self.best_workload.resource_shape,2)}")

        return z_new.tolist(), self.best_workload.environment

    def smart_perturb_sin_log(
            self,
            last_vector: List[float],                 # z_last (100-D, del wc recién evaluado)
            historical_latents: List[np.ndarray],     # H (n,100)
            historical_targets: List[float],          # y (n,) - T
            historical_ids: List[int],                # ids (n,)
            historical_phi: List[List[float]],        # φ (n,4)
            historical_suboptimal: List[bool],        # flags (n,)
            step_size: float = 0.20,
            eta_improve: float = 0.35,
            jitter_frac: float = 0.08
    ) -> Tuple[Optional[List[float]], Optional["WorkloadCharacterized.Environment"]]:

        if not self.best_workload:
            return None, None

        # ----- datos y guardarraíles -----
        z_last = np.asarray(last_vector, float)
        H  = np.asarray(historical_latents, float)   if len(historical_latents) else None
        y  = np.asarray(historical_targets, float)   if len(historical_targets) else None
        ids = np.asarray(historical_ids, float)      if len(historical_ids)     else None
        PHI = np.asarray(historical_phi, float)      if len(historical_phi)     else None
        sub = np.asarray(historical_suboptimal, bool) if len(historical_suboptimal) else None

        # Normaliza y sanea targets para pesos de rendimiento
        def _finite(a, fill):
            a = np.asarray(a)
            m = ~(np.isfinite(a))
            if m.any():
                a = a.copy()
                a[m] = fill
            return a

        # ---------- Caso 'improve': blend parcial + re-anclaje suave ----------
        if self.last_decision == "improve":
            z_prev = self.prev_best_vector if self.prev_best_vector is not None else z_last
            z_curr = np.asarray(self.best_workload.vector_metrics_garralda, float)
            z_new  = (1.0 - eta_improve) * z_prev + eta_improve * z_curr

            if H is not None and y is not None and ids is not None and PHI is not None and H.shape[0] > 0:
                y_s = _finite(y, np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 1.0)
                y_s = np.clip(y_s, 1e-12, None)
                w_perf = 1.0 / (y_s + 1e-8)
                sw = float(w_perf.sum())
                if sw <= 1e-12 or (not np.isfinite(sw)):
                    w_perf = np.ones_like(y_s) / y_s.size
                else:
                    w_perf /= sw

                ids_anchor = float(np.average(ids, weights=w_perf))
                phi_anchor = np.average(PHI, axis=0, weights=w_perf)
                alpha = 0.4
                self.best_workload.dataset_size = int(alpha * ids_anchor + (1 - alpha) * float(self.best_workload.dataset_size))
                self.best_workload.resource_shape = (
                        alpha * phi_anchor + (1 - alpha) * np.asarray(self.best_workload.resource_shape, float)
                ).tolist()

            self.best_workload.vector_metrics_garralda = z_new
            return z_new.tolist(), self.best_workload.environment

        # ---------- Fallback: poco histórico → paso mínimo hacia best ----------
        if H is None or y is None or ids is None or PHI is None or H.shape[0] == 0:
            z_best = np.asarray(self.best_workload.vector_metrics_garralda, float)
            d = z_best - z_last
            nrm = np.linalg.norm(d)
            if nrm == 0.0:
                return None, None
            z_new = z_last + float(np.clip(step_size, 0.05, 0.5)) * (d / nrm)
            self.best_workload.vector_metrics_garralda = z_new
            return z_new.tolist(), self.best_workload.environment

        # ---------- Pesos ids/φ/perf/subopt (centroide 'bueno') ----------
        # ids: similitud robusta en s=log1p(ids)
        s = np.log1p(ids)
        s_med = np.median(s)
        s_ref = np.log1p(float(self.best_workload.dataset_size))
        mad_s = 1.4826 * np.median(np.abs(s - s_med)) + 1e-12
        d_ids = np.abs(s - s_ref) / mad_s
        w_ids = np.exp(-d_ids)
        w_ids[~np.isfinite(w_ids)] = 0.0

        # φ: normalización robusta por columna + distancia L1
        phi_med = np.median(PHI, axis=0)
        phi_mad = 1.4826 * np.median(np.abs(PHI - phi_med), axis=0) + 1e-12
        PHI_std = (PHI - phi_med) / phi_mad
        phi_ref = (np.asarray(self.best_workload.resource_shape, float) - phi_med) / phi_mad
        d_phi = np.sum(np.abs(PHI_std - phi_ref), axis=1)
        mad_dphi = 1.4826 * np.median(np.abs(d_phi - np.median(d_phi))) + 1e-12
        w_phi = np.exp(-d_phi / mad_dphi)
        w_phi[~np.isfinite(w_phi)] = 0.0

        # rendimiento: mejor T → más peso
        y_s = _finite(y, np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 1.0)
        y_s = np.clip(y_s, 1e-12, None)
        w_perf = 1.0 / (y_s + 1e-8)
        swp = float(w_perf.sum())
        if swp <= 1e-12 or (not np.isfinite(swp)):
            w_perf = np.ones_like(y_s) / y_s.size
        else:
            w_perf /= swp

        # penalización por historial subóptimo (suave)
        if sub is not None and sub.size == H.shape[0]:
            pen_sub = np.where(sub, 0.6, 1.0)
        else:
            pen_sub = np.ones(H.shape[0])

        # Peso final y salvaguarda de suma
        w = w_ids * w_phi * w_perf * pen_sub
        w[~np.isfinite(w)] = 0.0
        sum_w = float(w.sum())
        if sum_w <= 1e-12:
            # Fallback robusto: pesos uniformes
            w = np.ones(H.shape[0], dtype=float) / float(H.shape[0])
        else:
            w = w / sum_w

        # ---------- Dirección y estabilidad por dimensión ----------
        try:
            z_centroid = np.average(H, axis=0, weights=w)
        except ZeroDivisionError:
            # Doble salvaguarda (por si algún backend de NumPy exige suma estricta)
            z_centroid = np.mean(H, axis=0)

        d = z_centroid - z_last
        nrm = np.linalg.norm(d)
        if nrm == 0.0:
            return None, None
        d /= nrm

        # estabilidad por dimensión (MAD ponderada): dims más estables → mayor peso
        abs_dev = np.abs(H - z_centroid)                 # (n,D)
        # media ponderada de |dev| (más estable y barata)
        mad_w = 1.4826 * np.average(abs_dev, axis=0, weights=w)
        mad_w[~np.isfinite(mad_w)] = 0.0
        max_mad = float(np.max(mad_w) + 1e-12)
        dim_w = 1.0 - (mad_w / max_mad)
        if np.all(dim_w <= 1e-12):
            dim_w = np.ones_like(dim_w)

        # ---------- Intensidad según decisión ----------
        if self.last_decision == "grey":
            step = float(np.clip(0.5 * step_size, 0.03, 0.25))
            z_new = z_last + step * dim_w * d

            k = max(1, int(0.10 * dim_w.size))
            idxs = np.argsort(-dim_w)[:k]
            ortho = np.random.randn(dim_w.size)
            # quitar componente en d para mantener jitter ortogonal
            ortho -= ortho.dot(d) * d
            n_ortho = np.linalg.norm(ortho)
            if n_ortho > 0:
                ortho /= n_ortho
                jitter = jitter_frac * step * ortho
                mask = np.zeros_like(jitter); mask[idxs] = 1.0
                z_new = z_new + jitter * mask
        else:  # "worse"
            step = float(np.clip(1.0 * step_size, 0.08, 0.50))
            z_new = z_last + step * dim_w * d

        # ---------- Re-anclaje ids/φ (centra TR/Sobol donde rinde) ----------
        ids_anchor = float(np.average(ids, weights=w)) if np.isfinite(np.average(ids, weights=w)) else float(np.median(ids))
        phi_anchor = np.average(PHI, axis=0, weights=w)
        if not np.all(np.isfinite(phi_anchor)):
            phi_anchor = np.median(PHI, axis=0)

        alpha = 0.6
        self.best_workload.dataset_size = int(alpha * ids_anchor + (1 - alpha) * float(self.best_workload.dataset_size))
        self.best_workload.resource_shape = (
                alpha * phi_anchor + (1 - alpha) * np.asarray(self.best_workload.resource_shape, float)
        ).tolist()

        self.best_workload.vector_metrics_garralda = z_new

        print(f"[ILS] smart_perturb({self.last_decision}): step={step:.3f}, "
              f"ids→{self.best_workload.dataset_size}, φ_avg={np.round(self.best_workload.resource_shape,2)}")

        return z_new.tolist(), self.best_workload.environment
