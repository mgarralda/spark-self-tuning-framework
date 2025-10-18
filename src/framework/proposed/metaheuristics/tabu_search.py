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

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Literal
from scipy.spatial.distance import cdist

Strategy = Literal["good", "avoid_bad", "mixed"]

Key = Tuple[int, ...]


class TabuSearch:
    """
    φ-aware Tabu Search with latent-space bands over standardized descriptors (L1 metric).

    Components:
      • Inclusion Band (IB): single locality gate in z-space (L1 around z_ref).
      • Exclusion Band (EB): excludes pockets of suboptimal regions (memory S).
      • Attraction Band (AB): attraction around recent bests (memory B; does not filter Sobol).
      • Per-configuration tabu with best-aware aspiration.
      • Independent auto-adjustment of IB/EB radii toward a target skip-rate.
      • Internal traces of radii and last skip-rate.

    API:
      - set_inclusion_band(Z_SL, z_ref, seed_best=True)
      - filter_candidates(X, phi_from_cs, phi_scaler) -> (mask_keep, skipped_IB, skipped_EB, n_pass_IB)
      - adapt_from_skip_rate(skipped, total, which={"EB","IB"})
      - is_tabu(...), step(...), observe_after_eval(...), rebuild_bands(), purge_expired(), clear()
      - debug_snapshot()
    """

    def __init__(self,
                 base_bounds: List[Tuple[int, int, int]],
                 tenure_L: int = 9,
                 q_sub: float = 0.75,
                 q_best: float = 0.50,

                 # v2 (kept as-is to remain paper-friendly)
                 target_skip_rate: float = 0.20, #0.05
                 adapt_rate: float = 0.15,
                 min_keep_frac: float = 0.10,
                 min_keep_abs: int = 12
                 ) -> None:
        self.base_bounds = list(base_bounds)

        # Per-configuration tabu memory
        self._tabu_until: Dict[Key, int] = {}
        self._t: int = 0
        self._last_key: Optional[Key] = None  # anti-repeat helper (no new knob)

        # Memories in z (standardized φ)
        self._Z_sub: np.ndarray = np.zeros((0, 4), float)   # S
        self._Z_best: np.ndarray = np.zeros((0, 4), float)  # B

        # EB/AB base/dynamic radii
        self.q_sub = float(q_sub)
        self.q_best = float(q_best)
        self._r_sub_base: float = 0.0
        self._r_best_base: float = 0.0
        self._r_sub_dyn: float = 0.0

        # Inclusion Band (IB)
        self._z_ref: Optional[np.ndarray] = None
        self._Z_SL: np.ndarray = np.zeros((0, 4), float)
        self._r_ib_base: float = 0.0
        self._r_ib_dyn: float = 0.0
        self._ib_enabled: bool = False

        # Skip-rate control
        self.target_skip_rate = float(np.clip(target_skip_rate, 0.0, 0.9))
        self.adapt_rate = float(np.clip(adapt_rate, 0.01, 0.5))
        self._last_skip_rate: float = 0.0

        # Tenure
        self.tenure_L = int(max(1, tenure_L))

        # Anti-empty guardrails
        self._min_keep_frac = float(np.clip(min_keep_frac, 0.0, 0.5))
        self._min_keep_abs = int(max(1, min_keep_abs))

    # ---------- helpers ----------
    @staticmethod
    def _key(x: np.ndarray | List[int]) -> Key:
        arr = np.asarray(x).ravel()
        return tuple(int(v) for v in arr)

    def _active(self, k: Key) -> bool:
        return (k in self._tabu_until) and (self._t < self._tabu_until[k])

    @staticmethod
    def _pairwise_L1_vals(Z: np.ndarray) -> np.ndarray:
        n = Z.shape[0]
        if n < 2:
            return np.array([0.0])
        vals = []
        for i in range(n - 1):
            d = np.sum(np.abs(Z[i+1:] - Z[i]), axis=1)
            vals.append(d)
        return np.concatenate(vals, axis=0)

    @staticmethod
    def _second_nn_row(di: np.ndarray) -> float:
        if di.size <= 1:
            return 0.0
        order = np.sort(di)
        return float(order[1])

    @staticmethod
    def _min_positive(di: np.ndarray) -> float:
        di_pos = di[di > 0.0]
        return float(np.min(di_pos)) if di_pos.size else np.inf

    # ---------- Inclusion Band (IB) ----------
    def set_inclusion_band(self, Z_SL: np.ndarray, z_ref: np.ndarray, *, seed_best: bool = True) -> None:
        self._Z_SL = np.asarray(Z_SL, float) if Z_SL is not None else np.zeros((0, 4), float)
        self._z_ref = np.asarray(z_ref, float) if z_ref is not None else None

        # Robust local scale from SL: cap by Q75 of 2-NN and median positive 1-NN
        if self._Z_SL.shape[0] >= 2:
            d2_list, d1p_list = [], []
            for i in range(self._Z_SL.shape[0]):
                di = np.sum(np.abs(self._Z_SL - self._Z_SL[i]), axis=1)
                d2_list.append(self._second_nn_row(di))
                d1p_list.append(self._min_positive(di))
            d2   = np.asarray(d2_list, float)
            d1p  = np.asarray(d1p_list, float)
            r75  = float(np.quantile(d2, 0.75))
            d1p_f = d1p[np.isfinite(d1p)]
            rmin = float(np.nanmedian(d1p_f)) if d1p_f.size else 0.0
            r_cap = max(r75, rmin, 1e-6)
        else:
            r_cap = 1e-6

        self._r_ib_base = float(max(r_cap, 1e-6))
        if self._r_ib_dyn <= 0.0:
            self._r_ib_dyn = self._r_ib_base
        self._ib_enabled = True

        # Seed AB at first iteration
        if seed_best and (self._z_ref is not None) and (self._Z_best.shape[0] == 0):
            self._Z_best = self._z_ref.reshape(1, -1)
            self.rebuild_bands()

    def in_IB(self, z: np.ndarray) -> bool:
        if (not self._ib_enabled) or (self._z_ref is None) or (self._r_ib_dyn <= 0.0):
            return True
        d = float(np.sum(np.abs(z - self._z_ref)))
        return d <= self._r_ib_dyn

    # ---------- EB/AB ----------
    def rebuild_bands(self) -> None:
        # EB radius from robust quantile over pairwise distances in S
        if self._Z_sub.shape[0] >= 2:
            r_sub = float(np.quantile(self._pairwise_L1_vals(self._Z_sub), self.q_sub))
        else:
            r_sub = 0.0
        self._r_sub_base = max(r_sub, 0.0)

        # AB radius from robust quantile over pairwise distances in B
        if self._Z_best.shape[0] >= 2:
            r_best = float(np.quantile(self._pairwise_L1_vals(self._Z_best), self.q_best))
        else:
            r_best = 0.0
        self._r_best_base = max(r_best, 0.0)

        if self._r_sub_dyn <= 0.0 and self._r_sub_base > 0.0:
            self._r_sub_dyn = self._r_sub_base

    def in_EB(self, z: np.ndarray) -> bool:
        if self._Z_sub.shape[0] == 0 or self._r_sub_dyn <= 0.0:
            return False
        dmin = float(np.min(np.sum(np.abs(self._Z_sub - z[None, :]), axis=1)))
        return dmin <= self._r_sub_dyn

    def in_AB(self, z: np.ndarray) -> bool:
        if self._Z_best.shape[0] == 0 or self._r_best_base <= 0.0:
            return False
        dmin = float(np.min(np.sum(np.abs(self._Z_best - z[None, :]), axis=1)))
        return dmin <= self._r_best_base

    # ---------- candidate filter (IB + EB) ----------
    def filter_candidates(self,
                          X: np.ndarray,
                          phi_from_cs: Callable[[np.ndarray], np.ndarray],
                          phi_scaler) -> Tuple[np.ndarray, int, int, int]:
        """
        Returns:
          - mask_keep: boolean mask of candidates passing IB and EB
          - skipped_IB: count filtered by IB
          - skipped_EB: count filtered by EB (among those passing IB)
          - n_pass_IB: number passing IB (denominator for EB adaptation)
        """
        if X.size == 0:
            return np.zeros(0, dtype=bool), 0, 0, 0

        Phi = np.asarray([phi_from_cs(x) for x in X], float)
        Z   = phi_scaler.transform(Phi)

        # IB gating
        if self._ib_enabled and (self._z_ref is not None) and (self._r_ib_dyn > 0.0):
            dIB = np.sum(np.abs(Z - self._z_ref[None, :]), axis=1)
            keep_ib = (dIB <= self._r_ib_dyn)
        else:
            dIB = None
            keep_ib = np.ones(X.shape[0], dtype=bool)

        # Emergency relax: ensure a minimum survivors after IB (anti-empty & anti-starvation)
        n_pass_ib = int(keep_ib.sum())
        min_keep_target = max(self._min_keep_abs, int(np.ceil(self._min_keep_frac * X.shape[0])))
        min_keep_target = min(min_keep_target, X.shape[0])

        if (dIB is not None) and (n_pass_ib < min_keep_target):
            # Threshold at k-th nearest distance to z_ref
            kth = np.partition(dIB, min_keep_target - 1)[min_keep_target - 1]
            # Proposed new radius bounded within [0.25, 2.0] * r_ib_base
            if self._r_ib_base > 0.0:
                r_new = float(np.clip(kth, 0.25 * self._r_ib_base, 2.0 * self._r_ib_base))
            else:
                r_new = float(kth)
            if r_new > self._r_ib_dyn:
                self._r_ib_dyn = r_new
                keep_ib = (dIB <= self._r_ib_dyn)
                n_pass_ib = int(keep_ib.sum())

        skipped_ib = int((~keep_ib).sum())

        # EB only on those that passed IB
        keep_eb = np.ones(X.shape[0], dtype=bool)
        skipped_eb = 0
        if self._Z_sub.shape[0] > 0 and self._r_sub_dyn > 0.0 and n_pass_ib > 0:
            Z_ib = Z[keep_ib]
            D = np.sum(np.abs(Z_ib[:, None, :] - self._Z_sub[None, :, :]), axis=2)
            dmin = D.min(axis=1)
            mask_ib_post_eb = (dmin > self._r_sub_dyn)
            keep_eb[keep_ib] = mask_ib_post_eb
            skipped_eb = int((~mask_ib_post_eb).sum())

        keep = keep_ib & keep_eb
        return keep, skipped_ib, skipped_eb, n_pass_ib

    # ---------- auto-adjustment ----------
    def adapt_from_skip_rate(self, skipped: int, total: int, which={"EB", "IB"}) -> None:
        """
        Proportional control on dynamic radii to approach target_skip_rate.
        If skip-rate is above target, radii are expanded (more permissive);
        if below target, radii are tightened (more selective).
        """
        if total <= 0:
            return
        sr = float(skipped) / float(total)
        self._last_skip_rate = sr

        def _adapt(radius_dyn: float, radius_base: float) -> float:
            if radius_base <= 0.0 or radius_dyn <= 0.0:
                return radius_dyn
            if sr > self.target_skip_rate:
                # Expand proportionally to the relative excess (≥1)
                factor = 1.0 + self.adapt_rate * ((sr / max(self.target_skip_rate, 1e-9)) - 1.0)
            else:
                # Shrink gently when already permissive (≤1)
                factor = 1.0 - self.adapt_rate * (1.0 - (sr / max(self.target_skip_rate, 1e-9)))
            return float(np.clip(radius_dyn * factor, 0.25 * radius_base, 2.0 * radius_base))

        # Accept set/str for backward compatibility
        if isinstance(which, str):
            which = {which}

        if "IB" in which and self._r_ib_base > 0.0:
            self._r_ib_dyn = _adapt(self._r_ib_dyn, self._r_ib_base)
        if "EB" in which and self._r_sub_base > 0.0 and self._r_sub_dyn > 0.0:
            self._r_sub_dyn = _adapt(self._r_sub_dyn, self._r_sub_base)

    # ---------- per-configuration tabu ----------
    def is_tabu(self,
                x_new: np.ndarray,
                f_new: Optional[float] = None,
                best_value: Optional[float] = None,
                tau_perc: Optional[float] = None) -> bool:
        k = self._key(x_new)
        active = self._active(k)
        if not active:
            return False
        # Aspiration: allow if significantly better than best (best-aware)
        if (f_new is not None) and (best_value is not None) and (tau_perc is not None):
            low = (1.0 - float(tau_perc) / 100.0) * float(best_value)
            if float(f_new) < low:
                return False
        return True

    def step(self, best_value: float, x_new: np.ndarray, f_new: float, tau_perc: float) -> Dict[str, bool]:
        """
        Updates time, applies tabu addition when clearly worse, and prevents
        immediate repetition without improvement (conservative anti-repeat).
        """
        self._t += 1
        evt = {'added_tabu': False, 'repeat_blocked': False}
        k = self._key(x_new)

        high = (1.0 + float(tau_perc) / 100.0) * float(best_value)
        low  = (1.0 - float(tau_perc) / 100.0) * float(best_value)

        # Clearly worse -> add full tenure
        if float(f_new) > high and not self._active(k):
            self._tabu_until[k] = self._t + int(self.tenure_L)
            evt['added_tabu'] = True

        # Conservative anti-repeat (no new knob): if same as last and not improved,
        # add a short tabu to force a different proposal next time.
        if (self._last_key is not None) and (k == self._last_key) and not (float(f_new) < low):
            if not self._active(k):
                self._tabu_until[k] = self._t + max(3, self.tenure_L // 2)
                evt['repeat_blocked'] = True

        self._last_key = k
        return evt

    # ---------- update with observation ----------
    def observe_after_eval(self,
                           x_new: np.ndarray,
                           phi_new: np.ndarray,
                           phi_scaler,
                           *,
                           is_improve: bool,
                           is_clearly_worse: bool,
                           best_k: int = 10) -> None:
        """
        Updates S/B memories from the new observation and rebuilds EB/AB bands.
        - If clearly worse: push to S (enables EB).
        - If improved: push to B (enables AB); keep last best_k entries.
        """
        z = phi_scaler.transform(phi_new.reshape(1, -1)).ravel()

        if is_clearly_worse:
            self._Z_sub = np.vstack([self._Z_sub, z[None, :]]) if self._Z_sub.size else z[None, :]

        if is_improve:
            self._Z_best = np.vstack([self._Z_best, z[None, :]]) if self._Z_best.size else z[None, :]
            if self._Z_best.shape[0] > int(max(3, best_k)):
                self._Z_best = self._Z_best[-int(max(3, best_k)):, :]

        self.rebuild_bands()

    # ---------- maintenance ----------
    def purge_expired(self) -> None:
        expired = [k for k, texp in self._tabu_until.items() if texp <= self._t]
        for k in expired:
            self._tabu_until.pop(k, None)

    def clear(self) -> None:
        self._tabu_until.clear()
        self._Z_sub = np.zeros((0, 4), float)
        self._Z_best = np.zeros((0, 4), float)
        self._r_sub_base = 0.0
        self._r_best_base = 0.0
        self._r_sub_dyn = 0.0
        self._Z_SL = np.zeros((0, 4), float)
        self._z_ref = None
        self._r_ib_base = 0.0
        self._r_ib_dyn = 0.0
        self._ib_enabled = False
        self._last_skip_rate = 0.0
        self._last_key = None
        self._t = 0

    # ---------- diagnostics ----------
    def debug_snapshot(self) -> Dict[str, float | int]:
        return {
            "r_ib_base": float(self._r_ib_base),
            "r_ib_dyn": float(self._r_ib_dyn),
            "r_sub_base": float(self._r_sub_base),
            "r_sub_dyn": float(self._r_sub_dyn),
            "r_best_base": float(self._r_best_base),
            "S_size": int(self._Z_sub.shape[0]),
            "B_size": int(self._Z_best.shape[0]),
            "last_skip_rate": float(self._last_skip_rate),
        }
