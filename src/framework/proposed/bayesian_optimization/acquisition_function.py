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
from jedi.api.helpers import infer
from scipy.stats import qmc, norm
from typing import List, Tuple, Optional, Any, Dict
from sklearn.preprocessing import StandardScaler
from framework.proposed.metaheuristics import TabuSearch
from framework.proposed.workload_characterization.workload import WorkloadCharacterized
from framework.proposed.bayesian_optimization.objective_function import OptimizationObjective

class CandidateSpace:
    """
    Real-executions-first candidate generator (target-centric).
    """

    def __init__(
            self,
            bounds: List[Tuple[float, float, float]],
            workload_target: WorkloadCharacterized,
            real_exec: np.ndarray,
            tabu_search: TabuSearch,
            seed: int = 24,
            max_cores: int = 64,
            max_memory: int = 256
    ) -> None:
        self.bounds = bounds
        self.max_cores = max_cores
        self.max_memory = max_memory
        self.workload_target = workload_target
        self.tabu_search = tabu_search

        # --- dimensión / grid ---
        self.d = len(self.bounds)
        self.lows_g  = np.array([l for l, _, _ in self.bounds], dtype=float)
        self.highs_g = np.array([h for _, h, _ in self.bounds], dtype=float)
        self.steps   = np.array([s for _, _, s in self.bounds], dtype=float)

        levels = [int((h - l) // s + 1) for (l, h, s) in bounds]
        grid_size = int(np.prod(levels))
        self.n_candidates = int(np.clip(np.ceil(np.sqrt(grid_size)), 32, 256))

        # --- target como vector 1D (d,) ---
        cs = np.array(self.workload_target.environment.to_vector(), dtype=int).reshape(-1)
        if cs.size != self.d:
            raise ValueError(f"[Candidate Space] cs_target size={cs.size} != d={self.d}. Got: {cs}")
        self.cs_target = cs  # (d,)

        # --- ejecuciones reales como matriz (m, d) ---
        if real_exec is None or len(real_exec) == 0:
            self.real_exec = np.empty((0, self.d), dtype=int)
        else:
            RX = np.array(real_exec, dtype=int).reshape(-1, self.d)
            self.real_exec = RX

        self.rng = np.random.default_rng(seed)

        print(f"[Candidate Space] grid_size={grid_size}, n_candidates={self.n_candidates}, "
              f"|real_exec|={len(self.real_exec)}")

    # ---------- utilidades ----------

    def _validate_sample(self, x: np.ndarray) -> bool:
        drv_c, drv_m, exc_c, exc_n, exc_m, _, task_c = x
        total_cores  = drv_c + exc_n * exc_c
        total_memory = drv_m + exc_n * exc_m
        return (total_cores <= self.max_cores
                and total_memory <= self.max_memory
                and exc_c >= task_c)

    def _sample_seed(self) -> int:
        return int(self.rng.integers(0, 2**32 - 1))

    def _ensure_strict_bounds(self, lows: np.ndarray, highs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Garantiza lower < upper por dimensión. Si high<=low:
          - expande ±1 paso dentro de [lows_g, highs_g];
          - si sigue igual (rejilla degenerada), añade epsilon al upper.
        """
        lows  = np.asarray(lows, dtype=float).copy()
        highs = np.asarray(highs, dtype=float).copy()
        for j in range(self.d):
            if highs[j] <= lows[j]:
                # intenta expandir ±1 paso
                l_try = max(self.lows_g[j],  lows[j]  - self.steps[j])
                h_try = min(self.highs_g[j], highs[j] + self.steps[j])
                if h_try <= l_try:
                    # añade un epsilon pequeño relativo al paso
                    eps = max(self.steps[j], 1.0) * 1e-6
                    h_try = l_try + eps
                lows[j], highs[j] = l_try, h_try
        return lows, highs

    def _scale_and_snap(self, U: np.ndarray, lows=None, highs=None) -> np.ndarray:
        lows_b  = self.lows_g if lows  is None else np.asarray(lows, dtype=float)
        highs_b = self.highs_g if highs is None else np.asarray(highs, dtype=float)
        # seguridad extra
        lows_b, highs_b = self._ensure_strict_bounds(lows_b, highs_b)
        X = qmc.scale(U, lows_b, highs_b)
        X = np.rint(X / self.steps) * self.steps
        X = np.clip(X, self.lows_g, self.highs_g)
        return X.astype(int)

    def _dedup_filter_tabu_feasible(self, X: np.ndarray) -> np.ndarray:
        if X is None or len(X) == 0:
            return np.empty((0, self.d), dtype=int)
        X = np.unique(X, axis=0)
        if len(X):
            X = np.array([x for x in X if not self.tabu_search.is_tabu(x)], dtype=int)
        if len(X):
            X = np.array([x for x in X if self._validate_sample(x)], dtype=int)
        return X

    # ---------- núcleo real: selección y caja local alrededor del target ----------

    def _select_real_anchors_near_target(self) -> np.ndarray:
        if self.real_exec.size == 0:
            return self.real_exec
        rng = np.maximum(self.highs_g - self.lows_g, 1.0)
        diff = np.abs(self.real_exec - self.cs_target.reshape(1, -1)) / rng.reshape(1, -1)
        d = np.sum(diff, axis=1)
        tau = np.median(d)
        anchors = self.real_exec[d <= tau + 1e-12]
        return self._dedup_filter_tabu_feasible(anchors)

    def _local_box_from_anchors(self, anchors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """[p20, p80] ±1 paso; fallback: target ±1 paso."""
        if anchors is None or len(anchors) == 0:
            lows  = np.maximum(self.lows_g,  self.cs_target - self.steps)
            highs = np.minimum(self.highs_g, self.cs_target + self.steps)
            return self._ensure_strict_bounds(lows, highs)

        q20 = np.percentile(anchors, 20, axis=0, method="linear")
        q80 = np.percentile(anchors, 80, axis=0, method="linear")
        lows  = np.maximum(self.lows_g,  q20 - self.steps)
        highs = np.minimum(self.highs_g, q80 + self.steps)
        # seguridad extra
        return self._ensure_strict_bounds(lows, highs)

    def _ring_neighbors(self, anchors: np.ndarray, radius_steps: int = 1, max_per_anchor: int = 24) -> np.ndarray:
        if anchors is None or len(anchors) == 0:
            return np.empty((0, self.d), dtype=int)
        neighs = []
        for a in anchors:
            for _ in range(max_per_anchor):
                mask = self.rng.integers(0, 2, size=self.d) * 2 - 1   # {-1,+1}
                x = a + mask * radius_steps * self.steps
                x = np.clip(x, self.lows_g, self.highs_g)
                if not np.array_equal(x, a):
                    neighs.append(x)
        if not neighs:
            return np.empty((0, self.d), dtype=int)
        N = np.unique(np.array(neighs, dtype=int), axis=0)
        return self._dedup_filter_tabu_feasible(N)

    # ---------- sobol block ----------

    def _sobol_block(self, n_need: int, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
        if n_need <= 0:
            return np.empty((0, self.d), dtype=int)
        # asegurar bounds válidos antes de escalar
        lows, highs = self._ensure_strict_bounds(lows, highs)
        sampler = qmc.Sobol(d=self.d, scramble=True, seed=self._sample_seed())
        m = int(np.ceil(np.log2(max(1, n_need))))
        U = sampler.random_base2(m)
        X = self._scale_and_snap(U, lows=lows, highs=highs)
        X = self._dedup_filter_tabu_feasible(X)
        if len(X) >= n_need:
            return X[:n_need]
        # top-up
        need = n_need - len(X)
        Ue = sampler.random(need * 2)
        Xe = self._scale_and_snap(Ue, lows=lows, highs=highs)
        Xe = self._dedup_filter_tabu_feasible(Xe)
        return np.vstack([X, Xe])[:n_need] if len(Xe) else X

    # ---------- API principal ----------

    def sobol_generate_candidates(self) -> np.ndarray:
        n = int(self.n_candidates)

        anchors = self._select_real_anchors_near_target()
        lows_L, highs_L = self._local_box_from_anchors(anchors)

        X_local = self._sobol_block(n, lows_L, highs_L)
        ring    = self._ring_neighbors(anchors, radius_steps=1, max_per_anchor=16)
        target1 = self._dedup_filter_tabu_feasible(self.cs_target.reshape(1, -1))

        pool = np.vstack([X_local, ring, target1]) if (len(ring) or len(target1)) else X_local
        pool = self._dedup_filter_tabu_feasible(pool)

        if len(pool) < n:
            Xg = self._sobol_block(n - len(pool), self.lows_g, self.highs_g)
            pool = np.vstack([pool, Xg]) if len(Xg) else pool
            pool = self._dedup_filter_tabu_feasible(pool)

        out = pool[:n] if len(pool) >= n else pool
        print(f"[Candidate Space|RealFirst] anchors={len(anchors)}, ring={len(ring)}, "
              f"local={len(X_local)}, final={len(out)}")
        return out


class CandidateSpaceGlobal:
    """
    Generates candidate configurations for acquisition function evaluation in Bayesian Optimization.

    This class supports low-discrepancy sampling (Sobol and LHS), tailored for structured parameter spaces
    like Spark configurations. It includes:
      - Weighted sampling across dimensions to emphasize parameters with higher influence,
      - Snapping candidates to valid discrete configuration values,
      - Constraint-aware filtering (e.g., total cores, memory, scheduling),
      - Adaptive candidate set sizing based on the grid resolution of the search space.

    Scientific Motivation:
    - Sampling follows adaptive BO design: start broad, then refine.
    - Weighting biases sampling toward sensitive parameters (e.g., memory, cores).
    - Constraint filtering ensures sampled candidates are deployable in real-world systems.
    """

    # def __init__(
    #         self,
    #         bounds: List[Tuple[float, float, float]],
    #         seed: int = 24,
    #         max_cores: int = 64,
    #         max_memory: int = 256
    # ) -> None:
    #     self.bounds = bounds
    #     self.max_cores = max_cores
    #     self.max_memory = max_memory
    #     self.seed = seed
    #
    #     # Estimate candidate size as sqrt of full grid (kept minimal).
    #     levels = [int((h - l) // s + 1) for (l, h, s) in bounds]
    #     grid_size = int(np.prod(levels))
    #     self.n_candidates = int(np.ceil(np.sqrt(grid_size)))  # keep your rule
    #     # Option: uncomment to be more robust:
    #     self.n_candidates = int(np.ceil(np.sqrt(grid_size)) * 2)
    #
    #     # Dimension weights (1.0 = no bias). Can be >1; we normalize internally.
    #     # self.dimension_weights = np.array([1, 1.3, 1.5, 1.8, 1.5, 1.0, 0.5])
    #     self.dimension_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    #     # self.dimension_weights = np.array([1.0, 1.2, 2.5, 2.5, 1.2, 1.0, 0.5])   # enfatiza cores/instances
    #
    #     # RNG kept as in your API
    #     self.rng = np.random.default_rng(seed)
    #
    #     # print(f"[Candidate Space] grid_size={grid_size}")

    def __init__(
            self,
            bounds: List[Tuple[float, float, float]],
            seed: int = 24,
            max_cores: int = 64,
            max_memory: int = 256,
            n: int = 2,   # multiplicative factor on sqrt(grid_size)
    ) -> None:
        """
        Candidate space generator.

        - Number of candidates is estimated as sqrt(grid_size) * n.
        - Rounded up to the next power of 2 to favor Sobol coverage.
        - Default n=2 reflects empirical observation that ~40% of Sobol
          points are filtered out by the trust-region; doubling ensures
          ~100+ usable candidates remain.
        """
        import numpy as np

        self.bounds = bounds
        self.max_cores = max_cores
        self.max_memory = max_memory
        self.seed = seed

        # Estimate candidate size from discrete grid
        levels = [int((h - l) // s + 1) for (l, h, s) in bounds]
        grid_size = int(np.prod(levels))
        base = int(np.ceil(np.sqrt(grid_size)) * n)

        # Round up to nearest power of 2 (Sobol-friendly)
        self.n_candidates = 1 << (int(np.ceil(np.log2(max(1, base)))))

        # Neutral weights by default
        self.dimension_weights = np.ones(len(bounds), dtype=float)

        # RNG for reproducibility
        self.rng = np.random.default_rng(seed)

        print(f"[Candidate Space] grid_size={grid_size}, "
              f"n_candidates={self.n_candidates} (n={n})")

    def _validate_sample(self, sample: np.ndarray) -> bool:
        """
        Validates a sample against system resource constraints.

        Constraints:
          - Total cores = driver.cores + (executor.instances * executor.cores)
          - Total memory = driver.memory + (executor.instances * executor.memory)
          - executor.cores ≥ task.cpus
        """
        driver_cores   = sample[0]
        driver_memory  = sample[1]
        exec_cores     = sample[2]
        exec_instances = sample[3]
        exec_memory    = sample[4]
        task_cpus      = sample[6]

        total_cores  = driver_cores + (exec_instances * exec_cores)
        total_memory = driver_memory + (exec_instances * exec_memory)

        return (
                total_cores <= self.max_cores
                and total_memory <= self.max_memory
                and exec_cores >= task_cpus
        )

    # def _sample_seed(self) -> int:
    #     """# NEW: derive a SciPy-compatible seed from self.rng (NumPy Generator)."""
    #     if hasattr(self.rng, "integers"):
    #         return int(self.rng.integers(0, 2**32 - 1))
    #     return int(self.rng)

    def _apply_dimension_warp(self, U: np.ndarray) -> np.ndarray:
        """# NEW: smooth weighting: U <- U*w + 0.5*(1-w), with w normalized to [0,1]."""
        w = np.asarray(self.dimension_weights, dtype=float)
        w = w / (np.max(w) + 1e-12)  # normalize -> max=1 (1: no bias, 0: pull to center 0.5)
        return U * w + 0.5 * (1.0 - w)

    def sobol_generate_candidates(self, seed: int | None = None) -> np.ndarray:
        d = len(self.bounds)
        n = int(self.n_candidates)

        # --- Sobol engine with proper seeding (scrambled) ---  # NEW
        s = self.seed if seed is None else int(seed)
        sampler = qmc.Sobol(d=d, scramble=True, seed=s)
        # seed = self._sample_seed()
        # sampler = qmc.Sobol(d=d, scramble=True, seed=seed)

        # Generate at least n points (power of 2), then extend if needed after filtering
        m = int(np.ceil(np.log2(max(1, n))))
        U = sampler.random_base2(m)  # (2**m, d) in [0,1)
        U = self._apply_dimension_warp(U)  # NEW

        X = self._scale_and_snap(U)
        # Deduplicate after snapping (important with discrete grids)  # NEW
        Xu = np.unique(X, axis=0)

        # Constraint filtering
        filtered = [x for x in Xu if self._validate_sample(x)]

        # If we didn't reach n, keep drawing from the same Sobol engine  # NEW
        while len(filtered) < n:
            need = n - len(filtered)
            extra = sampler.random(need * 2)           # draw a bit more to compensate duplicates
            extra = self._apply_dimension_warp(extra)   # NEW
            Xe = self._scale_and_snap(extra)
            Xe = np.unique(Xe, axis=0)
            for x in Xe:
                if self._validate_sample(x):
                    filtered.append(x)
                if len(filtered) >= n:
                    break
        candidates = np.array(filtered[:n])
        print(f"[Candidate Space]  Generated {len(candidates)} candidates via Sobol.")
        return candidates

    def lhs_generate_candidates(self) -> np.ndarray:
        d = len(self.bounds)
        n = int(self.n_candidates)

        # --- LHS with proper seeding (scrambled) ---  # NEW
        seed = self._sample_seed()
        sampler = qmc.LatinHypercube(d=d, scramble=True, seed=seed)
        U = sampler.random(n)
        U = self._apply_dimension_warp(U)  # NEW

        X = self._scale_and_snap(U)
        Xu = np.unique(X, axis=0)  # NEW

        filtered = [x for x in Xu if self._validate_sample(x)]

        # Top up if filtering removed too many  # NEW
        while len(filtered) < n:
            need = n - len(filtered)
            extra = sampler.random(need * 2)
            extra = self._apply_dimension_warp(extra)
            Xe = self._scale_and_snap(extra)
            Xe = np.unique(Xe, axis=0)
            for x in Xe:
                if self._validate_sample(x):
                    filtered.append(x)
                if len(filtered) >= n:
                    break

        candidates = np.array(filtered[:n])
        print(f"[Candidate Space]  Generated {len(candidates)} candidates via LHS.")
        return candidates

    def _scale_and_snap(self, unit_samples: np.ndarray) -> np.ndarray:
        """
        Converts unit cube samples [0,1]^d to discrete Spark configurations.
        Applies bounds, step rounding, and clipping for feasibility.
        """
        lows  = np.array([l for l, _, _ in self.bounds], dtype=float)
        highs = np.array([h for _, h, _ in self.bounds], dtype=float)
        steps = np.array([s for _, _, s in self.bounds], dtype=float)

        scaled = qmc.scale(unit_samples, lows, highs)
        snapped = np.rint(scaled / steps) * steps
        clipped = np.clip(snapped, lows, highs)
        return clipped.astype(int)


class AcquisitionFunction:
    """
    Minimal β-aware acquisition.

    Objective used by the acquisition (same as decision objective):
        Jβ(T,R) = expm1( β·log(1+T) + (1-β)·log(1+R) )

    Acquisition to minimize:
        AF(cs) = [ μ_Jβ(cs) + κ·σ_Jβ(cs) ] · ( 1 + ||z(cs)-z_ref|| / r_cap )^(1-β)

    where:
      - μ_Jβ from μ_T via Jβ(·), R is deterministic from x.
      - σ_Jβ from first-order propagation:  σ_Jβ ≈ |∂Jβ/∂T| · σ_T,
            with ∂Jβ/∂T = (Jβ+1)·β/(1+μ_T).
      - z(·) is φ standardized with a scaler learned on φ(S_L).
      - r_cap is a robust radius from S_L in z-space.

    Assumptions:
      - performance_model.predict(X) -> μ_T  (time).
      - uncertainty_model.predict(cs_new) -> σ_T  (time).
      - `phi` (n_SL,4) and `phi_ref` (4,) are ALWAYS provided and share scale.
    """

    # ---- deterministic resource usage from 7D config ----
    @staticmethod
    def _resource_from_x(x7: np.ndarray) -> int:
        dc, dm, ec, ei, em, *_ = map(int, np.asarray(x7).ravel())
        return int(dc * dm + ei * ec * em)

    # ---- simple geometry helpers ----
    @staticmethod
    def _second_nn(drow: np.ndarray) -> float:
        return float(np.partition(drow, 2)[2]) if drow.size >= 3 else float(np.partition(drow, 1)[1])

    @staticmethod
    def _min_positive(drow: np.ndarray) -> float:
        pos = drow[drow > 0]
        return float(pos.min()) if pos.size else float("inf")

    # ---- φ from 7D (for candidates) ----
    @staticmethod
    def _phi_from_cs(x7: np.ndarray) -> np.ndarray:
        # φ = (EP, MPC, PPC, DEB); x7 = [dc, dm, ec, ei, em, sp, tc]
        dc, dm, ec, ei, em, sp, _ = map(float, x7)
        EP  = max(ec, 1.0)
        MPC = max(em / EP, 1e-12)
        PPC = max(sp / max(ei * EP, 1.0), 1e-12)
        DEB = max((dc * dm) / max(ei * EP * em, 1.0), 1e-12)
        return np.array([EP, MPC, PPC, DEB], dtype=float)

    # ---- init ----
    def __init__(self,
                 bounds,
                 performance_model,
                 uncertainty_model,
                 nucleus: np.ndarray,
                 corona: Optional[np.ndarray] = None,     # kept for API compatibility
                 perf_model_scaler: Any = None,
                 kappa: float = 1.0,
                 phi: np.ndarray = None,                  # (n_SL,4) φ for S_L (same order as nucleus)
                 phi_ref: np.ndarray = None,              # (4,) reference φ (same scale)
                 *,
                 beta: float = 1.0
    ) -> None:

        self.performance_model = performance_model
        self.uncertainty_model = uncertainty_model
        self.perf_model_scaler = perf_model_scaler
        self.kappa = float(kappa)
        self.beta  = float(beta)

        self.nucleus = np.asarray(nucleus, dtype=float) if nucleus is not None else np.zeros((0, 7))
        self.cands   = CandidateSpaceGlobal(bounds, seed=24, n=2).sobol_generate_candidates()
        print(f"[Acq] Generated {len(self.cands)} candidates via Sobol (STL-aware).")


        # --- φ trust-region (assume phi & phi_ref are always provided) ---
        Phi_SL = np.asarray(phi, dtype=float)
        self._phi_scaler = StandardScaler().fit(Phi_SL)
        self._Z_SL = self._phi_scaler.transform(Phi_SL)

        self._z_ref = self._phi_scaler.transform(np.asarray(phi_ref, dtype=float).reshape(1, -1)).ravel()

        d2_list, d1p_list = [], []
        for i in range(self._Z_SL.shape[0]):
            di = np.linalg.norm(self._Z_SL - self._Z_SL[i], axis=1)
            d2_list.append(self._second_nn(di))
            d1p_list.append(self._min_positive(di))
        d2   = np.asarray(d2_list, dtype=float)
        d1p  = np.asarray(d1p_list, dtype=float)
        r75  = float(np.quantile(d2, 0.75))
        rmin = float(np.nanmedian(d1p[np.isfinite(d1p)]))
        self._r_cap = max(r75, rmin, 1e-6)

        print(f"[Acq|TR] enabled (φ-space): n_SL={len(Phi_SL)}, "
              f"r75_2NN={r75:.4g}, rmin_1NN+={rmin:.4g}, r_cap={self._r_cap:.4g}, z_ref=on")

    # ---- suggest ----
    def suggest(
            self,
            tabu_search: Optional["TabuSearch"] = None,
            diagnostics: bool = False,
            trust_region: bool = False,   # legacy hint; si hay TS, IB manda
    ):
        """
        β-LCB on Jβ con φ-aware Tabu (IB+EB) + tenure.
        Si hay TS, la IB (no el flag trust_region) es el gating real desde la iteración 1.
        """
        skipped_ib, skipped_eb, skipped_tabu_conf, negative_mu = 0, 0, 0, 0
        entries: List[Tuple[float, float, float, np.ndarray, int]] = []
        traces:  List[Dict[str, float]] = []

        # φ→Z con el mismo scaler que STL/AF
        Phi_C = np.asarray([self._phi_from_cs(x) for x in self.cands])
        Z_C   = self._phi_scaler.transform(Phi_C)

        # r_cap por defecto (para β-tilt) si no hay TS
        def _fallback_r_cap_from_SL() -> float:
            if self._Z_SL.shape[0] >= 2:
                d2_list, d1p_list = [], []
                for i in range(self._Z_SL.shape[0]):
                    di = np.sum(np.abs(self._Z_SL - self._Z_SL[i]), axis=1)
                    d2_list.append(self._second_nn_row(di))
                    d1p_list.append(self._min_positive(di))
                d2   = np.asarray(d2_list, dtype=float)
                d1p  = np.asarray(d1p_list, dtype=float)
                r75  = float(np.quantile(d2, 0.75))
                d1p_finite = d1p[np.isfinite(d1p)]
                rmin = float(np.nanmedian(d1p_finite)) if d1p_finite.size else 0.0
                return max(r75, rmin, 1e-6)
            return 1e-6

        # --- Gating de localidad: SI hay TS → IB manda (y EB después) ---
        if tabu_search is not None:
            tabu_search.set_inclusion_band(self._Z_SL, self._z_ref, seed_best=True)
            res = tabu_search.filter_candidates(
                X=self.cands, phi_from_cs=self._phi_from_cs, phi_scaler=self._phi_scaler
            )
            # aceptar (mask, skipped_IB, skipped_EB, n_pass_IB) o fallback más corto
            if isinstance(res, tuple) and len(res) == 4:
                mask_keep, skipped_ib, skipped_eb, n_pass_ib = res
            elif isinstance(res, tuple) and len(res) == 3:
                mask_keep, skipped_ib, skipped_eb = res
                n_pass_ib = int(mask_keep.sum()) + skipped_eb  # aproximación
            else:
                mask_keep = res  # type: ignore[assignment]
                skipped_ib = skipped_eb = 0
                n_pass_ib = int(mask_keep.sum())

            r_cap = getattr(tabu_search, "_r_ib_dyn", None) or _fallback_r_cap_from_SL()
            r_cap_src = "IB"

            # --- Seguro final: si aun así no queda nadie, forzar k más cercanos a z_ref ---
            if not np.any(mask_keep):
                k_force = max(8, int(0.05 * len(self.cands)))
                # Distancia L1 en Z al centro z_ref
                dIB_all = np.sum(np.abs(Z_C - self._z_ref[None, :]), axis=1)
                idx = np.argpartition(dIB_all, k_force - 1)[:k_force]
                mask_keep = np.zeros(len(self.cands), dtype=bool)
                mask_keep[idx] = True
                # Ajusta r_cap local para el β-tilt (no toca estados internos del TS)
                r_cap = max(r_cap, float(np.max(dIB_all[idx])))
                skipped_ib = int(len(self.cands) - k_force)
                skipped_eb = 0
                n_pass_ib = k_force
        else:
            # Sin TS: TR “clásica” por SL como gating (opcional)
            if trust_region:
                if self._Z_SL.shape[0] >= 1:
                    D_l1 = np.sum(np.abs(Z_C[:, None, :] - self._Z_SL[None, :, :]), axis=2)
                    Dmin = D_l1.min(axis=1) if D_l1.size else np.array([], dtype=float)
                    r_cap = _fallback_r_cap_from_SL()
                    mask_keep = (Dmin <= r_cap) if Dmin.size else np.zeros(len(self.cands), dtype=bool)
                else:
                    r_cap = _fallback_r_cap_from_SL()
                    mask_keep = np.zeros(len(self.cands), dtype=bool)
            else:
                r_cap = _fallback_r_cap_from_SL()
                mask_keep = np.ones(len(self.cands), dtype=bool)
            r_cap_src = "SL"
            n_pass_ib = int(mask_keep.sum())

            # Seguro final también en rama legacy si quedara vacío
            if not np.any(mask_keep):
                k_force = max(8, int(0.05 * len(self.cands)))
                dIB_all = np.sum(np.abs(Z_C - self._z_ref[None, :]), axis=1) if self._z_ref is not None else np.zeros(len(self.cands))
                if self._z_ref is not None:
                    idx = np.argpartition(dIB_all, k_force - 1)[:k_force]
                else:
                    idx = np.arange(k_force)
                mask_keep = np.zeros(len(self.cands), dtype=bool)
                mask_keep[idx] = True
                r_cap = max(r_cap, float(np.max(dIB_all[idx]))) if self._z_ref is not None else r_cap
                skipped_ib = int(len(self.cands) - k_force)
                skipped_eb = 0
                n_pass_ib = k_force

        # β-tilt distance a z_ref
        d_ref = (np.sum(np.abs(Z_C - self._z_ref[None, :]), axis=1)
                 if (self.beta < 1.0) else None)

        # Loop de candidatos
        for i, x in enumerate(self.cands):
            if not mask_keep[i]:
                if diagnostics:
                    traces.append({
                        "idx": i,
                        "score": float("inf"), "mu": float("nan"), "sigma": float("nan"),
                        "tabu": False,
                        "skipped_IB": True,
                        "skipped_EB": False,
                        "r_cap": float(r_cap), "r_cap_src": r_cap_src,
                    })
                continue

            # Tabú por configuración (tenure); aspiración se comprueba aguas arriba tras evaluar
            if (tabu_search is not None) and hasattr(tabu_search, "is_tabu") and tabu_search.is_tabu(x):
                if diagnostics:
                    traces.append({
                        "idx": i,
                        "score": float("inf"), "mu": float("nan"), "sigma": float("nan"),
                        "tabu": True, "skipped_IB": False, "skipped_EB": False,
                        "r_cap": float(r_cap), "r_cap_src": r_cap_src,
                    })
                skipped_tabu_conf += 1
                continue

            # μ_T, σ_T seguros
            mu_T = self._predict_mu(x)
            t_eff = float(mu_T)
            if (not np.isfinite(t_eff)) or (t_eff < 0.0):
                negative_mu += 1
                t_eff = 0.0
            sigma_T = self._predict_sigma(x)

            # Jβ y σ_J
            R = self._resource_from_x(x)
            j_mix = self.beta * np.log1p(t_eff) + (1.0 - self.beta) * np.log1p(float(R))
            j_mix = float(np.clip(j_mix, -50.0, 50.0))
            mu_J  = float(np.expm1(j_mix))
            dJ_dT = (mu_J + 1.0) * (self.beta / (1.0 + t_eff))
            sigma_J = abs(dJ_dT) * float(sigma_T)

            score = mu_J + self.kappa * sigma_J  # LCB on Jβ

            # β-tilt sólo si β<1, usando r_cap de IB/SL
            if d_ref is not None:
                tilt = (1.0 + (d_ref[i] / r_cap)) ** (1.0 - self.beta)
                score *= tilt

            if diagnostics:
                traces.append({
                    "idx": i, "score": float(score),
                    "mu_T": float(mu_T), "sigma_T": float(sigma_T),
                    "mu_J": float(mu_J), "sigma_J": float(sigma_J),
                    "t_eff": float(t_eff), "R": int(R), "beta": float(self.beta),
                    "tabu": False, "skipped_IB": False, "skipped_EB": False,
                    "r_cap": float(r_cap), "r_cap_src": r_cap_src,
                })

            entries.append((score, float(mu_T), float(sigma_T), x, i))

        # Auto-ajuste: EB con denominador = n_pass_ib; IB con total=candidatos
        if (tabu_search is not None) and hasattr(tabu_search, "adapt_from_skip_rate"):
            tabu_search.adapt_from_skip_rate(skipped=skipped_eb, total=max(1, n_pass_ib), which="EB")
            tabu_search.adapt_from_skip_rate(skipped=skipped_ib, total=max(1, len(self.cands)), which="IB")

        # Resumen
        print(f"\n[Acq|Suggest] skipped of {len(self.cands)} candidates:"
              f"\n\tskipped_IB={skipped_ib}"
              f"\n\tskipped_EB={skipped_eb}"
              f"\n\tskipped_tabu_conf={skipped_tabu_conf}"
              f"\n\tnegative_mu_T={negative_mu}"
              f"\n\tr_cap_source={r_cap_src}\n")

        if tabu_search is not None:
            try:
                print(f"[TS] bands: {tabu_search.debug_snapshot()}")
            except Exception:
                pass

        if not entries:
            raise ValueError("[Acq] All candidates filtered/tabu; nothing to suggest.")

        # Selección: AF asc, luego menor σ_T, luego índice
        entries.sort(key=lambda t: (t[0], t[2], t[4]))
        score_best, muT_best, sigT_best, x_best, _ = entries[0]
        result = (np.asarray(x_best), float(score_best), float(muT_best), float(sigT_best))
        return (result, traces) if diagnostics else result


    def suggest_last_OK_with_TS_last_OK(
            self,
            tabu_search: Optional["TabuSearch"] = None,
            diagnostics: bool = False,
            trust_region: bool = True,   # legacy hint; si hay TS, IB manda
    ):
        """
        β-LCB on Jβ con φ-aware Tabu (IB+EB) + tenure.
        Si hay TS, la IB (no el flag trust_region) es el gating real desde la iteración 1.
        """
        skipped_ib, skipped_eb, skipped_tabu_conf, negative_mu = 0, 0, 0, 0
        entries: List[Tuple[float, float, float, np.ndarray, int]] = []
        traces:  List[Dict[str, float]] = []

        # φ→Z con el mismo scaler que STL/AF
        Phi_C = np.asarray([self._phi_from_cs(x) for x in self.cands])
        Z_C   = self._phi_scaler.transform(Phi_C)

        # r_cap por defecto (para β-tilt) si no hay TS
        def _fallback_r_cap_from_SL() -> float:
            if self._Z_SL.shape[0] >= 2:
                d2_list, d1p_list = [], []
                for i in range(self._Z_SL.shape[0]):
                    di = np.sum(np.abs(self._Z_SL - self._Z_SL[i]), axis=1)
                    d2_list.append(self._second_nn_row(di))
                    d1p_list.append(self._min_positive(di))
                d2   = np.asarray(d2_list, dtype=float)
                d1p  = np.asarray(d1p_list, dtype=float)
                r75  = float(np.quantile(d2, 0.75))
                d1p_finite = d1p[np.isfinite(d1p)]
                rmin = float(np.nanmedian(d1p_finite)) if d1p_finite.size else 0.0
                return max(r75, rmin, 1e-6)
            return 1e-6

        # --- Gating de localidad: SI hay TS → IB manda (y EB después) ---
        if tabu_search is not None:
            tabu_search.set_inclusion_band(self._Z_SL, self._z_ref, seed_best=True)
            res = tabu_search.filter_candidates(
                X=self.cands, phi_from_cs=self._phi_from_cs, phi_scaler=self._phi_scaler
            )
            # aceptar (mask, skipped_IB, skipped_EB, n_pass_IB) o fallback más corto
            if isinstance(res, tuple) and len(res) == 4:
                mask_keep, skipped_ib, skipped_eb, n_pass_ib = res
            elif isinstance(res, tuple) and len(res) == 3:
                mask_keep, skipped_ib, skipped_eb = res
                n_pass_ib = int(mask_keep.sum()) + skipped_eb  # aproximación
            else:
                mask_keep = res  # type: ignore[assignment]
                skipped_ib = skipped_eb = 0
                n_pass_ib = int(mask_keep.sum())

            r_cap = getattr(tabu_search, "_r_ib_dyn", None) or _fallback_r_cap_from_SL()
            r_cap_src = "IB"
        else:
            # Sin TS: TR “clásica” por SL como gating (opcional)
            if trust_region:
                if self._Z_SL.shape[0] >= 1:
                    D_l1 = np.sum(np.abs(Z_C[:, None, :] - self._Z_SL[None, :, :]), axis=2)
                    Dmin = D_l1.min(axis=1) if D_l1.size else np.array([], dtype=float)
                    r_cap = _fallback_r_cap_from_SL()
                    mask_keep = (Dmin <= r_cap) if Dmin.size else np.zeros(len(self.cands), dtype=bool)
                else:
                    r_cap = _fallback_r_cap_from_SL()
                    mask_keep = np.zeros(len(self.cands), dtype=bool)
            else:
                r_cap = _fallback_r_cap_from_SL()
                mask_keep = np.ones(len(self.cands), dtype=bool)
            r_cap_src = "SL"
            n_pass_ib = int(mask_keep.sum())

        # β-tilt distance a z_ref
        d_ref = (np.sum(np.abs(Z_C - self._z_ref[None, :]), axis=1)
                 if (self.beta < 1.0) else None)

        # Loop de candidatos
        for i, x in enumerate(self.cands):
            if not mask_keep[i]:
                if diagnostics:
                    traces.append({
                        "idx": i,
                        "score": float("inf"), "mu": float("nan"), "sigma": float("nan"),
                        "tabu": False,
                        "skipped_IB": True,
                        "skipped_EB": False,
                        "r_cap": float(r_cap), "r_cap_src": r_cap_src,
                    })
                continue

            # Tabú por configuración (tenure)
            if (tabu_search is not None) and hasattr(tabu_search, "is_tabu") and tabu_search.is_tabu(x):
                if diagnostics:
                    traces.append({
                        "idx": i,
                        "score": float("inf"), "mu": float("nan"), "sigma": float("nan"),
                        "tabu": True, "skipped_IB": False, "skipped_EB": False,
                        "r_cap": float(r_cap), "r_cap_src": r_cap_src,
                    })
                skipped_tabu_conf += 1
                continue

            # μ_T, σ_T seguros
            mu_T = self._predict_mu(x)
            t_eff = float(mu_T)
            if (not np.isfinite(t_eff)) or (t_eff < 0.0):
                negative_mu += 1
                t_eff = 0.0
            sigma_T = self._predict_sigma(x)

            # Jβ y σ_J
            R = self._resource_from_x(x)
            j_mix = self.beta * np.log1p(t_eff) + (1.0 - self.beta) * np.log1p(float(R))
            j_mix = float(np.clip(j_mix, -50.0, 50.0))
            mu_J  = float(np.expm1(j_mix))
            dJ_dT = (mu_J + 1.0) * (self.beta / (1.0 + t_eff))
            sigma_J = abs(dJ_dT) * float(sigma_T)

            score = mu_J + self.kappa * sigma_J  # LCB on Jβ

            # β-tilt sólo si β<1, usando r_cap de IB/SL
            if d_ref is not None:
                tilt = (1.0 + (d_ref[i] / r_cap)) ** (1.0 - self.beta)
                score *= tilt

            if diagnostics:
                traces.append({
                    "idx": i, "score": float(score),
                    "mu_T": float(mu_T), "sigma_T": float(sigma_T),
                    "mu_J": float(mu_J), "sigma_J": float(sigma_J),
                    "t_eff": float(t_eff), "R": int(R), "beta": float(self.beta),
                    "tabu": False, "skipped_IB": False, "skipped_EB": False,
                    "r_cap": float(r_cap), "r_cap_src": r_cap_src,
                })

            entries.append((score, float(mu_T), float(sigma_T), x, i))

        # Auto-ajuste: EB con denominador = n_pass_ib; IB con total=candidatos
        if (tabu_search is not None) and hasattr(tabu_search, "adapt_from_skip_rate"):
            tabu_search.adapt_from_skip_rate(skipped=skipped_eb, total=max(1, n_pass_ib), which="EB")
            tabu_search.adapt_from_skip_rate(skipped=skipped_ib, total=max(1, len(self.cands)), which="IB")

        # Resumen
        print(f"\n[Acq|Suggest] skipped of {len(self.cands)} candidates:"
              f"\n\tskipped_IB={skipped_ib}"
              f"\n\tskipped_EB={skipped_eb}"
              f"\n\tskipped_tabu_conf={skipped_tabu_conf}"
              f"\n\tnegative_mu_T={negative_mu}"
              f"\n\tr_cap_source={r_cap_src}\n")

        if tabu_search is not None:
            try:
                print(f"[TS] bands: {tabu_search.debug_snapshot()}")
            except Exception:
                pass

        if not entries:
            raise ValueError("[Acq] All candidates filtered/tabu; nothing to suggest.")

        # Selección: AF asc, luego menor σ_T, luego índice
        entries.sort(key=lambda t: (t[0], t[2], t[4]))
        score_best, muT_best, sigT_best, x_best, _ = entries[0]
        result = (np.asarray(x_best), float(score_best), float(muT_best), float(sigT_best))
        return (result, traces) if diagnostics else result

    # ---- model adapters ----
    def _predict_mu(self, x_new: np.ndarray) -> float:
        X = self._maybe_scale_for_perf(np.asarray(x_new, dtype=float).reshape(1, -1))
        return float(np.asarray(self.performance_model.predict(X)).ravel()[0])

    def _predict_sigma(self, x_new: np.ndarray) -> float:
        return float(self.uncertainty_model.predict(cs_new=np.asarray(x_new, dtype=float).ravel()))

    def _maybe_scale_for_perf(self, X: np.ndarray) -> np.ndarray:
        return X if self.perf_model_scaler is None else self.perf_model_scaler.transform(X)
