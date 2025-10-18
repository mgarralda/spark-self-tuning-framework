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

from random import Random
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Tuple, List, Optional, Dict, Literal
from framework.proposed.parameters import SparkParameters
from framework.proposed.workload_characterization.workload import WorkloadCharacterized
from sklearn.linear_model import LinearRegression
from framework.experimental.metrics.optimization_model.evaluation import (
    SafeTransferLearningEvaluationMetrics,
    SearchSpace, WorkloadInvolved
)


class SafeTransferLearningStage:
    """
    STL–QGuard (sin residuos):
      Descriptor: z = [ w_25D , s=log1p(size/unit) , phi_4D(optional) ].
      1) Estandarización robusta por bloque (med/MAD) y distancias L1-avg por bloque.
      2) Pool local = intersección de caps por cuantiles de cada bloque (q* mínimo con |pool|>=k_min).
         Si no alcanza, se relaja el bloque menos informativo (menor |corr(d_B, |J-j*|)|).
      3) Guards de escala/tiempo (airbag) sobre ese pool (|Δs| y |J-j*| con MADs robustos).
      4) N/Ɔ por bandas robustas SIN residuo:
         - Núcleo: d_pat<=Q75 & d_s<=Q75
         - Corona: hasta Q95 en cualquiera, excluyendo Núcleo
      5) Pesos Tukey por bloques (patrón, escala, opcional phi) sin residuo.

    API: build_zones(...) -> (idx_T, idx_C)

    Atributos tras build_zones:
      - last_idx_ids (candidatos tras guards), last_idx_order (orden in-IDS por patrón)
      - last_T, last_C, last_thresholds, last_debug
      - last_weights = {"ids": (idx_ids, w_ids), "N": (idx_T, w_T), "C": (idx_C, w_C)}
    """

    def __init__(self, *, size_unit: float = 1024.0**3, k_min: int = 10):
        self.size_unit = float(size_unit)
        self.k_min = int(k_min)

        # trazas
        self.last_idx_ids: Optional[np.ndarray] = None
        self.last_idx_order: Optional[np.ndarray] = None
        self.last_d_pat: Optional[np.ndarray] = None
        self.last_d_r: Optional[np.ndarray] = None     # no se usa (sin residuo)
        self.last_r_ids: Optional[np.ndarray] = None   # no se usa (sin residuo)
        self.last_thresholds: Dict[str, float] = {}
        self.last_T: Optional[np.ndarray] = None
        self.last_C: Optional[np.ndarray] = None
        self.last_debug: Dict[str, object] = {}
        self.last_pool_idx: Optional[np.ndarray] = None
        self.last_ids_idx: Optional[np.ndarray] = None
        self.last_weights: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self.N_sizes = []
        self.C_sizes = []

    # -------------------- utils robustos --------------------
    @staticmethod
    def _winsor(x: np.ndarray, q=(0.025, 0.975)) -> np.ndarray:
        lo, hi = np.quantile(x, q)
        return np.clip(x, lo, hi)

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        """MAD respecto a la mediana (escalar robusto)."""
        return 1.4826 * float(np.median(np.abs(x - np.median(x))))

    @staticmethod
    def _mad0(x: np.ndarray) -> float:
        """MAD alrededor de 0 (útil en |J-j*|)."""
        return 1.4826 * float(np.median(np.abs(x)))

    @staticmethod
    def _stable_unique(idx: np.ndarray) -> np.ndarray:
        try:
            return np.unique(idx, kind="stable")
        except TypeError:
            _, first = np.unique(idx, return_index=True)
            return idx[np.sort(first)]

    @staticmethod
    def _block_minmax_normalize(X_block: np.ndarray, x_ref_block: np.ndarray):
        mn = X_block.min(axis=0)
        mx = X_block.max(axis=0)
        rng = np.clip(mx - mn, 1e-12, None)
        return (X_block - mn) / rng, (x_ref_block - mn) / rng, (mn, rng)

    # -------------------- pattern 25D --------------------
    @staticmethod
    def _reduce_pattern_25d(workload_descriptors: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Input: (n,100) = 25 métricas x [a,b,c,MAE].
        Output: X_pat (n,25) con score_i = mean(|a|,|b|,|c|) / MAE_pos,
        donde MAE_pos = MinMax(MAE) en (0,1], clip a >1e-6.
        """
        V = workload_descriptors.reshape(-1, 25, 4)
        coeff_mean = np.abs(V[:, :, :3]).mean(axis=2)   # (n,25)
        mae_mat = V[:, :, 3]                            # (n,25)
        mm = MinMaxScaler().fit(mae_mat)
        mae_pos = np.clip(mm.transform(mae_mat), 1e-6, None)
        X_pat = coeff_mean / mae_pos
        return X_pat, mm

    @staticmethod
    def _reduce_pattern_ref_25d(workload_ref: np.ndarray, mae_scaler: MinMaxScaler) -> np.ndarray:
        Vr = workload_ref.reshape(1, 25, 4)
        coeff_ref = np.abs(Vr[:, :, :3]).mean(axis=2)   # (1,25)
        mae_ref = Vr[:, :, 3]                           # (1,25)
        mae_ref_pos = np.clip(mae_scaler.transform(mae_ref), 1e-6, None)
        return coeff_ref / mae_ref_pos                  # (1,25)

    # -------------------- estandarización robusta y distancias por bloque --------------------
    @staticmethod
    def _robust_block_z(X: np.ndarray, x_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estandariza por bloque con mediana/MAD. Devuelve Z_pool, z_ref, (med, MAD).
        """
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        mad = np.clip(mad, 1e-12, None)
        Z = (X - med) / mad
        z = (x_ref - med) / mad
        return Z, z, (med, mad)

    @staticmethod
    def _avg_l1_block(Z: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
        """
        Distancia L1 media por fila respecto a z_ref en el bloque (evita ponderar por #dims).
        """
        return np.mean(np.abs(Z - z_ref), axis=1) if Z.ndim == 2 else np.abs(Z - z_ref).ravel()
        # return np.linalg.norm(Z - z_ref, axis=1) if Z.ndim == 2 else np.abs(Z - z_ref).ravel()

    # -------------------- intersección de cuantiles (pool local, sin k_budget) --------------------
    def _intersection_pool_by_quantiles(
            self,
            d_pat: np.ndarray,
            d_s: Optional[np.ndarray],
            d_phi: Optional[np.ndarray],
            J_pool: np.ndarray,
            j_ref: float,
            idx_pool: np.ndarray,
            *,
            q_grid: np.ndarray,
            k_min: int,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        Busca el menor q en q_grid tal que la intersección de caps por bloque alcanza k_min.
        Si no alcanza con q_max, relaja el bloque menos informativo (menor |corr(d_B, |J-j*|)|)
        y repite. Devuelve idx del pool seleccionado y trazas (q*, umbrales, bloques activos).
        """
        info: Dict[str, object] = {}
        blocks = {}
        if d_pat is not None:
            blocks["pat"] = d_pat
        if d_s is not None:
            blocks["s"] = d_s
        if d_phi is not None:
            blocks["phi"] = d_phi

        active = list(blocks.keys())
        d_t = np.abs(J_pool - j_ref)

        def try_sweep(act_keys):
            for q in q_grid:
                masks = []
                taus = {}
                for k in act_keys:
                    x = blocks[k]
                    tau = float(np.quantile(x, q))
                    taus[k] = tau
                    masks.append(x <= tau)
                mask_all = np.logical_and.reduce(masks) if masks else np.ones_like(idx_pool, bool)
                sel = idx_pool[mask_all]
                if sel.size >= k_min:
                    return sel, q, taus
            # último intento con q_max
            masks = []
            taus = {}
            for k in act_keys:
                x = blocks[k]
                tau = float(np.quantile(x, q_grid[-1]))
                taus[k] = tau
                masks.append(x <= tau)
            mask_all = np.logical_and.reduce(masks) if masks else np.ones_like(idx_pool, bool)
            sel = idx_pool[mask_all]
            return sel, q_grid[-1], taus

        sel, q_star, taus = try_sweep(active)
        if sel.size >= k_min or len(active) <= 1:
            info.update({"q_star": float(q_star), "taus": taus, "active_blocks": active, "relaxed": None})
            return sel, info

        # Relajar el bloque menos informativo: menor |corr(d_B, d_t)|
        corrs = {}
        for k in active:
            x = blocks[k]
            if np.std(x) < 1e-12 or np.std(d_t) < 1e-12:
                corrs[k] = 0.0
            else:
                c = np.corrcoef(x, d_t)[0, 1]
                corrs[k] = float(abs(c)) if np.isfinite(c) else 0.0
        drop = sorted(corrs.items(), key=lambda kv: kv[1])[0][0]
        active2 = [k for k in active if k != drop]

        sel2, q_star2, taus2 = try_sweep(active2)
        info.update({
            "q_star": float(q_star2), "taus": taus2, "active_blocks": active2,
            "relaxed": {"dropped": drop, "corrs_abs": corrs, "q_first": float(q_star), "taus_first": taus}
        })
        return sel2, info

    # -------------------- guards: escala + tiempo (airbag) --------------------
    def _apply_scale_time_guards(self,
                                 idx_pool: np.ndarray,
                                 S_pool: np.ndarray,
                                 J_pool: np.ndarray,
                                 s_ref: float,
                                 j_ref: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Guards robustos sobre el pool localizado:
          1) Scale guard: Δs <= median(Δs) + MAD(Δs).
          2) Time guard : |J - j_ref| <= MAD0 (o 2*MAD0).
        """
        # Δs global en este pool
        s_pool = np.log1p(S_pool / self.size_unit)
        delta_s_all = np.abs(s_pool - s_ref)
        med_s = float(np.median(delta_s_all))
        MAD_s = self._mad(delta_s_all)
        tau_s = med_s + MAD_s

        # mantener por escala
        mask_scale = (delta_s_all <= tau_s)
        base_idx = idx_pool[mask_scale]
        if base_idx.size == 0:
            # fallback: el más cercano en escala
            i_star = int(np.argmin(delta_s_all))
            base_idx = np.array([idx_pool[i_star]], dtype=int)

        # mapear base_idx -> posiciones en idx_pool (para alinear J_pool)
        posmap = {int(idx_pool[i]): i for i in range(idx_pool.size)}
        base_pos = np.array([posmap[int(i)] for i in base_idx], dtype=int)

        # time guard en el orden de base_idx
        d_t = np.abs(J_pool[base_pos] - j_ref)
        MAD_t = self._mad0(d_t)
        keep_mask = (d_t <= MAD_t)
        kept_idx = base_idx[keep_mask]
        tag = "MAD0"
        if kept_idx.size < max(self.k_min, 1):
            keep2_mask = (d_t <= 2.0 * MAD_t)
            kept2 = base_idx[keep2_mask]
            if kept2.size >= self.k_min:
                kept_idx = kept2; tag = "2*MAD0"
            else:
                kept_idx = base_idx; tag = "relaxed_base"

        info = {
            "med_ds": med_s, "MAD_s": MAD_s, "tau_s": tau_s,
            "MAD_t0": MAD_t, "time_guard": tag,
            "n_pool_in": int(idx_pool.size),
            "n_after_scale": int(base_idx.size),
            "n_after_time": int(kept_idx.size),
        }
        return kept_idx, info

    # -------------------- pesos robustos (Tukey bisquare, sin residuo) --------------------
    @staticmethod
    def _tukey_bisquare(u: np.ndarray, c: float) -> np.ndarray:
        x = u / (c + 1e-12)
        w = (1 - x**2)**2
        w[np.abs(x) >= 1.0] = 0.0
        return w

    @staticmethod
    def _safe_div(a: np.ndarray, b: float) -> np.ndarray:
        return a / (b + 1e-12)

    def debug_weights_summary(self, print_: bool = True) -> Dict[str, object]:
        """
        Resumen de correlaciones Spearman entre pesos y distancias, y conteos de no-cero.
        Requiere que `build_zones` se haya ejecutado.
        Devuelve:
          - n_ids, nz_ids, n_T, nz_T, n_C, nz_C
          - spearman(w_ids, |Δj| / |Δs| / d_pat / d_phi)
        """
        out: Dict[str, object] = {}
        if not getattr(self, "last_weights", None) or "ids" not in self.last_weights:
            if print_:
                print("[debug_weights_summary] No hay pesos almacenados (last_weights vacío).")
            return out

        idx_ids, w_ids = self.last_weights["ids"]
        w_T = self.last_weights.get("T", (np.array([], dtype=int), np.array([])))[1]
        w_C = self.last_weights.get("C", (np.array([], dtype=int), np.array([])))[1]

        n_ids = int(idx_ids.size)
        nz_ids = int(np.count_nonzero(w_ids)) if n_ids else 0
        n_T = int(w_T.size)
        nz_T = int(np.count_nonzero(w_T)) if n_T else 0
        n_C = int(w_C.size)
        nz_C = int(np.count_nonzero(w_C)) if n_C else 0

        dists = getattr(self, "last_dists_ids", None) or {}

        def _safe_spearman(a: np.ndarray, b: Optional[np.ndarray]) -> float:
            if b is None:
                return np.nan
            a = np.asarray(a, float); b = np.asarray(b, float)
            if a.size != b.size or a.size < 3:
                return np.nan
            try:
                from scipy.stats import spearmanr
                r = spearmanr(a, b).correlation
                return float(r) if np.isfinite(r) else np.nan
            except Exception:
                # Fallback sin scipy
                def _rank_avg(x: np.ndarray) -> np.ndarray:
                    order = np.argsort(x, kind="mergesort")
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.arange(1, x.size + 1, dtype=float)
                    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
                    if np.any(counts > 1):
                        for k, c in enumerate(counts):
                            if c > 1:
                                ranks[inv == k] = ranks[inv == k].mean()
                    return ranks
                ra = _rank_avg(a); rb = _rank_avg(b)
                va = np.var(ra); vb = np.var(rb)
                if va <= 0 or vb <= 0:
                    return np.nan
                r = np.corrcoef(ra, rb)[0, 1]
                return float(r) if np.isfinite(r) else np.nan

        rho_t   = _safe_spearman(w_ids, dists.get("d_t"))
        rho_s   = _safe_spearman(w_ids, dists.get("d_s"))
        rho_pat = _safe_spearman(w_ids, dists.get("d_pat"))
        rho_phi = _safe_spearman(w_ids, dists.get("d_phi"))

        out.update({
            "n_ids": n_ids, "nz_ids": nz_ids,
            "n_T": n_T, "nz_T": nz_T,
            "n_C": n_C, "nz_C": nz_C,
            "spearman(w, |Δj|)": rho_t,
            "spearman(w, |Δs|)": rho_s,
            "spearman(w, d_pat)": rho_pat,
            "spearman(w, d_phi)": rho_phi,
        })

        if print_:
            print(f"{out}")

        return out

    def build_zones_(self,
                    workload_descriptors: np.ndarray,   # (n,100)
                    objective_values: np.ndarray,       # (n,)
                    data_sizes: np.ndarray,             # (n,) bytes
                    workload_ref: np.ndarray,           # (100,)
                    objective_ref: float,               # scalar T (or T_R)
                    data_size_ref: float,               # bytes
                    *,
                    k_limit: Optional[int] = None,      # opcional: cap final si quieres
                    ids_mode: Literal["pat", "pat+size"] = "pat+size",
                    jackknife: bool = False,
                    X_phi_cs: Optional[np.ndarray] = None,  # (n,4) opcional
                    x_phi_ref: Optional[np.ndarray] = None, # (1,4) opcional
                    # rho_vec: Optional[np.ndarray] = None,   # ignorado aquí
                    # rho_ref: Optional[float] = None,        # ignorado
                    ) -> Tuple[np.ndarray, np.ndarray]:

        # Forzamos mínimo "pat+size" (modo "pat" degrada)
        if ids_mode != "pat+size":
            ids_mode = "pat+size"

        n = int(len(workload_descriptors))
        if n == 0:
            self.last_debug = {"n": 0}
            return np.array([], dtype=int), np.array([], dtype=int)

        # (1) patrón 25D
        X_pat, mm = self._reduce_pattern_25d(workload_descriptors)
        x_pat_ref = self._reduce_pattern_ref_25d(workload_ref, mm)

        # (2) coords y ref
        S = np.asarray(data_sizes, float).ravel()
        J = np.log1p(np.asarray(objective_values, float).ravel())
        j_ref = float(np.log1p(float(objective_ref)))
        s_all = np.log1p(S / self.size_unit)
        s_ref = float(np.log1p(float(data_size_ref) / self.size_unit))

        # (3) jackknife (opcional)
        if jackknife:
            desc_match = np.isclose(workload_descriptors, workload_ref, atol=1e-12).all(axis=1)
            size_match = np.isclose(S, float(data_size_ref), rtol=0.0, atol=0.0)
            cand = np.where(desc_match & size_match)[0]
            mask_pool = np.ones(n, dtype=bool)
            if cand.size > 0:
                j_cand = J[cand]
                i_star_local = int(np.argmin(np.abs(j_cand - j_ref)))
                mask_pool[cand[i_star_local]] = False
            excluded_idx = None
        else:
            mask_pool = np.ones(n, dtype=bool)
            excluded_idx = None

        idx_all = np.arange(n, dtype=int)
        idx_pool0 = idx_all[mask_pool]
        S_pool0, J_pool0, X_pat_pool0 = S[mask_pool], J[mask_pool], X_pat[mask_pool]
        n_pool0 = idx_pool0.size

        # (4) robust-z por bloque + distancias por bloque (pool inicial)
        Zp, zp, _ = self._robust_block_z(X_pat_pool0, x_pat_ref)         # (m,25)
        d_pat_pool = self._avg_l1_block(Zp, zp)                           # (m,)

        s_pool0 = s_all[mask_pool].reshape(-1, 1)
        Zs, zs, _ = self._robust_block_z(s_pool0, np.array([[s_ref]]))    # (m,1)
        d_s_pool = self._avg_l1_block(Zs, zs)                              # (m,)

        d_phi_pool = None
        if X_phi_cs is not None and x_phi_ref is not None:
            Zphi, zphi, _ = self._robust_block_z(X_phi_cs[mask_pool], x_phi_ref)
            d_phi_pool = self._avg_l1_block(Zphi, zphi)

        # (5) pool local: intersección de cuantiles
        q_grid = np.linspace(0.50, 0.99, 50)
        idx_pool_q, info_q = self._intersection_pool_by_quantiles(
            d_pat=d_pat_pool,
            d_s=d_s_pool,                   # ids_mode ya es pat+size
            d_phi=d_phi_pool,
            J_pool=J_pool0, j_ref=j_ref, idx_pool=idx_pool0,
            q_grid=q_grid, k_min=self.k_min
        )

        # Reducir arrays al pool localizado
        sel_mask = np.isin(idx_pool0, idx_pool_q)
        X_pat_pool = X_pat_pool0[sel_mask]
        S_pool = S_pool0[sel_mask]
        J_pool = J_pool0[sel_mask]
        d_pat_pool = d_pat_pool[sel_mask]
        d_s_pool = d_s_pool[sel_mask]
        if d_phi_pool is not None:
            d_phi_pool = d_phi_pool[sel_mask]

        # (6) guards escala/tiempo (airbag)
        idx_ids, info_guard = self._apply_scale_time_guards(
            idx_pool=idx_pool_q, S_pool=S_pool, J_pool=J_pool, s_ref=s_ref, j_ref=j_ref
        )
        self.last_pool_idx = idx_pool_q
        self.last_ids_idx = idx_ids

        # (7) cap final opcional
        if k_limit is not None and idx_ids.size > int(k_limit):
            posmap_pool = {int(i): k for k, i in enumerate(idx_pool_q.tolist())}
            ids_pos = np.array([posmap_pool[int(i)] for i in idx_ids], dtype=int)
            d_total_pool = d_pat_pool[ids_pos] + d_s_pool[ids_pos]
            if d_phi_pool is not None:
                d_total_pool = d_total_pool + d_phi_pool[ids_pos]
            order_local = np.argsort(d_total_pool, kind="stable")
            idx_ids = idx_ids[order_local[:int(k_limit)]]

        # (8) N/Ɔ SIN residuo
        Xp_ids = X_pat[idx_ids]
        Xp_s, x_ref_s, _ = self._block_minmax_normalize(Xp_ids, x_pat_ref)
        d_pat_ids = np.mean(np.abs(Xp_s - x_ref_s), axis=1)

        posmap_pool = {int(i): k for k, i in enumerate(idx_pool_q.tolist())}
        ids_pos = np.array([posmap_pool[int(i)] for i in idx_ids], dtype=int)
        d_s_ids_in = d_s_pool[ids_pos]

        order_pat = np.argsort(d_pat_ids, kind="stable")
        idx_order = idx_ids[order_pat]
        d_pat_ord = d_pat_ids[order_pat]
        d_s_ord   = d_s_ids_in[order_pat]

        q75_pat = float(np.quantile(d_pat_ord, 0.75)) if d_pat_ord.size else 1.0
        q95_pat = float(np.quantile(d_pat_ord, 0.95)) if d_pat_ord.size else 1.0
        q75_s   = float(np.quantile(d_s_ord,   0.75)) if d_s_ord.size   else 1.0
        q95_s   = float(np.quantile(d_s_ord,   0.95)) if d_s_ord.size   else 1.0

        mask_N = (d_pat_ord <= q75_pat) & (d_s_ord <= q75_s)
        idx_N = idx_order[mask_N]
        if idx_N.size < self.k_min:
            take = min(self.k_min, idx_order.size)
            idx_N = idx_order[:take]
            mask_N = np.isin(idx_order, idx_N)

        mask_C = ((d_pat_ord <= q95_pat) & (d_s_ord <= q95_s)) & (~mask_N)
        idx_C = idx_order[mask_C]
        if idx_C.size == 0 and idx_order.size > idx_N.size:
            rest = idx_order[~mask_N]
            idx_C = rest[: max(0, min(max(self.k_min, idx_order.size // 10), rest.size))]

        # -------- (9) PESOS anclados al target con esquema zona-aware --------
        s_ids = s_all[idx_ids]
        j_ids = J[idx_ids]
        delta_s_ids = np.abs(s_ids - s_ref)   # |Δs|
        delta_j_ids = np.abs(j_ids - j_ref)   # |Δj|

        if (X_phi_cs is not None) and (x_phi_ref is not None):
            Xphi_ids = X_phi_cs[idx_ids]
            Zphi_ids, zphi_ref, _ = self._robust_block_z(Xphi_ids, x_phi_ref)
            d_phi_ids = self._avg_l1_block(Zphi_ids, zphi_ref)
        else:
            d_phi_ids = None

        # Exponentes (prioriza tiempo y escala)
        # Reference       & (0.35, 0.30, 0.25, 0.10)  {alpha_t}, alpha_s, alpha_pat, alpha_phi
        # alpha_pat, alpha_s, alpha_t, alpha_phi = 0.25, 0.30, 0.35, (0.10 if d_phi_ids is not None else 0.0)
        alpha_pat, alpha_s, alpha_t, alpha_phi = 0.40, 0.20, 0.40, (0.10 if d_phi_ids is not None else 0.0) # OK v1


        ssum = alpha_pat + alpha_s + alpha_t + alpha_phi
        alpha_pat /= ssum; alpha_s /= ssum; alpha_t /= ssum
        if d_phi_ids is not None:
            alpha_phi /= ssum

        # Objetivo mínimo de no-cero en T
        nz_target_T = 0.40
        q_schedule = [0.75, 0.85, 0.90]

        pos_in_ids = {int(i): k for k, i in enumerate(idx_ids.tolist())}
        w_ids = None; w_T = None; w_C = None
        used_cutoffs = {}

        # Helper para construir pesos dados cuantiles por bloque
        def _build_weights(c_s, c_t, c_pat, c_phi):
            w_pat = self._tukey_bisquare(d_pat_ids,   c_pat)
            w_s   = self._tukey_bisquare(delta_s_ids, c_s)
            w_t   = self._tukey_bisquare(delta_j_ids, c_t)
            w_phi = self._tukey_bisquare(d_phi_ids,   c_phi) if d_phi_ids is not None else 1.0
            return (w_pat ** alpha_pat) * (w_s ** alpha_s) * (w_t ** alpha_t) * \
                (w_phi ** (alpha_phi if d_phi_ids is not None else 1.0))

        # 9.a) Elegimos cutoffs para IDS/T con agenda y objetivo de no-ceros
        c_s_used = c_t_used = c_pat_used = c_phi_used = None
        for q in q_schedule:
            c_s_T   = float(np.quantile(delta_s_ids, q)) if delta_s_ids.size else 1.0
            c_t_T   = float(np.quantile(delta_j_ids, q)) if delta_j_ids.size else 1.0
            c_pat_T = float(np.quantile(d_pat_ids,  min(q, 0.75))) if d_pat_ids.size else 1.0
            c_phi_T = float(np.quantile(d_phi_ids,  min(q, 0.75))) if (d_phi_ids is not None and d_phi_ids.size) else None

            w_ids_try = _build_weights(c_s_T, c_t_T, c_pat_T, c_phi_T)
            w_T_try = np.array([w_ids_try[pos_in_ids[int(i)]] for i in idx_N]) if idx_N.size else np.array([])
            nz_T = int(np.count_nonzero(w_T_try)) if w_T_try.size else 0
            min_nz = int(max(self.k_min, np.ceil(nz_target_T * max(1, idx_N.size))))
            if nz_T >= min_nz or q == q_schedule[-1]:
                w_ids = w_ids_try
                w_T = w_T_try
                used_cutoffs = {"q_ids_T": q, "q_pat_cap_T": min(q, 0.75)}
                c_s_used, c_t_used, c_pat_used, c_phi_used = c_s_T, c_t_T, c_pat_T, c_phi_T
                break

        # 9.b) Pesos específicos para C (más indulgentes) + suelo suave ε_C
        if idx_C.size:
            qC = max(0.90, used_cutoffs.get("q_ids_T", 0.75))
            c_s_C   = float(np.quantile(delta_s_ids, qC)) if delta_s_ids.size else 1.0
            c_t_C   = float(np.quantile(delta_j_ids, qC)) if delta_j_ids.size else 1.0
            c_pat_C = float(np.quantile(d_pat_ids,  max(0.85, min(qC, 0.90)))) if d_pat_ids.size else 1.0
            c_phi_C = float(np.quantile(d_phi_ids,  max(0.85, min(qC, 0.90)))) if (d_phi_ids is not None and d_phi_ids.size) else None

            w_ids_C = _build_weights(c_s_C, c_t_C, c_pat_C, c_phi_C)
            w_C = np.array([w_ids_C[pos_in_ids[int(i)]] for i in idx_C])
            eps_C = 0.00
            if w_C.size:
                w_C = (1.0 - eps_C) * w_C + eps_C
        else:
            w_C = np.array([])

        # === (9.c) Podado duro: no devolver vecinos con peso cero en N/C ===
        if idx_N.size:
            nzN = (w_T > 0)
            idx_N = idx_N[nzN]
            w_T   = w_T[nzN]

        if idx_C.size:
            nzC = (w_C > 0)
            idx_C = idx_C[nzC]
            w_C   = w_C[nzC]

        # (9.d) Guardamos pesos y distancias DIAGNÓSTICAS (post-podado)
        self.last_weights = {
            "ids": (idx_ids, w_ids),
            "N": (idx_N, w_T if w_T is not None else np.array([])),
            "C": (idx_C, w_C),
        }
        self.last_dists_ids = {
            "d_t":  delta_j_ids,
            "d_s":  delta_s_ids,
            "d_pat": d_pat_ids,
            "d_phi": (d_phi_ids if d_phi_ids is not None else None),
        }

        # (10) trazas y debug
        S_lin_ids = S[idx_ids] / self.size_unit
        S_ref_lin = float(data_size_ref) / self.size_unit

        # Spearman rápidos
        def _safe_spearman(a: np.ndarray, b: Optional[np.ndarray]) -> float:
            if b is None:
                return np.nan
            a = np.asarray(a, float); b = np.asarray(b, float)
            if a.size != b.size or a.size < 3:
                return np.nan
            try:
                from scipy.stats import spearmanr
                r = spearmanr(a, b).correlation
                return float(r) if np.isfinite(r) else np.nan
            except Exception:
                def _rank_avg(x: np.ndarray) -> np.ndarray:
                    order = np.argsort(x, kind="mergesort")
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.arange(1, x.size + 1, dtype=float)
                    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
                    if np.any(counts > 1):
                        for k, c in enumerate(counts):
                            if c > 1:
                                ranks[inv == k] = ranks[inv == k].mean()
                    return ranks
                ra = _rank_avg(a); rb = _rank_avg(b)
                va = np.var(ra); vb = np.var(rb)
                if va <= 0 or vb <= 0:
                    return np.nan
                r = np.corrcoef(ra, rb)[0, 1]
                return float(r) if np.isfinite(r) else np.nan

        rho_t = _safe_spearman(w_ids, self.last_dists_ids["d_t"])
        rho_s = _safe_spearman(w_ids, self.last_dists_ids["d_s"])
        rho_p = _safe_spearman(w_ids, self.last_dists_ids["d_pat"])
        rho_f = _safe_spearman(w_ids, self.last_dists_ids["d_phi"]) if (self.last_dists_ids["d_phi"] is not None) else np.nan

        self.last_idx_ids = idx_ids
        self.last_idx_order = idx_order
        self.last_d_pat = d_pat_ord
        self.last_d_r = None
        self.last_r_ids = None
        self.last_thresholds = {
            "q75_pat": q75_pat, "q95_pat": q95_pat,
            "q75_s": q75_s, "q95_s": q95_s,
            **info_guard,
            **({"q_star": info_q.get("q_star")} if "q_star" in info_q else {}),
            "cuts_T": {"q": used_cutoffs.get("q_ids_T"), "pat_cap": used_cutoffs.get("q_pat_cap_T")},
            "cuts_C": {
                "qC": (max(0.90, used_cutoffs.get("q_ids_T", 0.75))) if idx_C.size else None,
                "eps_C": (0.02 if idx_C.size else None),
            },
            "c_pat_used": float(c_pat_used) if 'c_pat_used' in locals() and c_pat_used is not None else np.nan,
            "c_s_used": float(c_s_used) if 'c_s_used' in locals() and c_s_used is not None else np.nan,
            "c_t_used": float(c_t_used) if 'c_t_used' in locals() and c_t_used is not None else np.nan,
            **({"c_phi_used": float(c_phi_used)} if (d_phi_ids is not None and c_phi_used is not None) else {}),
        }
        self.last_T = idx_N
        self.last_C = idx_C
        self.last_debug = {
            "n_total": int(n),
            "n_pool0": int(n_pool0),
            "n_pool_q": int(idx_pool_q.size),
            "n_ids": int(idx_ids.size),
            "|T|": int(idx_N.size),
            "|C|": int(idx_C.size),
            "ids_mode": ids_mode,
            "jackknife": bool(jackknife),
            "jackknife_excluded": None,
            "S_ref_lin": float(S_ref_lin),
            "q_info": info_q,
            "spearman": {
                "w_vs_|Δj|": rho_t,
                "w_vs_|Δs|": rho_s,
                "w_vs_d_pat": rho_p,
                "w_vs_d_phi": rho_f,
            },
            "nz_T": int(np.count_nonzero(self.last_weights["N"][1])) if idx_N.size else 0,
            "nz_C": int(np.count_nonzero(self.last_weights["C"][1])) if idx_C.size else 0,
            "len_T": int(idx_N.size),
            "len_C": int(idx_C.size),
        }

        self.N_sizes.append(idx_N.size)
        self.C_sizes.append(idx_C.size)
        return idx_N, idx_C


    def build_zones(self,
        workload_descriptors: np.ndarray,   # (n,100)
        objective_values: np.ndarray,       # (n,)
        data_sizes: np.ndarray,             # (n,) bytes
        workload_ref: np.ndarray,           # (100,)
        objective_ref: float,               # scalar T (or T_R)
        data_size_ref: float,               # bytes
        *,
        k_limit: Optional[int] = None,      # opcional: cap final si quieres
        ids_mode: Literal["pat", "pat+size"] = "pat+size",
        jackknife: bool = False,
        X_phi_cs: Optional[np.ndarray] = None,  # (n,4) opcional
        x_phi_ref: Optional[np.ndarray] = None, # (1,4) opcional
        # rho_vec: Optional[np.ndarray] = None,   # ignorado aquí
        # rho_ref: Optional[float] = None,        # ignorado
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Forzamos mínimo "pat+size" (modo "pat" degrada)
        if ids_mode != "pat+size":
            ids_mode = "pat+size"

        n = int(len(workload_descriptors))
        if n == 0:
            self.last_debug = {"n": 0}
            return np.array([], dtype=int), np.array([], dtype=int)

        # (1) patrón 25D
        X_pat, mm = self._reduce_pattern_25d(workload_descriptors)
        x_pat_ref = self._reduce_pattern_ref_25d(workload_ref, mm)

        # (2) coords y ref
        S = np.asarray(data_sizes, float).ravel()
        J = np.log1p(np.asarray(objective_values, float).ravel())
        j_ref = float(np.log1p(float(objective_ref)))
        s_all = np.log1p(S / self.size_unit)
        s_ref = float(np.log1p(float(data_size_ref) / self.size_unit))

        # (3) jackknife (opcional)
        if jackknife:
            desc_match = np.isclose(workload_descriptors, workload_ref, atol=1e-12).all(axis=1)
            size_match = np.isclose(S, float(data_size_ref), rtol=0.0, atol=0.0)
            cand = np.where(desc_match & size_match)[0]
            mask_pool = np.ones(n, dtype=bool)
            if cand.size > 0:
                j_cand = J[cand]
                i_star_local = int(np.argmin(np.abs(j_cand - j_ref)))
                mask_pool[cand[i_star_local]] = False
            excluded_idx = None
        else:
            mask_pool = np.ones(n, dtype=bool)
            excluded_idx = None

        idx_all = np.arange(n, dtype=int)
        idx_pool0 = idx_all[mask_pool]
        S_pool0, J_pool0, X_pat_pool0 = S[mask_pool], J[mask_pool], X_pat[mask_pool]
        n_pool0 = idx_pool0.size

        # (4) robust-z por bloque + distancias por bloque (pool inicial)
        Zp, zp, _ = self._robust_block_z(X_pat_pool0, x_pat_ref)         # (m,25)
        d_pat_pool = self._avg_l1_block(Zp, zp)                           # (m,)

        s_pool0 = s_all[mask_pool].reshape(-1, 1)
        Zs, zs, _ = self._robust_block_z(s_pool0, np.array([[s_ref]]))    # (m,1)
        d_s_pool = self._avg_l1_block(Zs, zs)                              # (m,)

        d_phi_pool = None
        if X_phi_cs is not None and x_phi_ref is not None:
            Zphi, zphi, _ = self._robust_block_z(X_phi_cs[mask_pool], x_phi_ref)
            d_phi_pool = self._avg_l1_block(Zphi, zphi)

        # (5) pool local: intersección de cuantiles
        q_grid = np.linspace(0.50, 0.99, 50)
        idx_pool_q, info_q = self._intersection_pool_by_quantiles(
            d_pat=d_pat_pool,
            d_s=d_s_pool,                   # ids_mode ya es pat+size
            d_phi=d_phi_pool,
            J_pool=J_pool0, j_ref=j_ref, idx_pool=idx_pool0,
            q_grid=q_grid, k_min=self.k_min
        )

        # Reducir arrays al pool localizado
        sel_mask = np.isin(idx_pool0, idx_pool_q)
        X_pat_pool = X_pat_pool0[sel_mask]
        S_pool = S_pool0[sel_mask]
        J_pool = J_pool0[sel_mask]
        d_pat_pool = d_pat_pool[sel_mask]
        d_s_pool = d_s_pool[sel_mask]
        if d_phi_pool is not None:
            d_phi_pool = d_phi_pool[sel_mask]

        # (6) guards escala/tiempo (airbag)
        idx_ids, info_guard = self._apply_scale_time_guards(
            idx_pool=idx_pool_q, S_pool=S_pool, J_pool=J_pool, s_ref=s_ref, j_ref=j_ref
        )
        self.last_pool_idx = idx_pool_q
        self.last_ids_idx = idx_ids

        # (7) cap final opcional
        if k_limit is not None and idx_ids.size > int(k_limit):
            posmap_pool = {int(i): k for k, i in enumerate(idx_pool_q.tolist())}
            ids_pos = np.array([posmap_pool[int(i)] for i in idx_ids], dtype=int)
            d_total_pool = d_pat_pool[ids_pos] + d_s_pool[ids_pos]
            if d_phi_pool is not None:
                d_total_pool = d_total_pool + d_phi_pool[ids_pos]
            order_local = np.argsort(d_total_pool, kind="stable")
            idx_ids = idx_ids[order_local[:int(k_limit)]]

        # (8) N/Ɔ SIN residuo
        Xp_ids = X_pat[idx_ids]
        Xp_s, x_ref_s, _ = self._block_minmax_normalize(Xp_ids, x_pat_ref)
        d_pat_ids = np.mean(np.abs(Xp_s - x_ref_s), axis=1)

        posmap_pool = {int(i): k for k, i in enumerate(idx_pool_q.tolist())}
        ids_pos = np.array([posmap_pool[int(i)] for i in idx_ids], dtype=int)
        d_s_ids_in = d_s_pool[ids_pos]

        order_pat = np.argsort(d_pat_ids, kind="stable")
        idx_order = idx_ids[order_pat]
        d_pat_ord = d_pat_ids[order_pat]
        d_s_ord   = d_s_ids_in[order_pat]

        q75_pat = float(np.quantile(d_pat_ord, 0.75)) if d_pat_ord.size else 1.0
        q95_pat = float(np.quantile(d_pat_ord, 0.95)) if d_pat_ord.size else 1.0
        q75_s   = float(np.quantile(d_s_ord,   0.75)) if d_s_ord.size   else 1.0
        q95_s   = float(np.quantile(d_s_ord,   0.95)) if d_s_ord.size   else 1.0

        mask_N = (d_pat_ord <= q75_pat) & (d_s_ord <= q75_s)
        idx_N = idx_order[mask_N]
        if idx_N.size < self.k_min:
            take = min(self.k_min, idx_order.size)
            idx_N = idx_order[:take]
            mask_N = np.isin(idx_order, idx_N)

        mask_C = ((d_pat_ord <= q95_pat) & (d_s_ord <= q95_s)) & (~mask_N)
        idx_C = idx_order[mask_C]
        if idx_C.size == 0 and idx_order.size > idx_N.size:
            rest = idx_order[~mask_N]
            idx_C = rest[: max(0, min(max(self.k_min, idx_order.size // 10), rest.size))]

        # -------- (9) PESOS anclados al target con esquema zona-aware --------
        s_ids = s_all[idx_ids]
        j_ids = J[idx_ids]
        delta_s_ids = np.abs(s_ids - s_ref)   # |Δs|
        delta_j_ids = np.abs(j_ids - j_ref)   # |Δj|

        if (X_phi_cs is not None) and (x_phi_ref is not None):
            Xphi_ids = X_phi_cs[idx_ids]
            Zphi_ids, zphi_ref, _ = self._robust_block_z(Xphi_ids, x_phi_ref)
            d_phi_ids = self._avg_l1_block(Zphi_ids, zphi_ref)
        else:
            d_phi_ids = None

        # Exponentes (prioriza tiempo y escala)
        # Flat
        # alpha_pat, alpha_s, alpha_t, alpha_phi = 0.25, 0.25, 0.25, (0.25 if d_phi_ids is not None else 0.0)

        # Shape–enhanced  & $(0.30,0.20,0.20,0.30)$  {\alpha_t}, \alpha_s, \alpha_{\mathrm{pat}}, \alpha_\phi$)
        # alpha_pat, alpha_s, alpha_t, alpha_phi = 0.30, 0.20, 0.30, (0.30 if d_phi_ids is not None else 0.0)

        #  Time–dominant   & $(0.55,0.20,0.20,0.05)$  {\alpha_t}, \alpha_s, \alpha_{\mathrm{pat}}, \alpha_\phi$)
        # alpha_pat, alpha_s, alpha_t, alpha_phi = 0.20, 0.20, 0.50, (0.10 if d_phi_ids is not None else 0.0) # OK v1

        # Reference       & $(0.35,0.30,0.25,0.10)$  {\alpha_t}, \alpha_s, \alpha_{\mathrm{pat}}, \alpha_\phi$)
        # alpha_pat, alpha_s, alpha_t, alpha_phi = 0.25, 0.30, 0.35, (0.10 if d_phi_ids is not None else 0.0) # OK v1
        alpha_pat, alpha_s, alpha_t, alpha_phi = 0.40, 0.20, 0.40, (0.10 if d_phi_ids is not None else 0.0) # OK v1
        # alpha_pat, alpha_s, alpha_t, alpha_phi = 0.50, 0.15, 0.35, (0.0 if d_phi_ids is not None else 0.0) # OK v1

        # alpha_pat, alpha_s, alpha_t, alpha_phi = .5, .2, .2, (.1 if d_phi_ids is not None else 0.0) # OK v2

        ssum = alpha_pat + alpha_s + alpha_t + alpha_phi
        alpha_pat /= ssum; alpha_s /= ssum; alpha_t /= ssum
        if d_phi_ids is not None:
            alpha_phi /= ssum

        # Objetivo mínimo de no-cero en T
        nz_target_T = 0.40
        q_schedule = [0.75, 0.85, 0.90]

        pos_in_ids = {int(i): k for k, i in enumerate(idx_ids.tolist())}
        w_ids = None; w_T = None; w_C = None
        used_cutoffs = {}

        # Helper para construir pesos dados cuantiles por bloque
        def _build_weights(c_s, c_t, c_pat, c_phi):
            w_pat = self._tukey_bisquare(d_pat_ids,   c_pat)
            w_s   = self._tukey_bisquare(delta_s_ids, c_s)
            w_t   = self._tukey_bisquare(delta_j_ids, c_t)
            w_phi = self._tukey_bisquare(d_phi_ids,   c_phi) if d_phi_ids is not None else 1.0
            return (w_pat ** alpha_pat) * (w_s ** alpha_s) * (w_t ** alpha_t) * \
                (w_phi ** (alpha_phi if d_phi_ids is not None else 1.0))

        # 9.a) Elegimos cutoffs para IDS/T con agenda y objetivo de no-ceros
        c_s_used = c_t_used = c_pat_used = c_phi_used = None
        for q in q_schedule:
            c_s_T   = float(np.quantile(delta_s_ids, q)) if delta_s_ids.size else 1.0
            c_t_T   = float(np.quantile(delta_j_ids, q)) if delta_j_ids.size else 1.0
            c_pat_T = float(np.quantile(d_pat_ids,  min(q, 0.75))) if d_pat_ids.size else 1.0
            c_phi_T = float(np.quantile(d_phi_ids,  min(q, 0.75))) if (d_phi_ids is not None and d_phi_ids.size) else None

            w_ids_try = _build_weights(c_s_T, c_t_T, c_pat_T, c_phi_T)
            w_T_try = np.array([w_ids_try[pos_in_ids[int(i)]] for i in idx_N]) if idx_N.size else np.array([])
            nz_T = int(np.count_nonzero(w_T_try)) if w_T_try.size else 0
            min_nz = int(max(self.k_min, np.ceil(nz_target_T * max(1, idx_N.size))))
            if nz_T >= min_nz or q == q_schedule[-1]:
                w_ids = w_ids_try
                w_T = w_T_try
                used_cutoffs = {"q_ids_T": q, "q_pat_cap_T": min(q, 0.75)}
                c_s_used, c_t_used, c_pat_used, c_phi_used = c_s_T, c_t_T, c_pat_T, c_phi_T
                break

        # 9.b) Pesos específicos para C (más indulgentes) + suelo suave ε_C
        if idx_C.size:
            qC = max(0.90, used_cutoffs.get("q_ids_T", 0.75))
            c_s_C   = float(np.quantile(delta_s_ids, qC)) if delta_s_ids.size else 1.0
            c_t_C   = float(np.quantile(delta_j_ids, qC)) if delta_j_ids.size else 1.0
            # patrón/phi más suaves pero razonables
            c_pat_C = float(np.quantile(d_pat_ids,  max(0.85, min(qC, 0.90)))) if d_pat_ids.size else 1.0
            c_phi_C = float(np.quantile(d_phi_ids,  max(0.85, min(qC, 0.90)))) if (d_phi_ids is not None and d_phi_ids.size) else None

            w_ids_C = _build_weights(c_s_C, c_t_C, c_pat_C, c_phi_C)
            w_C = np.array([w_ids_C[pos_in_ids[int(i)]] for i in idx_C])
            # suelo suave en C (evita 100% ceros; no afecta T)
            # eps_C = 0.02
            eps_C = 0.00
            if w_C.size:
                w_C = (1.0 - eps_C) * w_C + eps_C
        else:
            w_C = np.array([])

        self.last_weights = {
            "ids": (idx_ids, w_ids),
            "N": (idx_N, w_T if w_T is not None else np.array([])),
            "C": (idx_C, w_C),
        }
        # Distancias para diagnóstico
        self.last_dists_ids = {
            "d_t":  delta_j_ids,
            "d_s":  delta_s_ids,
            "d_pat": d_pat_ids,
            "d_phi": (d_phi_ids if d_phi_ids is not None else None),
        }

        # (10) trazas y debug
        S_lin_ids = S[idx_ids] / self.size_unit
        S_ref_lin = float(data_size_ref) / self.size_unit

        # Spearman rápidos
        def _safe_spearman(a: np.ndarray, b: Optional[np.ndarray]) -> float:
            if b is None:
                return np.nan
            a = np.asarray(a, float); b = np.asarray(b, float)
            if a.size != b.size or a.size < 3:
                return np.nan
            try:
                from scipy.stats import spearmanr
                r = spearmanr(a, b).correlation
                return float(r) if np.isfinite(r) else np.nan
            except Exception:
                def _rank_avg(x: np.ndarray) -> np.ndarray:
                    order = np.argsort(x, kind="mergesort")
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.arange(1, x.size + 1, dtype=float)
                    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
                    if np.any(counts > 1):
                        for k, c in enumerate(counts):
                            if c > 1:
                                ranks[inv == k] = ranks[inv == k].mean()
                    return ranks
                ra = _rank_avg(a); rb = _rank_avg(b)
                va = np.var(ra); vb = np.var(rb)
                if va <= 0 or vb <= 0:
                    return np.nan
                r = np.corrcoef(ra, rb)[0, 1]
                return float(r) if np.isfinite(r) else np.nan

        rho_t = _safe_spearman(w_ids, self.last_dists_ids["d_t"])
        rho_s = _safe_spearman(w_ids, self.last_dists_ids["d_s"])
        rho_p = _safe_spearman(w_ids, self.last_dists_ids["d_pat"])
        rho_f = _safe_spearman(w_ids, self.last_dists_ids["d_phi"]) if (self.last_dists_ids["d_phi"] is not None) else np.nan

        self.last_idx_ids = idx_ids
        self.last_idx_order = idx_order
        self.last_d_pat = d_pat_ord
        self.last_d_r = None
        self.last_r_ids = None
        self.last_thresholds = {
            "q75_pat": q75_pat, "q95_pat": q95_pat,
            "q75_s": q75_s, "q95_s": q95_s,
            **info_guard,
            **({"q_star": info_q.get("q_star")} if "q_star" in info_q else {}),
            "cuts_T": {"q": used_cutoffs.get("q_ids_T"), "pat_cap": used_cutoffs.get("q_pat_cap_T")},
            "cuts_C": {
                "qC": (max(0.90, used_cutoffs.get("q_ids_T", 0.75))) if idx_C.size else None,
                "eps_C": (0.02 if idx_C.size else None),
            },
            "c_pat_used": float(c_pat_used) if c_pat_used is not None else np.nan,
            "c_s_used": float(c_s_used) if c_s_used is not None else np.nan,
            "c_t_used": float(c_t_used) if c_t_used is not None else np.nan,
            **({"c_phi_used": float(c_phi_used)} if (d_phi_ids is not None and c_phi_used is not None) else {}),
        }
        self.last_T = idx_N
        self.last_C = idx_C
        self.last_debug = {
            "n_total": int(n),
            "n_pool0": int(n_pool0),
            "n_pool_q": int(idx_pool_q.size),
            "n_ids": int(idx_ids.size),
            "|T|": int(idx_N.size),
            "|C|": int(idx_C.size),
            "ids_mode": ids_mode,
            "jackknife": bool(jackknife),
            "jackknife_excluded": None,
            "S_ref_lin": float(S_ref_lin),
            "q_info": info_q,
            "spearman": {
                "w_vs_|Δj|": rho_t,
                "w_vs_|Δs|": rho_s,
                "w_vs_d_pat": rho_p,
                "w_vs_d_phi": rho_f,
            },
            "nz_T": int(np.count_nonzero(self.last_weights["N"][1])) if idx_N.size else 0,
            "nz_C": int(np.count_nonzero(self.last_weights["C"][1])) if idx_C.size else 0,
            "len_T": int(idx_N.size),
            "len_C": int(idx_C.size),
        }

        if idx_N.size:
            nzN = (w_T > 0)
            idx_N = idx_N[nzN]
            w_T   = w_T[nzN]

        if idx_C.size:
            nzC = (w_C > 0)
            idx_C = idx_C[nzC]
            w_C   = w_C[nzC]

        self.N_sizes.append(idx_N.size)
        self.C_sizes.append(idx_C.size)

        return idx_N, idx_C

    def get_last_debug(self) -> Dict[str, object]:
        return dict(self.last_debug)

    def reset_thresholds(self):
        """Clear historical sizes for N/C (useful when restarting experiments)."""
        self.N_sizes = []
        self.C_sizes = []


def get_safe_transfer_learning_space(
        safe_transfer_learning: SafeTransferLearningStage,
        target_workload: WorkloadCharacterized,
        characterized_workloads: List[WorkloadCharacterized],
        setting_workloads: List[List[int]],
        name_workloads: List[str],
        objective_values_workloads: List[float],  # J_beta (T or T_R)
        data_sizes_bytes_workloads: List[int],    # per-run input sizes (bytes)
        phi: Optional[List[List[int]]] = None,    # (n,4) optional extra features
        phi_ref: Optional[List[int]] = None,      # (4,) optional extra feature
        rho: Optional[np.ndarray] = None,         # (n,) opcional (ρ)
        rho_ref: Optional[float] = None,          # escalar ref
        *,
        size_unit: float = 1024.0**3
) -> tuple:
    """
    Materializa espacios STL en (X, y) y emite trazas compactas:
      - ESS y nº de pesos no-cero en N y C.
      - Deltas de anclas entre iteraciones: Δs_ref (GB) y Δφ_ref (L1).
    """
    k_limit = len(characterized_workloads)

    # ------- ancla actual (del target/ILS) -------
    s_ref_gb = float(target_workload.dataset_size / size_unit)
    phi_ref_now = np.array(phi_ref if phi_ref is not None else target_workload.resource_shape, dtype=float)

    # ------- trazas de ancla: delta vs. iteración previa (atributos estáticos) -------
    prev_s = getattr(get_safe_transfer_learning_space, "_prev_s_ref_gb", None)
    prev_phi = getattr(get_safe_transfer_learning_space, "_prev_phi_ref", None)
    if prev_s is None or prev_phi is None:
        print(f"[STL] anchors (init): s_ref_GB={s_ref_gb:.2f}, φ_ref={np.round(phi_ref_now, 3)}")
    else:
        d_s = abs(s_ref_gb - float(prev_s))
        d_phi = float(np.sum(np.abs(phi_ref_now - prev_phi)))
        print(f"[STL] anchors: Δs_ref_GB={d_s:.3f}, Δφ_ref_L1={d_phi:.3f} | s_ref_GB={s_ref_gb:.2f}, φ_ref={np.round(phi_ref_now, 3)}")
    # actualizar estáticos para la próxima llamada
    setattr(get_safe_transfer_learning_space, "_prev_s_ref_gb", s_ref_gb)
    setattr(get_safe_transfer_learning_space, "_prev_phi_ref", phi_ref_now.copy())

    # ------- log básico de referencia -------
    print(f"{target_workload.time_execution=} | ids={s_ref_gb:.2f} GB")

    # ------- construir zonas STL -------
    idx_SL, idx_SE = safe_transfer_learning.build_zones(
        workload_descriptors=np.array(characterized_workloads),
        objective_values=np.array(objective_values_workloads),
        data_sizes=np.array(data_sizes_bytes_workloads),      # bytes
        workload_ref=np.array(target_workload.vector_metrics_garralda),
        objective_ref=float(target_workload.time_execution),
        data_size_ref=float(target_workload.dataset_size),    # bytes
        X_phi_cs=np.array(phi) if phi is not None else None, # (n,4)
        x_phi_ref=phi_ref_now.reshape(1, -1)                  # (1,4)
    )

    # pesos/índices finales de STL
    idx_N, w_N = safe_transfer_learning.last_weights["N"]   # núcleo
    idx_C, w_C = safe_transfer_learning.last_weights["C"]   # corona

    # ------- LOG extra: ESS y no-cero en N y C -------
    def _ess(w):
        w = np.asarray(w, float)
        if w.size == 0:
            return 0.0
        s = w.sum(); q = (w**2).sum()
        return float((s * s) / (q + 1e-12))

    print(f"[STL] N: |{len(w_N)}| nz={np.count_nonzero(w_N)} ESS={_ess(w_N):.1f}")
    print(f"[STL] C: |{len(w_C)}| nz={np.count_nonzero(w_C)} ESS={_ess(w_C):.1f}")

    # ------- materializar espacios -------
    X_S_L  = np.array(setting_workloads)[idx_N]
    y_S_L  = np.array(objective_values_workloads)[idx_N]

    X_S_E  = np.array(setting_workloads)[idx_C]
    y_S_E  = np.array(objective_values_workloads)[idx_C]

    X_S_LE = np.vstack([X_S_L, X_S_E])
    y_S_LE = np.hstack([y_S_L, y_S_E])

    IDS_SL = np.array(data_sizes_bytes_workloads)[idx_N]
    IDS_SE = np.array(data_sizes_bytes_workloads)[idx_C]

    w_names_SL = np.array(name_workloads)[idx_N]
    w_names_SE = np.array(name_workloads)[idx_C]

    print("SafeTransferLearningStage:\n"
          f"{len(idx_SL)=} | {len(idx_SE)=}")

    # ------- artefactos -------
    stl_metrics = SafeTransferLearningEvaluationMetrics(
        search_space=[
            SearchSpace(
                name="nucleo",
                size=len(idx_N),
                workload_involved=[
                    WorkloadInvolved(
                        types=np.array(name_workloads)[idx_N].tolist(),
                        configurations=[SparkParameters.from_vector(x) for x in np.array(setting_workloads)[idx_N]]
                    )
                ]
            ),
            SearchSpace(
                name="corona",
                size=len(idx_C),
                workload_involved=[
                    WorkloadInvolved(
                        types=np.array(name_workloads)[idx_C].tolist(),
                        configurations=[SparkParameters.from_vector(x) for x in np.array(setting_workloads)[idx_C]]
                    )
                ]
            )
        ],
    )

    return (
        X_S_L,  y_S_L.reshape(-1, 1),
        stl_metrics,
        X_S_LE, y_S_LE.reshape(-1, 1),
        X_S_E,  y_S_E.reshape(-1, 1),
        IDS_SL,  IDS_SE, w_N, w_C,
        w_names_SL, w_names_SE
    )
