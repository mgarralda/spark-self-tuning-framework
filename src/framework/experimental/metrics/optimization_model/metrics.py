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

import statistics
from typing import List, Optional


class RunTimeOptimizationMetrics:
    """Metrics for runtime optimization, including speedup, tuning cost, and convergence analysis."""

    @staticmethod
    def speedup(time_default: float, time_optimized: float) -> float:
        """
        Calculate speedup percentage (SU).
        Positive means improvement (faster than default).
        """
        if time_default <= 0:
            raise ValueError("Default execution time must be > 0 for speedup calculation.")
        return ((time_default - time_optimized) / time_default) * 100.0

    @staticmethod
    def tuning_cost(execution_times: List[float]) -> float:
        """
        Total accumulated execution time across all iterations (TC).
        """
        if not execution_times:
            raise ValueError("Execution times list is empty.")
        return float(sum(execution_times))

    @staticmethod
    def aocc(execution_times: List[float]) -> float:
        """
        Area Over the Convergence Curve (AOCC), i.e., cumulative overhead
        relative to the best execution time observed across iterations.
        (Unnormalized version kept for reference/compat.)
        """
        if not execution_times:
            raise ValueError("Execution times list is empty.")
        best_time = min(execution_times)
        return float(sum(t - best_time for t in execution_times))

    @staticmethod
    def naocc(execution_times: List[float]) -> float:
        """
        Normalized AOCC (nAOCC): AOCC divided by (best_time * n).
        Lower is better; 0 means all iterations at the best time.

            nAOCC = sum_i (t_i - best) / (best * n)
        """
        if not execution_times:
            raise ValueError("Execution times list is empty.")
        n = len(execution_times)
        best_time = min(execution_times)
        if best_time <= 0:
            raise ValueError("Best execution time must be > 0 to compute nAOCC.")
        aocc_val = sum(t - best_time for t in execution_times)
        return float(aocc_val / (best_time * n))

    @staticmethod
    def hit_at_epsilon(
            execution_times: List[float],
            epsilon: float = 0.10,
            best_time: Optional[float] = None,
            as_percentage: bool = True,
    ) -> float:
        """
        Hit@ε: fraction (or %) of evaluations within ε-tolerance of the best.

            Hit@ε = (1/n) * sum_i 1{ (t_i - best) / best <= ε }

        Args:
            execution_times: per-iteration times.
            epsilon: relative tolerance (e.g., 0.05 or 0.10).
            best_time: optional reference best; if None uses min(execution_times).
            as_percentage: if True, returns 0..100 (%); else 0..1.

        Returns:
            float: Hit@ε.
        """
        if not execution_times:
            raise ValueError("Execution times list is empty.")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0.")
        ref_best = min(execution_times) if best_time is None else float(best_time)
        if ref_best <= 0:
            raise ValueError("Reference best time must be > 0.")
        n = len(execution_times)
        hits = sum(1 for t in execution_times if (t - ref_best) / ref_best <= epsilon)
        frac = hits / n
        return float(frac * 100.0) if as_percentage else float(frac)

    @staticmethod
    def tte(
            execution_times: List[float],
            epsilon: float = 0.10,
            best_time: Optional[float] = None,
    ) -> int:
        """
        Time-to-ε (TTE): earliest iteration index i such that the relative gap to the incumbent
        optimum falls below ε, i.e.:

            TTE(ε) = min { i ≤ n : (t_i - best) / best ≤ ε }

        Returns:
            int: 1-based iteration index. If the condition is never met within n iterations,
                 returns n+1 (sentinel value useful for tables: "not reached within budget").

        Args:
            execution_times: per-iteration execution times (length n).
            epsilon: relative tolerance (e.g., 0.10 for 10%).
            best_time: optional reference best; if None uses min(execution_times).
        """
        if not execution_times:
            raise ValueError("Execution times list is empty.")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0.")

        ref_best = min(execution_times) if best_time is None else float(best_time)
        if ref_best <= 0:
            raise ValueError("Reference best time must be > 0.")

        for i, t in enumerate(execution_times, start=1):
            if (t - ref_best) / ref_best <= epsilon:
                return i

        # Not reached within budget -> return n+1
        return len(execution_times) + 1

    @staticmethod
    def median_normalized_to_baseline(
            values: List[float],
            baseline: float
    ) -> float:
        """
        Compute the median of metric values normalized to a single workload baseline.

        Args:
            values: list of metric values for a workload (across methods).
            baseline: baseline value for that workload.

        Returns:
            float: median of normalized values (value / baseline).
        """
        if not values:
            raise ValueError("Values list must not be empty.")
        if baseline <= 0:
            raise ValueError("Baseline must be > 0.")

        ratios = [v / baseline for v in values]
        return statistics.median(ratios)


class CostAwareOptimizationMetrics:
    """Metrics for cost-aware optimization (β=0.5), including resource/time deltas and TOR."""

    # ---------- internal checks ----------
    @staticmethod
    def _check_positive(name: str, value: float):
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")

    @staticmethod
    def _resource_savings_fraction(resources_default: float, resources_optimized: float) -> float:
        """
        ΔR (fraction): positive means resource savings relative to the baseline.
        """
        CostAwareOptimizationMetrics._check_positive("Default resource usage", resources_default)
        CostAwareOptimizationMetrics._check_positive("Optimized resource usage", resources_optimized)
        return (resources_default - resources_optimized) / resources_default

    @staticmethod
    def _runtime_overhead_fraction(time_default: float, time_optimized: float) -> float:
        """
        ΔT (fraction): positive means runtime overhead (slower than baseline).
        """
        CostAwareOptimizationMetrics._check_positive("Default execution time", time_default)
        CostAwareOptimizationMetrics._check_positive("Optimized execution time", time_optimized)
        return (time_optimized - time_default) / time_default

    # ---------- public API ----------
    @staticmethod
    def resource_reduction(
            resources_default: float,
            resources_optimized: float
    ) -> float:
        """
        Resource reduction percentage:
        ΔR(%) = 100 * (R_default - R_optimized) / R_default.
        Positive means resource savings.
        """
        frac = CostAwareOptimizationMetrics._resource_savings_fraction(resources_default, resources_optimized)
        return frac * 100.0

    @staticmethod
    def time_reduction(
            time_default: float,
            time_optimized: float
    ) -> float:
        """
        Time reduction percentage (speed-up):
        ΔT_red(%) = 100 * (T_default - T_optimized) / T_default.
        Positive means faster than baseline.
        Note: this is not the overhead sign used in TOR; TOR uses ΔT_over = (T_opt - T_def)/T_def.
        """
        CostAwareOptimizationMetrics._check_positive("Default execution time", time_default)
        CostAwareOptimizationMetrics._check_positive("Optimized execution time", time_optimized)
        tr = ((time_default - time_optimized) / time_default) * 100.0
        return tr

    @staticmethod
    def trade_off_ratio_monotone_old(
            time_default: float, time_optimized: float,
            resources_default: float, resources_optimized: float
    ) -> dict:
        """
        Monotone Trade-off Ratio (TOR): higher is better in all cases.

        Definitions (baseline 'default' vs candidate 'optimized'):
          ΔT_frac = (T_opt - T_def) / T_def     # runtime change; ΔT>0 is overhead
          ΔR_frac = (R_def - R_opt) / R_def     # resource change; ΔR>0 is savings

        TOR (dimensionless):
          - If ΔT > 0 (overhead): TOR = ΔR / ΔT  (savings per unit overhead)
          - If ΔT <= 0:           TOR = ΔR       (pure savings)

        Returns a dict with:
          - delta_t: ΔT in percent (overhead sign convention)
          - delta_r: ΔR in percent (savings sign convention)
          - tor:     TOR in fraction (dimensionless, monotone)
          - tor_for_table: None if |ΔT| < 1% (optional masking for tables), else tor
        """
        # validations
        CostAwareOptimizationMetrics._check_positive("Default execution time", time_default)
        CostAwareOptimizationMetrics._check_positive("Optimized execution time", time_optimized)
        CostAwareOptimizationMetrics._check_positive("Default resource usage", resources_default)
        CostAwareOptimizationMetrics._check_positive("Optimized resource usage", resources_optimized)

        # fractions
        dT = (time_optimized - time_default) / time_default      # ΔT (fraction, overhead sign)
        dR = (resources_default - resources_optimized) / resources_default  # ΔR (fraction, savings sign)

        # percentages for reporting
        delta_t_pct = dT * 100.0
        delta_r_pct = dR * 100.0

        # robust handling near-zero overhead to avoid exploding ratios
        EPS_FRAC = 1e-6
        if dT > EPS_FRAC:
            tor_val = dR / dT
        elif dT < -EPS_FRAC:
            tor_val = dR  # pure savings in speed-up/no-overhead regime
        else:
            # effectively no overhead: treat as pure savings for TOR; tor_for_table may hide it
            tor_val = dR

        # optional masking for pretty tables (hide TOR when |ΔT| < 1%)
        HIDE_TOR_IF_DT_LT_PERCENT = 1.0
        tor_for_table = None if abs(delta_t_pct) < HIDE_TOR_IF_DT_LT_PERCENT else tor_val

        return {
            "delta_t": delta_t_pct,
            "delta_r": delta_r_pct,
            "tor_mono": tor_val,   # keep original key for backward compatibility
            "tor": tor_val,        # alias aligned with paper naming
            "tor_for_table": tor_for_table
        }

    @staticmethod
    def trade_off_ratio_monotone(
            time_default: float, time_optimized: float,
            resources_default: float, resources_optimized: float
    ) -> dict:
        """
        Monotone Trade-off Ratio (TOR): higher is better in all cases.

        Definitions (baseline 'default' vs candidate 'optimized'):
          ΔT_frac = (T_opt - T_def) / T_def     # runtime change; ΔT>0 is overhead
          ΔR_frac = (R_def - R_opt) / R_def     # resource change; ΔR>0 is savings

        TOR (dimensionless, monotone):
          - If ΔT > 0 (overhead): TOR = ΔR / ΔT             # savings per unit overhead
          - If ΔT <= 0:           TOR = (1 - ΔT) * ΔR       # joint improvement reward

        Returns a dict with:
          - delta_t: ΔT in percent (overhead sign convention)
          - delta_r: ΔR in percent (savings sign convention)
          - tor:     TOR in fraction (dimensionless, monotone)
          - tor_for_table: None if |ΔT| < 1% (optional masking for tables)
        """
        # validations
        CostAwareOptimizationMetrics._check_positive("Default execution time", time_default)
        CostAwareOptimizationMetrics._check_positive("Optimized execution time", time_optimized)
        CostAwareOptimizationMetrics._check_positive("Default resource usage", resources_default)
        CostAwareOptimizationMetrics._check_positive("Optimized resource usage", resources_optimized)

        # fractions
        dT = (time_optimized - time_default) / time_default      # ΔT (fraction, overhead sign)
        dR = (resources_default - resources_optimized) / resources_default  # ΔR (fraction, savings sign)

        # percentages for reporting
        delta_t_pct = dT * 100.0
        delta_r_pct = dR * 100.0

        EPS_FRAC = 1e-6
        if dT > EPS_FRAC:
            # overhead case → savings per unit overhead
            tor_val = dR / dT
        else:
            # speed-up or negligible overhead → joint improvement
            tor_val = (1 - dT) * dR

        # optional masking for pretty tables (hide TOR when |ΔT| < 1%)
        HIDE_TOR_IF_DT_LT_PERCENT = 1.0
        tor_for_table = None if abs(delta_t_pct) < HIDE_TOR_IF_DT_LT_PERCENT else tor_val

        return {
            "delta_t": delta_t_pct,
            "delta_r": delta_r_pct,
            "tor_mono": tor_val,   # backward compatibility
            "tor": tor_val,        # paper naming
            "tor_for_table": tor_for_table,
        }
