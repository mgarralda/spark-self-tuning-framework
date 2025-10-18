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

import asyncio
from typing import List, Tuple
import numpy as np
from framework.proposed.bayesian_optimization.acquisition_function import CandidateSpaceGlobal
from framework.proposed.workload_characterization.workload import WorkloadRepository, WorkloadCharacterized
from runners.optimization_models import RESTRICTION_BOUNDS, WORKLOAD_TARGET_ID, BUDGET, WORKLOAD_TARGET_TYPE, \
    LHS_BEST_TIME
from framework.experimental.baselines.turbo.optimization_model import turbo_bo_loop


async def main(
        workload_id: str,
        budget: int,
        best_lhs_time: int,
        bounds: List[Tuple[int, int, int]],
        config: dict
):
    """
    Paper-faithful runner for TurBO (Dou et al., 2023).

    Key points for faithfulness:
      • The meta-learning oracle (CASampling) must be trained ONLY on historical data,
        with per-sample weights derived from workload similarity (Eq.(6) in the paper).
      • To enable that, we pass BOTH historical descriptors (w_meta) and the target
        descriptor (w_ref) to the BO loop via `config`. The loop will compute
        similarity weights and train the Random Forest oracle accordingly.
      • The BO surrogate (GP+EI) is trained on historical + target-reference pairs
        (as you already do), and candidates come from a FIXED Sobol pool shared
        across methods (fairness).
    """
    repo = WorkloadRepository(
        collection=config.get("collection_historical_dataset")
    )

    # ---- Target workload (held-out) ----
    target: WorkloadCharacterized = repo.get_characterized_workload(workload_id)
    print(f"Target workload: {target}")

    # Extract target vectors for LOGO:
    # NOTE: get_data_for_regression(...) returns a tuple; we assume:
    #   idx 0 -> workload descriptors (w)
    #   idx 3 -> configuration settings (cs)
    #   idx 5 -> execution time (t)
    # If your helper returns them in a different order, adjust the indices below.
    (_w_ref, _, _, workload_setting_ref, _, execution_time_ref, *_) = \
        WorkloadCharacterized.get_data_for_regression(workloads=[target])
    w_ref = np.array(_w_ref)  # descriptor of the target workload (1 × D or D,)

    # ---- Historical dataset (exclude the target workload) ----
    characterized_workloads: List[WorkloadCharacterized] = repo.get_characterized_workloads(
        workloads=[config.get("workload_target_type")],
        include=False,
        app_benchmark_group_id=config.get("app_benchmark_group_id")
    )

    (_w_meta, _, _, all_setting_features, _, all_time_execution, *_) = \
        WorkloadCharacterized.get_data_for_regression(workloads=characterized_workloads)
    w_meta = np.array(_w_meta)                     # descriptors per historical sample (N_hist × D)
    X_meta = np.array(all_setting_features)        # historical configurations (N_hist × C)
    y_meta = np.array(all_time_execution)          # historical runtimes (N_hist,)

    # ---- BO surrogate training set: historical + target reference ----
    # (as in your current setup; these are the pairs used by GP+EI)
    X_train = np.vstack([X_meta, workload_setting_ref])
    y_train = np.append(y_meta, execution_time_ref)

    # ---- Shared candidate pool (fairness across methods) ----
    cand_pool = CandidateSpaceGlobal(bounds, seed=24, n=2).sobol_generate_candidates()

    # ---- Pass similarity information to BO loop (enables CASampling-style weighting) ----
    config_with_w = dict(config)
    config_with_w["w_meta"] = w_meta  # historical descriptors (aligned with X_meta / y_meta)
    config_with_w["w_ref"] = w_ref    # target descriptor

    # ---- Run TurBO baseline with CASampling (weighted RF) + BO-AdaPP + GP+EI ----
    await turbo_bo_loop(
        target_workload=target,
        X_init=X_train,
        y_init=y_train,
        bounds=bounds,
        budget=budget,
        best_lhs_time=best_lhs_time,
        config=config_with_w,   # contains w_meta / w_ref for similarity weights
        X_meta=X_meta,
        y_meta=y_meta,
        candidate_pool=cand_pool
    )



if __name__ == "__main__":
    CONFIG = {
        "app_benchmark_group_id": "lhs",
        "framework": "turbo_bo_baseline",
        "experiment_id": "lda_q2_evaluation",
        "collection_historical_dataset": "historical_dataset",
        "collection_save_results": "turbo_bo_opt_model_evaluations",
        "workload_target_type": WORKLOAD_TARGET_TYPE,
        "beta": 1,
        "enable_pseudopts": True,
        "tau": 0.01,
        "lmax": 10
    }

    asyncio.run(
        main(
            workload_id=WORKLOAD_TARGET_ID,
            budget=BUDGET,
            best_lhs_time=LHS_BEST_TIME,
            bounds=RESTRICTION_BOUNDS,
            config=CONFIG,
        )
    )
