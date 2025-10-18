# -# -----------------------------------------------------------------------------
# #  Project: Spark Self-Tuning Framework (STL-PARN-ILS-TS-BO)
# #  File: yoro_perf_model_runner.py
# #  Copyright (c) 2025 Mariano Garralda Barrio
# #  Affiliation: Universidade da Coruña
# #  SPDX-License-Identifier: CC-BY-NC-4.0 OR LicenseRef-Commercial
# #
# #  Associated publication:
# #    "A hybrid metaheuristics–Bayesian optimization framework with safe transfer learning for continuous Spark tuning"
# #    Mariano Garralda Barrio, Verónica Bolón Canedo, Carlos Eiras Franco
# #    Universidade da Coruña, 2025.
# #
# #  Academic & research use: CC BY-NC 4.0
# #    https://creativecommons.org/licenses/by-nc/4.0/
# #  Commercial use: requires prior written consent.
# #    Contact: mariano.garralda@udc.es
# #
# #  Distributed on an "AS IS" basis, without warranties or conditions of any kind.
# # -----------------------------------------------------------------------------

import asyncio
from typing import List, Tuple
from framework.experimental.baselines.yoro.optimization_model import yoro_loop
from framework.proposed.bayesian_optimization.acquisition_function import CandidateSpaceGlobal
from framework.proposed.workload_characterization.workload import WorkloadRepository, WorkloadCharacterized
from runners.optimization_models import WORKLOAD_TARGET_TYPE, WORKLOAD_TARGET_ID, BUDGET, RESTRICTION_BOUNDS, \
    LHS_BEST_TIME


async def main(
        workload_id: str,
        budget: int,
        best_lhs_time: int,
        bounds: List[Tuple[int, int, int]],
        config: dict
):
    repo = WorkloadRepository(
        collection=config.get("collection_historical_dataset")
    )

    target: WorkloadCharacterized = repo.get_characterized_workload(workload_id)
    print(f"Target workload: {target}")

    (
        _,
        _,
        workload_ref,
        workload_setting_ref,
        _,
        execution_time_ref,
        *_
    ) = WorkloadCharacterized.get_data_for_regression(workloads=[target])

    print(f"Workload reference {type(workload_ref)=} | {len(workload_ref[0])=}\n{workload_ref}")
    print(f"Workload sd setting reference: {workload_setting_ref}")
    print(f"Workload execution time reference: {execution_time_ref}")

    characterized_workloads: List[WorkloadCharacterized] = (
        repo.get_characterized_workloads(
            workloads=[config.get("workload_target_type")],  # LOGO
            include=False,
            app_benchmark_group_id=config.get("app_benchmark_group_id")
        )
    )
    (
        _,
        _,
        workload_characterization_extended_features_v2,
        workload_setting_features,
        _,
        workload_time_execution,
        workload_names,
        *_
    ) = WorkloadCharacterized.get_data_for_regression(workloads=characterized_workloads)

    # Pool común de candidatos (Sobol), compartido con TurBO para equidad:
    cand_pool = CandidateSpaceGlobal(bounds, seed=24, n=2).sobol_generate_candidates()

    print(f"Total characterized workloads: {len(workload_characterization_extended_features_v2)} *******")
    print(f"Workload reference {type(workload_characterization_extended_features_v2)=} | {len(workload_characterization_extended_features_v2[0])=}")

    await yoro_loop(
        target_workload=target,
        target_full_vector=workload_ref,
        X_init=workload_characterization_extended_features_v2,
        y_init=workload_time_execution,
        bounds=bounds,
        budget=budget,
        best_lhs_time=best_lhs_time,
        config=config,
        candidate_pool=cand_pool,
        mode="sbo",             #  "greedy" o "sbo"
    )


if __name__ == "__main__":

    CONFIG = {
        "app_benchmark_group_id": "lhs",
        "framework": "yoro_bo_baseline",
        "experiment_id": "lda_q2_evaluation",
        "collection_historical_dataset": "historical_dataset",
        "collection_save_results": "yoro_bo_opt_model_evaluations",
        "workload_target_type": WORKLOAD_TARGET_TYPE,
        "beta": 1, # 0.75
        "alpha": 1,
    }

    # Run the Bayesian Optimization loop
    asyncio.run(
        main(
            workload_id=WORKLOAD_TARGET_ID,
            budget=BUDGET,
            best_lhs_time=LHS_BEST_TIME,
            bounds=RESTRICTION_BOUNDS,
            config=CONFIG
        )
    )
