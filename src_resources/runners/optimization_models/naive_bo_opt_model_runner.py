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
from framework.proposed.bayesian_optimization.acquisition_function import CandidateSpaceGlobal
from framework.proposed.workload_characterization.workload import WorkloadRepository
from runners.optimization_models import RESTRICTION_BOUNDS, BUDGET, WORKLOAD_TARGET_ID, LHS_BEST_TIME
from framework.experimental.baselines.naivebo.optimization_model.naivebo_opt_model import naive_bo_loop


async def main(
        workload_ref: str,
        budget: int,
        best_lhs_time: int,
        restriction_bounds,
        config: dict
) -> None:

    # --- Target workload ---
    repo = WorkloadRepository(collection=config.get("collection_historical_dataset"))
    target = repo.get_characterized_workload(workload_ref)

    # --- FAIRNESS: The same candidate pool for all baselines ---
    candidate_pool = CandidateSpaceGlobal(restriction_bounds, seed=42, n=2).sobol_generate_candidates()

    await naive_bo_loop(
        target_workload=target,
        bounds=restriction_bounds,
        budget=budget,
        best_lhs_time=best_lhs_time,
        config=config,
        candidate_pool=candidate_pool,    # <- el mismo pool para todos los baselines
    )


if __name__ == "__main__":

    CONFIG = {
        "app_benchmark_group_id": "lhs",
        "framework": "naive_bo_baseline",
        "experiment_id": "lda_q1_evaluation_1",
        "collection_historical_dataset": "historical_dataset",
        "collection_save_results": "naive_bo_opt_model_evaluations",
        "beta": 1,
        "alpha": 1.0
    }

    asyncio.run(
        main(
            workload_ref=WORKLOAD_TARGET_ID,
            budget=BUDGET,
            best_lhs_time=LHS_BEST_TIME,
            restriction_bounds=RESTRICTION_BOUNDS,
            config=CONFIG
        )
    )
