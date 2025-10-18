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
from typing import List, Tuple, Counter
from framework.proposed.adaptive_optimization_framework import adaptive_ils_ts_bo_loop
from framework.proposed.bayesian_optimization.objective_function import OptimizationObjective
from framework.proposed.parameters import SparkParameters
from framework.proposed.workload_characterization.workload import WorkloadRepository, WorkloadCharacterized
from runners.optimization_models import RESTRICTION_BOUNDS, WORKLOAD_TARGET_ID, BUDGET, WORKLOAD_TARGET_TYPE, \
    LHS_BEST_TIME


async def main(
        workload_id: str,
        budget: int,
        best_lhs_time: int,
        bounds: List[Tuple[int, int, int]],
        config: dict
):
    # todo: we need to think about how to get the firts workload characterized
    # For now, we will use a predefined workload ID

    # todo: think about how to implement the warn-start.
    # Use only the first space S_L (without the ) for training the Gaussian Process Regression model

    repo = WorkloadRepository(
        collection=config.get("collection_historical_dataset")
    )

    characterized_workloads: List[WorkloadCharacterized] = (
        repo.get_characterized_workloads(
            workloads=[
                config.get("workload_target_type")  # The target workload LOGO
            ],
            include=False,
            app_benchmark_group_id=config.get("app_benchmark_group_id")
        )
    )

    (
        workload_characterization_features_v1,
        _,
        _,
        workload_setting_features_v1,
        _,
        workload_time_execution,
        workload_names,
        # workload_time_resources
        workload_input_data_sizes,
        workload_resources,
        workload_configuration_shapes
    ) = WorkloadCharacterized.get_data_for_regression(workloads=characterized_workloads)

    print(f"{Counter(workload_names)=}")

    # Add dynamically the time resources to the characterized workloads
    # This is the objective function value T^β · R^(1−β)
    workload_time_resources = [
        OptimizationObjective.objective_function(
            T=workload.time_execution,
            R=OptimizationObjective.calculate_resource_usage(
                SparkParameters.from_vector(
                    workload.environment.to_vector()
                )
            ),
            beta=config.get("beta")
        )
        for workload in characterized_workloads
    ]

    print(f"Characterized workloads: {len(workload_characterization_features_v1)=}")

    # if adaptive_loop_version:
    await adaptive_ils_ts_bo_loop(
        target_workload_id=workload_id,
        characterized_workloads=workload_characterization_features_v1,
        setting_workloads=workload_setting_features_v1,
        time_execution_workloads=workload_time_execution,
        time_resources=workload_time_resources,
        input_data_sizes=workload_input_data_sizes,
        workload_names=workload_names,
        phi=workload_configuration_shapes,
        rho=workload_resources,
        bounds=bounds,
        budget=budget,
        best_lhs_time=best_lhs_time,
        config=config
    )


if __name__ == "__main__":
    CONFIG = {
        "app_benchmark_group_id": "lhs",
        "framework": "garralda_ils_ts_bo_baseline",
        # "experiment_id": "linear_q1_continuous_evaluation_7",
        "experiment_id": "linear_q1_evaluation_new_2",
        "collection_historical_dataset": "historical_dataset",
        "collection_save_results": "garralda_opt_model_evaluations",
        "workload_target_type": WORKLOAD_TARGET_TYPE,
        "enable_update_real_executions": True,
        "beta": 1, # 0.75
        "alpha": 1,  # 0.3
        "adaptive_alpha_lcb": False,
        "tolerance_perc": 3,
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