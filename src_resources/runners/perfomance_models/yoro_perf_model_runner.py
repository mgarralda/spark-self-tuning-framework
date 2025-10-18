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
from numpy import unique
from typing import List
import numpy as np
from framework.experimental.baselines.yoro.performance_model.yoro_perf_model import CrossValidation
from framework.proposed.bayesian_optimization.objective_function import OptimizationObjective
from framework.proposed.parameters import SparkParameters
from framework.proposed.workload_characterization.workload import WorkloadRepository, WorkloadCharacterized
from runners.perfomance_models import T_R_BETA


if __name__ == "__main__":

    repo = WorkloadRepository(
        collection="historical_dataset",
        database_name="spark_event_log"
    )
    characterized_workloads: List[WorkloadCharacterized] = repo.get_characterized_workloads(
        app_benchmark_group_id="lhs",
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
            beta=T_R_BETA
        )
        for workload in characterized_workloads
    ]

    print(f"Characterized workloads:\n"
          f"\t# samples: {len(workload_characterization_extended_features_v2)}\n"
          f"\t# workload characterization vector size: {len(workload_characterization_extended_features_v2[0])}\n"
          f"\t# setting vector size: {len(workload_setting_features[0])}\n"
          f"\t# unique workloads: {len(unique(workload_names))}")

    async def run():
        validation = CrossValidation(
            w=np.array(workload_characterization_extended_features_v2),
            w_groups=np.array(workload_names),
            # t=np.array(workload_time_execution)
            t=np.array(workload_time_resources)
        )

        # validation.leave_one_out()
        validation.leave_one_group_out()

    asyncio.run(run())


# Residual Std. Dev.: 236.87
