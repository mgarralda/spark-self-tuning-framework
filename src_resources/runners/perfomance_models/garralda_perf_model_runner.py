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
import numpy as np
from typing import Counter, List
from numpy import unique
from framework.experimental.baselines.garralda.performance_model.garralda_perf_model import CrossValidation
from framework.proposed.bayesian_optimization.objective_function import OptimizationObjective
from framework.proposed.parameters import SparkParameters
from framework.proposed.workload_characterization.workload import WorkloadRepository, WorkloadCharacterized
from runners.perfomance_models import T_R_BETA


if __name__ == "__main__":

    async def run():

        repo = WorkloadRepository(
            collection="historical_dataset",
            database_name="spark_event_log"
        )
        characterized_workloads: List[WorkloadCharacterized] = (
            repo.get_characterized_workloads(
                # app_benchmark_group_id="lhs",
                #     workloads=[WorkloadType.WORDCOUNT],
                #     include=False  # Exclude the target workload to avoid data leakage
                app_benchmark_group_id="lhs"
            )
        )
        (
            workload_characterization_features_v1,
            workload_characterization_features_v2,
            _,
            workload_setting_features_v1,
            workload_setting_features_v2,
            workload_time_execution,
            workload_names,
            workload_input_data_sizes,
            workload_resources,
            workload_configuratin_shapes
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
        # V1: Garralda parser descriptor
        k_limit = int(len(workload_characterization_features_v1))

        print(f"Characterized workloads:\n"
              f"\t# samples: {len(workload_characterization_features_v1)}\n"
              f"\t# workload characterization vector size v1: {len(workload_characterization_features_v1[0])}\n"
              f"\t# workload characterization vector size v2: {len(workload_characterization_features_v2[0])}\n"
              f"\t# setting vector size v1: {len(workload_setting_features_v1[0])}\n"
              f"\t# setting vector size v2: {len(workload_setting_features_v2[0])}\n"
              f"\t# unique workloads: {len(unique(workload_names))}\n"
              f"\t# limit of transfer learning (k_limit): {k_limit}\n")

        #noinspectiton
        print(f"Workload counter per group: {Counter(workload_names)}")

        # for cs, t, wn in zip(workload_setting_features_v1, workload_time_resources, workload_names):
        #     print(f"{cs=}, {t=}, {wn=}")

        validation = CrossValidation(
            w=np.array(workload_characterization_features_v1),
            w_groups=np.array(workload_names),
            cs=np.array(workload_setting_features_v1),
            # t=np.array(workload_time_execution),
            t=np.array(workload_time_resources),
            ids=np.array(workload_input_data_sizes),
            phi=np.array(workload_configuratin_shapes),
            rho=np.array(workload_resources),
            k_limit=k_limit,
        )

        # validation.leave_one_out()

        validation.leave_one_group_out()

    asyncio.run(run())
