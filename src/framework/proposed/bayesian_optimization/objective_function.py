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
from framework.proposed.parameters import SparkParameters


class OptimizationObjective:

    @staticmethod
    def objective_function(
            # cs: SparkParameters,
            T: int,
            R: int,
            beta: float
    ) -> float:
        """
        compute geometric cost objective:
            T_R = µ^β · R^(1−β)
        Log-space for numerical stability.
        Computes the multiplicative cost-aware objective function:
        Reflects multiplicative cost per executor.
        Aligns better with cost in cloud environments or sustainability scenarios (e.g., GB×core-hours).
        """
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"Beta must be in [0, 1], got {beta}")

        # R_pred = OptimizationObjective.calculate_resource_usage(cs)
        # return T ** beta * R ** (1 - beta)

        # Use log1p for numerical stability with small values
        log_of = beta * np.log1p(T) + (1.0 - beta) * np.log1p(R)

        return np.expm1(log_of)  # inverse of log1p to keep scale

    @staticmethod
    def calculate_resource_usage(cs: SparkParameters) -> int:
        """
        Reflects multiplicative cost per executor.
        Aligns better with cost in cloud environments or sustainability scenarios (e.g., GB×core-hours).
        """

        return (
                cs.driver_cores * cs.driver_memory +
                cs.executor_instances * cs.executor_cores * cs.executor_memory
        )
