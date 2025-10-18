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

from framework.experimental.dataset import LatinHypercubeSampling
from framework.proposed.parameters import SparkRangeParameters


if __name__ == "__main__":

    # Create an instance of SparkParameters using default bounds.
    spark_params = SparkRangeParameters()

    # Define maximum available resources.
    max_cores = 22       # e.g., total cores available.
    max_memory = 50      # e.g., total memory (GB) available.

    # Define scale mapping: number of configurations for each tier.
    scale_mapping = {'small': 15, 'large': 25}
    # scale_mapping = {'small': 15, 'large': 25, 'huge': 30}

    # Instantiate the LatinHypercubeSampling class.
    lhs_sampler = LatinHypercubeSampling(spark_params, max_cores, max_memory, scale_mapping, random_state=42)
    print(f"Total possible combinations (ignoring resource constraints): {lhs_sampler.total_valid_combinations()}\n")

    # Generate all configuration spaces in one call.
    spark_config_spaces = lhs_sampler.initialize_config_spaces()

    # # Print the generated configuration spaces.
    # print(f"Generated configuration spaces:")
    print(f"{spark_config_spaces.model_dump_json(indent=2)}")
