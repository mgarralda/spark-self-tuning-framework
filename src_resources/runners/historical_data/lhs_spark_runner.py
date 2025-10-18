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
# -----------------------------------------------------------------------------

import asyncio
from pathlib import Path
from runners.historical_data import CONFIG
from utils.spark.hibench import run_lhs_hibench_batch


if __name__ == "__main__":
    spark_configs = Path(
        "G:\studies\doctorate\Source Code\\3-article\\resources\dataset\latin_hypercube_sampling\lhs_initialization_development_large_.json"
    )

    asyncio.run(
        run_lhs_hibench_batch(
            spark_configs,
            characterization=False,
            config=CONFIG
        )
    )