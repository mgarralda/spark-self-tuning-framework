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

import time
import traceback
from pathlib import Path
from runners.historical_data import CONFIG
from framework.proposed.workload_characterization.pipeline import get_files_in_folder, run_characterization


if __name__ == '__main__':
    spark_event_log_path = Path(
        "D:\Spark\hadoop-spark-cluster\shared-master\HiBench2\logs"
    )

    for log_file in get_files_in_folder(spark_event_log_path):

        start_time = time.time()
        try:
            run_characterization(
                log_file,
                CONFIG
            )
        except Exception:
            print(traceback.format_exc())
        finally:
            print('Duration: {}'.format(time.time() - start_time))
