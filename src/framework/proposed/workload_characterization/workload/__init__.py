# -----------------------------------------------------------------------------
#  Project: Spark Autotuning Framework
#  File: __init__.py
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

from framework.proposed.workload_characterization.workload.repository import WorkloadRepository
from framework.proposed.workload_characterization.workload.vector_descriptor import WorkloadCharacterized
from framework.proposed.workload_characterization.workload.base import (
    WorkloadType,
    InputDataSizeType,
    WorkloadEntity,
)

__all__ = [
    "WorkloadEntity",
    "WorkloadCharacterized",
    "WorkloadType",
    "InputDataSizeType",
    "WorkloadRepository"
]
