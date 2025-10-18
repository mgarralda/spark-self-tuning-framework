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

import math
from typing import List, Tuple, Optional
from framework.proposed.workload_characterization.workload.base import WorkloadEntity


class WorkloadCharacterized(WorkloadEntity):
    """
    Workload Characterization according to compared version of paper:
    You Only Run Once: Spark Auto-tuning from a Single Run
    """

    vector_metrics_yoro: List[float]
    vector_metrics_garralda: List[float]
    resource_usage: Optional[float] = None
    resource_shape: Optional[List[float]] = None


    def get_garralda_vector_descriptor(self, adding_data_size: bool = False) -> List[float]:
        if adding_data_size:
            # If we want to add the data size to the Garralda parser descriptor
            # This is required for Prats characterization (YORO paper)
            return self.vector_metrics_garralda + [self.get_data_size()]
        return self.vector_metrics_garralda

    def get_garralda_vector_descriptor_17(self, adding_data_size: bool = False) -> List[float]:
        vector_17_features = self.pick_17_features_from_garralda_vector(self.vector_metrics_garralda)
        if adding_data_size:
            # If we want to add the data size to the Garralda parser descriptor
            # This is required for Prats characterization (YORO paper)
            return vector_17_features + [self.get_data_size()]
        return vector_17_features

    def pick_17_features_from_garralda_vector(self, vector_metrics_garralda: list[float]) -> list[float]:
        """
        Given `vector_metrics_garralda` of length 25*4=100 (in the order
        [raw#0, raw#1, …, raw#24], each expanded to 4 slots),
        return exactly those 17 raw‐features {0–8, 11, 15–20, 22} (each 4 stats).
        """
        vector_17: list[float] = []

        # 0..8  → pick raw features 0 through 8
        for raw_idx in range(0, 9):
            start_slot = raw_idx * 4
            end_slot = start_slot + 4
            vector_17.extend(vector_metrics_garralda[start_slot : end_slot])
            # e.g. raw_idx=0 → slots[0:4]
            #      raw_idx=1 → slots[4:8]
            #      …
            #      raw_idx=8 → slots[32:36]

        # raw feature 11 → slots [44..47]
        raw_idx = 11
        vector_17.extend(vector_metrics_garralda[raw_idx*4 : raw_idx*4 + 4])
        # raw_idx*4 = 11*4 = 44, so take [44:48)

        # raw features 15..20 → slots [60..83]
        for raw_idx in range(15, 21):
            start_slot = raw_idx * 4
            end_slot = start_slot + 4
            vector_17.extend(vector_metrics_garralda[start_slot : end_slot])
            # raw_idx=15 → slots[60:64]
            # raw_idx=16 → slots[64:68]
            # ...
            # raw_idx=20 → slots[80:84]

        # raw feature 22 → slots [88..91]
        raw_idx = 22
        vector_17.extend(vector_metrics_garralda[raw_idx*4 : raw_idx*4 + 4])
        # raw_idx*4 = 22*4 = 88 → slots[88:92)

        # Sanity check:
        assert len(vector_17) == 17 * 4, f"Expected 68 values, got {len(vector_17)}"
        return vector_17

    def get_yoro_vector_descriptor(self, adding_data_size: bool = False) -> List[float]:
        # todo: Get only the 17 features mentioned in the YORO paper + dataset size and check the results
        # yoro_vector = self.vector_metrics_yoro

        if adding_data_size:
            # If we want to add the data size to the YORO parser descriptor
            # This is required for Prats characterization (YORO/Turbo paper)
            return self.vector_metrics_yoro + [self.get_data_size()]

        return self.vector_metrics_yoro

    def get_data_size(self) -> float:
        """
        Data size feature for the workload as the total data size read from disk.
        """
        return self.dataset_size / (1024 ** 3) # Convert bytes to GB

    def to_configuration_shape_vector(self) -> List[float]:
        """
        Configuration shape representation as a 4-D vector according to paper definition:
        Beta-aware, scales, and configuration shape
        """

        # Number of executor instances
        I =   self.environment.executor_instances

        # executor parallelism
        EP =  self.environment.executor_cores

        # Memory per core
        MPC = self.environment.executor_memory_gb/EP

        # Partitions per core
        PPC = self.environment.sql_shuffle_partitions/(I*EP)

        # #executor concurrency
        # Theta = EP / self.environment.task_cpus

        # Driver-executor balance
        DEB = self.environment.driver_cores * self.environment.driver_memory_gb / (I * EP * self.environment.executor_memory_gb)

        # Return the 5-D vector configuration shape epresentation
        return [
            EP,
            MPC,
            PPC,
            # Theta,
            DEB
        ]


    def to_configuration_shape_vector_(self) -> List[float]:
        """
        Configuration shape representation as a 4-D vector φ = (EP, MPC, PPC, DEB),
        returned in log1p-space for robust, scale-invariant distances.

        Uses only current class members (no extra knobs).
        """

        # Number of executor instances
        I = self.environment.executor_instances

        # Executor parallelism
        EP = self.environment.executor_cores

        # Memory per core
        MPC = self.environment.executor_memory_gb / EP

        # Partitions per core
        PPC = self.environment.sql_shuffle_partitions / (I * EP)

        # Driver–executor balance
        DEB = (
                self.environment.driver_cores * self.environment.driver_memory_gb
                / (I * EP * self.environment.executor_memory_gb)
        )

        # Return φ in log1p-space (component-wise)
        return [
            math.log1p(float(EP)),
            math.log1p(float(MPC)),
            math.log1p(float(PPC)),
            math.log1p(float(DEB)),
        ]



