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
from pydantic import BaseModel, Field
from typing import Tuple, List
from framework.proposed.workload_characterization.workload import WorkloadEntity


class SizeType(str):
    """
    Enum-like class for size types.
    """
    SMALL = "small"
    LARGE = "large"
    HUGE = "huge"


class SparkParameters(BaseModel):
    """
    This class is a placeholder for Spark configuration as well as default values.
    It can be extended to include additional configuration settings as needed.
    """

    driver_cores: int = Field(2, description="Number of cores for the driver")
    driver_memory: int = Field(5, description="Memory for the driver in GB")
    executor_cores: int = Field(5, description="Number of cores for the executor")
    executor_instances: int = Field(2, description="Number of executor instances")
    executor_memory: int = Field(3, description="Memory for the executor in GB")
    sql_shuffle_partitions: int = Field(200, description="Number of shuffle partitions")
    task_cpus: int = Field(1, description="Number of CPUs for tasks")

    @classmethod
    # def from_vector(cls, vector: Tuple[int, ...]) -> "SparkParameters":
    def from_vector(cls, vector: List[int] | np.ndarray) -> "SparkParameters":
        """
        Create a SparkParameters instance from a vector of integers.
        The vector should contain values in the order defined by the class attributes.
        """
        if len(vector) != 7:
            raise ValueError("Vector must contain exactly 7 elements.")

        return cls(
            driver_cores=vector[0],
            driver_memory=vector[1],
            executor_cores=vector[2],
            executor_instances=vector[3],
            executor_memory=vector[4],
            sql_shuffle_partitions=vector[5],
            task_cpus=vector[6]
        )

    @classmethod
    def from_dict(cls, conf: dict) -> "SparkParameters":
        """
        Create a SparkParameters instance from dictionary
        The vector should contain values in the order defined by the class attributes.
        """
        # if len(conf) != 7:
        #     raise ValueError("Configuration must contain exactly 7 elements.")

        return cls(
            driver_cores=conf.get('driver_cores'),
            driver_memory=conf.get('driver_memory_gb'),
            executor_cores=conf.get('executor_cores'),
            executor_instances=conf.get('executor_instances'),
            executor_memory=conf.get('executor_memory_gb'),
            sql_shuffle_partitions=conf.get('sql_shuffle_partitions', 200),
            task_cpus=conf.get('task_cpus', 1)
        )

    def __eq__(self, other: WorkloadEntity.Environment) -> bool:
        """
        Compare SparkParameters with WorkloadEntity.Environment.
        This method checks if the SparkParameters match the environment settings.
        """
        return (
            self.driver_cores == other.driver_cores and
            self.driver_memory == other.driver_memory_gb and
            self.executor_cores == other.executor_cores and
            self.executor_instances == other.executor_instances and
            self.executor_memory == other.executor_memory_gb and
            self.sql_shuffle_partitions == other.sql_shuffle_partitions and
            self.task_cpus == other.task_cpus
        )


class Range(BaseModel):
    """
    Base class for parameters. This class can be extended to define specific parameter types.
    """
    min: int
    max: int
    step: int


class SparkRangeParameters(BaseModel):
    """
    Spark parameters and their bounds for configuration space sampling.
    Each parameter is defined as a tuple: (min, max, step).

    Parameter order (following Table~\ref{tab:parallelism-config}):
      0: driver.cores         (cores)
      1: driver.memory        (GB)
      2: executor.cores       (cores)
      3: executor.instances   (instances)
      4: executor.memory      (GB)
      5: sql.shuffle.partitions (not used for resource constraints)
      6: task.cpus            (cores)
    """
    driver_cores: Range = Field(Range(
        min=1, max=3, step=1),
        description="Range for driver.cores"
    )
    driver_memory: Range = Field(
        Range(min=2, max=4, step=1),
        description="Range for driver.memory (GB)"
    )
    executor_cores: Range = Field(
        Range(min=1, max=5, step=1),
        description="Range for executor.cores"
    )
    executor_instances: Range = Field(
        Range(min=1, max=5, step=1),
        description="Range for executor.instances"
    )
    executor_memory: Range = Field(
        Range(min=2, max=5, step=1),
        description="Range for executor.memory (GB)"
    )
    sql_shuffle_partitions: Range = Field(
        Range(min=50, max=350, step=50),
        description="Range for sql.shuffle.partitions"
    )
    task_cpus: Range = Field(
        Range(min=1, max=2, step=1),
        description="Range for task.cpus"
    )

    def get_bounds_list(self) -> List[Range]:
        """
        Returns the list of bounds in the expected order.
        """
        return [
            self.driver_cores,
            self.driver_memory,
            self.executor_cores,
            self.executor_instances,
            self.executor_memory,
            self.sql_shuffle_partitions,
            self.task_cpus
        ]


# Example usage:
if __name__ == "__main__":
    # Instantiate SparkParameters with default bounds (executor_instances maximum can be adjusted according to cluster capacity)
    spark_params = SparkRangeParameters()
    bounds_list = spark_params.get_bounds_list()
    print("Spark parameter bounds:", bounds_list)

