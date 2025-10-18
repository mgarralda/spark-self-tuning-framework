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

from pathlib import Path
import numpy as np
from typing import List, Dict
from pydantic import BaseModel, Field, model_validator
from framework.proposed.parameters.spark_parameters import SparkRangeParameters, SparkParameters


class SparkDataScaleConfigurations(BaseModel):
    """
    This class is a placeholder for Spark configuration as well as default values.
    It can be extended to include additional configuration settings as needed.
    """
    # data_scale: Literal["small", "large", "huge"] = Field(
    data_scale: str = Field(
        default="small",
        description="Data scale for the configuration"
    )
    num_configs: int = Field(
        default=0,
        description="Number of configurations for the data scale"
    )
    configurations: List[SparkParameters] = Field(
        default_factory=list,
        description="List of Spark configurations for different data scales"
    )

    @model_validator(mode="after")
    def _set_num_configs(cls, values):
        values.num_configs = len(values.configurations)
        return values


class SparkConfigurations(BaseModel):
    """
    This class is a placeholder for Spark configuration as well as default values.
    It can be extended to include additional configuration settings as needed.
    """
    framework: str = Field(default="lhs", description="Own ID for the framework's configuration")
    config_spaces: List[SparkDataScaleConfigurations] = Field(
        default_factory=list,
        description="List of Spark configurations for different data scales"
    )

    @classmethod
    def load(cls, path):
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


class LatinHypercubeSampling:
    """
    Generates configuration samples using Latin Hypercube Sampling (LHS) while enforcing resource
    constraints defined by the Spark configuration space.

    Expected order of parameters (according to Table~\ref{tab:parallelism-config}):
      0: driver.cores         (cores)
      1: driver.memory        (GB)
      2: executor.cores       (cores)
      3: executor.instances   (instances)
      4: executor.memory      (GB)
      5: sql.shuffle.partitions (used for sampling but not subject to resource constraints)
      6: task.cpus            (cores)

    Resource constraints:
      Total cores = driver.cores + (executor.instances * executor.cores) + task.cpus
         must not exceed max_cores.
      Total memory = driver.memory + (executor.instances * executor.memory)
         must not exceed max_memory.

    This module first generates a large candidate set via LHS, then sorts the valid samples by a
    normalized resource usage score, and finally partitions them into tiers based on the provided scale mapping.
    """

    def __init__(
            self,
            spark_params: SparkRangeParameters,
            max_cores: int,
            max_memory: int,
            scale_mapping: dict,
            random_state: int = None
    ) -> None:
        """
        Args:
            spark_params (SparkRangeParameters): Instance containing bounds for the configuration space.
            max_cores (int): Maximum total cores available.
            max_memory (int): Maximum total memory (in GB) available.
            scale_mapping (dict): Mapping from input scales (e.g., {'small': 15, 'large': 25, 'huge': 30})
                                  that specifies the number of configurations for each tier.
            random_state (int, optional): Seed for reproducibility.
        """
        self.config_bounds = spark_params.get_bounds_list()  # List of RangeParameter objects.
        self.n_dimensions = len(self.config_bounds)
        self.max_cores = max_cores
        self.max_memory = max_memory
        self.scale_mapping = scale_mapping
        if random_state is not None:
            np.random.seed(random_state)

    def _generate(self, n_samples: int) -> np.ndarray:
        """
        Generates LHS samples in unit space and scales them to the configuration bounds.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of shape (n_samples, n_dimensions) with scaled configuration samples.
        """
        unit_samples = np.zeros((n_samples, self.n_dimensions))
        for dim in range(self.n_dimensions):
            # Divide [0,1) into n_samples strata and add a random offset within each stratum.
            intervals = np.linspace(0, 1, n_samples, endpoint=False)
            offsets = np.random.rand(n_samples) / n_samples
            unit_samples[:, dim] = np.random.permutation(intervals + offsets)

        scaled_samples = np.zeros_like(unit_samples)
        for i, bound in enumerate(self.config_bounds):
            min_val, max_val, step = bound.min, bound.max, bound.step
            # Scale unit values to [min_val, max_val]
            scaled = min_val + unit_samples[:, i] * (max_val - min_val)
            # Round to the nearest multiple of 'step'
            scaled = np.round(scaled / step) * step
            # Ensure values lie within bounds
            scaled_samples[:, i] = np.clip(scaled, min_val, max_val)
        return scaled_samples

    def _validate_sample(self, sample: np.ndarray) -> bool:
        """
        Validates a sample against the overall resource constraints.

        Resource constraints are:
          Total cores = driver.cores + (executor.instances * executor.cores) + task.cpus
          Total memory = driver.memory + (executor.instances * executor.memory)

        Args:
            sample (np.ndarray): A configuration sample (expected order as described above).

        Returns:
            bool: True if the sample meets resource constraints, False otherwise.
        """
        total_cores = sample[0] + (sample[3] * sample[2])
        total_memory = sample[1] + (sample[3] * sample[4])
        task_cpus = sample[6]
        executor_cores = sample[2] # The number of cores per executor (=1) has to be >= the number of cpus per task = 2.
        return (total_cores <= self.max_cores) and (total_memory <= self.max_memory) and (executor_cores >= task_cpus)

    def _compute_resource_score(self, sample: np.ndarray) -> float:
        """
        Computes a normalized resource usage score for a configuration sample.
        Lower scores indicate configurations with lower resource usage.

        Args:
            sample (np.ndarray): A configuration sample.

        Returns:
            float: The computed resource score.
        """
        total_cores = sample[0] + (sample[3] * sample[2])
        total_memory = sample[1] + (sample[3] * sample[4])
        normalized_cores = total_cores / self.max_cores
        normalized_memory = total_memory / self.max_memory
        return normalized_cores + normalized_memory

    def _compute_weighted_score(self, sample: np.ndarray) -> float:
        """
        Computes a weighted resource usage score to prioritize sampling over the executor parameters.

        Weights:
            - executor_cores, executor_instances, executor_memory, task_cpus → weight 2
            - driver_cores, driver_memory, sql_shuffle_partitions → weight 1

        Returns:
            float: Weighted resource score (lower = less resource-intensive).
        """
        weights = np.array([1, 1, 3, 3, 2, 1, 1])
        max_values = np.array([b.max for b in self.config_bounds])
        norm_sample = sample / max_values  # Normalize
        return float(np.dot(weights, norm_sample))


    def total_valid_combinations(self) -> int:
        """
        Calculates the total number of discrete combinations in the configuration space (ignoring resource constraints).

        Returns:
            int: Total number of combinations.
        """
        total = 1
        for bound in self.config_bounds:
            total *= int((bound.max - bound.min) / bound.step) + 1
        return total

    # def initialize_config_spaces(self) -> Dict[str, List[SparkParameters]]:
    def initialize_config_spaces(self) -> SparkConfigurations:
        """
        Generates a candidate set of configuration samples using LHS, filters them against resource constraints,
        sorts the valid samples by resource usage, and partitions them into tiers based on the scale mapping.

        The partitioning is as follows:
          - 'small': configurations with the lowest resource usage.
          - 'large': configurations in the middle range.
          - 'huge': configurations with the highest resource usage.

        Returns:
            Dict[str, List[SparkParameters]]: Dictionary mapping each scale ('small', 'large', 'huge')
                                                  to a list of SparkConfiguration objects.
        """
        total_required = sum(self.scale_mapping.values())
        # Generate a large candidate batch (ensure enough valid samples)
        n_candidates = max(40, int(total_required * 1.2))
        candidate_samples = self._generate(n_candidates)

        # Filter valid samples.
        valid_samples = [sample for sample in candidate_samples if self._validate_sample(sample)]
        if len(valid_samples) < total_required:
            raise ValueError("Not enough valid samples generated; consider increasing the candidate batch size.")

        # Sort valid samples by resource score (ascending: lower resource usage first).
        # valid_samples.sort(key=self._compute_resource_score)
        valid_samples.sort(key=self._compute_weighted_score)

        # Partition the sorted samples according to the scale mapping.
        small_count = self.scale_mapping.get('small', 0)
        large_count = self.scale_mapping.get('large', 0)
        # huge_count = self.scale_mapping.get('huge', 0)

        small_configs = valid_samples[:small_count]
        large_configs = valid_samples[small_count:small_count + large_count]
        # huge_configs  = valid_samples[small_count + large_count:small_count + large_count + huge_count]

        def to_spark_config(sample: np.ndarray) -> SparkParameters:
            return SparkParameters(
                driver_cores=int(sample[0]),
                driver_memory=int(sample[1]),
                executor_cores=int(sample[2]),
                executor_instances=int(sample[3]),
                executor_memory=int(sample[4]),
                sql_shuffle_partitions=int(sample[5]),
                task_cpus=int(sample[6])
            )

        def to_spark_scale_config(scale: str, configs: list) -> SparkDataScaleConfigurations:
            """
            Prepares HiBench configuration from the given file.
            This method would typically read a configuration file and set up the environment.
            For simplicity, we will just print the config file path.
            """
            return SparkDataScaleConfigurations(
                data_scale=scale,
                configurations=configs
        )

        return SparkConfigurations(
            config_spaces=[
                to_spark_scale_config('small', [to_spark_config(s) for s in small_configs]),
                to_spark_scale_config('large', [to_spark_config(s) for s in large_configs]),
                # to_spark_scale_config('huge',  [to_spark_config(s) for s in huge_configs])
            ]
        )

