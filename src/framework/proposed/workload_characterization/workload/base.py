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

from enum import Enum
from typing import List, NewType, Tuple, Optional
import uuid
from abc import ABC
from datetime import datetime
from typing import TypeVar
from pydantic import BaseModel, Field


class BaseEntity(BaseModel, ABC):
    # id: str = Field(default_factory=lambda: str(uuid.uuid1()), alias='_id', exclude=True)
    id: str = Field(default_factory=lambda: str(uuid.uuid1()), alias='_id')
    time_stamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        # This config allows bidirectional work with the name of the fields but using the alias as well
        populate_by_name = True

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__.lower()


Entity = TypeVar('Entity', bound=BaseEntity)
# T = TypeVar("T")
# T_ru = TypeVar("T_ru", bound='RuleService')
# T_co = TypeVar('T_co', covariant=True)


RegressionData = NewType(
    "RegressionData",
    Tuple[
        List[List[float]],
        List[List[float]],
        List[List[float]],
        List[List[float]],
        List[int],
        List[str],
        List[float],
        List[float],
        List[List[float]],
    ]
)


class WorkloadCharacterizationEnum(Enum):
    @classmethod
    def to_list(cls) -> List[str]:
        return [e.value for e in cls]


class WorkloadType(WorkloadCharacterizationEnum):
    SLEEP = "sleep"
    SORT = "sort"
    TERASORT = "terasort"
    WORDCOUNT = "wordcount"
    REPARTITION = "repartition"
    AGGREGATION = "aggregation"
    JOIN = "join"
    SCAN = "scan"
    BAYES = "bayes"
    KMEANS = "kmeans"
    LR = "lr"
    ALS = "als"
    PCA = "pca"
    GBT = "gbt"
    RF = "rf"
    SVD = "svd"
    LINEAR = "linear"
    LDA = "lda"
    SVM = "svm"
    GMM = "gmm"
    CORRELATION = "correlation"
    SUMMARIZER = "summarizer"
    PAGERANK = "pagerank"
    NWEIGHT = "nweight"


class InputDataSizeType(WorkloadCharacterizationEnum):
    TINY = "tiny"
    SMALL = "small"
    LARGE = "large"
    HUGE = "huge"


class WorkloadEntity(BaseEntity):
    class Environment(BaseModel):
        # Note: Some papers use these parameters to characterize the workloads and others not
        driver_cores: int
        driver_memory_gb: int
        dynamic_allocation: bool
        executor_cores: int
        executor_instances: int
        executor_memory_gb: int
        sql_adaptive: bool
        sql_shuffle_partitions: int
        task_cpus: int

        def to_vector(self) -> List[int]:
            """
            Convert the workload to a vector representation.
            :return: List of float values representing the workload.
            """
            return [
                self.driver_cores,
                self.driver_memory_gb,
                self.executor_cores,
                self.executor_instances,
                self.executor_memory_gb,
                self.sql_shuffle_partitions,
                self.task_cpus,
            ]

    _id: str
    app_name: str
    app_benchmark_workload: str
    time_execution: int | float
    dataset_size: float = 0.0 #Input Metrics: Bytes Read accross all tasks
    app_benchmark_data_size: InputDataSizeType
    environment: Environment
    time_resources: Optional[float] = None

    @classmethod
    def get_data_for_regression(
            cls,
            workloads: List["WorkloadEntity"]
    ) -> RegressionData:
        # todo work with numphy
        def calculate_resource_usage(cs: List[float]) -> int:
            """
            Reflects multiplicative cost per executor.
            Aligns better with cost in cloud environments or sustainability scenarios (e.g., GB×core-hours).
            """
            # vector =
            #      driver_cores,
            #      driver_memory_gb,
            #      executor_cores,
            #      executor_instances,
            #      executor_memory_gb,
            #      sql_shuffle_partitions,
            #      task_cpus,
            #  ]

            return int(
                cs[0] * cs[1] +
                cs[3] * cs[2] * cs[4]
            )

        """
        Split data for regression problem
        :param version:
        :param workloads:
        :return: workload_characterization_features, workload_setting_features, workload_time_execution, workload_names
        """
        workload_characterization_features_v1 = [workload.get_garralda_vector_descriptor(adding_data_size=False) for workload in workloads]
        # workload_characterization_features_v1 = [workload.get_garralda_vector_descriptor_17(adding_data_size=False) for workload in workloads]
        workload_characterization_features_v2 = [workload.get_yoro_vector_descriptor(adding_data_size=False) for workload in workloads]

        # workload_setting_features = [
        #     [
        #         workload.environment.driver_cores,
        #         workload.environment.driver_memory_gb,
        #         workload.environment.executor_cores,
        #         workload.environment.executor_memory_gb,
        #         workload.environment.executor_instances,
        #         workload.environment.sql_shuffle_partitions,
        #         workload.environment.task_cpus,
        #     ]
        #     for workload in workloads
        # ]
        #
        # workload_setting_features_v2 = [
        #     features + [workload.dataset_size]
        #     for features, workload in zip(workload_setting_features, workloads)
        # ]

        workload_setting_features = []
        workload_setting_features_v2 = []

        for workload in workloads:
            features = [
                workload.environment.driver_cores,
                workload.environment.driver_memory_gb,
                workload.environment.executor_cores,
                workload.environment.executor_instances,
                workload.environment.executor_memory_gb,
                workload.environment.sql_shuffle_partitions,
                workload.environment.task_cpus,
            ]
            workload_setting_features.append(features)
            workload_setting_features_v2.append(features + [workload.dataset_size])

        # Following the YORO paper: features is the concatenation of spark log features, dataset size and the configuration settings
        workload_characterization_extended_features_v2 = [workload.get_yoro_vector_descriptor(adding_data_size=True) for workload in workloads]
        workload_characterization_extended_features_v2 = [sublist1 + sublist2 for sublist1, sublist2 in
                             zip(workload_characterization_extended_features_v2, workload_setting_features)]

        workload_names = [workload.app_benchmark_workload for workload in workloads]
        workload_time_execution = [workload.time_execution for workload in workloads]
        workload_input_data_sizes = [workload.dataset_size for workload in workloads]
        workload_resources = [calculate_resource_usage(workload.environment.to_vector()) for workload in workloads]
        workload_configuratin_shapes = [workload.to_configuration_shape_vector() for workload in workloads]

        # workload_time_resources = [
        #     OptimizationObjective.objective_function(
        #         T=workload.time_execution,
        #         R=OptimizationObjective.calculate_resource_usage(
        #             SparkParameters.from_vector(
        #                 workload.environment.to_vector()
        #             )
        #         ),
        #         beta=beta_cost_function
        #     )
        #     for workload in workloads
        # ]

        return RegressionData((
            workload_characterization_features_v1,
            workload_characterization_features_v2,
            workload_characterization_extended_features_v2,
            workload_setting_features,
            workload_setting_features_v2, # to test in Garralda
            workload_time_execution,
            workload_names,
            workload_input_data_sizes,
            workload_resources,
            workload_configuratin_shapes
        ))

        # return RegressionData((
        #     np.array(workload_characterization_features),
        #     np.array(workload_characterization_extended_features),
        #     np.array(workload_setting_features),
        #     np.array(combined_features),
        #     np.array(workload_names),
        #     np.array(workload_time_execution),
        # ))
