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

from io import StringIO
from pathlib import Path
from typing import List, Optional, Dict, Any, NewType
import zipfile
from pydantic import BaseModel
from framework.proposed.workload_characterization.parser.feature import SparkEventLogFeatureStatistics, \
    SparkEventLogFeature
from framework.proposed.workload_characterization.workload.base import WorkloadType, InputDataSizeType
from framework.proposed.workload_characterization.workload.vector_descriptor import WorkloadCharacterized
from utils.persistence.mongo import MongoDriverConnCfg
from utils.persistence.mongo.driver import MongoDriver

MONGO_DATABASE = "spark_event_log"

Predicate = NewType("Predicate", Dict[str, Any])


class WorkloadRepository:
    """ Workload Repository """

    def __init__(
            self,
            collection: str,
            database_name: str= MONGO_DATABASE
    ) -> None:
        self.__driver_mongo = MongoDriver(
            MongoDriverConnCfg(
                database=database_name,
                collection=collection
            )
        )
        self.database_name = database_name
        self.collection = collection

    def get_characterized_workloads(
            self,
            workloads: Optional[List[WorkloadType]] = None,
            input_data_size: Optional[List[InputDataSizeType]] = None,
            include: bool = True,
            exclude_id: Optional[str] = None,
            app_benchmark_group_id: Optional[str] = None,
            exclude_errors: Optional[bool] = None
    ) -> Optional[List[WorkloadCharacterized]]:
        """
        Retrieve characterized workloads based on specified workloads and input data sizes.

        If `include` is True, filters will include the specified values.
        If `include` is False, filters will exclude the specified values.
        """

        workload_filter = {
            "app_benchmark_workload": {
                "$in" if include else "$nin": [w.value for w in workloads]
            }
        } if workloads is not None else {}

        data_size_filter = {
            "app_benchmark_data_size": {
                "$in" if include else "$nin": [i.value for i in input_data_size]
            }
        } if input_data_size is not None else {}

        if exclude_id:
            workload_id_filter = {"_id": {"$ne": exclude_id}}
        else:
            workload_id_filter = {}

        if app_benchmark_group_id:
            app_benchmark_group_id_filter = {"app_benchmark_group_id": app_benchmark_group_id}
        else:
            app_benchmark_group_id_filter = {}

        # Avoiding use workloads with execution errors.
        if exclude_errors:
            execution_error_filter = {"summary_counters.tasks_error_count": {"$eq": 0}}
        else:
            execution_error_filter = {}

        predicate = Predicate({
            **workload_filter,
            **data_size_filter,
            **workload_id_filter,
            **app_benchmark_group_id_filter,
            **execution_error_filter
        })

        results = self.__driver_mongo.read(predicate)

        if results:
            return [WorkloadCharacterized.model_validate(result) for result in results]
        else:
            print("No characterized workloads found with the specified filters.")

        return None

    def get_characterized_workload(
            self,
            workload_id: str
    ) -> Optional[WorkloadCharacterized]:
        """
        Retrieve characterized workloads based on specified workloads and input data sizes.

        If `include` is True, filters will include the specified values.
        If `include` is False, filters will exclude the specified values.
        """

        workload_filter = {"_id": workload_id}
        predicate = Predicate({**workload_filter})
        results = self.__driver_mongo.read(predicate)

        if results:
            return WorkloadCharacterized.model_validate(results[0])

        return None

    def get_optimization_metrics(
            self,
            experiment_id: Optional[str] = None
    ) -> List[dict]:
        """
        Retrieve optimization metrics from MongoDB.
        :return: List of optimization metrics.
        """

        if experiment_id:
            filter = {"experiment_id": experiment_id}
        else:
            filter = {}

        predicate = Predicate({**filter})
        results = self.__driver_mongo.read(predicate)

        return results

    def save_event_log_instances_into_mongo(
            self,
            elf: List[SparkEventLogFeature]
    ) -> None:
        # Prepare data collection before to insert into MongoDB
        # documents: List = [el.to_json(), elp.to_json()]
        documents: List = []
        for e in elf:
            documents.append(e.to_json())

        results = self.__driver_mongo.upsert_many(documents=documents)

        inserted: int = len(results)
        if inserted > 0:
            print(f"INSERT {inserted} documents into MongoDB")
        else:
            print(f"UPDATE (already exits) {len(documents)} documents into MongoDB")

    def save_statistic_vectors_into_mongo(
            self,
            vector: SparkEventLogFeatureStatistics
    ) -> None:

        documents: List = [vector.to_json()]

        # driver_mongo = MongoDriver(MongoDriverConnCfg(database="spark_event_log", collection="app_event_logs_statistics"))
        driver_mongo = MongoDriver(
            MongoDriverConnCfg(
                collection=self.collection,
                database=self.database_name
            )
        )
        results = driver_mongo.upsert_many(documents=documents)

        inserted: int = len(results)
        if inserted > 0:
            print(f"INSERT {inserted} documents into MongoDB")
        else:
            print(f"UPDATE (already exits) {len(documents)} documents into MongoDB")

    def save_optimized_workload_into_mongo(
            self,
            evaluation_opt_metrics: BaseModel,
    ) -> str:
        """
        Save the optimized workload into MongoDB.
        :param evaluation_opt_metrics: EvaluationOptimizationMetrics
        :return: None
        """
        document =  evaluation_opt_metrics.model_dump(
            mode="json",
            by_alias=True
        )
        print(f"Saving optimized workload into MongoDB: {document}")

        result = self.__driver_mongo.write(document=document)
        print(f"INSERT {result} documents into MongoDB")

        return result

    def save_event_log_into_file_system(
            seff,
            log: StringIO,
            base_processed_path: Path,
            original_app_path: Path,
            processed_app: str,
            save_processed_app: bool = True
    ) -> None:

        # Check if the base path exists, if not create it
        if not base_processed_path.exists():
            base_processed_path.mkdir(parents=True, exist_ok=True)

        # Save the log into a file in the processed folder
        if save_processed_app:
            processed_app_path = Path(base_processed_path, processed_app)
            log.seek(0)
            with processed_app_path.open(mode='w') as archive:
                archive.write(log.getvalue())

        # Move the original application file (original_app_path) to the processed folder (base_processed_path)
        destination_path = Path(base_processed_path, original_app_path.name)
        if destination_path.exists():
            destination_path.unlink()  # Eliminar el archivo existente
        original_app_path.rename(destination_path)

    def get_application_from_file_system(
            self,
            path: Path
    ) -> StringIO:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                # Extraer el primer archivo del ZIP
                file_name = zip_ref.namelist()[0]
                with zip_ref.open(file_name) as file:
                    log = file.read().decode('utf-8')
        else:
            with path.open(mode="r") as file:
                log = file.read()

        return StringIO(log)

