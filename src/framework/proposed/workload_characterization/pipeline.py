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

# -*- coding: utf-8 -*-

import time
import traceback
from io import StringIO
from pathlib import Path
from typing import List
import pandas as pd
from framework.proposed.workload_characterization.parser.eventlog import SparkEventLog
from framework.proposed.workload_characterization.parser.eventlog_restapi import SparkEventLogRestApi
from framework.proposed.workload_characterization.parser.feature import (
    SparkEventLogFeature,
    SparkEventLogFeatureStatistics, PolynomialRegressionErrorType
)
from framework.proposed.workload_characterization.parser.eventlog_instrumentation import SparkEventLogInstrumentation
from framework.proposed.workload_characterization.parser.feature_engineering import FeatureEngineering
from framework.proposed.workload_characterization.parser.profile import SparkEventLogProfile
from framework.proposed.workload_characterization.workload import WorkloadRepository
import warnings

# Ignorar todas las advertencias
warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")
# todo
# change counter = counter[0] by counter = counter.iloc[0]

# Ignorar todas las advertencias
warnings.filterwarnings("ignore")
# pd.set_option('max_columns', None)
pd.set_option('display.max_columns', None)

def get_files_in_folder(folder_path):
    """
    Get a list of all files in a folder.

    Args:
        folder_path (str or Path): The path to the folder.

    Returns:
        list: A list of Path objects representing files in the folder.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not a valid directory path.")

    # Use a list comprehension to get all files in the folder
    files = [file for file in folder_path.iterdir() if file.is_file()]

    return files


def get_applications_from_spark_api(min_data: str, max_date: str) -> List:
    api = SparkEventLogRestApi()

    # todo: Control so far app getting: filter
    return api.get_applications(min_data, max_date)


def get_applications_from_file_system(path: Path) -> List:
    return [log for log in path.iterdir() if log.is_file()]


def get_application_from_spark_api(id: str) -> StringIO:
    api = SparkEventLogRestApi()

    # todo: Control so far app getting: filter
    return api.get_application(id)


def get_application_from_file_system_(path: Path) -> StringIO:
    with path.open(mode="r") as file:
        log = file.read()

    return StringIO(log)


def create_event_log_profile_instance(log: StringIO) -> SparkEventLogProfile:
    return SparkEventLogInstrumentation(log).get_application_profiled()


def create_event_log_instance(profile: SparkEventLogProfile, log: StringIO) -> SparkEventLog:
    return SparkEventLog(
        _id=profile._id,
        app_name=profile.app_name,
        date_started=profile.date_started,
        app_benchmark_workload=profile.app_benchmark_workload,
        app_benchmark_group_id=profile.app_benchmark_group_id,
        app_benchmark_data_size=profile.app_benchmark_data_size,
        event_log=log.getvalue(),
        data_generator=profile.data_generator,
        time_execution=profile.time_execution
    )


def create_event_log_feature_instance() -> SparkEventLogFeature:
    # path = Path(PATH_LOG, NAME_LOG)
    #
    # spark_instrumentation = SparkEventLogInstrumentation(path)
    # profile = spark_instrumentation.get_application_profiled()
    # fe = FeatureEngineering(profile, transformation)
    pass


def run_characterization(
        spark_event_log_path: Path,
        config: dict
) -> str:
    """
    Main Pipeline in order to get log, parser, create features
    :return:
    """

    try:
        print(f"{'='*100}\nProcessing:\n{spark_event_log_path}\n{'='*100}", flush=True)
        repo = WorkloadRepository(
            # database_name=config.get("database_name"),
            collection=config.get("collection_historical_dataset")
        )

        log = repo.get_application_from_file_system(spark_event_log_path)

        ################################################################################################################
        # First action: parsing the spark event log
        ################################################################################################################
        start_time = time.time()
        profile = create_event_log_profile_instance(log)
        time_profile = time.time()-start_time
        # event_log = create_event_log_instance(profile, log)
        # features: List[SparkEventLogFeature] = []
        print(f"{'='*100}\n1) Elapsed time to create profile of the sparkeventlog \n{time.time()-start_time:2f}\n{'='*100}", flush=True)

        ################################################################################################################
        # Processing the statistics parser characterization for baseline comparisons
        ################################################################################################################
        # Init: To get statistics parser (compare with the paper)
        tasks_by_stage = [stage.tasks_count for stage in profile.stages.list]
        statistics_vector = SparkEventLogFeatureStatistics(
            _id = profile._id,
            app_name=profile.app_benchmark_workload,
            app_benchmark_data_size=profile.app_benchmark_data_size,
            vector_metrics_yoro=profile.tasks.counters_metrics.get_statistics_vector_metrics(tasks_by_stage)
        )
        repo.save_statistic_vectors_into_mongo(statistics_vector)
        print(f"{'='*100}\n2) Elapsed time to create statistic vectors \n{time.time()-start_time:2f}\n{'='*100}", flush=True)

        ################################################################################################################
        # Processing the parser characterization for our framework
        ################################################################################################################
        start_time = time.time()
        feature = FeatureEngineering(
            profile,
            PolynomialRegressionErrorType.MAE
        ).get_feature()
        # features.append(feature)
        repo.save_event_log_instances_into_mongo([feature])
        repo.save_event_log_into_file_system(
            log,
            base_processed_path=Path(spark_event_log_path.parent, "processed"),
            original_app_path=spark_event_log_path,
            processed_app=f"{profile._id}_{profile.app_benchmark_workload}_{profile.app_benchmark_data_size}",
            save_processed_app=False
        )
        print(f"{'='*100}\n3) Elapsed time to create Workload characterization\n{(time.time()-start_time) + time_profile:2f}\n{'='*100}", flush=True)
        return feature._id

    except Exception as e:
        print(e)
        print(traceback.format_exc())
