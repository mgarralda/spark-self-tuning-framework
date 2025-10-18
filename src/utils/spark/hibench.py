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

import asyncio
import re
from typing import Optional
import requests
from pathlib import Path
from config import HiBenchSparkSubmitConfig, SSHConfig
from framework.experimental.dataset.latin_hypercube_sampling import SparkConfigurations
from framework.proposed.parameters import SparkParameters
from framework.proposed.workload_characterization.pipeline import run_characterization
from framework.proposed.workload_characterization.workload import WorkloadRepository, WorkloadCharacterized
from utils.ssh import connect_via_async_ssh


class HiBenchSparkSubmit:
    """ SparkSubmit class to submit Spark jobs using HiBench structure """

    def __init__(
            self,
            ssh_config: SSHConfig,
            settings: HiBenchSparkSubmitConfig,
            queue: asyncio.Queue = None
    ):
        # Initialize from config settings
        self.settings = settings
        self.ssh_config = ssh_config
        self.queue = queue

        # Cretea a asyncio background task to monitor the queue

        # self.spark_home = spark_home
        # self.app_name = app_name
        # self.master = master

    async def run_lhs_batch(
            self,
            spark_configs: SparkConfigurations,
            *args
    ) -> None:
        # pas the files of the Hibench
        """ Submit a Latin Hypercube Sampling (LHS) batch of Spark jobs using HiBench framework """

        app_id_pattern = re.compile(r"Submitted application application_\d+_\d+")
        # app_status_pattern = re.compile(r"final status: \w+")
        app_status_failed_pattern = "final status: FAILED"
        # app_finished_pattern = "final status: SUCCEEDED"
        app_finished_pattern = "Shutdown hook called"
        app_status_failed = False

        # final status: UNDEFINED
        # final status: FAILED
        # final status: SUCCEEDED

        # This is a custom added to the stdio from our custom hibench
        #app_finished_pattern = "Extracted application ID:"

        # Main iteration over the configurations
        for config_scales in spark_configs.config_spaces: # SparkDataScaleConfigurations
            # print(f"{type(config_scales)=} | {config_scales=}")
            # if isinstance(config_scales, SparkDataScaleConfigurations):
            data_scale = config_scales.data_scale

            # First, prepare the data scale for workloads
            self._prepare_hibench_data(
                data_scale=data_scale,
                run_prepare_data=True
            )

            #result = subprocess.run(run_all, capture_output=True, text=True)
            # application_id = None
            print(f"{'='*50}\nPreparing data workloads with {data_scale} data\n{'='*50}")
            async for line in connect_via_async_ssh( self.ssh_config):
                # todo: We can check when a data prepare workload is finished: having into account that it could be a
                # mapreduce or spark work.
                # pass
                if "RUNNING" in line:
                    continue
                print(line, end="")
            #
            # with SSHClientManager(self.ssh_config) as client:
            #     # print(f"Running workload prepare data: {ssh_config.command=}")
            #     stdin, stdout, stderr = client.exec_command(ssh_config.command)
            #     output = stdout.read().decode('utf-8')
            #     # error = stderr.read().decode('utf-8')  # Leer y decodificar la salida de error
            #
            #     # if error:
            #     #     print(f"Error: {error}")
            #     # else:
            #     #     print(f"Output: {output}")
            print(f"{'='*125}\nFINISHED PREPARE DATA\n{'='*125}")

           # Block prepare data, so only focus on the execution of the workloads
            self._prepare_hibench_data(
                data_scale=data_scale,
                run_prepare_data=False
            )

            for config in config_scales.configurations: # SparkParameters
                self._prepare_hibench_config(
                    data_scale=data_scale,
                    group_id=spark_configs.framework,
                    # group_id="group_id_test",
                    config=config
                )

                # result = subprocess.Popen(run_all, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(f"{'='*50}\nRunning workloads with config:\n{config}\n{'='*50}")
                #result = subprocess.run(run_all, capture_output=True, text=True)
                application_id = None

                # todo: how to control error in the application due to the configuration?
                async for line in connect_via_async_ssh(self.ssh_config):
                    if "RUNNING" in line:
                        continue
                    print(line, end="")
                    if application_id is None:
                        match = app_id_pattern.search(line)
                        if match:
                            application_id = match.group()
                            application_id = application_id.replace("Submitted application ", "")
                            print(f"\n{'='*65}\nDetected {application_id}\n{'='*65}\n", flush=True)
                    else:
                        if app_status_failed_pattern in line:
                            print(f"\n{'='*65}\nFailed {application_id} with {data_scale} data\n{'='*65}\n", flush=True)
                            app_status_failed = True

                        elif app_finished_pattern in line:
                            if not app_status_failed:
                                await self.queue.put(application_id)
                                print(f"\n{'='*65}\nFinished {application_id} with {data_scale} data and queued\n{'='*65}\n", flush=True)

                             # using some strategy to queue that we can characterize the workload and save into mongoDB
                            application_id = None
                            app_status_failed = False

                # with SSHClientManager(self.ssh_config ) as client:
                #     stdin, stdout, stderr = client.exec_command(ssh_config.command)
                #     for line in iter(stdout.readline, ""):
                #         print(line, end="")  # Imprime cada línea en tiempo real
                #         if application_id is None:
                #             match = app_id_pattern.search(line)
                #             if match:
                #                 application_id = match.group()
                #                 print(f"\nDetected: {application_id} *********************************\n")
                #         else:
                #             if app_finished_pattern in line:
                #                 await self.queue.put(application_id)
                #                 print(f"\n{'='*65}\nFinished {application_id} with {data_scale} data and queued\n{'='*65}\n", flush=True)
                #
                #                 # using some strategy to queue that we can characterize the workload and save into mongoDB
                #                 application_id = None

        print(f"{'='*125}\nFINISHED RUNNING WORKLOADS\n{'='*125}")

    async def run_once(
            self,
            data_scale: str,
            framework: str,
            config: SparkParameters
    ) -> Optional[str]:
        """
        Run a single Spark job with the given configuration.
        :return : The application ID of the submitted Spark job.
        """

        app_id_pattern = re.compile(r"Submitted application application_\d+_\d+")
        # app_status_pattern = re.compile(r"final status: \w+")
        app_status_failed_pattern = "final status: FAILED"
        # app_finished_pattern = "final status: SUCCEEDED"
        app_finished_pattern = "Shutdown hook called"
        app_status_failed = False
        application_id = None


        # First, if so not,  block the prepare the data scale for workloads.
        self._prepare_hibench_data(
            data_scale=data_scale,
            run_prepare_data=False
        )

        # #result = subprocess.run(run_all, capture_output=True, text=True)
        # # application_id = None
        # print(f"{'='*50}\nPreparing data workloads with {data_scale} data\n{'='*50}")
        # async for line in connect_via_async_ssh( self.ssh_config):
        #     # todo: We can check when a data prepare workload is finished: having into account that it could be a
        #     # mapreduce or spark work.
        #     # pass
        #     if "RUNNING" in line:
        #         continue
        #     print(line, end="")
        # #
        # # with SSHClientManager(self.ssh_config) as client:
        # #     # print(f"Running workload prepare data: {ssh_config.command=}")
        # #     stdin, stdout, stderr = client.exec_command(ssh_config.command)
        # #     output = stdout.read().decode('utf-8')
        # #     # error = stderr.read().decode('utf-8')  # Leer y decodificar la salida de error
        # #
        # #     # if error:
        # #     #     print(f"Error: {error}")
        # #     # else:
        # #     #     print(f"Output: {output}")
        # print(f"{'='*125}\nFINISHED PREPARE DATA\n{'='*125}")


        self._prepare_hibench_config(
            data_scale=data_scale,
            group_id=framework,
            config=config
        )

        # todo: how to control error in the application due to the configuration?
        async for line in connect_via_async_ssh(self.ssh_config ):
            if "RUNNING" in line:
                continue
            print(line, end="")
            if application_id is None:
                match = app_id_pattern.search(line)
                if match:
                    application_id = match.group()
                    application_id = application_id.replace("Submitted application ", "")
                    print(f"\n{'='*65}\nDetected {application_id}\n{'='*65}\n", flush=True)
            else:
                if app_status_failed_pattern in line:
                    print(f"\n{'='*65}\nFailed {application_id} with {data_scale} data\n{'='*65}\n", flush=True)
                    app_status_failed = True

                elif app_finished_pattern in line:
                    if not app_status_failed:
                        # await self.queue.put(application_id)
                        print(f"\n{'='*65}\nFinished {application_id} with {data_scale} data and queued\n{'='*65}\n", flush=True)

                    # # using some strategy to queue that we can characterize the workload and save into mongoDB
                    # application_id = None
                    # app_status_failed = False

        return application_id

    def _prepare_spark_submit_command(
            self,
            scale_data: str,
            config: SparkParameters
    ) -> list:
        """ Prepare the spark-submit command """

        # # Prepare the command
        # command = [
        #     f"{self.spark_home}/bin/spark-submit",
        #     "--name", self.app_name,
        #     "--master", self.master,
        #     "--conf", f"spark.driver.cores={config.driver_cores}",
        #     "--conf", f"spark.driver.memory={config.driver_memory}g",
        #     "--conf", f"spark.executor.cores={config.executor_cores}",
        #     "--conf", f"spark.executor.instances={config.executor_instances}",
        #     "--conf", f"spark.executor.memory={config.executor_memory}g",
        #     "--conf", f"spark.sql.shuffle.partitions={config.sql_shuffle_partitions}",
        #     "--conf", f"spark.task.cpus={config.task_cpus}",
        #     "--conf", f"spark.app.benchmark.group.id=own",
        #     "--conf", f"spark.app.benchmark.workload={self.app_name}",
        #     "--conf", f"spark.app.benchmark.data.size={scale_data}",

        # command = [
        #     f"{self.spark_home}/bin/spark-submit",
        #     "--name", self.app_name,
        #     "--master", self.master,
        #     app_file,
        # ] + list(args)
        # print("Running command:", " ".join(command))
        # result = subprocess.run(command, capture_output=True, text=True)
        # if result.returncode != 0:
        #     raise Exception(f"Spark job failed: {result.stderr}")
        # return result.stdout
        # ]
        # return command
        pass

    def _prepare_hibench_config(
            self,
            data_scale: str,
            group_id: str,
            config: SparkParameters
    ) -> None:
        """
        Actualiza las propiedades específicas en el archivo hibench.conf sin modificar el resto.

        :param file_path: Ruta al archivo hibench.conf.
        :param scale_profile: Nuevo valor para hibench.scale.profile.
        :param map_parallelism: Nuevo valor para hibench.default.map.parallelism.
        :param shuffle_parallelism: Nuevo valor para hibench.default.shuffle.parallelism.
        """
        hibench_conf_file = Path(self.settings.base_path_hibench, "conf/hibench.conf")
        # Leer el contenido del archivo
        with hibench_conf_file.open("r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if not line.startswith("#") and line.strip():  # Ignorar líneas comentadas y líneas vacías
                match line.split()[0]:  # Compara el inicio de cada línea
                    # case "hibench.scale.profile":
                    #     lines[i] = f"hibench.scale.profile                 {data_scale}\n"
                    case "hibench.default.map.parallelism":
                        lines[i] = f"hibench.default.map.parallelism       {config.executor_cores * config.executor_instances * 2}\n"
                    case "hibench.default.shuffle.parallelism":
                        lines[i] = f"hibench.default.shuffle.parallelism   {config.sql_shuffle_partitions}\n"

        # Escribir los cambios de vuelta al archivo
        with hibench_conf_file.open("w", newline="\n") as file:
            file.writelines(lines)

        spark_conf_file= Path(self.settings.base_path_hibench, "conf/spark.conf")
        # Leer el contenido del archivo
        with spark_conf_file.open("r") as file:
            lines = file.readlines()

        # Modificar solo las líneas específicas
        for i, line in enumerate(lines):
            if not line.startswith("#") and line.strip():  # Ignorar líneas comentadas y líneas vacías
                # print(f"line: {line}")
                match line.split()[0]:
                    case "hibench.yarn.executor.num":
                        lines[i] = f"hibench.yarn.executor.num     {config.executor_instances}\n"
                    case "hibench.yarn.executor.cores":
                        lines[i] = f"hibench.yarn.executor.cores   {config.executor_cores}\n"
                    case "spark.executor.memory":
                        lines[i] = f"spark.executor.memory         {config.executor_memory}g\n"
                    case "spark.driver.memory":
                        lines[i] = f"spark.driver.memory           {config.driver_memory}g\n"
                    case "spark.driver.cores":
                        lines[i] = f"spark.driver.cores            {config.driver_cores}\n"
                    case "spark.task.cpus":
                        lines[i] = f"spark.task.cpus               {config.task_cpus}\n"
                    case "spark.app.benchmark.group.id":
                        lines[i] = f"spark.app.benchmark.group.id  {group_id}\n"

        # Escribir los cambios de vuelta al archivo
        with spark_conf_file.open("w",  newline="\n") as file:
            file.writelines(lines)

    def _prepare_hibench_data(
            self,
            data_scale: str,
            run_prepare_data: bool = True,
    ) -> None:

        """
        Prepares the data for HiBench workloads.
        This function is a placeholder and should be implemented based on your requirements.
        """
        hibench_conf_file = Path(self.settings.base_path_hibench, "conf/hibench.conf")

        # Leer el contenido del archivo
        with hibench_conf_file.open("r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if not line.startswith("#") and line.strip():  # Ignorar líneas comentadas y líneas vacías
                # print(f"line: {line}")
                match line.split()[0]:  # Compara el inicio de cada línea
                    case "hibench.scale.profile":
                        lines[i] = f"hibench.scale.profile                 {data_scale}\n"
                        # print(f"Updated line: {lines[i]}")

        # Escribir los cambios de vuelta al archivo
        with hibench_conf_file.open("w", newline="\n") as file:
            file.writelines(lines)

        benchmarks_conf_file= Path(self.settings.base_path_hibench, "conf/benchmarks.lst")

        with benchmarks_conf_file.open("r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if run_prepare_data:
                if "skip.prepare.data" in line.strip():
                    lines[i] = f"#skip.prepare.data\n"
                elif "skip.workloads.run" in line.strip():
                    lines[i] = f"skip.workloads.run\n"
            else:
                # Descomentar la línea de preparación de datos
                if "skip.prepare.data" in line.strip():
                    lines[i] = f"skip.prepare.data\n"
                elif "skip.workloads.run" in line.strip():
                    lines[i] = f"#skip.workloads.run\n"

        # Escribir los cambios de vuelta al archivo
        with benchmarks_conf_file.open("w",  newline="\n") as file:
            file.writelines(lines)


class HiBenchWorkloadCharacterization:
    """
    This class is responsible for characterizing a workload and generating a parser representation.
    It uses the HiBench framework to run the workloads and collect the necessary data.
    """

    def __init__(
            self,
            settings: HiBenchSparkSubmitConfig,
            queue: asyncio.Queue = None
    ) -> None:
        self.settings = settings
        self.queue = queue
        self.stop_queue = False

    async def run_lhs_batch(
            self,
            characterization: bool,
            config: dict
    ) -> None:
        """ Run the workload characterization Latin of Hypercube Sampling (LHS) batch batch process."""
        time_out = 10
        retry_list = []
        retried = 0

        while not self.stop_queue:
            try:
                if retry_list:
                    if   retried > 3:
                        # We obviate the element from the retry list
                        retry_list.pop(0)
                        application_id = None
                        retried = 0
                    else:
                        # Prioritize retry items
                        application_id = retry_list.pop(0)
                        print(f"Got retry application: {application_id}", flush=True)

                else:
                    # Wait for new application_id from the shared queue
                    application_id = await asyncio.wait_for(self.queue.get(), timeout=time_out)

                # print(f"{'='*50}\nDownloading logs of application ID: {application_id}\n{'='*50}", flush=True)

                try:
                    if application_id:
                        # Download the Spark event logs as a ZIP file
                        spark_event_log_path = Path(self.settings.base_path_hibench, "logs", f"{application_id}.zip")
                        await self._download_spark_config_log(
                            application_id=application_id,
                            to_path=spark_event_log_path
                        )
                        retried = 0
                        print(f"{'='*65}\nDownloaded {application_id}.\nWe proceed to vectorization the application ...\n{'='*65}", flush=True)

                        # Here we can call the characterization method
                        if characterization:
                            run_characterization(
                                spark_event_log_path,
                                config=config
                            )

                            print(f"{'='*65}\nFinished application vectorization\n{'='*65}", flush=True)

                except Exception as e:
                    retried += 1
                    print(f"Failed to download Spark logs, we'll retry in next iteration {retried}: {e}")
                    retry_list.append(application_id)  # Retry on next iteration
                    await asyncio.sleep(time_out * 2)  # Wait before retrying


            except asyncio.TimeoutError:
                # print(f"{'='*50}\nQueue is still empty... waiting...\n{'='*50}", flush=True)
                pass

        # After running the workloads, we can characterize them
        # parser = await self.characterize()
        # await self._save_vector(parser)

    async def run_once(
            self,
            application_id: str,
            config: dict
    ) -> str:

        """
        Run a single workload characterization for the given application ID.
        :param application_id: The Spark application ID (e.g., 'application_1234567890123_0001')
        :param characterization: Whether to run the characterization process
        """

        time_out = 10
        for attempt in range(1, 4):
            try:
                spark_event_log_path = Path(self.settings.base_path_hibench, "logs", f"{application_id}.zip")
                await self._download_spark_config_log(
                    application_id=application_id,
                    to_path=spark_event_log_path
                )

                application_id =  run_characterization(
                    spark_event_log_path,
                    config
                )
                print(f"Finished application vectorization for {application_id}")
                return application_id

            except Exception as e:
                print(f"Failed to download Spark logs, we'll retry in next iteration {attempt}: {e}")
                await asyncio.sleep(time_out * 2)  # Wait before retrying
                if attempt > 3:
                    raise Exception(f"Failed to download Spark logs after {attempt} attempts, skipping application {application_id}")

    async def _save_vector(self, vector):
        # Placeholder for saving the parser to a database or file
        # This should save the generated parser representation of the workload
        pass

    async def _download_spark_config_log(
            self,
            application_id: str,
            to_path: Path
    ) -> None:
        """
        Download the Spark event logs as a ZIP file for the given application ID.

        :param application_id: The Spark application ID (e.g., 'application_1234567890123_0001')
        :param path: The local file path where the ZIP file will be saved
        """
        # Define the base URL of your Spark History Server

        # Construct the URL to download the event logs
        log_url = f"{self.settings.spark_history_server_url}{application_id}/logs"

        # for attempt in range(1, 4):  # Retry up to 3 times
        # try:
            # Send a GET request to download the ZIP file
        response = requests.get(log_url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Write the content to the specified local path
        with open(to_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # print(f"Successfully downloaded Spark logs to: {to_path}")
        #

        # except requests.exceptions.RequestException as e:
        #     print(f"Failed to download Spark logs in attempt {attempt}: {e}")
            # await asyncio.sleep(10)
            # print(f"Sleeping for 10 seconds before retrying...")


async def run_lhs_hibench_batch(
        spark_config_json_path: Path,
        characterization: bool,
        config: dict
) -> None:

    # Global queue for managing finished application IDs
    # global shared_queue  # Declarar que se usará la variable global
    shared_queue = asyncio.Queue()

    settings = HiBenchSparkSubmitConfig(); print(settings)
    ssh_config = SSHConfig(); print(ssh_config)

    spark_submit = HiBenchSparkSubmit(ssh_config, settings, shared_queue)
    spark_vector = HiBenchWorkloadCharacterization(settings, shared_queue)

    spark_configs = SparkConfigurations.load(spark_config_json_path)

    # Concurrently run the Spark HiBench workload submits and workload characterization
    await asyncio.gather(
        spark_submit.run_lhs_batch(spark_configs),
        spark_vector.run_lhs_batch(characterization, config)
    )


async def run_once_workload_hibench(
        data_scale: str,
        framework: str,
        parameters: SparkParameters,
        config: dict
) -> Optional[WorkloadCharacterized]:
    """
    Run a single workload characterization for the given application ID.
    :param application_id: The Spark application ID (e.g., 'application_1234567890123_0001')
    :param characterization: Whether to run the characterization process
    """

    settings = HiBenchSparkSubmitConfig(); print(settings)
    ssh_config = SSHConfig(); print(ssh_config)

    # Submit the execution of the Spark job using HiBench framework
    spark_submit = HiBenchSparkSubmit(ssh_config, settings)
    print(f"Running HiBench Spark job with data scale: {data_scale}, framework: {framework}, config: {parameters}")
    application_id =  await spark_submit.run_once(data_scale, framework, parameters)

    # Characterize the workload and save into mongoDB
    spark_vector = HiBenchWorkloadCharacterization(settings)
    print(f"Characterizing workload for application ID: {application_id}")
    application_id = await spark_vector.run_once(
        application_id,
        config
    )

    # Get the application ID from the repository to get the features
    repo = WorkloadRepository(
        # database_name=config.get("collection_historical_dataset"),
        collection=config.get("collection_historical_dataset")
    )

    workload_characterized: WorkloadCharacterized = repo.get_characterized_workload(application_id)

    return workload_characterized