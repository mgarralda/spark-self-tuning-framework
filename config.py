# -----------------------------------------------------------------------------
#  Project: Spark Autotuning Framework
#  File: config.py
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

from pydantic_settings import BaseSettings, SettingsConfigDict
import dotenv
dotenv.load_dotenv()

class HiBenchSparkSubmitConfig(BaseSettings):
    """Configuration settings for HiBench Spark Submit."""
    base_path_hibench: str = "/home/hibench"
    # hibench_run_all_path: str = "hibench/bin/run_all.sh"
    spark_history_server_url: str = "http://localhost:18080/api/v1/applications/"
    # Example: http://localhost:18080/api/v1/applications/app-20250519063925-0001/logs --> eventLogs-app-20250519063925-0001.zip

    # class Config:
    #     env_file = ".env"  # Especifica el archivo .env


class SSHConfig(BaseSettings):
    """Configuration settings for SSH connection."""
    hostname: str
    port: int
    username: str
    password: str
    command: str

    model_config = SettingsConfigDict(
        env_prefix='SSH_',         # match with SSH_ variables
        env_file=".env",           # specify the .env file to load
        env_file_encoding="utf-8"  # optional
    )
