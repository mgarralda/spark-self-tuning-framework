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

# -*- coding: utf-8 -*-

"""Module documentation goes here
   and here
   and ...
"""

from dataclasses import dataclass
from enum import Enum


class PredicateType(Enum):
    ONE = "one"
    MANY = "many"


@dataclass
class MongoDriverConnCfg:
    database: str
    host: str = "localhost"
    port: int = 27017
    collection: str = None
    user: str = None
    password: str = None
    replica_set: str = None
    auth_source: str = "admin"
    auth_mechanism: str = "SCRAM-SHA-256"
    ssl: bool = False

    def get_uri(self) -> str:
        """
        :return:  Example
        "mongodb://<userName>:<password>@127.0.0.1:27017/?authSource=admin"
        "mongodb://127.0.0.1:27017/database.collection
        """

        if not self.database or not self.host:
            raise Exception(__class__, "Database or host name are not specified")

        # Adding host, port and database to the uri
        uri = "{0}:{1}/{2}".format(self.host, str(self.port), self.database)

        if self.user and self.password:
            # Adding user, pass and auth database to uri
            ssl = "true" if self.ssl else "false"
            uri = f"mongodb://{self.user}:{self.password}@{uri}?authSource={self.auth_source}&ssl={ssl}"
        else:
            uri = f"mongodb://{uri}"

        return uri