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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


@dataclass
class DriverConnection(ABC):
    host: str
    port: int


class Persistence(ABC):
    """
    Interface - Create, Read, Update, Delete
    """

    PLATFORM_PATTERN_PREFIX_COLUMNS = ("contextData")
    # PLATFORM_PATTERN_PREFIX_COLUMNS = ("_id", "contextData")
    # _id: struct (nullable = true)
    # |    |-- oid: string (nullable = true)
    # contextData: struct (nullable = true)
    # |    |-- deviceTemplate: string (nullable = true)
    # |    |-- device: string (nullable = true)
    # |    |-- clientConnection: string (nullable = true)
    # |    |-- clientSession: string (nullable = true)
    # |    |-- user: string (nullable = true)
    # |    |-- timezoneId: string (nullable = true)
    # |    |-- timestamp: string (nullable = true)
    # |    |-- timestampMillis: long (nullable = true)
    # |    |-- source: string (nullable = true)

    # def __init__(self, conn: str):
    def __init__(self):
        # self._conn = conn
        super().__init__()

    @abstractmethod
    # def read(self, predicate: str): pass
    def read(self, **kwargs): pass

    @abstractmethod
    def write(self, **kwargs): pass

    @abstractmethod
    def update(self, **kwargs): pass

    @abstractmethod
    def delete(self, **kwargs): pass

    @abstractmethod
    def get_underlying_driver(self, **kwargs): pass


# class FactoryPersistence:
#     """
#     The Abstract Factory Repository"
#     """
#
#     @staticmethod
#     def builder(repository: Persistence):
#     # def builder(repository: IPersistence) -> IPersistence:
#         try:
#             # @singletown patter??
#             # Create connection object before
#             return repository
#             # if factory in ['aa', 'ab', 'ac']:
#             #     return FactoryA().create_object(factory[1])
#             # if factory in ['ba', 'bb', 'bc']:
#             #     return FactoryB().create_object(factory[1])
#             # raise Exception('No Factory Found')
#         except Exception as _e:
#             # Exception('No Factory Found')
#             print(_e)
#         return None








# import importlib
# def str_to_class(module_name, class_name):
#     """Return a class instance from a string reference"""
#     try:
#         module_ = importlib.import_module(module_name)
#         try:
#             class_ = getattr(module_, class_name)()
#         except AttributeError:
#             logging.error('Class does not exist')
#     except ImportError:
#         logging.error('Module does not exist')
#     return class_ or None

# def to_class(path:str):
#     try:
#         from pydoc import locate
#         class_instance = locate(path)
#     except ImportError:
#         print('Module does not exist')
#     return class_instance or None
#
# path = "my_app.models.MyClass"
# my_class = to_class(path)
