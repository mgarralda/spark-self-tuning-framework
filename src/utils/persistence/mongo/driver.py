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

import atexit
from typing import List, Dict
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.results import InsertOneResult, UpdateResult, BulkWriteResult
from utils.persistence.mongo.base import Persistence
from utils.persistence.mongo import MongoDriverConnCfg


class MongoDriver(Persistence):
    """
    Wrapper official Python MongoDB Driver
    """

    def __init__(self, conn: MongoDriverConnCfg):
        # todo: check
        # self._client = MongoClient(conn.get_uri(), replicaSet=conn.replica_set)
        # initialize the set we need to connect directly to a single node and run the initiate command using the
        # directConnection option:
        self._conn = conn
        self._client = MongoClient(conn.get_uri(), directConnection=True)
        # Setting database to the client
        self._db_client = self._client[conn.database]
        atexit.register(self._cleanup)

    def read(self, predicate: List[Dict] = None, aggs=None, collection=None, exclude_platform_columns: bool = True) \
            -> List[Dict]:
        """
        :param collection:
        :param aggs:
        :param predicate:
        :param exclude_platform_columns:
        :return:
        """

        # Setting collection to the client
        if collection is None:
            if self._conn.collection is None:
                raise Exception("You need to set a collection name before read or in the read method pass it")
            collection = self._conn.collection

        collection_client = self._db_client[collection]

        if aggs and predicate:
            raise AttributeError("You cannot pass predicate and aggs at the same time, only make a find or aggregation")

        if aggs:
            data_find = list(collection_client.aggregate(aggs))
        else:
            data_find = list(collection_client.find(predicate))  # # Pay attention because consumes a lot of memory

        return data_find

    def write(self, document: Dict, collection: str = None) -> str:
        """
        :param document:
        :param collection:
        :return:
        """
        # Setting collection to the client
        collection_client: MongoClient
        if collection:
            collection_client = self._db_client[collection]
        else:
            if self._conn.database:
                collection_client = self._db_client[self._conn.collection]
            else:
                raise Exception(__class__, "Collection name is not specified")

        result: InsertOneResult = collection_client.insert_one(document=document)

        return str(result.inserted_id)

    def upsert(self, filter: Dict, update: Dict, collection: str) -> str:
        collection_client = self._db_client[collection]
        result: UpdateResult = collection_client.update_one(filter=filter, update=update, upsert=True)

        return str(result.upserted_id)

    def upsert_many(self, documents: List[Dict], collection: str = None) -> Dict:

        collection_client: Collection
        if collection:
            collection_client = self._db_client[collection]
        else:
            if self._conn.database:
                collection_client = self._db_client[self._conn.collection]
            else:
                raise Exception(__class__, "Collection name is not specified")

        updates = []
        for document in documents:
            updates.append(UpdateOne({'_id': document.get("_id")}, {'$set': document}, upsert=True))

        results: BulkWriteResult = collection_client.bulk_write(updates)

        return results.upserted_ids

    def update(self):
        raise NotImplementedError(
            "Not implemented yet. Available methods under-the-hook on the pymongo client: get_underlying_driver()"
        )

    def delete(self):
        raise NotImplementedError(
            "Not implemented yet. Available methods under-the-hook on the pymongo client: get_underlying_driver()"
        )

    def get_underlying_driver(self, collection: str) -> Collection:
        """
        Under-the-hood spark instance
        :return: MongoClient underlying instance created
         The supported write operations by PyMongo are:
        bulk_write(), as long as UpdateMany or DeleteMany are not included.
        delete_one()
        insert_one()
        insert_many()
        replace_one()
        update_one()
        find_one_and_delete()
        find_one_and_replace()
        find_one_and_update()
        """
        return self._db_client[collection]

    def _cleanup(self) -> None:
        self._client.close()
