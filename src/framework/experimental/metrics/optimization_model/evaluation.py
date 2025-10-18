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
import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field, PrivateAttr
from framework.proposed.parameters import SparkParameters
from framework.proposed.workload_characterization.workload import WorkloadEntity


class WorkloadInvolved(BaseModel):
    types: List[str] = Field(default_factory=list)
    configurations: List[SparkParameters] = Field(default_factory=list)


class SearchSpace(BaseModel):
    name: str # local | local extended | uninformative
    size: int
    error: Optional[float] = None
    workload_involved: List[WorkloadInvolved] = Field(default_factory=list)


class SafeTransferLearningEvaluationMetrics(BaseModel):
    """
    Métricas y artefactos de STL enriquecidos para el bucle principal,
    sin romper los usos existentes.
    """
    search_space: List[SearchSpace] = Field(default_factory=list)

    # Private attributes for STL metrics (not serialized)
    _z_repo: Optional[np.ndarray] = PrivateAttr(default=None)
    _z_ref: Optional[np.ndarray] = PrivateAttr(default=None)
    _idx_S_L: Optional[np.ndarray] = PrivateAttr(default=None)
    _idx_S_LE: Optional[np.ndarray] = PrivateAttr(default=None)
    _idx_S_E: Optional[np.ndarray] = PrivateAttr(default=None)
    _k_opt: Optional[int] = PrivateAttr(default=None)
    _k_knee: Optional[int] = PrivateAttr(default=None) # Field(None, exclude=True)


    class Config:
        arbitrary_types_allowed = True

    def get_latents_for_S_LE(self) -> np.ndarray:
        """
        Latentes (26d) para S_{L+E}, alineados con (X_S_L_LE, y_S_L_LE)
        tal como los devuelve get_safe_transfer_learning_space.
        """
        if self._z_repo is None or self._idx_S_LE is None:
            raise RuntimeError("STL metrics missing z_repo or idx_S_L_LE.")
        # return self._z_repo[self._idx_S_LE]
        return self._z_repo

    def get_target_latent(self) -> np.ndarray:
        """Latente (26d) del target."""
        if self._z_ref is None:
            raise RuntimeError("STL metrics missing z_star.")
        return self._z_ref

    def get_counts(self) -> tuple:
        """Devuelve (k_opt, k_knee)."""
        if self._k_opt is None or self._k_knee is None:
            raise RuntimeError("STL metrics missing k_opt/k_knee.")
        return self._k_opt, self._k_knee


class TargetWorkloadOptimization(BaseModel):
    id: str
    execution_time: int
    name: str
    input_data_size: str
    configuration: WorkloadEntity.Environment


class EvaluationOptimizationMetrics(BaseModel):
    id: str = Field(default_factory=str, alias="_id")
    experiment_id: str
    experiment_iteration: int
    target_workload: TargetWorkloadOptimization
    objective_function_real: float
    acquisition_function_score: float
    resource_usage_value: float
    execution_time: int
    configuration: SparkParameters
    execution_time_error: Optional[int] = None
    objective_function_predict: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    repeated_config: Optional[bool] = None
    suboptimal: Optional[bool] = None
    converged: Optional[bool] = None
    safe_transfer_learning: Optional[SafeTransferLearningEvaluationMetrics] = None

    class Config:
        validate_by_name = True



