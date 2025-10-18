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

"""
Validation Module
Cross-Validation Techniques for Performance Evaluation of Big Data Workloads

This module implements various cross-validation techniques to evaluate the performance of predictive models for big data workloads, particularly in Apache Spark environments.

Author: Mariano Garralda-Barrio, Carlos Eiras-Franco, Verónica Bolón-Canedo
License: MIT License (see LICENSE file for details)
Date: 2024

Usage:
- Import the `CrossValidation` class and use the `leave_one_out` or `leave_one_group_out` methods to perform cross-validation.
- See the documentation or README for more details on how to use this module.

Example:
    validation = CrossValidation(
        w=garralda_workload_descriptors,
        w_groups=worload_groups,
        cs=configuration_settings
        t=execution_times,
        k_min=k_min,
        k_max=k_max
        eebc_weight=0.5
    )

    loocv_eval = validation.leave_one_out()
    logocv_eval = validation.leave_one_group_out()

    print(f"{loocv_eval}")
    print(f"{logocv_eval}")

    ame = EvaluationMetrics.AME(
        loocv_eval.HME(),
        logocv_eval.HME()
    )

    print(f"AME: {ame:.2f}")
"""

import warnings
import traceback
from typing import Tuple, List
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from framework.experimental.metrics.performance_model.evaluation import EvaluationMetrics
from framework.proposed.bayesian_optimization import PerformanceModel
from runners.optimization_models import RESTRICTION_BOUNDS


class CrossValidation:
    """ Cross validation techniques """

    def __init__(
            self,
            w: np.ndarray,
            w_groups: np.ndarray,
            cs: np.ndarray,
            t: np.ndarray,
            ids: np.ndarray,
            phi: np.ndarray,
            rho: np.ndarray,
            # k_min: int,
            k_limit: int,
            bounds: List[Tuple[int, int, int]] = RESTRICTION_BOUNDS
    ) -> None:
        """ Initialize the cross validation attributes
        Args:
            w (np.ndarray): Garralda workload descriptor.
            w_groups (np.ndarray): Unique groups of workloads.
            cs (np.ndarray): Configuration settings.
            t (np.ndarray): Execution times (target).
            ids (np.ndarray): Input data size
            k_min (int): Minimum value in range K.
            k_max (int): Maximum value in range K.
            eebc_weight (float): Weight to use on EEBC. Default is 0.5.
        """

        self.w = w
        self.w_groups = w_groups
        self.cs = cs
        self.t = t
        self.ids = ids
        self.phi = phi
        self.rho = rho
        # self.k_min = k_min
        self.k_limit = k_limit
        self.bounds =  bounds

        self.perf_model = PerformanceModel()

    def leave_one_out(self) -> EvaluationMetrics:
        """
        Leave-one-out cross-validation (LOOCV).
        Imprime micro (pooled) y macro (avg across folds), análogo a LOGOCV.
        """

        y_pred, y_true = [], []
        fold_metrics = []  # para macro (por fold)

        max_abs_residual = -np.inf
        worst_workload = None

        loocv = LeaveOneOut()

        for train_index, test_index in loocv.split(X=self.w):
            # Reestablece estado interno por fold
            self.perf_model.safe_transfer_learning_stage.reset_thresholds()

            # Conjuntos train/test
            w_train, w_test = self.w[train_index], self.w[test_index]
            cs_train, cs_test = self.cs[train_index], self.cs[test_index]
            t_train, t_test = self.t[train_index], self.t[test_index]
            ids_train, ids_test = self.ids[train_index], self.ids[test_index]
            wn_train, wn_test = self.w_groups[train_index], self.w_groups[test_index]

            phi_train = self.phi[train_index] if getattr(self, "phi", None) is not None else None
            phi_test  = self.phi[test_index]  if getattr(self, "phi", None) is not None else None
            rho_train = self.rho[train_index] if getattr(self, "rho", None) is not None else None
            rho_test  = self.rho[test_index]  if getattr(self, "rho", None) is not None else None

            # Punto de test (escalares/filas)
            w_ref   = w_test[0]
            cs_ref  = cs_test[0]
            t_ref   = float(t_test[0])
            ids_ref = float(ids_test[0])
            wn_ref  = wn_test[0]
            phi_ref = phi_test[0] if phi_test is not None else None
            rho_ref = float(rho_test[0]) if rho_test is not None else None

            try:
                pred = self.perf_model.fit_predict(
                    workload_descriptors=w_train,
                    configuration_settings=cs_train,
                    execution_times=t_train,
                    input_data_sizes=ids_train,
                    workload_ref=w_ref,
                    configuration_settings_ref=cs_ref,
                    execution_time_ref=t_ref,
                    input_data_size_ref=ids_ref,
                    workload_names=wn_train,
                    worload_config_shapes=phi_train,
                    workload_config_shape_ref=phi_ref,
                    workload_resources=rho_train,
                    execution_resources_ref=rho_ref,
                    k_limit=self.k_limit,
                    cv="LOOCV"
                )
                pred = float(np.asarray(pred).ravel()[0])

                # acumular pooled
                y_pred.append(pred)
                y_true.append(t_ref)

                # métricas por fold (macro)
                fold_metrics.append(EvaluationMetrics([t_ref], [pred]))

                # peor residual (debug)
                abs_res = abs(pred - t_ref)
                if abs_res > max_abs_residual:
                    max_abs_residual = abs_res
                    worst_workload = wn_ref

            except Exception as e:
                traceback.print_exc()
                warnings.warn(f"leave_one_out: {e}", UserWarning)

        # métricas pooled (micro)
        em = EvaluationMetrics(y_true=y_true, y_pred=y_pred)

        # resumen macro (avg across folds)
        cv_summary = EvaluationMetrics.summarize_loocv(fold_metrics, print_summary=True)

        # # --- impresión alineada con LOGOCV ---
        # print("LOOCV micro (pooled): "
        #       f"MAE={em.MAE():.2f} RMSE={em.RMSE():.2f} "
        #       f"Med={em.residual_median():.2f} P95={em.residual_p95():.2f} "
        #       f"P98={em.residual_p98():.2f} Max={em.residual_max():.2f}")
        #
        # print("LOOCV macro (avg across folds): "
        #       f"MAE={cv_summary.macro_mae:.2f} RMSE={cv_summary.macro_rmse:.2f}")

        # extra debug consistente con tu versión previa
        print(f"LOOCV worst absolute residual: {max_abs_residual:.2f} (workload: {worst_workload})")
        self._print_k_bounds()
        print(f"{self.perf_model.weights_diff_threshold=}")

        return em

    def leave_one_group_out(self) -> EvaluationMetrics:
        """
        Leave-One-Group-Out CV (LOGOCV).
        Retorna métricas *pooled/micro* (todas las predicciones concatenadas).
        """
        y_true_all, y_pred_all = [], []
        per_group_metrics = []  # opcional: para inspección/log

        negative = 0

        logocv = LeaveOneGroupOut()

        # split dejando fuera cada grupo de workloads
        for train_index, test_index in logocv.split(self.w, self.t, self.w_groups):
            # --- reestablece estado del STL por fold externo
            self.perf_model.safe_transfer_learning_stage.reset_thresholds()

            # conjuntos de entrenamiento
            w_train, cs_train = self.w[train_index], self.cs[train_index]
            t_train, ids_train = self.t[train_index], self.ids[train_index]
            w_groups_train = self.w_groups[train_index]
            phi_train = self.phi[train_index] if self.phi is not None else None
            rho_train = self.rho[train_index] if self.rho is not None else None

            # conjuntos de test (grupo dejado fuera)
            w_test, cs_test = self.w[test_index], self.cs[test_index]
            t_test, ids_test = self.t[test_index], self.ids[test_index]
            w_groups_test = self.w_groups[test_index]
            phi_test = self.phi[test_index] if self.phi is not None else None
            rho_test = self.rho[test_index] if self.rho is not None else None

            # nombre del grupo (todos iguales en test_index)
            try:
                group_name = np.unique(w_groups_test).tolist()
                group_name = group_name[0] if len(group_name) == 1 else str(group_name)
            except Exception:
                group_name = str(w_groups_test[0])

            # acumular predicciones de este fold (para logging opcional)
            y_true_fold, y_pred_fold = [], []

            # para cada muestra del grupo dejado fuera
            for i in range(len(w_test)):
                try:
                    w_ref   = w_test[i]
                    cs_ref  = cs_test[i]
                    t_ref   = float(t_test[i])
                    ids_ref = float(ids_test[i])
                    phi_ref = phi_test[i] if phi_test is not None else None
                    rho_ref = float(rho_test[i]) if rho_test is not None else None

                    pred = self.perf_model.fit_predict(
                        workload_descriptors=w_train,
                        configuration_settings=cs_train,
                        execution_times=t_train,
                        input_data_sizes=ids_train,
                        workload_ref=w_ref,
                        configuration_settings_ref=cs_ref,
                        execution_time_ref=t_ref,
                        input_data_size_ref=ids_ref,
                        workload_names=w_groups_train,
                        worload_config_shapes=phi_train,
                        workload_config_shape_ref=phi_ref,
                        workload_resources=rho_train,
                        execution_resources_ref=rho_ref,
                        k_limit=self.k_limit,
                        cv="LOGOCV"
                    )


                    if pred[0] < 0:
                        negative += 1
                        print(f"[WARNING LOGOCV] Negative prediction: {pred[0]} (workload: {group_name})")

                    pred = float(np.asarray(pred).ravel()[0])  # escalar
                    y_pred_all.append(pred)
                    y_true_all.append(t_ref)
                    y_pred_fold.append(pred)
                    y_true_fold.append(t_ref)

                except Exception as e:
                    traceback.print_exc()
                    warnings.warn(f"leave_one_group_out (grupo={group_name}): {e}", UserWarning)

            # métricas por fold (opcional; no afectan al retorno)
            if len(y_true_fold) > 0:
                m_fold = EvaluationMetrics(y_true=y_true_fold, y_pred=y_pred_fold)
                per_group_metrics.append((group_name, m_fold))
                print(f"[LOGOCV] Grupo fuera: {group_name} | "
                      f"MAE={m_fold.MAE():.2f} RMSE={m_fold.RMSE():.2f} "
                      f"Med={m_fold.residual_median():.2f} P95={m_fold.residual_p95():.2f} "
                      f"P98={m_fold.residual_p98():.2f} Max={m_fold.residual_max():.2f}")

        print(f"LOGOCV total negative predictions: {negative}")

        # métricas *pooled/micro* (todas las predicciones)
        final_metrics = EvaluationMetrics(y_true=y_true_all, y_pred=y_pred_all)


        EvaluationMetrics.summarize_logocv([m for _, m in per_group_metrics], print_summary=True)

        self._print_k_bounds()

        return final_metrics

    def _print_k_bounds(self):
        N_sizes = self.perf_model.safe_transfer_learning_stage.N_sizes
        print(f"N_sizes:\n"
              # np.median(self.perf_model.unsupervised_stage.k_opts), np.percentile(self.perf_model.unsupervised_stage.k_opts,90), max(self.perf_model.unsupervised_stage.k_opts))
              f"\tmin: {min(N_sizes)}\n"
              f"\tmedian: {np.median(N_sizes):.2f}\n"
              f"\tmean: {np.mean(N_sizes):.2f}\n"
              f"\tmax: {max(N_sizes)}")
        print(f"Vector:{len(N_sizes)}-dimension = {N_sizes}\n")

        C_sizes = self.perf_model.safe_transfer_learning_stage.C_sizes
        print(f"C_sizes:\n"
              f"\tmin: {min(C_sizes)}\n"
              f"\tmedian: {np.median(C_sizes):.2f}\n"
              f"\tmean: {np.mean(C_sizes):.2f}\n"
              f"\tmax: {max(C_sizes)}")
        print(f"Vector {len(C_sizes)}-dimension = {C_sizes}\n")
