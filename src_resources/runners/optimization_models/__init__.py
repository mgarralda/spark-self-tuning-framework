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

from framework.proposed.workload_characterization.workload import WorkloadType

SEED = 42
BUDGET = 10
RESTRICTION_BOUNDS = [(1, 3, 1), (2, 4, 1), (1, 5, 1), (1, 4, 1), (2, 5, 1), (50, 350, 50), (1, 2, 1)]


################################################################################################
# LR LARGE
################################################################################################
# LHS Best Execution Time: 1675 (s)
#     driver_cores=2
#     driver_memory_gb=2
#     executor_cores=4
#     executor_instances=5
#     executor_memory_gb=2
#     sql_shuffle_partitions=100
#     task_cpus=1
#     Resources=4+40=44
#     T_R beta =.5 => 1675^0.5 * 44^0.5 = 274

# LHS Best T_R Value: 257
LHS_BEST_LR_TIME = 1675
###################################################################
# Under Q1 of the LHS, used for optimization experiments
###################################################################
# time_execution=2304
# driver_cores=2
# driver_memory_gb=4
# executor_cores=4
# executor_instances=5
# executor_memory_gb=2
# sql_shuffle_partitions=200
# task_cpus=2
# Resources=8+40=48
# T_R=335
WORKLOAD_LR_LARGE_Q1_TARGET_ID = "application_1753629954149_0020_lr_large"

###################################################################
# Under Q2 of the LHS, used for optimization experiments
###################################################################
# time_execution=3479
# driver_cores=1
# driver_memory_gb=2
# executor_cores=4
# executor_instances=2
# executor_memory_gb=4
# sql_shuffle_partitions=300
# task_cpus=2
# Resources=2+32=34
# T_R = 3479^0.5 * 34^0.5 = 348
WORKLOAD_LR_LARGE_Q2_TARGET_ID = "application_1753112283118_0466_lr_large"

###################################################################
# Under Q3 of the LHS, used for optimization experiments
###################################################################
# time_execution=4594
# driver_cores=2
# driver_memory_gb=3
# executor_cores=3
# executor_instances=3
# executor_memory_gb=3
# sql_shuffle_partitions=200
# task_cpus=2
# Resources=6+37=33
# T_R =4594^0.5 * 33^0.5 = 390
WORKLOAD_LR_LARGE_Q3_TARGET_ID = "application_1753112283118_0464_lr_large"

###################################################################
# TEST
###################################################################
WORKLOAD_LR_SMALL_TEST_TARGET_ID = "application_1752991012573_0035_lr_small"


################################################################################################
# LINEAR LARGE
################################################################################################
# LHS Best Execution Time: 1795 (s)
#     driver_cores=2
#     driver_memory_gb=3
#     executor_cores=2
#     executor_instances=5
#     executor_memory_gb=3
#     sql_shuffle_partitions=300
#     task_cpus=1
#     Resources=6+30=36
#     T_R: beta=.5 => T_R=1795^0.5 * 36^0.5 = 254

# LHS Best TR Value: 225
LHS_BEST_LINEAR_TIME = 1795
###################################################################
# Under Q1 of the LHS, used for optimization experiments
###################################################################
# time_execution=2296
# driver_cores=2
# driver_memory_gb=4
# executor_cores=4
# executor_instances=5
# executor_memory_gb=2
# sql_shuffle_partitions=200
# task_cpus=2
# Resources=8+40=48
# T_R = 2296^0.5 * 48^0.5 = 332
WORKLOAD_LINEAR_LARGE_Q1_TARGET_ID="application_1753629954149_0013_linear_large"

###################################################################
# Under Q2 of the LHS, used for optimization experiments
###################################################################
# time_execution=2976
# driver_cores=2
# driver_memory_gb=2
# executor_cores=4
# executor_instances=3
# executor_memory_gb=3
# sql_shuffle_partitions=250
# task_cpus=1
# Resources=4+36=40
# T_R = 2976^0.5 * 40^0.5 = 345
WORKLOAD_LINEAR_LARGE_Q2_TARGET_ID="application_1753554264659_0011_linear_large"

###################################################################
# Under Q3 of the LHS, used for optimization experiments (FALTA)
###################################################################
# time_execution=2345
# driver_cores=2
# driver_memory_gb=3
# executor_cores=4
# executor_instances=4
# executor_memory_gb=3
# sql_shuffle_partitions=300
# task_cpus=11
# Resources=6+48=54
WORKLOAD_LINEAR_LARGE_Q3_TARGET_ID=""


################################################################################################
# SVM LARGE (application_1753112283118_0442_svm_large)
################################################################################################
# LHS Best Execution Time: 1792 (s)
#     driver_cores=3
#     driver_memory_gb=4
#     executor_cores=2
#     executor_instances=4
#     executor_memory_gb=5
#     sql_shuffle_partitions=150
#     task_cpus=1
#     Resources=12+40=52
#     T_R beta =.5 => 1792^0.5 * 52^0.5 = 305

LHS_BEST_T_R = 208
LHS_BEST_SVM_TIME = 1792
###################################################################
# Under Q1 of the LHS, used for optimization experiments
###################################################################
# time_execution=2020
# driver_cores=1
# driver_memory_gb=2
# executor_cores=5
# executor_instances=3
# executor_memory_gb=3
# sql_shuffle_partitions=200
# task_cpus=1
# Resources=2+45=47
# T_R = 2020^0.5 * 47^0.5 = 301
WORKLOAD_SVM_LARGE_Q1_TARGET_ID="application_1753112283118_0437_svm_large"

###################################################################
# Under Q2 of the LHS, used for optimization experiments
###################################################################
# time_execution=3326
# driver_cores=2
# driver_memory_gb=4
# executor_cores=4
# executor_instances=2
# executor_memory_gb=4
# sql_shuffle_partitions=250
# task_cpus=2
# Resources=8+32=40
# T_R = 3326^0.5 * 40^0.5 = 365
WORKLOAD_SVM_LARGE_Q2_TARGET_ID="application_1753112283118_0443_svm_large"

###################################################################
# Under Q3 of the LHS, used for optimization experiments (FALTA)
###################################################################
# time_execution=
# driver_cores=
# driver_memory_gb=
# executor_cores=
# executor_instances=
# executor_memory_gb=
# sql_shuffle_partitions=
# task_cpus=
# Resources=
# T_R=
WORKLOAD_SVM_LARGE_Q3_TARGET_ID=""


################################################################################################
# LDA LARGE (application_1753112283118_0251_lda_large)
################################################################################################
# LHS Best Execution Time: 125 (s)
#     driver_cores=1
#     driver_memory_gb=4
#     executor_cores=5
#     executor_instances=2
#     executor_memory_gb=4
#     sql_shuffle_partitions=200
#     task_cpus=1
#     Resources=4+40=44
#     T_R beta =.5 => 125^0.5 * 44^0.5 = 74

# LHS Best T_R Value: 68
LHS_BEST_LDA_TIME = 125
###################################################################
# Under Q1 of the LHS, used for optimization experiments
###################################################################
# time_execution=210
# driver_cores=2
# driver_memory_gb=3
# executor_cores=2
# executor_instances=5
# executor_memory_gb=3
# sql_shuffle_partitions=300
# task_cpus=1
# Resources=6+30=36
# T_R = 210^0.5 * 36^0.5 = 87
WORKLOAD_LDA_LARGE_Q1_TARGET_ID="application_1753112283118_0246_lda_large"

###################################################################
# Under Q2 of the LHS, used for optimization experiments (FALTA)
###################################################################
# time_execution=434
# driver_cores=2
# driver_memory_gb=3
# executor_cores=3
# executor_instances=3
# executor_memory_gb=3
# sql_shuffle_partitions=200
# task_cpus=2
# Resources=6+27=33
# T_R=434^0.5 * 33^0.5 = 108
WORKLOAD_LDA_LARGE_Q2_TARGET_ID="application_1753112283118_0221_lda_large"

###################################################################
# Under Q3 of the LHS, used for optimization experiments (FALTA)
###################################################################
# time_execution=
# driver_cores=
# driver_memory_gb=
# executor_cores=
# executor_instances=
# executor_memory_gb=
# sql_shuffle_partitions=
# task_cpus=
# R=esources=
# T_R=
WORKLOAD_LDA_LARGE_Q3_TARGET_ID=""

################################################################################################
###################################################################
# FIX WORKLOAD TARGETS
###################################################################
################################################################################################

WORKLOAD_TARGET_TYPE = WorkloadType.LINEAR
LHS_BEST_TIME = LHS_BEST_LINEAR_TIME
WORKLOAD_TARGET_ID = WORKLOAD_LINEAR_LARGE_Q1_TARGET_ID
# WORKLOAD_TARGET_ID = WORKLOAD_LINEAR_LARGE_Q2_TARGET_ID

# WORKLOAD_TARGET_TYPE = WorkloadType.LR
# LHS_BEST_TIME = LHS_BEST_LR_TIME
# WORKLOAD_TARGET_ID = WORKLOAD_LR_LARGE_Q1_TARGET_ID
# WORKLOAD_TARGET_ID = WORKLOAD_LR_LARGE_Q2_TARGET_ID
# WORKLOAD_TARGET_ID = WORKLOAD_LR_LARGE_Q3_TARGET_ID
# WORKLOAD_TARGET_ID = WORKLOAD_LR_SMALL_TEST_TARGET_ID

# WORKLOAD_TARGET_TYPE = WorkloadType.SVM
# LHS_BEST_TIME = LHS_BEST_SVM_TIME
# # WORKLOAD_TARGET_ID = WORKLOAD_SVM_LARGE_Q1_TARGET_ID
# WORKLOAD_TARGET_ID = WORKLOAD_SVM_LARGE_Q2_TARGET_ID
#
# WORKLOAD_TARGET_TYPE = WorkloadType.LDA
# LHS_BEST_TIME = LHS_BEST_LDA_TIME
# WORKLOAD_TARGET_ID = WORKLOAD_LDA_LARGE_Q1_TARGET_ID
# WORKLOAD_TARGET_ID = WORKLOAD_LDA_LARGE_Q2_TARGET_ID
