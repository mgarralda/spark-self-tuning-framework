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

# import loguru
from typing import List, Tuple, Counter
import numpy as np
import warnings
from framework.experimental.metrics.optimization_model.metrics import RunTimeOptimizationMetrics, \
    CostAwareOptimizationMetrics
from framework.proposed.bayesian_optimization import AcquisitionFunction
from framework.proposed.bayesian_optimization.objective_function import OptimizationObjective
from framework.proposed.bayesian_optimization.performance_model import PerformanceModel
from framework.proposed.bayesian_optimization.uncertainty_model import UncertaintyModel
from framework.proposed.metaheuristics import TabuSearch
from framework.proposed.metaheuristics.iterated_local_search import IteratedLocalSearch
from framework.proposed.parameters import SparkParameters
from framework.proposed.safe_transfer_learning import get_safe_transfer_learning_space
from framework.proposed.workload_characterization.workload import WorkloadCharacterized, WorkloadRepository
from utils.spark.hibench import run_once_workload_hibench
from framework.experimental.metrics.optimization_model.evaluation import (
    EvaluationOptimizationMetrics, TargetWorkloadOptimization
)


def _get_executed_workload(repo: WorkloadRepository, workload_id: str) -> WorkloadCharacterized | None:
    """ Get the executed workload from the repository."""
    wc = repo.get_characterized_workload(workload_id)
    if wc is None:
        print(f"Workload {workload_id} not found in the repository.")
        return None
    wc.resource_shape = wc.to_configuration_shape_vector()
    print(f"Workload {workload_id} found in the repository.")
    print(f"Workload characterized:\n\t{wc}")
    return wc

def _calculate_time_resources(workload: WorkloadCharacterized, beta: int) -> float:
    return OptimizationObjective.objective_function(
        T=workload.time_execution,
        R=OptimizationObjective.calculate_resource_usage(
            SparkParameters.from_vector(
                workload.environment.to_vector()
            )
        ),
        beta=beta
    )

async def adaptive_ils_ts_bo_loop(
        target_workload_id: str,
        characterized_workloads: List["WorkloadCharacterized"],
        setting_workloads: List[List[int]],
        time_execution_workloads: List[int | float],
        time_resources: List[int | float],
        input_data_sizes: List[int],
        workload_names: List[str],
        phi: List[List[int]],
        rho: List[int],
        bounds: List[Tuple[int, int, int]],
        budget: int,
        best_lhs_time: int,
        config: dict
):
    """ Adaptive Iterated Local Search with Bayesian Optimization loop (TS φ-aware, IB+EB govern from start). """
    repo = WorkloadRepository(
        collection=config.get("collection_historical_dataset")
    )

    # -------------------- Initialization --------------------
    perf_model = PerformanceModel()
    ils = IteratedLocalSearch(
        tau_perc=float(config.get("tolerance_perc", 4))
    )

    # TS φ-aware: IB (replaces TR), EB/AB, tenure and self-adaptation
    ts = TabuSearch(
        base_bounds=bounds,
        tenure_L=int(config.get("tabu_tenure_L", 9)),
        q_sub=float(config.get("q_sub", 0.75)),
        q_best=float(config.get("q_best", 0.50)),
        target_skip_rate=float(config.get("target_skip_rate", 0.20)),
        adapt_rate=float(config.get("adapt_rate", 0.15)),
    )

    # History
    historical_latents, historical_parameters, historical_targets = [], [], []
    historical_ids, historical_phi, historical_subotimal = [], [], []

    # State
    it = 0
    wc = None
    current_failures = 0
    converged = False
    initial_target_workload = None
    max_allowed_failures = int(budget * 1.5)

    # === JPDC metrics collection (real runtimes only; TR = T) ===
    # We will compute metrics over the first 10 successful evaluations, matching JPDC tables.
    per_iter_times: List[float] = []
    list_eval_opt_metrics: List[EvaluationOptimizationMetrics] = []

    print(f"[ILS] beta={config.get('beta')}")

    while it < budget:
        it += 1
        print(f"[BO] Iter {it} **************************************************************************************")

        # -------------------- 1) Load initial target --------------------
        if wc is None:
            print(f"Initial {len(characterized_workloads)=}")
            print(f"[BO] Initial iteration. Looking for target workload {target_workload_id} in the repository.")
            wc = _get_executed_workload(repo, target_workload_id)
            if wc is None:
                raise ValueError(f"Target workload {target_workload_id} not found in the repository.")

            wc.resource_usage = OptimizationObjective.calculate_resource_usage(
                SparkParameters.from_vector(wc.environment.to_vector())
            )
            wc.resource_shape = wc.to_configuration_shape_vector()
            wc.time_resources = _calculate_time_resources(wc, beta=config.get('beta'))

            initial_target_workload = wc.model_copy()

            ils.update_best(
                workload=wc,
                tau_perc=config.get("tolerance_perc", 5)
            )
            if config.get("enable_update_real_executions"): # False -> Pure LOGO
                print("[BO] Enable update historical dataset with target workload.")
                historical_latents.append(wc.vector_metrics_garralda)
                historical_parameters.append(wc.environment.to_vector())
                # historical_targets.append(wc.time_execution)
                historical_targets.append(wc.time_resources)
                historical_ids.append(wc.dataset_size)
                historical_phi.append(wc.resource_shape)
                historical_subotimal.append(ils.last_is_suboptimal)

        print(f"{historical_targets=}")
        print(f"historical_latens: len: {len(historical_latents)}")
        for i, (s, p) in enumerate(zip(historical_subotimal, historical_phi)):
            print(f"iter: {i}: suboptimal: {s} | phi: {p}")

        # -------------------- 2) Update in-memory DS_H  --------------------
        if config.get("enable_update_real_executions"): # False -> Pure LOGO
            characterized_workloads.append(wc.vector_metrics_garralda)
            setting_workloads.append(wc.environment.to_vector())
            # time_execution_workloads.append(wc.time_execution)
            time_resources.append(wc.time_resources)
            input_data_sizes.append(wc.dataset_size)
            workload_names.append(wc.app_benchmark_workload)
            phi.append(wc.resource_shape)
            rho.append(wc.resource_usage)
            print("[BO] Enable update real execution; updating DS_H")
        else:
            print("[BO] Disable update real execution: skipping DS_H")

        print(f"\n[BO] Last configuration: {wc.environment.to_vector()} with T_R={wc.time_resources:.0f}.")
        print(f"[BO] Best known configuration so far: {ils.best_workload.environment.to_vector()} with T_R={ils.best_workload.time_resources:.0f}.")


        # -------------------- 3) STL neighborhoods --------------------
        X_N, y_N, stl, X_NC, y_NC, X_C, y_C, IDS_N, IDS_C, w_N, w_C, wn_N, wn_C  = (
            get_safe_transfer_learning_space(
                safe_transfer_learning=perf_model.safe_transfer_learning_stage,
                target_workload=ils.best_workload,
                characterized_workloads=characterized_workloads,
                setting_workloads=setting_workloads,
                name_workloads=workload_names,
                # objective_values_workloads=time_execution_workloads,
                objective_values_workloads=time_resources,
                data_sizes_bytes_workloads=input_data_sizes,
                # phi=phi,
                # phi_ref=wc.resource_shape
            )
        )

        print(f"\n[BO] |N|={len(X_N)}")
        print(f"\n[BO] |N+C|={len(X_C)}")

        # -------------------- 4) Train performance model + UM --------------------
        # supervised_model, scaler, *_ = perf_model.supervised_stage.get_gradientboosting_regressor_model(
        supervised_model, scaler, *_ = perf_model.supervised_stage.get_ridge_regressor_model(
            X_N, y_N, sample_weight=w_N,
        )
        um = UncertaintyModel(
            model=supervised_model,
            X_SL=X_N, y_SL=y_N,
            X_LE=X_NC, y_LE=y_NC,
            scaler=scaler
        )

        # -------------------- 5) Acquisition function --------------------
        acquisition_function = AcquisitionFunction(
            bounds=bounds,
            performance_model=supervised_model,
            perf_model_scaler=scaler,
            nucleus=X_N,
            corona=X_NC,
            uncertainty_model=um,
            phi=phi,                                   # (n,4)
            phi_ref=ils.best_workload.resource_shape,  # (4,)
            beta=float(config.get("beta", 1.0))
        )

        # -------------------- 5.1) TS: initialize IB with S_L and z_ref --------------------
        # NOTE: IB replaces classical trust region from the very first iteration
        try:
            ts.set_inclusion_band(
                Z_SL=acquisition_function._Z_SL,   # S_L cloud in z(φ)
                z_ref=acquisition_function._z_ref  # reference center in z(φ)
            )
            print(f"[TS] IB initialized. Radii: {ts.debug_snapshot()}")
        except Exception as e:
            print(f"[TS] Warning initializing IB: {e}")

        # Seed AB with the initial target as a slight "improve" (iter 1 only)
        if it == 1:
            try:
                ts.observe_after_eval(
                    x_new=np.asarray(wc.environment.to_vector()),
                    phi_new=np.asarray(wc.resource_shape, float),
                    phi_scaler=acquisition_function._phi_scaler,
                    is_improve=True,             # initial target is the best-known
                    is_clearly_worse=False,
                    best_k=int(config.get("best_k", 10))
                )
                print(f"[TS] Seeded AB with initial best. Radii: {ts.debug_snapshot()}")
            except Exception as e:
                print(f"[TS] Warning seeding AB: {e}")

        # -------------------- 6) Suggest next cfg (SUGGEST + TS IB/EB) --------------------
        # Important: TR disabled (IB is managed by TS.filter_candidates)
        scores, dig = acquisition_function.suggest(
            tabu_search=ts,
            diagnostics=True,
            trust_region=False
        )

        x_next, acq_score, mu_best, sigma_best = scores
        print(
            "\n[BO] Proposed next configuration:\n"
            f"\t x_next={x_next}\n"
            f"\t AF_score={acq_score:.2f}\n"
            f"\t mu_T_R_pred={mu_best:.2f}\n"
            f"\t sigma={sigma_best:.2f}"
        )

        print(f"\n[BO] Diagnostics:\n{dig}")
        cfg = SparkParameters.from_vector(x_next)

        repeated_config = (cfg == wc.environment)
        if repeated_config:
            print(f"\n[BO WARNING] Proposed next configuration repeated: {cfg}\n")

        # -------------------- 7) Real system evaluation --------------------
        wc = await run_once_workload_hibench(
            data_scale=wc.app_benchmark_data_size.value,
            framework=config.get("framework"),
            parameters=cfg,
            config=config
        )
        print(f"Executed workload characterized:\n\t{wc}")

        if wc:
            # Update R and φ
            wc.resource_usage = OptimizationObjective.calculate_resource_usage(
                SparkParameters.from_vector(wc.environment.to_vector())
            )
            wc.resource_shape = wc.to_configuration_shape_vector()
            wc.time_resources = _calculate_time_resources(wc, beta=config.get('beta'))

            print(f"{'='*30}\n[BO] Real execution completed. T={wc.time_execution:.0f} sec. T_R={wc.time_resources:.0f}. R={wc.resource_usage}")

            # === runtime collection (count only successful real runs) ===
            # per_iter_times.append(float(wc.time_execution))
            per_iter_times.append(float(wc.time_resources))

            # --- Best-aware acceptance (defines improve/grey/worse) ---
            ils.update_best(
                workload=wc,
                tau_perc=config.get("tolerance_perc", 4)
            )
            is_improve = (getattr(ils, "last_decision", "grey") == "improve")
            is_clearly_worse = (getattr(ils, "last_decision", "grey") == "worse")

            print(f"{'='*30}\n[BO] Real execution completed. T={wc.time_execution:.0f} sec. T_R={wc.time_resources:.0f}. R={wc.resource_usage}")

            # History
            historical_latents.append(wc.vector_metrics_garralda)
            historical_parameters.append(wc.environment.to_vector())
            # historical_targets.append(wc.time_execution)
            historical_targets.append(wc.time_resources)
            historical_ids.append(wc.dataset_size)
            historical_phi.append(wc.resource_shape)
            historical_subotimal.append(ils.last_is_suboptimal)

            # --- ILS perturbation (always active; intensity depends on improve/grey/worse) ---
            vector_perturbed, best_environment = ils.smart_perturb(
                last_vector=wc.vector_metrics_garralda,
                historical_targets=historical_targets,
                historical_latents=historical_latents,
                historical_ids=historical_ids,
                historical_phi=historical_phi,
                historical_suboptimal=historical_subotimal,
                step_size=0.20,
            )

            if vector_perturbed is None or repeated_config:
                converged = True

            # -------------------- TS φ-aware: feed bands and tenure --------------------
            # 1) Feed EB/AB with evaluated point
            ts.observe_after_eval(
                x_new=np.asarray(wc.environment.to_vector()),
                phi_new=np.asarray(wc.resource_shape, float),
                phi_scaler=acquisition_function._phi_scaler,
                is_improve=is_improve,
                is_clearly_worse=is_clearly_worse,
                best_k=int(config.get("best_k", 10))
            )
            # 2) Tenure by clear worsening (best-aware)
            ts.step(
                # best_value=float(ils.best_workload.time_execution),
                best_value=float(ils.best_workload.time_resources),
                x_new=np.asarray(wc.environment.to_vector()),
                # f_new=float(wc.time_execution),
                f_new=float(wc.time_resources),
                tau_perc=float(config.get("tolerance_perc", 4))
            )
            # 3) Purge expired entries
            ts.purge_expired()

            print(f"[TS] Radii after observe/step: {ts.debug_snapshot()}")

            # -------------------- Persist evaluation --------------------
            print(f"{'='*30}\n[BO] Real execution completed. T={wc.time_execution:.0f} sec. T_R={wc.time_resources:.0f}. R={wc.resource_usage}")
            eval_opt_metrics = EvaluationOptimizationMetrics(
                id=wc.id,
                experiment_id=config.get("experiment_id"),
                experiment_iteration=it,
                target_workload=TargetWorkloadOptimization(
                    id=target_workload_id,
                    execution_time=initial_target_workload.time_execution,
                    name=initial_target_workload.app_benchmark_workload,
                    input_data_size=initial_target_workload.app_benchmark_data_size.value,
                    configuration=initial_target_workload.environment
                ),
                # NOTE: TR = T (runtime-centric),
                objective_function_real=wc.time_resources,
                acquisition_function_score=acq_score,
                resource_usage_value=OptimizationObjective.calculate_resource_usage(cfg),
                execution_time=wc.time_execution,
                execution_time_error=int(abs(wc.time_resources - int(mu_best))),
                objective_function_predict=mu_best,
                configuration=cfg,
                beta=config.get("beta"),
                alpha=config.get("alpha"),
                repeated_config=repeated_config,
                suboptimal=ils.last_is_suboptimal,
                converged=converged,
                safe_transfer_learning=stl
            )
            WorkloadRepository(collection=config.get("collection_save_results")).save_optimized_workload_into_mongo(eval_opt_metrics)

            list_eval_opt_metrics.append(eval_opt_metrics)
            converged = False  # reset for next iteration

            print(
                f"\n{'=' * 150}\n"
                f"[BO] Evaluated:\n\t"
                f"{eval_opt_metrics}\n"
                f"{'=' * 150}\n"
            )

        else:
            warnings.warn(f"[BO] Iter {it} - Workload execution failed for {cfg}.")
            print(f"\n{'=' * 150}\n[BO] Iter {it} FAILED for {cfg}. Skipping.\n{'=' * 150}\n")
            it -= 1
            current_failures += 1
            if current_failures >= max_allowed_failures:
                print(f"[BO] Max failures ({max_allowed_failures}) reached. Stopping.")
                break

    # -------------------- JPDC-style metrics (first 10 successful real runs) --------------------
    """
    JPDC fairness:
      • Metrics are computed on REAL executions only (oracle or model calls excluded).
      • We aggregate over the first 10 successful evaluations to match TurBO/Naive/YORO tables.
      • TR = T (pure runtime); Speedup is w.r.t. the initial target runtime.
    """
    if len(per_iter_times) == 0:
        print("[Adaptive-ILS-TS-BO] No successful evaluations; cannot compute metrics.")
        return

    print(f"{'='*30}\n[BO] {len(per_iter_times)=} successful evaluations: {per_iter_times}\n")

    times10 = per_iter_times[:10]
    T_best = float(np.min(times10))
    i_best = int(np.argmin(times10))          # 1-based index
    # T_first_i = float(np.sum(times10[:i_best]))   # cumulative until first best
    T_first = float(times10[0])   # time first iteration

    # Default (baseline) time: initial target workload runtime
    # T_default = float(initial_target_workload.time_execution) if initial_target_workload is not None else float(times10[0])
    T_default = float(initial_target_workload.time_resources) if initial_target_workload is not None else float(times10[0])

    # Metrics
    SU = RunTimeOptimizationMetrics.speedup(T_default, T_best)
    TC = RunTimeOptimizationMetrics.tuning_cost(times10)
    nAOCC = RunTimeOptimizationMetrics.naocc(times10)

    print("\n===  RunTime metrics (10 iterations) — Adaptive ILS-TS-BO ===")
    print(f"T best ↓   : {T_best:.2f}  (found at i={i_best+1})")
    print(f"T first ↓  : {T_first:.2f}")
    print(f"SU (%) ↑   : {SU:.2f}")
    print(f"TC ↓       : {TC:.2f}")
    print(f"nAOCC ↓    : {nAOCC:.4f}")


    # Cost-aware metrics (best vs initial target)
    R_default = OptimizationObjective.calculate_resource_usage(
        SparkParameters.from_vector(
            initial_target_workload.environment.to_vector()
        )
    )
    T_r = T_best
    R_best = list_eval_opt_metrics[i_best].resource_usage_value
    time_reduction = CostAwareOptimizationMetrics.time_reduction(
        time_default=T_default,
        time_optimized=T_best
    )
    resource_reduction = CostAwareOptimizationMetrics.resource_reduction(
        resources_default=R_default,
        resources_optimized=R_best
    )
    TR = CostAwareOptimizationMetrics.trade_off_ratio_monotone(
        time_default=T_default,
        time_optimized=time_reduction,
        resources_default=R_default,
        resources_optimized=R_best
    )

    print("\n=== Cost-aware metrics (best found vs initial target) ===")
    print(f"R initial target ↓ : {R_default:.2f}")
    print(f"R best found ↓    : {R_best:.2f}")
    print(f"T_R (best) ↓      : {T_r:.2f}")
    print(f"Time reduction (%) ↑     : {time_reduction:.2f}")
    print(f"Resource reduction (%) ↑ : {resource_reduction:.2f}")
    print(f"Trade-off ratio (detailed) ↑ : {TR}")
