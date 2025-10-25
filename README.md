# Spark Self-Tuning Framework (STL-PARN–ILS–TS–BO)

Implementation of the framework proposed in  
**“A hybrid metaheuristics–Bayesian Optimization framework with safe transfer learning for continuous Spark tuning”**  
(*Future Generation Computer Systems*, 2025).  
[![DOI](https://zenodo.org/badge/DOI/10.1016/j.future.2025.107730.svg)](https://doi.org/10.1016/j.future.2025.107730)

---

## 🧠 Overview

The **Spark Self-Tuning Framework** provides continuous and adaptive optimization of Apache Spark configurations by combining:

- **Bayesian Optimization (BO)** with a custom acquisition function (`LCB`)
- **Compositional surrogate models** for performance and uncertainty estimation
- **Iterated Local Search + Tabu Search (ILS–TS)** for guided exploration and local refinement
- **Safe Transfer Learning (STL-PARN)** to reuse historical workload executions
- **Baseline implementations**: *Garralda*, *TurBO*, *YORO*, and *Naïve BO*

This framework enables cost-aware, knowledge-driven configuration tuning for complex Spark workloads.

---

## 📁 Project Structure

```
project-root/
├── src/                       # Core framework
│   └── framework/
│       ├── proposed/          # Main optimization method
│       ├── metaheuristics/    # Tabu + ILS modules
│       ├── bayesian_optimization/
│       └── safe_transfer_learning/
├── src_resources/             # Experiment runners
├── resources/                 # Datasets & results
```

---

## 📊 Data

Experimental data and benchmarks are provided under:

```
resources/
├── dataset/
│   ├── historical_dataset.json
│   ├── lhs_initialization.json
├── experiment_results/
│   ├── performance_model/
│   ├── optimization_model/
```

---

## 📜 License

This project is dual-licensed under:

- **CC BY-NC 4.0** for academic and research use  
  <https://creativecommons.org/licenses/by-nc/4.0/>
- **Commercial use is not allowed.**  
  Any use of this software or its derivatives for commercial purposes is strictly prohibited.

Distributed on an “AS IS” basis, without warranties or conditions of any kind.  
See the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

Please cite the following article when using this framework or its components:

```bibtex
@article{GarraldaBarrio2025,
    title     = {A hybrid metaheuristics–Bayesian Optimization framework with safe transfer learning for continuous Spark tuning},
    author    = {Mariano Garralda-Barrio and Carlos Eiras-Franco and Verónica Bolón-Canedo},
    journal   = {Future Generation Computer Systems},
    pages     = {107730},
    year      = {2025},
    issn      = {0167-739X},
    doi       = {https://doi.org/10.1016/j.future.2025.107730},
    url       = {https://www.sciencedirect.com/science/article/pii/S0167739X25000251},
    publisher = {Elsevier},
    note      = {Code available at \url{https://github.com/mgarralda/spark-self-tuning-framework}},
    keywords  = {Performance modeling, Big data, Machine learning, Apache Spark, Distributed computing}
}
```

---

## 📬 Contact

For questions, collaborations, or feedback, please contact:  
**Mariano Garralda**  
[mariano.garralda@udc.es](mailto:mariano.garralda@udc.es)  
Universidade da Coruña (UDC) · CITIC

---
