# Spark Self-Tuning Framework (STL–ILS–TS–BO)

Implementation of the framework proposed in  
**“A hybrid metaheuristics–Bayesian Optimization framework with safe transfer learning for continuous Spark tuning”**  
(*Future Generation Computer Systems*, 2025).  
DOI: https://doi.org/10.1016/j.future.2025.108325

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

If you use this framework, its methodology, infrastructure, datasets, or derived components in research, benchmarking studies, technical documentation, or industrial reports, please cite the associated article and/or doctoral thesis.

### Article

```bibtex
@article{GarraldaBarrio2025,
    title     = {A hybrid metaheuristics–Bayesian optimization framework with safe transfer learning for continuous spark tuning},
    author    = {Mariano Garralda-Barrio and Carlos Eiras-Franco and Verónica Bolón-Canedo},
    journal   = {Future Generation Computer Systems},
    pages     = {108325},
    year      = {2025},
    issn      = {0167-739X},
    doi       = {https://doi.org/10.1016/j.future.2025.108325},
    publisher = {Elsevier},
    note      = {Code available at \url{https://github.com/mgarralda/spark-self-tuning-framework}},
    keywords  = {Performance modeling, Big data, Machine learning, Apache Spark, Distributed computing}
}
```

### Doctoral Thesis

```bibtex
@phdthesis{GarraldaBarrio2026,
    author    = {Mariano Garralda Barrio},
    title     = {AI-Driven Optimization in Distributed Computing Systems: A Self-Tuning Framework},
    school    = {University of Coruña},
    year      = {2026},
    type      = {Doctoral Thesis},
    url       = {https://hdl.handle.net/2183/48114}
}
```

### References

- Garralda-Barrio, M., Eiras-Franco, C., & Bolón-Canedo, V. (2025).  
  *A hybrid metaheuristics–Bayesian optimization framework with safe transfer learning for continuous Spark tuning*.  
  Future Generation Computer Systems.  
  https://doi.org/10.1016/j.future.2025.108325

- Garralda Barrio, M. (2026).  
  *AI-Driven Optimization in Distributed Computing Systems: A Self-Tuning Framework*.  
  Doctoral Thesis, University of Coruña.  
  https://hdl.handle.net/2183/48114

---

## 📬 Contact

For questions, collaborations, or feedback, please contact:  
**Mariano Garralda**  
[mariano.garralda@udc.es](mailto:mariano.garralda@udc.es)  
Universidade da Coruña (UDC)

---
