<div align="center">

# SelfFulfillingNKPC.jl

**SMM estimation of a self-fulfilling New Keynesian Phillips curve with central-bank learning.**

[![CI](https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nilufarslh.github.io/NKPC_SelfFullfilling_Econ622-Project/dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<br/>

<img src="results/figures/fig_self_fulfilling_calibrated.png" width="780" alt="Self-fulfilling NKPC — calibrated Case 2 output."/>

<sub><b>Case 2 (active intervention)</b>, US quarterly data, <i>1984Q2–2025Q2</i>.</sub>

</div>

---

## What it does

Estimates the structural slope $\kappa$ of the NKPC

$$
\pi_t = \beta\,\mathbb{E}_t\pi_{t+1} + \kappa\,y_t + \nu_t
$$

when the central bank sets its Taylor coefficient $\phi_\pi^\*(\hat\kappa_t)$ from a real-time estimate of $\kappa$ — a feedback loop between beliefs, policy, and the data they generate. Two cases are estimated by single-stage SMM with ForwardDiff gradients, Fminbox(L-BFGS), and threaded bootstrap standard errors.

| Case                       | Policy rule                                              | Weight $W$                                       | Starts |
|----------------------------|----------------------------------------------------------|--------------------------------------------------|-------:|
| `case1_no_intervention`    | $\phi_\pi$ frozen at $\phi_\pi^\*(\hat\kappa_0)$         | $\mathrm{diag}(1/\lvert\hat m\rvert)$            |     20 |
| `case2_with_intervention`  | $\phi_\pi^\*(\hat\kappa_t)$ re-optimised every period    | $I$                                              |     10 |

Written for ECON 622 at UBC. Full report: [`REPORT.pdf`](REPORT.pdf). Full API: [documentation](https://nilufarslh.github.io/NKPC_SelfFullfilling_Econ622-Project/dev/).

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project")
```

Requires Julia ≥ 1.10.

## Reproduce

| # | Step              | Command                                                      | Runtime |
|---|-------------------|--------------------------------------------------------------|---------|
| 1 | Instantiate env   | `julia --project=. -e 'using Pkg; Pkg.instantiate()'`        | once    |
| 2 | Estimate (both cases) | `julia --project=. -t auto run_estimation.jl`            | ~8 min  |
| 3 | Tests             | `julia --project=. -t auto test/runtests.jl`                 | ~2 min  |
| 4 | Figures           | `julia --project=. -t auto scripts/fig_report.jl`            | ~30 s   |

Outputs land in `results/<case>/params.csv`, `results/<case>/moments.csv`, and a timestamped log `results/estimation_<YYYYMMDD_HHMMSS>.log`. Seeds are fixed.

## Citation

<details>
<summary>BibTeX</summary>

```bibtex
@misc{eslahi2026selffulfillingnkpc,
  author       = {Eslahi, Niloufar},
  title        = {{SelfFulfillingNKPC.jl}: SMM estimation of a self-fulfilling New Keynesian Phillips curve with central-bank learning},
  year         = {2026},
  note         = {ECON 622 final project, University of British Columbia},
  howpublished = {\url{https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project}}
}
```

</details>

## License

[MIT](LICENSE) © 2026 Niloufar Eslahi.
