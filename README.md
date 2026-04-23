# SelfFulfillingNKPC.jl

[![CI](https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nilufarslh.github.io/NKPC_SelfFullfilling_Econ622-Project/dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Estimates the structural slope $\kappa$ of the New Keynesian Phillips curve on US quarterly data 1984Q2–2025Q2, under two policy regimes that differ only in whether the central bank's Taylor coefficient $\phi_\pi$ responds to its own real-time estimate $\hat\kappa_t$ of the slope. Comparing the two is a direct empirical test of the self-fulfilling channel of Beaudry, Hou, and Portier (2020).

## Headline result

| Regime                                             | $\hat\kappa$ | bootstrap SE | $t$ vs $0$ | verdict                              |
|----------------------------------------------------|-------------:|-------------:|-----------:|--------------------------------------|
| **Passive**: $\phi_\pi$ frozen at $\phi_\pi(\hat\kappa_0)$ |        0.051 |        0.303 |       0.17 | cannot reject a flat Phillips curve |
| **Active**: $\phi_\pi(\hat\kappa_t)$ updates each period  |        0.611 |        0.274 |       2.23 | rejects $\kappa = 0$ at the 5% level |

A factor-of-twelve jump in the point estimate, driven entirely by whether the central bank reacts to its own learning. The passive-regime moments are governed by the shock process alone and leave $\kappa$ unidentified; the active-regime moments carry the signature of a belief-responsive Taylor rule, which is what pins the structural slope down. Full fit diagnostics and identification analysis in [`REPORT.pdf`](REPORT.pdf); API in the [docs](https://nilufarslh.github.io/NKPC_SelfFullfilling_Econ622-Project/dev/).

ECON 622 final project, UBC.

## The model

$$
\pi_t = \beta \mathbb{E}_t \pi_{t+1} + \kappa y_t + \nu_t
$$

$$
r_t = \phi_\pi(\hat\kappa_t) \pi_t + \phi_y y_t + u_t
$$

| Symbol | Meaning |
|---|---|
| $\pi_t, y_t, r_t$ | CPI inflation, CBO output gap, Fed Funds rate (quarterly, demeaned) |
| $\kappa$ | structural NKPC slope; **the object of interest** |
| $\beta$ | household discount factor (calibrated) |
| $\nu_t$ | AR(1) cost-push shock, innovation $\sigma_\nu$ |
| $u_t$ | composite policy residual $\phi_\pi m^\pi_t + \phi_y m^y_t$ |
| $\hat\kappa_t$ | central bank's real-time estimate of $\kappa$ |
| $\phi_\pi(\hat\kappa_t)$ | Taylor coefficient, set optimally given the CB's belief |

The feedback: the CB chooses $\phi_\pi$ using its best guess $\hat\kappa_t$ of the slope; that policy shapes the equilibrium data $(\pi_t, y_t, r_t)$; and $\hat\kappa_t$ is itself estimated from that data. Different beliefs sustain different equilibria, hence *self-fulfilling*. The econometric task is to recover the structural $\kappa$ from inside the loop.

## Two specifications

| Case                        | Policy rule                                                        | Weight $W$                              | Starts |
|-----------------------------|--------------------------------------------------------------------|-----------------------------------------|-------:|
| `case1_no_intervention`     | $\phi_\pi$ frozen at $\phi_\pi(\hat\kappa_0)$                      | $\mathrm{diag}(1/\lvert\hat m\rvert)$   |     20 |
| `case2_with_intervention`   | $\phi_\pi(\hat\kappa_t)$ re-optimised every period                 | $I$                                     |     10 |

Both normalise $\theta_y = 1$, calibrate $\theta_r = 0.3$, and estimate seven structural parameters: $\kappa, \rho_d, \rho_\nu, \sigma_d, \sigma_\nu, \sigma_{m^y}, \sigma_{m^\pi}$. Eleven VAR(1) moments from the trivariate system on $(\pi, y, r)$ are matched.

<p align="center">
  <img src="results/figures/fig_self_fulfilling_calibrated.png" width="760" alt="Terminal κ̂ as a function of initial belief κ₀ at the illustrative calibration (κ = 0.10)."/>
</p>

Terminal belief $\hat\kappa$ as a function of the initial belief $\kappa_0$, averaged over 40 shock seeds per grid point, illustrative calibration ($\kappa = 0.10$). The $45^\circ$ line marks the passive-regime self-confirming locus; the dotted horizontal marks the true slope.

- **Passive policy traces the $45^\circ$ line.** Terminal $\hat\kappa$ rises with $\kappa_0$: the bank ends where it started. The data it observes are shaped by a policy that never reacts to learning, so any initial belief is self-confirming.
- **Active policy collapses the curve to a horizontal line at $\hat\kappa \approx 0.23$.** Once the bank's Taylor coefficient moves with $\hat\kappa_t$, the equilibrium stops depending on $\kappa_0$. The feedback loop is broken by endogenous policy, not by accumulating data.
- **Both regimes converge above the true $\kappa = 0.10$**, a small upward recursive-IV bias under persistent shocks. It is flat across $\kappa_0$ under active policy and rising in $\kappa_0$ under passive policy, so the bias does not confound the comparison.

## Method

Single-stage SMM. Exact gradients via ForwardDiff, including through the CB's inner policy problem, approximated by a 31-point linear interpolant of $\phi_\pi(\hat\kappa)$ to keep the objective differentiable. Multi-start Fminbox(L-BFGS) with fixed multiplier seeds. Standard errors from a 20-replicate nonparametric bootstrap, parallelised across threads.

## Run

```julia
using Pkg
Pkg.add(url = "https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project")
```

Or, from a clone:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -t auto run_estimation.jl
julia --project=. -t auto test/runtests.jl
```

Requires Julia ≥ 1.10. Writes `results/<case>/params.csv`, `results/<case>/moments.csv`, and a timestamped `estimation_<YYYYMMDD_HHMMSS>.log`. The full pipeline takes ~8 min on 4 threads; seeds are fixed.

## Cite

```bibtex
@misc{eslahi2026selffulfillingnkpc,
  author       = {Eslahi, Niloufar},
  title        = {{SelfFulfillingNKPC.jl}: SMM estimation of a self-fulfilling New Keynesian Phillips curve with central-bank learning},
  year         = {2026},
  note         = {ECON 622 final project, University of British Columbia},
  howpublished = {\url{https://github.com/nilufarslh/NKPC_SelfFullfilling_Econ622-Project}}
}
```

MIT © 2026 Niloufar Eslahi.
