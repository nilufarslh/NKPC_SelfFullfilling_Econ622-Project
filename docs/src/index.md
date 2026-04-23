# SelfFulfillingNKPC.jl

SMM estimation of a self-fulfilling New Keynesian Phillips curve with central-bank learning, in the spirit of Beaudry, Hou and Portier (2020).

A central bank that estimates the slope of the NKPC from the data it observes, while simultaneously setting the policy rule that shapes that data, creates a feedback loop between beliefs, policy, and outcomes. This package estimates the structural slope ``\kappa`` under two policy regimes on US quarterly data from 1984Q2 through 2025Q2.

## The two cases

| Case                     | Policy rule                                                                 | Weighting ``W``                     | Starts |
|:-------------------------|:----------------------------------------------------------------------------|:------------------------------------|-------:|
| `case1_no_intervention`  | Passive: ``\phi_\pi`` frozen at ``\phi_\pi^*(\hat\kappa_0)``                | ``\operatorname{diag}(1/|\hat m|)`` | 20     |
| `case2_with_intervention`| Active:  ``\phi_\pi^*(\hat\kappa_t)`` re-optimised every period             | ``I``                               | 10     |

Both cases fix ``\theta_y = 1`` as the identification normalisation and calibrate ``\theta_r = 0.3``.

## Quick start

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Full pipeline: estimation + report figures
include("run_all.jl")
```

The driver writes per-case CSVs to `results/<case>/` and three report figures to `results/figures/`. A timestamped log in `results/estimation_<YYYYMMDD_HHMMSS>.log` records each start's objective and parameter vector.

## Navigating the documentation

- [Methodology](@ref) — the structural model, the central bank's learning recursion, and the SMM criterion
- [Reproducibility](@ref) — step-by-step guide to regenerating the estimates in `REPORT.pdf`
- [API Reference](@ref) — function-level documentation for the exported symbols

## Reference implementation

The project report (`REPORT.pdf` in the repository root) contains the full model derivation, estimation details, and empirical results, with Figures 1–3 illustrating the self-fulfilling mechanism at a clean non-IID calibration.

## License

MIT. See `LICENSE` at the repository root.
