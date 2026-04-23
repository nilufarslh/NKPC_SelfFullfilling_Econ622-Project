# Reproducibility

## One-command reproduction

From the repository root:

```bash
julia --project=. -t auto run_all.jl
```

This runs, in order:
1. SMM estimation for both cases (`run_estimation.jl`)
2. The three report figures (`scripts/fig_report.jl`)

Outputs land in `results/<case>/` (CSVs) and `results/figures/` (PDFs + PNGs).

## Step-by-step

### 1. Install dependencies

Julia ≥ 1.10 is required. The package's `Project.toml` pins compatible versions; `Pkg.instantiate()` resolves them against `Manifest.toml` for bit-reproducible results.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. Run the estimation

```bash
julia --project=. -t auto run_estimation.jl
```

Produces:

- `results/case1_no_intervention/params.csv`, `moments.csv`
- `results/case2_with_intervention/params.csv`, `moments.csv`, `learning.csv`
- `results/estimation_<YYYYMMDD_HHMMSS>.log` — line-by-line record of each start's objective and parameter vector.

### 3. Generate the figures

```bash
julia --project=. scripts/fig_report.jl
```

Produces three PDFs + PNGs in `results/figures/`:

- `fig_policy_map` — the optimal policy reaction ``\phi_\pi^*(\hat\kappa)``
- `fig_learning_paths` — median learning path with IQR bands for both cases
- `fig_self_fulfilling_calibrated` — terminal ``\hat\kappa`` as a function of ``\kappa_0``

### 4. Run the test suite

```bash
julia --project=. -t auto test/runtests.jl
```

Expected result: 94 passed, 1 broken (documented AD bias on the intervention path), 0 failed.

## Optional diagnostics

### Benchmarks

```bash
julia --project=. scripts/benchmark.jl
```

Times `simulate`, `smm_objective`, and `ForwardDiff.gradient(smm_objective)` for both cases via BenchmarkTools.

### Profile

```bash
julia --project=. scripts/profile.jl
```

Writes a text-form flat + tree profile of 30 000 `simulate` calls per case to `results/figures/profile_simulate.txt`.

## Estimates

The Apr 23 production run lands at

| Case | ``\hat\kappa`` | Bootstrap SE | Stage-1 ``Q(\hat\theta)`` |
|:-----|---------------:|-------------:|---------------------------:|
| Case 1 (passive) | 0.051 | 0.303 |  2.798 |
| Case 2 (active)  | 0.611 | 0.274 | 10.574 |

The full parameter vectors and moment fits are in `REPORT.md` / `REPORT.pdf`.

## Hardware and wall-clock

Two cases together take roughly eight minutes on a four-thread machine. The bootstrap is parallelised via `Threads.@threads`; pass `-t auto` to use all available threads.
