# API Reference

```@meta
CurrentModule = SelfFulfillingNKPC
```

Docstrings for every exported symbol.

## Configuration

```@docs
ModelParams
ParamSpec
EstimationConfig
case_non_iid
case_non_iid_free_phi
param_names
theta0_vec
lower_bounds
upper_bounds
unpack
```

## Data

```@docs
TargetData
load_targets
```

## Model primitives

```@docs
taylor_rule
uncond_var
policy_numeric
compute_policy
```

## Policy interpolant

```@docs
PolicyInterp
build_policy_interp
evaluate_policy_itp
```

## Simulation

```@docs
SimulationResult
simulate
```

## Moments

```@docs
MomentVector
compute_moments
moments_from_theta
var1_ols
vec_mat
vech
cov_lag
```

## Objective and estimation

```@docs
smm_objective
estimate_smm
OptimResult
```

## Inference and reporting

```@docs
CaseResult
bootstrap_se
print_results
save_results
```

## Index

```@index
```
