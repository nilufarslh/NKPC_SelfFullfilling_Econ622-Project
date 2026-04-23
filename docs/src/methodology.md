# Methodology

## Structural equations

The economy has three equations. The IS block is
```math
y_t = -\tfrac{1}{\gamma}\, r_t + d_t,
```
the hybrid New Keynesian Phillips curve is
```math
\pi_t = \beta\, E_t \pi_{t+1} + \kappa\,(\theta_y\, y_t + \theta_r\, r_t) + \nu_t,
```
and monetary policy follows a Taylor rule with two measurement errors,
```math
r_t = \phi_\pi\,(\pi_t + m^\pi_t) + \phi_y\,(y_t + m^y_t).
```

The demand shock ``d_t`` and the cost-push shock ``\nu_t`` are AR(1) with persistences ``\rho_d, \rho_v`` and innovation standard deviations ``\sigma_d, \sigma_v``. The measurement errors ``m^\pi_t, m^y_t`` are iid with scales ``\sigma_{m\pi}, \sigma_{my}``. Throughout, ``\beta = 0.99``, ``\gamma = 1.0``, ``\lambda_y = 0.5``, ``\theta_r = 0.3``; the normalisation ``\theta_y = 1`` resolves the non-identification of ``\kappa \theta_y``, so the estimand is ``\kappa``.

## Central bank learning

The CB's subjective model omits the cost channel:
```math
\pi_t = \beta\, E_t \pi_{t+1} + \hat\kappa_t\, y_t + \nu_t.
```
It updates ``\hat\kappa_t`` by recursive instrumental variables with ``d_t`` as the instrument for ``y_t``,
```math
R_t = R_{t-1} + \gamma_t\,(d_t y_t - R_{t-1}),\quad
\hat\kappa_t = \hat\kappa_{t-1} + \gamma_t R_t^{-1} d_t\,(\pi_t - \beta \hat\kappa_{t-1} y_t - y_t \hat\kappa_{t-1}),
```
with decreasing Evans–Honkapohja gain ``\gamma_t = 1/t``. Initial belief ``\hat\kappa_0 = 0.02``; burn-in is 200 periods.

## Two policy regimes

- **Passive (Case 1).** ``\phi_\pi`` is fixed at ``\phi_\pi^*(\hat\kappa_0)`` for all ``t`` — the CB's belief evolves in the background but does not feed back into policy.
- **Active (Case 2).** ``\phi_\pi`` is re-optimised every period to minimise the subjective loss
  ```math
  L(\phi_\pi; \hat\kappa_t) = \operatorname{Var}(\pi_t \mid \hat\kappa_t) + \lambda_y \operatorname{Var}(y_t \mid \hat\kappa_t),
  ```
  closing the belief → policy → data → belief loop.

For estimation, the inner problem is precomputed on a 31-point grid in ``\hat\kappa`` and replaced by a linear interpolant. This is what makes the outer objective differentiable under ForwardDiff: Brent's minimiser strips ``\operatorname{Dual}`` types, the interpolant preserves them.

## SMM criterion

With sample moments ``\hat m`` and simulated moments ``m(\theta)``,
```math
\hat\theta = \arg\min_\theta (m(\theta) - \hat m)'\, W\, (m(\theta) - \hat m).
```
The moment vector has eleven entries: four VAR(1) cross-coefficients on ``(\pi, y, r)``, the residual variances of ``\pi`` and ``y``, three contemporaneous cross-covariances, and the first-order autocovariances of ``\pi`` and ``y``. The estimated parameter vector is ``\theta = (\kappa, \rho_d, \rho_v, \sigma_d, \sigma_v, \sigma_{my}, \sigma_{m\pi})``.

## Inference

Standard errors come from a nonparametric bootstrap: twenty independent shock paths at ``\hat\theta``, L-BFGS restarted from ``\hat\theta`` on each path, reported SEs are the sample standard deviations of the twenty replicate estimates. The procedure captures simulation variance and finite-sample criterion curvature in a single pass.
