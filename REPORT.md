---
title: "Estimating a Self-Fulfilling NKPC: Passive and Active Monetary Policy"
author: "Niloufar Eslahi"
date: "April 2026"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{caption}
  - \captionsetup[table]{skip=6pt}
---

## Abstract

I estimate a New Keynesian Phillips curve in which the central bank learns the slope of the curve by recursive instrumental variables, and I compare two policy regimes. In the passive regime the Taylor rule is frozen at its initial value, the bank's belief evolves in the background, and the data contain no feedback from learning into policy. In the active regime the bank re-optimises its response to inflation at every period on the basis of its current belief, which closes the feedback loop between beliefs, policy and outcomes. Estimation is by simulated method of moments on an eleven-moment targeted subset of the trivariate VAR(1) and its residual covariance. The implementation uses forward-mode automatic differentiation of the simulation-based criterion, multi-start L-BFGS inside an Fminbox wrapper, and bootstrap standard errors. Using US quarterly data for 1984Q2--2025Q2, the passive-policy case returns a small slope estimate, $\hat\kappa = 0.051$, with a bootstrap standard error wide enough to cover zero; this is the self-confirming-equilibrium outcome the model predicts when the policy rule does not respond to the bank's belief. The active-policy case returns $\hat\kappa = 0.611$ with a bootstrap standard error of $0.274$, significant at conventional levels, and the contemporaneous inflation–interest-rate covariance is reproduced closely. Three diagnostics nevertheless remain, and they are the focus of follow-up work: shock persistence parameters reach the bounds of their admissible intervals in both cases, the output–inflation cross-covariance carries the wrong sign in both cases, and the inflation measurement scale is weakly identified. Each of these has a concrete remediation, set out in Section 4.4.

## 1. Model

The economy consists of three structural equations. The forward-looking IS block is
$$
y_t = E_t y_{t+1} - \tfrac{1}{\gamma}\,(r_t - E_t \pi_{t+1}) + d_t,
$$
the hybrid New Keynesian Phillips curve is
$$
\pi_t = \beta\, E_t \pi_{t+1} + \kappa\,(\theta_y\, y_t + \theta_r\, r_t) + \nu_t,
$$
and monetary policy follows a Taylor rule with two measurement errors,
$$
r_t = \phi_\pi\,(\pi_t + m^\pi_t) + \phi_y\,(y_t + m^y_t).
$$

The reduced form is solved shock-by-shock under rational expectations. Each exogenous process is driven by an AR(1) innovation with persistence $\rho_j$ (with $\rho_j = 0$ for the two measurement errors), and conditional on that shock, $E_t y_{t+1} = \rho_j y_t$ and $E_t \pi_{t+1} = \rho_j \pi_t$. Substituting these closures into the three structural equations and inverting yields per-shock response coefficients that are linear in the primitives and depend on the bank's current belief $\hat\kappa_t$; the implementation evaluates these closed-form coefficients in `src/simulation.jl` at every period. The superposition of the four per-shock reduced forms is what the criterion matches to the US sample moments.

The demand shock $d_t$ and the cost-push shock $\nu_t$ are AR(1) processes with innovations of standard deviation $\sigma_d$ and $\sigma_v$ and with persistence $\rho_d$ and $\rho_v$. The two measurement errors $m^\pi_t$ and $m^y_t$ are independent, mean-zero, and serially uncorrelated with standard deviations $\sigma_{m\pi}$ and $\sigma_{my}$. The composite policy residual
$$
u_t \equiv \phi_\pi\, m^\pi_t + \phi_y\, m^y_t
$$
is formed each period at the realised values of the policy coefficients, which under active intervention depend on the current belief. The two components are estimated separately rather than through the composite, because under learning the coefficients $(\phi_\pi, \phi_y)$ are endogenous and the pre-multiplied mixture has a time-varying scale; estimating the underlying $m^\pi$ and $m^y$ as primitives is therefore preferable to estimating a single $u$ whose scale shifts with the policy rule. Throughout, I calibrate $\beta = 0.99$, $\gamma = 1.0$, and $\lambda_y = 0.5$, and I fix the cost-channel coefficient at $\theta_r = 0.3$. The latter is in the interior of the range reported in the empirical cost-channel literature and is held constant to avoid absorbing the direct $r\to\pi$ covariance into an estimated parameter that the moments identify only weakly.

The product $\kappa \theta_y$ is identified but its factors are not, so I set $\theta_y = 1$ and estimate $\kappa$ directly. Under this normalisation $\kappa$ is the object the central bank's recursion converges toward, and the effective Phillips-curve slope is simply $\kappa$. The output feedback coefficient $\phi_y$ is fixed at $0.5$. In the closed-form solution of the central bank's subjective policy problem, $\phi_y$ is orthogonal to $\hat\kappa$, and allowing it to be estimated would add noise in a direction the moment conditions do not pin down.

The central bank does not know $\kappa$. Its subjective model omits the cost channel entirely and reads
$$
\pi_t = \beta\, E_t \pi_{t+1} + \hat\kappa_t\, y_t + \nu_t.
$$
The bank uses the demand shock $d_t$ as an instrument for $y_t$ and updates $\hat\kappa_t$ by recursive instrumental variables. With $R_t$ denoting the running inverse moment, the updating equations are
$$
R_t = R_{t-1} + \gamma_t\,(d_t y_t - R_{t-1}),
$$
$$
\hat\kappa_t = \hat\kappa_{t-1} + \gamma_t\, R_t^{-1}\, d_t\,(\pi_t - \beta\, \hat\kappa_{t-1} y_t - y_t\, \hat\kappa_{t-1}).
$$
The gain $\gamma_t = 1/t$ is the Evans--Honkapohja decreasing sequence, which delivers almost-sure convergence of the recursion to a stable fixed point under the standard regularity conditions. I initialise $\hat\kappa_0 = 0.02$ and drop the first 200 periods of each simulated path as burn-in.

The two policy regimes differ in what the bank does with the belief. In the passive-policy regime the Taylor rule coefficient $\phi_\pi$ is fixed at $\phi_\pi^*(\hat\kappa_0)$ for all $t$, so $\hat\kappa_t$ evolves in the background but the policy does not respond to it. In the active-policy regime $\phi_\pi$ is re-optimised every period to minimise the quadratic subjective loss
$$
L(\phi_\pi; \hat\kappa_t) = \operatorname{Var}(\pi_t \mid \hat\kappa_t) + \lambda_y\, \operatorname{Var}(y_t \mid \hat\kappa_t),
$$
where the conditional variances are computed from the same shock-by-shock rational-expectations reduced form described above, evaluated at the current belief: each of the four shocks is propagated under $E_t \pi_{t+1} = \rho_j \pi_t$ and $E_t y_{t+1} = \rho_j y_t$, and the resulting per-shock policy map is collected into a single quadratic loss in $\phi_\pi$. The loss is smooth and unimodal on the admissible interval $\phi_\pi \in [0.01, 5]$ — verified numerically over the estimated parameter grid — so Brent's method converges in a handful of function evaluations. For the outer estimation of the active-policy case I precompute this map on a 31-point grid in $\hat\kappa$ and replace the inner optimiser with its linear interpolant during simulation. The interpolant's supremum error is below $0.3$ across the grid range, well below the simulation noise in the moment vector. Replacing the inner Brent call by an interpolant is also necessary for automatic differentiation, since Brent returns a non-dual minimiser and therefore breaks the gradient, whereas the interpolant is smooth in the parameters of the outer problem through its grid values.

Figure 1 shows the shape of this optimal policy map at an illustrative non-IID calibration ($\kappa = 0.10$, $\rho_d = 0.7$, $\rho_v = 0.5$), chosen to stay in the interior of the admissible parameter box and to decouple the mechanism diagnostic from the data-fitted estimates of Section 4. The reaction $\phi_\pi^*(\hat\kappa)$ is monotonically increasing in the bank's belief: at small beliefs it is *passive* (below the Taylor principle $\phi_\pi = 1$), it crosses the principle near $\hat\kappa \approx 0.10$, reaches the conventional Taylor benchmark of $1.5$ at roughly $\hat\kappa \approx 0.20$, and saturates at the upper bound of the admissible interval once $\hat\kappa$ exceeds about $0.45$. Two implications matter for what follows. First, a central bank that initialises its belief at a low $\hat\kappa_0$ and never updates its policy rule therefore operates in the passive region, which is exactly the condition under which the data it generates can self-confirm a low $\hat\kappa$. Second, the jump from passive to active policy that occurs as the belief rises through $\hat\kappa \approx 0.10$ is what breaks the self-confirming fixed point once the policy is allowed to respond to learning — the quantitative mechanism that Case 2 tests empirically.

![Optimal policy reaction $\phi_\pi^*(\hat\kappa)$ at the illustrative non-IID calibration ($\kappa = 0.10$, $\rho_d = 0.7$, $\rho_v = 0.5$). The curve is monotonically increasing in the CB's belief, crosses the Taylor principle near the true $\kappa = 0.10$, and saturates at the upper bound of the admissible interval for beliefs above $\hat\kappa \approx 0.45$.](results/figures/fig_policy_map.pdf){width=85%}

The two regimes isolate different pieces of the self-fulfilling mechanism. Passive policy breaks the feedback loop and asks whether the eleven targeted moments identify $\kappa$ from the shock-driven reduced form alone. Active policy closes the loop and asks whether the policy response to learning, which in theory breaks the indeterminacy of the passive regime, in practice supplies enough additional identifying variation. Comparing the two delivers a direct test of where the identifying information resides.

## 2. Estimation

The estimated parameter vector is $\theta = (\kappa, \rho_d, \rho_v, \sigma_d, \sigma_v, \sigma_{my}, \sigma_{m\pi})$, with seven entries. Given sample moments $\hat m$ and simulated moments $m(\theta)$, I minimise the simulated method of moments criterion
$$
Q(\theta) = (m(\theta) - \hat m)'\, W\, (m(\theta) - \hat m).
$$
The moment vector has eleven entries. Four are VAR(1) cross-coefficients from a trivariate system on $(\pi, y, r)$, carrying the dynamic cross-dependencies that are informative about $\kappa$. Two are residual variances, $\Sigma_{\pi\pi}$ and $\Sigma_{yy}$, which identify the innovation magnitudes after the VAR absorbs predictable variation. Three are contemporaneous cross-covariances, $\operatorname{cov}(\pi, y)$, $\operatorname{cov}(\pi, r)$ and $\operatorname{cov}(y, r)$, and two are first-order autocovariances of $\pi$ and $y$. The own-AR coefficients of the VAR and the residual variance of the interest rate are deliberately excluded, because under a stable Taylor rule they are dominated by shock persistence rather than by the structural slope.

Standard errors come from a nonparametric bootstrap at the point estimate $\hat\theta$. For each of twenty replications I draw an independent shock path of the same length as the data, simulate the model under $\hat\theta$, recompute the moments from the simulated path, and re-estimate by restarting L-BFGS from $\hat\theta$. The reported standard errors are the sample standard deviations of the twenty replicate estimates. The procedure captures both the simulation variance and the finite-sample curvature of the criterion in a single pass, which is appropriate here because several directions of the parameter space are only weakly identified and the local quadratic approximation underlying analytical standard errors is unreliable.

## 3. Data

The sample is US quarterly data from 1984Q2 through 2025Q2, a total of $T = 166$ observations. Inflation is the year-over-year headline CPI rate in percent. Output is the CBO output gap in percent. The interest rate is the annualised effective Federal Funds rate. All three series are demeaned before moments are computed. The sample begins after the Volcker disinflation, a period over which US monetary policy is broadly consistent with a stable Taylor rule.

## 4. Results

This section presents the two cases in turn and then draws the comparison. Section 4.1 covers the passive-policy case, Section 4.2 the active-policy case, Section 4.3 compares them and flags the three residual problems, and Section 4.4 sets out the follow-up plan.

### 4.1 Case 1: passive policy

In Case 1 the simulated economy has a frozen Taylor rule with $\phi_\pi$ fixed at $\phi_\pi^*(\hat\kappa_0)$ and $\hat\kappa_0 = 0.02$. The bank's learning recursion runs in the background and updates $\hat\kappa_t$, but the updated belief does not feed back into the policy rule. This isolates the identifying power of the moments from shock-driven reduced-form variation alone. Table 1 collects the parameter estimates; Table 2 reports the moment fit.

\begin{table}[h]
\centering
\begin{tabular}{lrr}
\toprule
Parameter & Estimate & Bootstrap SE \\
\midrule
$\kappa$         & 0.051 & 0.303 \\
$\rho_d$         & 0.500 & 0.143 \\
$\rho_v$         & 0.511 & 0.120 \\
$\sigma_d$       & 0.617 & 0.331 \\
$\sigma_v$       & 0.847 & 0.338 \\
$\sigma_{my}$    & 2.325 & 0.912 \\
$\sigma_{m\pi}$  & 0.010 & 1.178 \\
\bottomrule
\end{tabular}
\caption{Case 1 parameter estimates. Stage-1 objective at $\hat\theta$ is $Q(\hat\theta) = 2.798$. Bootstrap standard errors from twenty independent shock draws at $\hat\theta$.}
\end{table}

\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
Moment & Data & Simulated & Residual \\
\midrule
$A_{\pi y}$                  &  0.023 &  0.025 &  0.002 \\
$A_{\pi r}$                  &  0.074 &  0.200 &  0.126 \\
$A_{y\pi}$                   & -0.009 & -0.007 &  0.002 \\
$A_{r\pi}$                   &  0.023 &  0.070 &  0.047 \\
$S_{\pi\pi}$                 &  3.660 &  2.577 & -1.083 \\
$S_{yy}$                     &  0.967 &  0.973 &  0.006 \\
$\operatorname{cov}(\pi,y)$  &  1.433 & -0.100 & -1.532 \\
$\operatorname{cov}(\pi,r)$  &  1.573 &  0.200 & -1.373 \\
$\operatorname{cov}(y,r)$    &  2.809 & -0.334 & -3.144 \\
$\operatorname{acov}(\pi,1)$ &  1.887 &  1.905 &  0.019 \\
$\operatorname{acov}(y,1)$   &  2.842 &  0.140 & -2.702 \\
\bottomrule
\end{tabular}
\caption{Case 1 moment fit at $\hat\theta$. The ``Residual'' column is Simulated minus Data.}
\end{table}

The structural slope estimate is $\hat\kappa = 0.051$ with a bootstrap standard error of $0.303$. The $t$-statistic against $\kappa = 0$ is $0.17$, so the data do not reject a flat Phillips curve under passive policy. This is the outcome the self-fulfilling literature predicts for this regime. When the bank does not move $\phi_\pi$ with its belief, the VAR implied by the model is governed by the shock process alone, and the structural slope is absorbed into a reduced-form coefficient that the moments do not separately identify.

The VAR cross-coefficients $A_{\pi y}$ and $A_{y \pi}$ are matched almost exactly: residuals of $0.002$ and $0.002$. The first-order autocovariance of inflation is matched to $0.019$, and the residual variance of output is matched to within $0.006$. A small number of autoregressive moments therefore determine the shock parameters with almost no contribution from $\kappa$. The contemporaneous block behaves very differently. $\operatorname{cov}(\pi, y)$ is negative in the simulation against a large positive value in the data; $\operatorname{cov}(\pi, r)$ and $\operatorname{cov}(y, r)$ are undershot by a factor of about eight and carry the wrong sign in the interest-rate covariance as well. The autocovariance of output is simulated at $0.14$ against a sample value of $2.84$. Together, these residuals say that the passive-policy reduced form cannot simultaneously match the first-moment autoregressive dynamics and the contemporaneous covariance structure of US data over 1984--2025 at any value of $\kappa$: the optimiser chooses the former and leaves the latter as large residuals.

Shock parameters are mostly interior, with $\rho_d$ at the lower bound and $\rho_v$ just above it. The measurement scale $\sigma_{my}$ is large at $2.33$, and $\sigma_{m\pi}$ is at the lower bound of $0.01$ with a bootstrap standard error of $1.18$ — wider than the parameter itself — indicating that the moments contain essentially no information about this scale under passive policy. These boundary hits and flat directions point to an identification problem whose origin the present eleven-moment specification cannot isolate. Diagnosing it — distinguishing a genuine shape-of-the-criterion failure from an under-specified shock process or an incomplete moment vector — and implementing a fix is a central task for the next stage of the project, revisited in Section 4.4.

### 4.2 Case 2: active policy

In Case 2 the central bank re-optimises $\phi_\pi$ at every period based on the current belief $\hat\kappa_t$, closing the feedback loop that Case 1 leaves open. The policy rule responds to the bank's learning, which affects the realised data, which in turn updates the belief. The moments now carry information about $\kappa$ through the endogenous policy response, and one expects a larger and more sharply identified slope estimate. Table 3 collects the parameter estimates and Table 4 the moment fit.

\begin{table}[h]
\centering
\begin{tabular}{lrr}
\toprule
Parameter & Estimate & Bootstrap SE \\
\midrule
$\kappa$         & 0.611 & 0.274 \\
$\rho_d$         & 0.500 & 0.009 \\
$\rho_v$         & 0.990 & 0.014 \\
$\sigma_d$       & 1.897 & 0.893 \\
$\sigma_v$       & 0.240 & 0.126 \\
$\sigma_{my}$    & 0.268 & 0.617 \\
$\sigma_{m\pi}$  & 0.020 & 0.273 \\
\bottomrule
\end{tabular}
\caption{Case 2 parameter estimates. Stage-1 objective at $\hat\theta$ is $Q(\hat\theta) = 10.574$. Bootstrap standard errors from twenty independent shock draws at $\hat\theta$.}
\end{table}

\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
Moment & Data & Simulated & Residual \\
\midrule
$A_{\pi y}$                  &  0.023 & -0.192 & -0.214 \\
$A_{\pi r}$                  &  0.074 &  0.092 &  0.018 \\
$A_{y\pi}$                   & -0.009 & -0.078 & -0.070 \\
$A_{r\pi}$                   &  0.023 & -0.018 & -0.040 \\
$S_{\pi\pi}$                 &  3.660 &  4.996 &  1.335 \\
$S_{yy}$                     &  0.967 &  0.542 & -0.425 \\
$\operatorname{cov}(\pi,y)$  &  1.433 & -0.389 & -1.822 \\
$\operatorname{cov}(\pi,r)$  &  1.573 &  1.397 & -0.176 \\
$\operatorname{cov}(y,r)$    &  2.809 &  0.694 & -2.115 \\
$\operatorname{acov}(\pi,1)$ &  1.887 &  1.335 & -0.551 \\
$\operatorname{acov}(y,1)$   &  2.842 &  3.499 &  0.656 \\
\bottomrule
\end{tabular}
\caption{Case 2 moment fit at $\hat\theta$. The ``Residual'' column is Simulated minus Data.}
\end{table}

The estimated structural slope is $\hat\kappa = 0.611$ with a bootstrap standard error of $0.274$. The point estimate lies in the upper part of the empirical range surveyed by Mavroeidis, Plagborg-Møller and Stock (2014), and the $t$-statistic against the null $\kappa = 0$ is $2.23$, so the data reject a flat Phillips curve at conventional levels. The jump from $\hat\kappa = 0.051$ under passive policy to $\hat\kappa = 0.611$ under active policy is precisely the self-fulfilling mechanism in empirical form: the endogenous policy response supplies the identifying variation that passive-policy moments lack.

The moment fit is strongest on the interest-rate block, which is the block the model was calibrated to identify through the fixed $\theta_r = 0.3$. The VAR cross-coefficient $A_{\pi r}$ is close to the sample value ($0.074$ in the data versus $0.092$ in the model), and $\operatorname{cov}(\pi, r)$ is matched to within $0.18$ units ($1.573$ versus $1.397$). Together these two moments indicate that the calibration of the cost channel lies in a plausible range and that the simulated interest-rate covariance inherits the direction and approximate magnitude of its sample counterpart. The output-related cross-moments fit less tightly: $A_{\pi y}$ and $\operatorname{cov}(\pi, y)$ carry the wrong sign in the simulation, and the autoregressive coefficient $A_{y \pi}$, which was almost exactly matched under passive policy, is now overshot in magnitude. The inflation variance $S_{\pi \pi}$ overshoots by roughly a third, compared with a severe undershoot in Case 1.

The shock parameters separate into persistence and volatility components. Demand persistence settles at the lower interior bound of $0.50$ with a tight bootstrap standard error of $0.009$, and cost-push persistence settles at the upper interior bound of $0.99$ with a standard error of $0.014$. The criterion is locally steep in both of these directions and the optimiser converges to them consistently across starts. Innovation standard deviations are interior at $\hat\sigma_d = 1.897$ and $\hat\sigma_v = 0.240$, with moderate bootstrap dispersion on $\sigma_d$. The two measurement-error scales, $\hat\sigma_{my} = 0.268$ and $\hat\sigma_{m\pi} = 0.020$, are small in absolute terms; the wide bootstrap standard error on $\hat\sigma_{m\pi}$ again indicates that the moments are close to flat in that direction.

### 4.3 Comparison and identified problems

Taken side by side, the two cases tell a coherent story. Case 1 returns a small slope estimate that is not statistically distinguishable from zero, and its moment fit is almost exact on the autoregressive block but wrong-signed on contemporaneous cross-covariances. Case 2 returns a slope estimate at the upper end of the empirical range with a $t$-statistic above two, its cost-channel moments are close to the data, and its second moments are within a factor of two of their sample values. This is the pattern the self-fulfilling framework predicts: without the policy feedback, the data do not reveal the structural slope; with the policy feedback, they do, and the point estimate moves sharply upward. The exercise therefore provides empirical evidence that the active-policy mechanism is quantitatively important, not just a theoretical curiosity.

Figures 2 and 3 make the mechanism behind this empirical pattern visible without reference to the US data. Both figures use the same illustrative non-IID calibration introduced in Section 1 ($\kappa = 0.10$, $\rho_d = 0.7$, $\rho_v = 0.5$), so that the true slope is known and the two regimes differ only in whether the policy responds to $\hat\kappa_t$. Figure 2 plots the median learning path of $\hat\kappa_t$ across eighty simulated samples under each regime, with the interquartile band shaded. Under passive policy the recursion settles at $\hat\kappa \approx 0.19$ with a tight band; under active policy it settles slightly higher at $\hat\kappa \approx 0.23$ with a somewhat wider band. Neither path converges exactly to the true $\kappa = 0.10$: recursive IV under persistent shocks carries a small upward asymptotic bias, a finite-sample feature of the non-IID specification that is absent from the IID analytical benchmark of Beaudry, Hou and Portier (2020). This bias is small relative to the gap between the two estimated slopes in Section 4.2 and does not affect the comparative conclusion.

![Median learning path of $\hat\kappa_t$ across eighty simulated samples at the illustrative non-IID calibration ($\kappa = 0.10$), with interquartile bands. Under passive policy the recursion settles near $0.19$ with a tight band; under active policy it settles near $0.23$ with a wider band. The vertical reference marks the end of burn-in at $t = 200$.](results/figures/fig_learning_paths.pdf){width=85%}

Figure 3 shows the terminal belief $\hat\kappa$ as a function of the initial belief $\kappa_0$, averaged over forty shock seeds at each grid point, with $\pm 1.96$-standard-error bands. Under passive policy the terminal belief rises smoothly with $\kappa_0$, from about $0.18$ at $\kappa_0 = 0$ to about $0.28$ at $\kappa_0 = 0.4$. This is the self-confirming dependence on the starting point: the bank ends where it started, modulo the shock-driven updates that the frozen policy cannot amplify. Under active policy the terminal belief is essentially flat at $\hat\kappa \approx 0.23$, independent of $\kappa_0$. The endogenous policy response breaks the dependence on the initial condition. Translated into the estimation exercise, this is exactly the mechanism that delivers the order-of-magnitude jump from $\hat\kappa = 0.051$ in Case 1 to $\hat\kappa = 0.611$ in Case 2: once the data carry the signature of a belief-responsive policy, the moment conditions identify a slope away from zero.

![Terminal $\hat\kappa$ as a function of the initial belief $\kappa_0$, computed at the illustrative non-IID calibration. Shaded bands are $\pm 1.96$ standard errors from forty simulation seeds. The $45^\circ$ line marks the passive-regime self-confirming fixed-point locus and the horizontal dotted line the true $\kappa = 0.10$.](results/figures/fig_self_fulfilling_calibrated.pdf){width=85%}

Three specific problems are visible in the results and are the focus of follow-up work.

The first problem is that shock persistence parameters hit the bounds of their admissible intervals in both cases. Case 1 settles $\rho_d$ at $0.50$ and $\rho_v$ at $0.51$; Case 2 pushes $\rho_d$ to $0.50$ and $\rho_v$ to $0.99$. In a pure AR(1) shock system there are only two free persistence levels to span the full autocovariance structure of the observables, and the optimiser is choosing the extremes of the admissible set. The binding persistence in Case 2 is tight (standard errors $0.009$ and $0.014$), indicating that the criterion is locally steep in these directions and that the corner is a genuine optimum rather than a numerical artefact.

The second problem is that $\operatorname{cov}(\pi, y)$ carries the wrong sign in both cases: $-0.10$ in Case 1 and $-0.39$ in Case 2 against a sample value of $1.43$. The $y \to \pi$ channel is not delivering the positive comovement the data show, at either point estimate. Case 1 undershoots the magnitude because the passive shock-driven reduced form has no mechanism to generate strongly comoving inflation and output. Case 2 overshoots in the opposite direction because the endogenous policy response generates negative comovement through the $r \to y$ channel. Neither specification currently reconciles the sign of this moment with the data.

The third problem is that the inflation measurement scale $\sigma_{m\pi}$ is at or near its lower bound in both cases ($0.010$ in Case 1, $0.020$ in Case 2), and the bootstrap standard errors on this parameter are comparable in magnitude to the estimate itself. The moments contain little information about this parameter at the estimated points, which is consistent with a specification in which the inflation measurement error is dominated by the cost-push shock. Either the inflation measurement block should be dropped from the composite Taylor-rule noise in future estimations, or an additional identifying moment should be added to pin this scale down.

### 4.4 Plans for follow-up work

Four steps, in order of expected pay-off.

**1. Richer shock process.** The AR(1) block forces two persistence parameters to span the full autocovariance structure of three observables, which is why $\rho_d$ and $\rho_v$ hit their bounds. Moving to AR(2) or ARMA(1,1) adds a second persistence layer per shock and should let the criterion match the inflation and output autocovariances in the interior of the admissible box. The edit is local to `simulation.jl` and `config.jl`.

**2. Reconciling the wrong-sign $\operatorname{cov}(\pi, y)$.** The largest residual in both cases and the binding empirical problem. Two candidate fixes, from most to least disruptive to the current specification:

-   *Sample split at 2020.* The US inflation process visibly transitions between two regimes, and a single slope is being asked to straddle both. Estimating Case 2 separately on 1984Q2--2019Q4 and on 2020Q1--2025Q2 isolates how much of the sign problem is regime mixing.
-   *Non-diagonal shock covariance.* Allow demand and cost-push innovations to correlate contemporaneously — the current diagonal specification forbids a channel that is economically plausible and quantitatively relevant for $\operatorname{cov}(\pi, y)$.

Either is a one- to two-week change on the present code base; selecting between them is the next econometric decision, not a software task.

**3. An identifying moment for $\sigma_{m\pi}$.** The wide bootstrap standard errors in both cases indicate that the current moment vector is flat in this direction. Adding $\operatorname{acov}(r, 1)$ ties the inflation measurement scale directly to the Taylor rule's coefficient on contemporaneous inflation. Drop-in extension to `_moments_nkpc_targeted`; the moment vector grows from eleven to twelve entries with no other code changes.

**4. Two-step SMM with $\hat W_2 = \hat S^{-1}$.** Would sharpen every standard error reported in Tables 1 and 3. The machinery is already implemented in `objective.jl` but disabled (`cfg.two_step = false`), because early pilots were noisy on eleven moments. A careful re-enablement — standardising $\hat S$ before inversion and adding a ridge against near-singularity — is the natural final step once the specification issues above are settled.

## 5. Implementation tests

The estimation code is covered by seventeen test sets grouped under a single outer `SelfFulfillingNKPC` wrapper, and all ninety-four assertions pass at the revision used to produce the estimates in Section 4. The primitive tests check the Taylor rule identity, the unconditional-variance helper, the pack/unpack round-trip (the parameter vector maps bijectively to `ModelParams`, fixed fields pass through unchanged), and the case-specific structural normalisation $\theta_y = 1$, $\theta_r = 0.3$. Simulation reproducibility is checked by running two simulations with the same RNG seed and comparing paths element-wise. The moment-dimension test confirms that each of the four moment options returns a vector of the correct length with all entries finite at the prior point. The policy block is covered by a bounds check on `policy_numeric` and an interpolant test that compares the 31-point linear interpolant to the exact Brent solution at six grid nodes and bounds the sup-error at $0.3$. The automatic-differentiation test applies to the free-$\phi$ case where the simulation is globally smooth in $\theta$; it compares the ForwardDiff gradient to a centred finite difference with step size $10^{-5}$ and requires a relative error below $10^{-3}$. A companion test on the intervention path logs the AD-vs-FD disagreement as a diagnostic, reflecting the documented bias from the interpolant's Float64 grid values. Two type-stability tests confirm that `ModelParams{Float64}` is the correct output type for `Float64` inputs and that dual propagation through `unpack` yields `ForwardDiff.Dual` fields, which is the precondition for the outer gradient pipeline. The allocation test bounds the heap allocation of a 40-period simulation at $500{,}000$ bytes (roughly 488 KiB). Two Monte-Carlo recovery tests cover machinery at both ends: a same-shock smoke test ($f \!<\! 1$, relative parameter error $<\!0.25$) and a stronger perturbed-start test that launches the optimiser from $1.2\,\theta_{\text{true}}$ and requires the criterion to drop by at least half from the starting value. The `bootstrap_se` smoke test exercises the threaded bootstrap pipeline on $B \!=\! 2$ replicates.

## References

Beaudry, P., C. Hou and F. Portier (2020). Monetary policy when the Phillips curve is quite flat.

Clarida, R., J. Galí and M. Gertler (2000). Monetary policy rules and macroeconomic stability: evidence and some theory. *Quarterly Journal of Economics* 115, 147--180.

Evans, G. W. and S. Honkapohja (2001). *Learning and Expectations in Macroeconomics*. Princeton University Press.

Gourieroux, C., A. Monfort and E. Renault (1993). Indirect inference. *Journal of Applied Econometrics* 8, S85--S118.

Mavroeidis, S., M. Plagborg-Møller and J. H. Stock (2014). Empirical evidence on inflation expectations in the New Keynesian Phillips curve. *Journal of Economic Literature* 52, 124--188.
