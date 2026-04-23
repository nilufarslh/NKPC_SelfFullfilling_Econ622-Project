# ──────────────────────────────────────────────────────────────────────────────
# Policy interpolation (φ_π as function of k̂, φ_y fixed)
# ──────────────────────────────────────────────────────────────────────────────

"""
    PolicyInterp

Linear interpolant for `φ_π(κ̂)`: the belief grid and the interpolation
object. Built once per SMM objective evaluation so that ForwardDiff can flow
through the policy map instead of through Brent's non-differentiable minimiser.
"""
struct PolicyInterp
    k_grid    ::Vector{Float64}
    phi_pi_itp
end

"""
    build_policy_interp(p, cfg; n_grid=31) -> PolicyInterp

Solve the CB's Brent problem at `n_grid` equally-spaced beliefs in `cfg.k_bounds`
and wrap the result in a linear interpolant with linear extrapolation.
"""
function build_policy_interp(p::ModelParams{Float64}, cfg::EstimationConfig;
                             n_grid::Int=31)
    k_grid = range(cfg.k_bounds[1], cfg.k_bounds[2]; length=n_grid)
    phi_pi_vals = [policy_numeric(k, p, cfg) for k in k_grid]
    itp = linear_interpolation(collect(k_grid), phi_pi_vals, extrapolation_bc=Line())
    PolicyInterp(collect(k_grid), itp)
end

"""
    evaluate_policy_itp(itp, k̂, cfg) -> (φ_π, φ_y)

Evaluate the interpolant at belief `k̂`, clamp `φ_π` to `[phi_min, phi_max]`,
and return alongside the fixed `φ_y` from `cfg.fixed`.
"""
function evaluate_policy_itp(itp::PolicyInterp, k_hat, cfg::EstimationConfig)
    ppi = itp.phi_pi_itp(k_hat)
    (clamp(ppi, cfg.phi_min, cfg.phi_max), cfg.fixed.phi_y)
end
