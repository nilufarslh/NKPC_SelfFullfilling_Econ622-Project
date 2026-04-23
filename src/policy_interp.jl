# ──────────────────────────────────────────────────────────────────────────────
# Policy interpolation (φ_π as function of k̂, φ_y fixed)
# ──────────────────────────────────────────────────────────────────────────────

struct PolicyInterp
    k_grid    ::Vector{Float64}
    phi_pi_itp
end

function build_policy_interp(p::ModelParams{Float64}, cfg::EstimationConfig;
                             n_grid::Int=31)
    k_grid = range(cfg.k_bounds[1], cfg.k_bounds[2]; length=n_grid)
    phi_pi_vals = [policy_numeric(k, p, cfg) for k in k_grid]
    itp = linear_interpolation(collect(k_grid), phi_pi_vals, extrapolation_bc=Line())
    PolicyInterp(collect(k_grid), itp)
end

function evaluate_policy_itp(itp::PolicyInterp, k_hat, cfg::EstimationConfig)
    ppi = itp.phi_pi_itp(k_hat)
    (clamp(ppi, cfg.phi_min, cfg.phi_max), cfg.fixed.phi_y)
end
