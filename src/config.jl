# ──────────────────────────────────────────────────────────────────────────────
# Parameter containers and case-profile configuration
#
# All cases estimate κ with θ_y = 1 fixed. θ_r = 0.3 calibrated.
# NKPC: π = β E π + κ(θ_y y + θ_r r) + ν
# CB's subjective model: π = β E π + k̂ y + ν  (omits θ_r)
#
# Shocks: d (demand), ν (cost-push), m_y (output measurement), m_π (inflation measurement)
# u_t = φ_π m_π_t + φ_y m_y_t composed each period with current policy.
# σ_my and σ_mπ estimated separately.
# ──────────────────────────────────────────────────────────────────────────────

"""
    ModelParams{T}

All structural and policy parameters of the model. Mutable so that
[`unpack`](@ref) can overwrite the estimated fields while the rest stay at
their calibrated values. Parametric in `T` so that ForwardDiff duals
propagate through the entire simulation.
"""
Base.@kwdef mutable struct ModelParams{T<:Real}
    beta      ::T = 0.99
    gamma     ::T = 1.00
    lambda_y  ::T = 0.50
    theta_y   ::T = 1.00    # FIXED at 1 in all cases
    theta_r   ::T = 0.30    # CALIBRATED
    kappa     ::T = 0.20    # estimated in all cases

    rho_d     ::T = 0.85
    rho_v     ::T = 0.85
    rho_u     ::T = 0.00    # unused (u composed from m_y, m_π)

    sigma_d   ::T = 1.00
    sigma_v   ::T = 0.50
    sigma_u   ::T = 0.50    # unused
    sigma_my  ::T = 0.24
    sigma_mpi ::T = 0.05

    phi_pi    ::T = 1.50
    phi_y     ::T = 0.50
end

"""
    ParamSpec(name, theta0, lb, ub)

Estimation spec for one parameter: its symbol, starting value, and box
bounds. A vector of these lives on `EstimationConfig.estimated`.
"""
struct ParamSpec
    name  ::Symbol
    theta0::Float64
    lb    ::Float64
    ub    ::Float64
end

"""
    EstimationConfig

Everything that defines one estimation case: fixed vs estimated parameters,
learning settings, CB policy bounds, burn-in and seed, which moment set to
match, and how to compute standard errors. Build one via [`case_non_iid`](@ref)
or [`case_non_iid_free_phi`](@ref).
"""
struct EstimationConfig
    environment  ::Symbol
    intervention ::Bool
    fixed        ::ModelParams{Float64}
    estimated    ::Vector{ParamSpec}

    use_learning   ::Bool
    constant_gain  ::Bool
    gain           ::Float64
    k0             ::Float64
    R0             ::Float64
    k_bounds       ::Tuple{Float64,Float64}

    free_phi             ::Bool
    use_unconditional_var::Bool
    phi_min              ::Float64
    phi_max              ::Float64

    burn_in   ::Int
    rng_seed  ::Int
    moment_option ::Symbol

    n_starts      ::Int
    two_step      ::Bool
    w2_replications::Int
    w2_ridge       ::Float64

    se_method      ::Symbol
    bootstrap_reps ::Int
end

# ──────────────────────────────────────────────────────────────────────────────
# Case profiles — ALL estimate κ (θ_y=1, θ_r=0.3 fixed)
# ──────────────────────────────────────────────────────────────────────────────

function _base_config(;environment, intervention, free_phi, estimated)
    fixed = ModelParams{Float64}(
        theta_y  = 1.0,
        theta_r  = 0.3,
        kappa    = 0.20,
        phi_y    = 0.5,
        rho_u    = 0.0,
    )

    EstimationConfig(
        environment,
        intervention,
        fixed,
        estimated,
        true,           # use_learning
        false,          # constant_gain
        0.05,           # gain (unused)
        0.02,           # k0
        1.0,            # R0
        (-5.0, 5.0),    # k_bounds
        free_phi,
        true,           # use_unconditional_var
        0.01,           # phi_min
        5.00,           # phi_max
        200,            # burn_in
        123,            # rng_seed
        :nkpc_targeted,
        20,             # n_starts (Case 1,3); overridden to 10 for Case 2
        false,          # two_step OFF
        150,            # w2_replications (unused)
        1e-6,           # w2_ridge (unused)
        :bootstrap,
        20,             # bootstrap_reps
    )
end

"""
    case_non_iid(; intervention) -> EstimationConfig

Build the standard non-iid case: seven estimated parameters (κ, the two AR(1)
persistences and volatilities, and the two measurement-error volatilities).
`intervention=false` freezes the Taylor coefficient at `φ_π(κ̂₀)` (Case 1);
`intervention=true` re-optimises it every period as the belief updates (Case 2).
"""
function case_non_iid(;intervention::Bool)
    estimated = [
        ParamSpec(:kappa,     0.20, 0.001,  1.00),
        ParamSpec(:rho_d,     0.85, 0.50,   0.99),
        ParamSpec(:rho_v,     0.85, 0.50,   0.99),
        ParamSpec(:sigma_d,   1.00, 0.10,   5.00),
        ParamSpec(:sigma_v,   0.50, 0.10,   5.00),
        ParamSpec(:sigma_my,  0.24, 0.01,   3.00),
        ParamSpec(:sigma_mpi, 0.05, 0.01,   3.00),
    ]
    _base_config(environment=:non_iid, intervention=intervention,
                 free_phi=false, estimated=estimated)
end

"""
    case_non_iid_free_phi(; intervention=true) -> EstimationConfig

Variant that treats `φ_π` and `φ_y` as free parameters rather than solving
the CB's inner policy problem. The objective is globally smooth in `θ`, so
this is the clean path for the AD-vs-finite-difference gradient test.
"""
function case_non_iid_free_phi(;intervention::Bool=true)
    estimated = [
        ParamSpec(:kappa,     0.20, 0.001,  1.00),
        ParamSpec(:rho_d,     0.85, 0.50,   0.99),
        ParamSpec(:rho_v,     0.85, 0.50,   0.99),
        ParamSpec(:sigma_d,   1.00, 0.10,   5.00),
        ParamSpec(:sigma_v,   0.50, 0.10,   5.00),
        ParamSpec(:sigma_my,  0.24, 0.01,   3.00),
        ParamSpec(:sigma_mpi, 0.05, 0.01,   3.00),
        ParamSpec(:phi_pi,    1.50, 0.01,   5.00),
        ParamSpec(:phi_y,     0.50, 0.01,   5.00),
    ]
    _base_config(environment=:non_iid_free_phi, intervention=intervention,
                 free_phi=true, estimated=estimated)
end

# ──────────────────────────────────────────────────────────────────────────────
# Pack / unpack
# ──────────────────────────────────────────────────────────────────────────────

"""
    param_names(cfg) -> Vector{Symbol}

Names of the estimated parameters in `cfg`, in vector order.
"""
param_names(cfg::EstimationConfig) = [s.name for s in cfg.estimated]

"""
    theta0_vec(cfg) -> Vector{Float64}

Starting values for the estimated parameters, in vector order.
"""
theta0_vec(cfg::EstimationConfig)  = [s.theta0 for s in cfg.estimated]

"""
    lower_bounds(cfg) -> Vector{Float64}

Lower box bounds for the estimated parameters.
"""
lower_bounds(cfg::EstimationConfig) = [s.lb for s in cfg.estimated]

"""
    upper_bounds(cfg) -> Vector{Float64}

Upper box bounds for the estimated parameters.
"""
upper_bounds(cfg::EstimationConfig) = [s.ub for s in cfg.estimated]

"""
    unpack(theta, cfg) -> ModelParams{T}

Turn an estimated parameter vector into a fully-populated `ModelParams`,
preserving the element type `T` so ForwardDiff duals flow through end-to-end.
Fixed fields are taken from `cfg.fixed`; estimated fields are overwritten
from `theta` in the order given by `param_names(cfg)`.
"""
function unpack(theta::AbstractVector{T}, cfg::EstimationConfig) where T
    p = ModelParams{T}(
        beta      = T(cfg.fixed.beta),
        gamma     = T(cfg.fixed.gamma),
        lambda_y  = T(cfg.fixed.lambda_y),
        theta_y   = T(cfg.fixed.theta_y),
        theta_r   = T(cfg.fixed.theta_r),
        kappa     = T(cfg.fixed.kappa),
        rho_d     = T(cfg.fixed.rho_d),
        rho_v     = T(cfg.fixed.rho_v),
        rho_u     = T(cfg.fixed.rho_u),
        sigma_d   = T(cfg.fixed.sigma_d),
        sigma_v   = T(cfg.fixed.sigma_v),
        sigma_u   = T(cfg.fixed.sigma_u),
        sigma_my  = T(cfg.fixed.sigma_my),
        sigma_mpi = T(cfg.fixed.sigma_mpi),
        phi_pi    = T(cfg.fixed.phi_pi),
        phi_y     = T(cfg.fixed.phi_y),
    )
    names = param_names(cfg)
    for (i, nm) in enumerate(names)
        setfield!(p, nm, theta[i])
    end
    p
end
