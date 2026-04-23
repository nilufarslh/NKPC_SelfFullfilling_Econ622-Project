# ──────────────────────────────────────────────────────────────────────────────
# SMM objective: innovations held fixed, no priors
# ──────────────────────────────────────────────────────────────────────────────

"""
    smm_objective(theta, cfg, m_data, W, T_obs; inn_d, inn_v, inn_my, inn_mpi, policy_itp=nothing)

The SMM criterion `(m(θ) - m̂)' W (m(θ) - m̂)`. Innovations are held fixed
across evaluations so the objective is a deterministic function of `θ`;
that is what lets ForwardDiff produce exact gradients. If the simulated
moments go non-finite, a large quadratic penalty in `θ - θ₀` is returned so
the outer optimiser can still descend.
"""
function smm_objective(theta::AbstractVector{T}, cfg::EstimationConfig,
                       m_data::Vector{Float64}, W::Matrix{Float64},
                       T_obs::Int;
                       inn_d::Vector{Float64}, inn_v::Vector{Float64},
                       inn_my::Vector{Float64}, inn_mpi::Vector{Float64},
                       policy_itp=nothing) where T
    sim = simulate(theta, cfg, T_obs;
                   inn_d=inn_d, inn_v=inn_v, inn_my=inn_my, inn_mpi=inn_mpi,
                   policy_itp=policy_itp)
    mv = compute_moments(sim.X; option=cfg.moment_option)
    m_sim = mv.values

    if length(m_sim) != length(m_data) || any(!isfinite, m_sim)
        theta0 = T.(theta0_vec(cfg))
        return T(1e10) + sum(abs2, theta .- theta0)
    end

    diff = m_sim .- T.(m_data)
    dot(diff, T.(W) * diff)
end
