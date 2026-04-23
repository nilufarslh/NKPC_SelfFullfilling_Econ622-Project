# ──────────────────────────────────────────────────────────────────────────────
# Simulation: persistent RF, decomposed m_y/m_π, innovations held fixed
# ──────────────────────────────────────────────────────────────────────────────

"""
    SimulationResult{T}

One simulated path: the post-burn-in observable matrix `X = [π y r]`, the
three series, and the full-length paths of the belief `κ̂` and the applied
policy coefficients `φ_π, φ_y`.
"""
struct SimulationResult{T<:Real}
    X            ::Matrix{T}
    pi           ::Vector{T}
    y            ::Vector{T}
    r            ::Vector{T}
    kappa_path   ::Vector{T}
    phi_pi_path  ::Vector{T}
    phi_y_path   ::Vector{T}
end

"""
    simulate(theta, cfg, T_obs; inn_d, inn_v, inn_my, inn_mpi, policy_itp)

Persistent RF simulation with decomposed measurement errors.
Innovations (N(0,1)) are held fixed; AR(1) processes constructed with current σ,ρ.
u_t = φ_π(t) × m_π_t + φ_y × m_y_t composed each period with current policy.
"""
function simulate(theta::AbstractVector{T}, cfg::EstimationConfig, T_obs::Int;
                  inn_d::Vector{Float64}, inn_v::Vector{Float64},
                  inn_my::Vector{Float64}, inn_mpi::Vector{Float64},
                  policy_itp=nothing) where T

    p = unpack(theta, cfg)
    burn_in = cfg.burn_in
    TT = T_obs + burn_in

    # AR(1) from innovations with CURRENT σ, ρ
    rd = clamp(p.rho_d, T(-0.9999), T(0.9999))
    rv = clamp(p.rho_v, T(-0.9999), T(0.9999))
    d_t = zeros(T, TT); v_t = zeros(T, TT)
    d_t[1] = p.sigma_d / sqrt(1 - rd^2) * inn_d[1]
    v_t[1] = p.sigma_v / sqrt(1 - rv^2) * inn_v[1]
    for t in 2:TT
        d_t[t] = rd * d_t[t-1] + p.sigma_d * inn_d[t]
        v_t[t] = rv * v_t[t-1] + p.sigma_v * inn_v[t]
    end
    # m_y, m_π: iid
    my_t = p.sigma_my .* inn_my
    mpi_t = p.sigma_mpi .* inn_mpi

    pi_t = zeros(T, TT); y_t = zeros(T, TT); r_t = zeros(T, TT)
    k_path = zeros(T, TT); phi_p = zeros(T, TT); phi_y_p = zeros(T, TT)

    k_hat = T(cfg.k0); R = T(cfg.R0)

    # Policy setup
    tv = eltype(theta) <: ForwardDiff.Dual ? ForwardDiff.value.(theta) : Float64.(theta)
    pf = unpack(tv, cfg)

    local itp_pi
    if cfg.intervention && !cfg.free_phi
        if policy_itp !== nothing
            itp_pi = policy_itp
        else
            kg = collect(range(cfg.k_bounds[1], cfg.k_bounds[2]; length=31))
            pv = [policy_numeric(k, pf, cfg) for k in kg]
            itp_pi = linear_interpolation(kg, pv, extrapolation_bc=Line())
        end
    end

    if !cfg.intervention && !cfg.free_phi
        phi_pi_fix = policy_numeric(cfg.k0, pf, cfg)
        phi_y_fix = pf.phi_y
    elseif cfg.free_phi
        phi_pi_fix = p.phi_pi
        phi_y_fix = p.phi_y
    else
        phi_pi_fix = T(NaN); phi_y_fix = T(NaN)
    end

    for t in 1:TT
        if cfg.free_phi
            phi_pi = phi_pi_fix; phi_y = phi_y_fix
        elseif cfg.intervention
            k_val = eltype(k_hat) <: ForwardDiff.Dual ? ForwardDiff.value(k_hat) : Float64(k_hat)
            phi_pi = T(clamp(itp_pi(k_val), cfg.phi_min, cfg.phi_max))
            phi_y = p.phi_y
        else
            phi_pi = T(phi_pi_fix); phi_y = phi_y_fix
        end

        # Compose u with CURRENT φ
        u_now = phi_pi * mpi_t[t] + phi_y * my_t[t]
        e = (d_t[t], v_t[t], u_now)

        # Persistent RF (per-shock Θ_j, correct Cramer determinant)
        rho_vec = (p.rho_d, p.rho_v, zero(T))
        yn = zero(T); pn = zero(T)
        for j in 1:3
            rj = rho_vec[j]; brj = p.beta * rj
            a11 = (1 - rj) * p.gamma + phi_y
            a12 = phi_pi
            a21 = -p.kappa * (p.theta_y + p.theta_r * phi_y)
            a22 = (1 - brj) - p.kappa * p.theta_r * phi_pi
            dv = a11 * a22 - a12 * a21
            id = dv / (dv^2 + T(1e-14))
            if j == 1;     by = p.gamma * e[j]; bp = zero(T)
            elseif j == 2; by = zero(T);        bp = e[j]
            else;          by = -e[j];          bp = p.kappa * p.theta_r * e[j]
            end
            yn += (a22 * by - a12 * bp) * id
            pn += (a11 * bp - a21 * by) * id
        end
        rn = phi_pi * pn + phi_y * yn + e[3]

        # Learning
        if cfg.use_learning
            gt = T(1 / max(t, 1))
            d = e[1]; kk = k_hat
            R = R + gt * (d * yn - R)
            Rs = sqrt(R^2 + T(1e-14))
            k_hat = kk + gt / Rs * d * (pn - p.beta * kk * yn - kk * yn)
            k_hat = clamp(k_hat, T(cfg.k_bounds[1]), T(cfg.k_bounds[2]))
        end

        y_t[t] = yn; pi_t[t] = pn; r_t[t] = rn
        k_path[t] = k_hat; phi_p[t] = phi_pi; phi_y_p[t] = phi_y
    end

    idx = (burn_in + 1):TT
    X = hcat(pi_t[idx], y_t[idx], r_t[idx])
    SimulationResult(X, pi_t[idx], y_t[idx], r_t[idx],
                     k_path, phi_p, phi_y_p)
end
