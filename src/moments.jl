# ──────────────────────────────────────────────────────────────────────────────
# Moment computation for SMM estimation.
#
# Default set (:var1_bivariate) stacks vec(A) and vech(Σ) from a bivariate
# VAR(1) on (π, y) with marginal variances and Cov(π,y) — 11 moments total.
# Alternative sets add the r equation or swap in targeted cross-covariances.
# ──────────────────────────────────────────────────────────────────────────────

# ── Linear algebra utilities ─────────────────────────────────────────────────

"""Column-stack vectorisation."""
vec_mat(A::AbstractMatrix) = A[:]

"""Half-vectorisation (lower triangle including diagonal)."""
function vech(S::AbstractMatrix)
    n = size(S, 1)
    [S[i, j] for i in 1:n for j in 1:i]
end

# ── VAR(1) estimation by OLS ────────────────────────────────────────────────

"""
    var1_ols(X) -> (A, Σ)

Estimate X_t = A X_{t-1} + ε_t by OLS.
Returns the coefficient matrix A and the residual covariance Σ.
"""
function var1_ols(X::AbstractMatrix)
    Y    = X[2:end, :]
    Xlag = X[1:end-1, :]
    B    = Xlag \ Y          # B = (Xlag'Xlag)⁻¹ Xlag'Y
    A    = transpose(B)
    resid = Y - Xlag * B
    Sigma = (transpose(resid) * resid) / size(resid, 1)
    (A, Sigma)
end

# ── Lag covariance ──────────────────────────────────────────────────────────

"""
    cov_lag(x, y, L)

Sample cross-covariance at lag `L`, computed on the aligned slices
`x[1+L:end]` and `y[1:end-L]` after demeaning each slice. `L=0` gives the
contemporaneous covariance.
"""
function cov_lag(x::AbstractVector, y::AbstractVector, L::Int)
    if L == 0
        x1, y1 = x, y
    else
        x1 = x[1+L:end]
        y1 = y[1:end-L]
    end
    x1 = x1 .- mean(x1)
    y1 = y1 .- mean(y1)
    mean(x1 .* y1)
end

# ── Named moment vector ─────────────────────────────────────────────────────

"""
    MomentVector{T}

Named vector of moments: the numeric `values` and matching `names`. Supports
`length` and integer indexing.
"""
struct MomentVector{T<:Real}
    values ::Vector{T}
    names  ::Vector{String}
end

Base.length(m::MomentVector)  = length(m.values)
Base.getindex(m::MomentVector, i) = m.values[i]

# ── Compute moments from a T×3 data matrix ──────────────────────────────────

"""
    compute_moments(X; option=:var1_bivariate) -> MomentVector

Compute the empirical moment vector from a T×3 matrix [π, y, r].
"""
function compute_moments(X::AbstractMatrix{T};
                         option::Symbol=:var1_bivariate) where T
    if option == :var1_bivariate
        return _moments_var1_bivariate(X)
    elseif option == :var1_trivariate
        return _moments_var1_trivariate(X)
    elseif option == :nkpc_targeted
        return _moments_nkpc_targeted(X)
    elseif option == :cov_matrix
        return _moments_cov_matrix(X)
    else
        error("Unknown moment option: $option")
    end
end

"""
Targeted moments for NKPC slope (κ) and cost channel (θ_r) identification.

Uses 11 moments: VAR(1) cross-coefficients linking π↔y and π↔r (4), residual
variances of π and y (2), contemporaneous cross-covariances (3), and first-lag
autocovariances of π and y (2).  Drops the own-AR coefficients and Var(r)
which mainly pin down shock persistence rather than structural slopes.
"""
function _moments_nkpc_targeted(X::AbstractMatrix{T}) where T
    A, Sigma = var1_ols(X)
    vals = T[
        A[1,2],        # A_pi_y  — how y predicts π (slope channel)
        A[1,3],        # A_pi_r  — how r predicts π (cost channel)
        A[2,1],        # A_y_pi  — how π predicts y (demand channel)
        A[3,1],        # A_r_pi  — how π predicts r (policy response)
        Sigma[1,1],    # S_pi_pi — residual inflation variance
        Sigma[2,2],    # S_y_y   — residual output variance
        cov(X[:,1], X[:,2], corrected=false),   # cov(π,y)
        cov(X[:,1], X[:,3], corrected=false),   # cov(π,r)
        cov(X[:,2], X[:,3], corrected=false),   # cov(y,r)
        cov_lag(X[:,1], X[:,1], 1),             # acov(π,1)
        cov_lag(X[:,2], X[:,2], 1),             # acov(y,1)
    ]
    mnames = ["A_pi_y", "A_pi_r", "A_y_pi", "A_r_pi",
              "S_pi_pi", "S_y_y",
              "cov_pi_y", "cov_pi_r", "cov_y_r",
              "acov_pi_1", "acov_y_1"]
    MomentVector(vals, mnames)
end

"""
    _moments_var1_trivariate(X)

VAR(1) on the full (π, y, r) system plus cross-covariances at lag 0. This
gives 9 + 6 + 3 = 18 moments — richer identification of the interest-rate
channel (θ_r) than the bivariate set, which ignores r entirely.
"""
function _moments_var1_trivariate(X::AbstractMatrix{T}) where T
    A, Sigma = var1_ols(X)
    va = vec_mat(A)          # 9 coefficients
    vs = vech(Sigma)         # 6 residual (co)variances
    c01 = cov(X[:,1], X[:,2], corrected=false)
    c02 = cov(X[:,1], X[:,3], corrected=false)
    c12 = cov(X[:,2], X[:,3], corrected=false)
    vals = vcat(va, vs, [c01, c02, c12])
    vnames = ("pi","y","r")
    mnames = String[]
    for i in 1:3, j in 1:3
        push!(mnames, "A_$(vnames[i])_$(vnames[j])")
    end
    for i in 1:3, j in 1:i
        push!(mnames, "S_$(vnames[i])_$(vnames[j])")
    end
    push!(mnames, "cov_pi_y", "cov_pi_r", "cov_y_r")
    MomentVector(vals, mnames)
end

"""
    _moments_cov_matrix(X)

Lightweight moment set: variances of (π, y, r), pairwise covariances, and
first-lag autocovariances of π and y. 8 moments total — used for quick
diagnostics and MC recovery where a compact, well-conditioned set is useful.
"""
function _moments_cov_matrix(X::AbstractMatrix{T}) where T
    pi_s = X[:, 1]; y_s = X[:, 2]; r_s = X[:, 3]
    vals = T[
        var(pi_s, corrected=false),
        var(y_s,  corrected=false),
        var(r_s,  corrected=false),
        cov(pi_s, y_s, corrected=false),
        cov(pi_s, r_s, corrected=false),
        cov(y_s,  r_s, corrected=false),
        cov_lag(pi_s, pi_s, 1),
        cov_lag(y_s,  y_s,  1),
    ]
    names = ["var_pi", "var_y", "var_r",
             "cov_pi_y", "cov_pi_r", "cov_y_r",
             "acov_pi_1", "acov_y_1"]
    MomentVector(vals, names)
end

function _moments_var1_bivariate(X::AbstractMatrix{T}) where T
    Xxy = X[:, 1:2]       # π, y only
    A, Sigma = var1_ols(Xxy)
    va = vec_mat(A)
    vs = vech(Sigma)

    var_pi = var(X[:, 1], corrected=false)
    var_y  = var(X[:, 2], corrected=false)
    var_r  = var(X[:, 3], corrected=false)

    C = cov(X[:, 1], X[:, 2], corrected=false)

    vals = vcat(va, vs, [var_pi, var_y, var_r, C])

    varnames = ["pi", "y"]
    n = 2
    mnames = String[]
    for i in 1:n, j in 1:n
        push!(mnames, "A_$(varnames[i])_$(varnames[j])")
    end
    for i in 1:n, j in 1:i
        push!(mnames, "S_$(varnames[i])_$(varnames[j])")
    end
    push!(mnames, "var_pi", "var_y", "var_r", "cov_pi_y_0")

    MomentVector(vals, mnames)
end

# ── Convenience: moments from parameter vector ──────────────────────────────

"""
    moments_from_theta(theta, cfg, T_obs; inn_d, inn_v, inn_my, inn_mpi) -> MomentVector

Simulate the model at θ and return the simulated moment vector.
"""
function moments_from_theta(theta::AbstractVector, cfg::EstimationConfig,
                            T_obs::Int;
                            inn_d::Vector{Float64}, inn_v::Vector{Float64},
                            inn_my::Vector{Float64}, inn_mpi::Vector{Float64},
                            policy_itp=nothing)
    sim = simulate(theta, cfg, T_obs;
                   inn_d=inn_d, inn_v=inn_v, inn_my=inn_my, inn_mpi=inn_mpi,
                   policy_itp=policy_itp)
    compute_moments(sim.X; option=cfg.moment_option)
end
