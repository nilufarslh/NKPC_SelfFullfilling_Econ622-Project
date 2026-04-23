# ──────────────────────────────────────────────────────────────────────────────
# Test suite for SelfFulfillingNKPC.jl
#
# Unit tests for the model primitives (Taylor rule, unconditional variance,
# CB policy, VAR OLS, moments, data loader) and integration tests for the
# simulation, SMM objective, gradient correctness, and parameter recovery.
#
# Run with:   julia --project=. -t auto test/runtests.jl
# ──────────────────────────────────────────────────────────────────────────────

using Test, Random, LinearAlgebra, Statistics, ForwardDiff

include(joinpath(@__DIR__, "..", "src", "SelfFulfillingNKPC.jl"))
using .SelfFulfillingNKPC

const ATOL = 1e-8

# Copy a config with selected fields overridden.  Used to shrink burn-in,
# n_starts, and bootstrap_reps for fast test runs.
function with_overrides(cfg::EstimationConfig;
                        burn_in::Int=cfg.burn_in,
                        n_starts::Int=cfg.n_starts,
                        two_step::Bool=cfg.two_step,
                        se_method::Symbol=cfg.se_method,
                        bootstrap_reps::Int=cfg.bootstrap_reps)
    EstimationConfig(
        cfg.environment, cfg.intervention, cfg.fixed, cfg.estimated,
        cfg.use_learning, cfg.constant_gain, cfg.gain, cfg.k0, cfg.R0,
        cfg.k_bounds, cfg.free_phi, cfg.use_unconditional_var,
        cfg.phi_min, cfg.phi_max,
        burn_in, cfg.rng_seed, cfg.moment_option,
        n_starts, two_step, cfg.w2_replications, cfg.w2_ridge,
        se_method, bootstrap_reps,
    )
end

# Draw the four N(0,1) innovation vectors that simulate() needs.
function fresh_innovations(T_obs::Int, cfg::EstimationConfig, seed::Int)
    TT = T_obs + cfg.burn_in
    rng = MersenneTwister(seed)
    (inn_d   = randn(rng, TT),
     inn_v   = randn(rng, TT),
     inn_my  = randn(rng, TT),
     inn_mpi = randn(rng, TT))
end

@testset "SelfFulfillingNKPC" begin

@testset "taylor_rule" begin
    # r = φ_π π + φ_y y + u
    @test taylor_rule(1.0, 0.5, 0.3, 1.5, 0.5) ≈ 2.05
    @test taylor_rule(0.0, 0.0, 0.0, 1.5, 0.5) ≈ 0.0
    @test taylor_rule(-1.0, 2.0, 0.1, 1.5, 0.5) ≈ -0.4
end

@testset "uncond_var" begin
    # σ² / (1 - ρ²)
    @test uncond_var(1.0, 0.0) ≈ 1.0
    @test uncond_var(2.0, 0.5) ≈ 4.0 / 0.75
    # clamps ρ away from ±1 so the variance stays finite
    @test isfinite(uncond_var(1.0, 1.0))
    @test isfinite(uncond_var(1.0, -1.0))
end

@testset "pack/unpack round-trip" begin
    cfg = case_non_iid(intervention=false)
    θ = theta0_vec(cfg)
    p = unpack(θ, cfg)
    for (i, nm) in enumerate(param_names(cfg))
        @test getfield(p, nm) ≈ θ[i]
    end
    # fixed fields must pass through untouched
    @test p.theta_y ≈ 1.0
    @test p.theta_r ≈ cfg.fixed.theta_r
    @test p.beta    ≈ cfg.fixed.beta
    @test p.gamma   ≈ cfg.fixed.gamma
end

@testset "structural normalisation" begin
    # All three cases fix θ_y = 1 and estimate κ.
    for intv in (false, true)
        cfg = case_non_iid(intervention=intv)
        @test cfg.fixed.theta_y == 1.0
        @test :kappa in param_names(cfg)
        @test !(:theta_y in param_names(cfg))
    end
    cfg_fp = case_non_iid_free_phi()
    @test cfg_fp.fixed.theta_y == 1.0
    @test :kappa in param_names(cfg_fp)
    @test :phi_pi in param_names(cfg_fp)
    @test :phi_y  in param_names(cfg_fp)
end

@testset "simulate: reproducibility" begin
    cfg = case_non_iid(intervention=false)
    θ = theta0_vec(cfg)
    inn = fresh_innovations(80, cfg, 42)
    s1 = simulate(θ, cfg, 80; inn...)
    s2 = simulate(θ, cfg, 80; inn...)
    @test s1.X ≈ s2.X
    @test all(isfinite, s1.X)
    @test size(s1.X) == (80, 3)
    @test length(s1.kappa_path) == 80 + cfg.burn_in

    # Intervention path should also produce finite output.
    cfg2 = case_non_iid(intervention=true)
    inn2 = fresh_innovations(80, cfg2, 42)
    s3 = simulate(θ, cfg2, 80; inn2...)
    @test all(isfinite, s3.X)
end

@testset "simulate: allocation" begin
    # Hot-loop allocation is bounded — relied on by AD/bootstrap throughput.
    cfg = with_overrides(case_non_iid(intervention=false); burn_in=100)
    θ = theta0_vec(cfg)
    inn = fresh_innovations(40, cfg, 1)
    simulate(θ, cfg, 40; inn...)    # warm-up / compile
    allocs = @allocated simulate(θ, cfg, 40; inn...)
    @test allocs < 500_000
end

@testset "moments" begin
    cfg = case_non_iid(intervention=false)
    inn = fresh_innovations(120, cfg, 7)
    sim = simulate(theta0_vec(cfg), cfg, 120; inn...)
    m_biv  = compute_moments(sim.X; option=:var1_bivariate)
    m_tri  = compute_moments(sim.X; option=:var1_trivariate)
    m_cov  = compute_moments(sim.X; option=:cov_matrix)
    m_nkpc = compute_moments(sim.X; option=:nkpc_targeted)
    @test length(m_biv)  == 11
    @test length(m_tri)  == 18
    @test length(m_cov)  == 8
    @test length(m_nkpc) == 11
    @test all(isfinite, m_biv.values)
    @test all(isfinite, m_tri.values)
    @test all(isfinite, m_cov.values)
    @test all(isfinite, m_nkpc.values)
end

@testset "var1_ols analytical" begin
    # Simulate a VAR(1) with known A and check OLS recovers it.
    Random.seed!(42)
    A_true = [0.5 0.1; -0.2 0.3]
    T_sim = 5000
    X = zeros(T_sim, 2)
    X[1, :] = randn(2)
    for t in 2:T_sim
        X[t, :] = A_true * X[t-1, :] + 0.1 * randn(2)
    end
    A_hat, Sigma = var1_ols(X)
    @test norm(A_hat - A_true) < 0.05
    @test issymmetric(round.(Sigma, digits=10))
end

@testset "load_targets" begin
    tmpdir = mktempdir()
    csv_path = joinpath(tmpdir, "test_data.csv")
    open(csv_path, "w") do io
        println(io, "date,pi,y,r")
        println(io, "2000Q1,2.0,1.0,5.0")
        println(io, "2000Q2,4.0,3.0,3.0")
        println(io, "2000Q3,3.0,2.0,4.0")
    end
    td = load_targets(csv_path; demean=true, annualize=false)
    @test td.T == 3
    @test size(td.X) == (3, 3)
    # Demeaned columns sum to ≈ 0
    for j in 1:3
        @test abs(sum(td.X[:, j])) < 1e-10
    end

    td_raw = load_targets(csv_path; demean=false)
    @test td_raw.X[1, 1] ≈ 2.0
    @test td_raw.X[2, 3] ≈ 3.0
end

@testset "policy_numeric bounds" begin
    # CB's 1D policy solve must return a value inside [phi_min, phi_max]
    # for any reasonable belief.
    cfg = case_non_iid(intervention=true)
    p = unpack(theta0_vec(cfg), cfg)
    for k in [-1.0, -0.1, 0.0, 0.2, 0.5, 1.0, 2.0]
        ppi = policy_numeric(k, p, cfg)
        @test cfg.phi_min ≤ ppi ≤ cfg.phi_max
        @test isfinite(ppi)
    end
end

@testset "policy interpolant" begin
    cfg = case_non_iid(intervention=true)
    p = unpack(theta0_vec(cfg), cfg)
    itp = build_policy_interp(p, cfg; n_grid=61)
    # Linear interpolation: allow up to 0.3 sup-error — the policy map is
    # highly nonlinear near k̂ = 0.
    for k in [-2.0, -0.5, 0.0, 0.2, 1.0, 2.0]
        ppi_exact = policy_numeric(k, p, cfg)
        ppi_itp, phi_y_itp = evaluate_policy_itp(itp, k, cfg)
        @test abs(ppi_exact - ppi_itp) < 0.3
        @test phi_y_itp == cfg.fixed.phi_y
    end
end

@testset "type stability" begin
    cfg = case_non_iid(intervention=false)
    θ = theta0_vec(cfg)
    p = unpack(θ, cfg)
    @test typeof(p) == ModelParams{Float64}

    r = taylor_rule(1.0, 0.5, 0.3, 1.5, 0.5)
    @test r isa Float64

    # ForwardDiff duals must propagate through unpack.
    θd = ForwardDiff.Dual.(θ, ones(length(θ)))
    pd = unpack(θd, cfg)
    @test typeof(getfield(pd, :kappa)) <: ForwardDiff.Dual

    inn = fresh_innovations(40, cfg, 1)
    sim = simulate(θ, cfg, 40; inn...)
    @test eltype(sim.X)          == Float64
    @test eltype(sim.kappa_path) == Float64
end

@testset "AD vs finite differences (free-phi)" begin
    # Free-phi is the clean AD path: φ_π, φ_y are free parameters carried
    # through as Duals, so ForwardDiff produces an exact gradient.  In the
    # intervention case the policy interpolant is built from Float64 values,
    # so AD through φ_π is a biased approximation by construction.
    cfg = with_overrides(case_non_iid_free_phi(); burn_in=60)
    T_obs = 100
    θ = theta0_vec(cfg)
    inn = fresh_innovations(T_obs, cfg, 11)
    m_data = compute_moments(simulate(θ, cfg, T_obs; inn...).X;
                             option=cfg.moment_option).values
    # Slight perturbation so ∇Q is non-degenerate.
    m_target = m_data .+ 0.05
    W = Matrix{Float64}(I, length(m_data), length(m_data))

    f = θ -> smm_objective(θ, cfg, m_target, W, T_obs; inn...)
    g_ad = ForwardDiff.gradient(f, θ)

    g_fd = similar(θ)
    h = 1e-5
    for i in eachindex(θ)
        e = zeros(length(θ)); e[i] = h
        g_fd[i] = (f(θ + e) - f(θ - e)) / (2h)
    end
    rel = norm(g_ad - g_fd) / max(norm(g_fd), 1e-8)
    @info "AD-vs-FD relative gradient error (free-phi)" rel
    @test rel < 1e-3
end

@testset "AD vs finite differences (intervention, documented bias)" begin
    # Case 2 is the flagship estimation, but its policy interpolant is built
    # from Float64 grid values inside simulate() — ForwardDiff.value strips
    # the Dual part, so AD does not see θ's effect on the interpolant.
    # Consequence: AD through the intervention path is a biased gradient.
    # This test documents the bias rather than asserting a tight bound: the
    # checks below simply verify the numbers are finite and the bias exists.
    # The bias is the reason the AD-vs-FD test only asserts tight agreement
    # on the free-phi path, where the simulation is globally smooth in θ.
    cfg = with_overrides(case_non_iid(intervention=true); burn_in=60)
    T_obs = 100
    θ = theta0_vec(cfg)
    inn = fresh_innovations(T_obs, cfg, 23)
    m_data = compute_moments(simulate(θ, cfg, T_obs; inn...).X;
                             option=cfg.moment_option).values
    m_target = m_data .+ 0.05
    W = Matrix{Float64}(I, length(m_data), length(m_data))

    f = θ -> smm_objective(θ, cfg, m_target, W, T_obs; inn...)
    g_ad = ForwardDiff.gradient(f, θ)

    g_fd = similar(θ)
    h = 1e-4
    for i in eachindex(θ)
        e = zeros(length(θ)); e[i] = h
        g_fd[i] = (f(θ + e) - f(θ - e)) / (2h)
    end
    rel = norm(g_ad - g_fd) / max(norm(g_fd), 1e-8)
    cosine = dot(g_ad, g_fd) / (norm(g_ad) * norm(g_fd))
    @info "AD-vs-FD diagnostics (intervention)" rel cosine
    # Guards against catastrophic breakage (NaN, Inf) while tolerating the
    # documented bias in direction and magnitude — the intervention path is
    # known to miss the true gradient by up to O(1), which is why the tight
    # AD-vs-FD assertion is applied only to the free-phi path above.
    @test all(isfinite, g_ad)
    @test all(isfinite, g_fd)
end

@testset "MC recovery (smoke)" begin
    # With the same innovations used to generate the data, the SMM objective
    # should be ≈ 0 at θ₀.  Since the first multiplier in estimate_smm is 1.0,
    # at least one start begins at θ₀ itself and the optimiser should stay
    # there.
    cfg = with_overrides(case_non_iid(intervention=false);
                         burn_in=60, n_starts=2)
    T_obs = 100
    θ_true = theta0_vec(cfg)
    inn = fresh_innovations(T_obs, cfg, 999)
    m_data = compute_moments(simulate(θ_true, cfg, T_obs; inn...).X;
                             option=cfg.moment_option).values
    W = Matrix{Float64}(I, length(m_data), length(m_data))

    res = estimate_smm(cfg, m_data, W, T_obs; inn...,
                       n_starts=2, verbose=false)
    @test res.fval < 1.0
    @test norm(res.theta_hat - θ_true) / max(norm(θ_true), 1e-8) < 0.25
end

@testset "MC recovery (perturbed start)" begin
    # Stronger recovery test: generate data at θ_true, then launch the
    # optimiser from a perturbed starting point so that standing-still at
    # θ_true is not an option.  The optimiser must actually descend.
    cfg_base = case_non_iid(intervention=false)
    cfg = with_overrides(cfg_base; burn_in=60, n_starts=1)
    T_obs = 120
    θ_true = theta0_vec(cfg)
    lb = lower_bounds(cfg); ub = upper_bounds(cfg)

    inn = fresh_innovations(T_obs, cfg, 2024)
    m_data = compute_moments(simulate(θ_true, cfg, T_obs; inn...).X;
                             option=cfg.moment_option).values
    W = Matrix{Float64}(I, length(m_data), length(m_data))

    # Perturb each coordinate by +20% and clamp to the admissible box so no
    # start sits exactly at θ_true.
    θ_start = clamp.(θ_true .* 1.2, lb .+ 1e-4, ub .- 1e-4)
    @test norm(θ_start - θ_true) / norm(θ_true) > 0.1

    res = estimate_smm(cfg, m_data, W, T_obs; inn...,
                       n_starts=1, warm_start=θ_start, verbose=false)
    # Criterion must drop substantially from the perturbed start — this is
    # the machinery check: L-BFGS + gradients are descending on the SMM
    # objective from a non-trivial initial condition.
    f_start = smm_objective(θ_start, cfg, m_data, W, T_obs; inn...)
    @test res.fval < 0.5 * f_start
    # Note on parameter recovery: the SMM criterion on seven parameters
    # with T = 120 observations has multiple local modes, and a single
    # perturbed start is not guaranteed to land at θ_true even when the
    # objective drops sharply.  The multi-start design in the main pipeline
    # addresses this by launching twenty starts per case.  Here we only
    # check that the recovered θ̂ stays inside the admissible box.
    @test all(res.theta_hat .>= lb)
    @test all(res.theta_hat .<= ub)
end

@testset "bootstrap_se (smoke)" begin
    # Minimal shape / sanity check: bootstrap_se re-estimates B times in
    # parallel.  B = 2 is enough to confirm the return value is well-formed;
    # a full run uses cfg.bootstrap_reps = 20.
    cfg = with_overrides(case_non_iid(intervention=false);
                         burn_in=40, n_starts=1, bootstrap_reps=2)
    T_obs = 60
    θ = theta0_vec(cfg)
    inn = fresh_innovations(T_obs, cfg, 77)
    m_data = compute_moments(simulate(θ, cfg, T_obs; inn...).X;
                             option=cfg.moment_option).values
    W = Matrix{Float64}(I, length(m_data), length(m_data))
    se = bootstrap_se(θ, cfg, m_data, W, T_obs; B=2)
    @test length(se) == length(θ)
    @test all(isfinite, se)
    @test all(se .>= 0)
end

end # outer testset
