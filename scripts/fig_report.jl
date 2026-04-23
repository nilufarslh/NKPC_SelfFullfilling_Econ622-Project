# ──────────────────────────────────────────────────────────────────────────────
# Three report figures for the non-IID self-fulfilling NKPC:
#   1. Learning paths κ̂_t  (case 1 vs case 2)
#   2. Self-confirming κ̂(κ₀) — non-IID calibration
#   3. Policy reaction φ_π*(k̂)
#
# All figures use a common moderate non-IID calibration that avoids the
# boundary pathologies of the estimated parameters (ρ_v = 0.99 boundary).
# ──────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots, Random, Statistics, Printf, DataFrames, CSV

include(joinpath(@__DIR__, "..", "src", "SelfFulfillingNKPC.jl"))
using .SelfFulfillingNKPC

const RESDIR = joinpath(@__DIR__, "..", "results")
const FIGDIR = joinpath(RESDIR, "figures")
mkpath(FIGDIR)
gr()

# Moderate non-IID calibration:
#   κ = 0.10  (true slope, away from bounds)
#   ρ_d = 0.70, ρ_v = 0.50  (persistent but not at unit root)
#   σ's chosen to produce empirically plausible variability
const CAL = Dict(
    :kappa     => 0.10,
    :rho_d     => 0.70,
    :rho_v     => 0.50,
    :sigma_d   => 1.00,
    :sigma_v   => 0.50,
    :sigma_my  => 0.24,
    :sigma_mpi => 0.05,
)

function theta_cal(cfg::EstimationConfig)
    [CAL[nm] for nm in param_names(cfg)]
end

function with_k0(cfg::EstimationConfig, new_k0::Float64)
    EstimationConfig(
        cfg.environment, cfg.intervention, cfg.fixed, cfg.estimated,
        cfg.use_learning, cfg.constant_gain, cfg.gain,
        new_k0, cfg.R0, cfg.k_bounds,
        cfg.free_phi, cfg.use_unconditional_var, cfg.phi_min, cfg.phi_max,
        cfg.burn_in, cfg.rng_seed, cfg.moment_option,
        cfg.n_starts, cfg.two_step, cfg.w2_replications, cfg.w2_ridge,
        cfg.se_method, cfg.bootstrap_reps,
    )
end

function draw_innovations(TT::Int, seed::Int)
    rng = MersenneTwister(seed)
    (randn(rng, TT), randn(rng, TT), randn(rng, TT), randn(rng, TT))
end

# Common plot defaults
const PLOT_DEFAULTS = (
    titlefontsize=13,
    guidefontsize=12,
    tickfontsize=10,
    legendfontsize=10,
    framestyle=:box,
    grid=true,
    gridalpha=0.25,
)

# ── 1. Learning paths κ̂_t (median + IQR) ────────────────────────────────────
function plot_learning_paths()
    T_obs = 400
    cfg1 = case_non_iid(intervention=false)
    cfg2 = case_non_iid(intervention=true)
    th = theta_cal(cfg1)

    n_seeds = 80
    TT = T_obs + cfg1.burn_in
    paths1 = zeros(TT, n_seeds)
    paths2 = zeros(TT, n_seeds)
    for s in 1:n_seeds
        id_d, id_v, id_my, id_mpi = draw_innovations(TT, 7000 + s)
        sim1 = simulate(th, cfg1, T_obs;
                        inn_d=id_d, inn_v=id_v,
                        inn_my=id_my, inn_mpi=id_mpi)
        sim2 = simulate(th, cfg2, T_obs;
                        inn_d=id_d, inn_v=id_v,
                        inn_my=id_my, inn_mpi=id_mpi)
        paths1[:, s] = sim1.kappa_path
        paths2[:, s] = sim2.kappa_path
    end
    med1 = [median(paths1[t, :]) for t in 1:TT]
    q25_1 = [quantile(paths1[t, :], 0.25) for t in 1:TT]
    q75_1 = [quantile(paths1[t, :], 0.75) for t in 1:TT]
    med2 = [median(paths2[t, :]) for t in 1:TT]
    q25_2 = [quantile(paths2[t, :], 0.25) for t in 1:TT]
    q75_2 = [quantile(paths2[t, :], 0.75) for t in 1:TT]
    tvec = 1:TT

    plt = plot(; xlabel="period t", ylabel="κ̂_t",
               title="Learning path of CB's belief κ̂  (non-IID, κ=0.10)",
               legend=:topright, size=(900, 450),
               xlim=(0, TT), ylim=(-0.1, 0.6), PLOT_DEFAULTS...)
    plot!(plt, tvec, med1,
          ribbon=(med1 .- q25_1, q75_1 .- med1), fillalpha=0.20,
          label="no intervention (Case 1)", color=:steelblue, lw=2.5)
    plot!(plt, tvec, med2,
          ribbon=(med2 .- q25_2, q75_2 .- med2), fillalpha=0.20,
          label="with intervention (Case 2)",
          color=:crimson, ls=:dash, lw=2.5)
    hline!(plt, [CAL[:kappa]], label="true κ = $(CAL[:kappa])",
           color=:black, ls=:dot, lw=1.5)
    vline!(plt, [cfg1.burn_in], label="end of burn-in",
           color=:gray, ls=:dashdot, lw=1)

    savefig(plt, joinpath(FIGDIR, "fig_learning_paths.pdf"))
    savefig(plt, joinpath(FIGDIR, "fig_learning_paths.png"))
    println("[1] learning paths saved")
end

# ── 2. Self-confirming κ̂(κ₀) — non-IID calibration ─────────────────────────
function plot_self_fulfilling_calibrated()
    cfg1 = case_non_iid(intervention=false)
    cfg2 = case_non_iid(intervention=true)
    th = theta_cal(cfg1)

    k0_grid = collect(range(0.0, 0.4; length=31))
    T_obs = 2000
    n_seeds = 40
    kh1 = zeros(length(k0_grid)); se1 = zeros(length(k0_grid))
    kh2 = zeros(length(k0_grid)); se2 = zeros(length(k0_grid))

    println("Sweeping κ₀ ∈ [0, 0.4]  (non-IID, κ=0.10)...")
    for (i, k0) in enumerate(k0_grid)
        c1 = with_k0(cfg1, k0); c2 = with_k0(cfg2, k0)
        v1 = Float64[]; v2 = Float64[]
        for s in 1:n_seeds
            TT = T_obs + c1.burn_in
            id_d, id_v, id_my, id_mpi = draw_innovations(TT, 20_000 + s)
            s1 = simulate(th, c1, T_obs;
                          inn_d=id_d, inn_v=id_v,
                          inn_my=id_my, inn_mpi=id_mpi)
            s2 = simulate(th, c2, T_obs;
                          inn_d=id_d, inn_v=id_v,
                          inn_my=id_my, inn_mpi=id_mpi)
            push!(v1, s1.kappa_path[end]); push!(v2, s2.kappa_path[end])
        end
        kh1[i] = mean(v1); se1[i] = std(v1) / sqrt(n_seeds)
        kh2[i] = mean(v2); se2[i] = std(v2) / sqrt(n_seeds)
    end

    plt = plot(; xlabel="initial belief κ₀", ylabel="terminal κ̂",
               title="Self-confirming κ̂ as a function of κ₀  (non-IID, κ=0.10)",
               legend=:topleft, size=(800, 500), PLOT_DEFAULTS...)
    plot!(plt, k0_grid, kh1, ribbon=1.96 .* se1, fillalpha=0.15,
          label="no intervention",
          color=:steelblue, lw=2.5)
    plot!(plt, k0_grid, kh2, ribbon=1.96 .* se2, fillalpha=0.15,
          label="with intervention",
          color=:crimson, ls=:dash, lw=2.5)
    plot!(plt, k0_grid, k0_grid, label="45° line",
          color=:orange, ls=:dashdot, lw=1.5)
    hline!(plt, [CAL[:kappa]], label="true κ = $(CAL[:kappa])",
           color=:black, ls=:dot, lw=1.5)

    savefig(plt, joinpath(FIGDIR, "fig_self_fulfilling_calibrated.pdf"))
    savefig(plt, joinpath(FIGDIR, "fig_self_fulfilling_calibrated.png"))
    println("[2] self-confirming κ̂(κ₀) saved")
end

# ── 3. Policy reaction function φ_π*(k̂) ─────────────────────────────────────
function plot_policy_map()
    cfg = case_non_iid(intervention=true)
    p = unpack(theta_cal(cfg), cfg)

    ks = collect(range(0.05, 1.5; length=201))
    pol = [clamp(policy_numeric(k, p, cfg),
                 cfg.phi_min, cfg.phi_max) for k in ks]

    plt = plot(; xlabel="CB belief k̂", ylabel="optimal φ_π*(k̂)",
               title="Optimal policy reaction: passive vs. active regions",
               legend=:topright, size=(800, 450),
               ylim=(0, max(2.5, maximum(pol) * 1.05)),
               PLOT_DEFAULTS...)
    plot!(plt, ks, pol, color=:steelblue, lw=2.5,
          label="φ_π*(k̂) at non-IID calibration")
    hline!(plt, [1.0], color=:black, ls=:dash, lw=1.2,
           label="Taylor principle (φ_π = 1)")
    hline!(plt, [1.5], color=:gray, ls=:dashdot, lw=1.0,
           label="Taylor benchmark (φ_π = 1.5)")
    vline!(plt, [CAL[:kappa]], color=:green, ls=:dot, lw=1.5,
           label="true κ = $(CAL[:kappa])")

    savefig(plt, joinpath(FIGDIR, "fig_policy_map.pdf"))
    savefig(plt, joinpath(FIGDIR, "fig_policy_map.png"))
    println("[3] policy map saved")
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Generating report figures (non-IID calibration)...")
    plot_learning_paths()
    plot_self_fulfilling_calibrated()
    plot_policy_map()
    println("\nAll three figures saved to ", FIGDIR)
end
