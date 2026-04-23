# ──────────────────────────────────────────────────────────────────────────────
# BenchmarkTools timing of the hot paths:
#   • simulate() at the prior point
#   • smm_objective() at the prior point
#   • ForwardDiff.gradient of smm_objective (the outer optimiser's per-iter cost)
#
#   julia --project=. scripts/benchmark.jl
# ──────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BenchmarkTools, Random, LinearAlgebra, ForwardDiff, Printf

include(joinpath(@__DIR__, "..", "src", "SelfFulfillingNKPC.jl"))
using .SelfFulfillingNKPC

function draw(TT, seed)
    rng = MersenneTwister(seed)
    (inn_d=randn(rng, TT), inn_v=randn(rng, TT),
     inn_my=randn(rng, TT), inn_mpi=randn(rng, TT))
end

function bench_case(label, cfg; T_obs=166)
    θ = theta0_vec(cfg)
    TT = T_obs + cfg.burn_in
    inn = draw(TT, 42)

    mv = compute_moments(simulate(θ, cfg, T_obs; inn...).X;
                         option=cfg.moment_option)
    m_data = mv.values
    W = Matrix{Float64}(I, length(m_data), length(m_data))
    obj = θv -> smm_objective(θv, cfg, m_data, W, T_obs; inn...)

    println("\n── $label ───────────────────────────────────────────")
    @printf("simulate(T=%d):\n", T_obs)
    display(@benchmark simulate($θ, $cfg, $T_obs; $inn...) samples=30 evals=1)
    println()
    println("smm_objective:")
    display(@benchmark $obj($θ) samples=30 evals=1)
    println()
    println("ForwardDiff.gradient(smm_objective):")
    display(@benchmark ForwardDiff.gradient($obj, $θ) samples=15 evals=1)
    println()
end

bench_case("Case 1 — passive policy", case_non_iid(intervention=false))
bench_case("Case 2 — active policy", case_non_iid(intervention=true))
