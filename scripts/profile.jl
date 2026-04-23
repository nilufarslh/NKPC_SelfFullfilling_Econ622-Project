# ──────────────────────────────────────────────────────────────────────────────
# @profile of simulate() on Case 1 and Case 2.
#
# Produces a text-form profile at results/figures/profile_simulate.txt,
# sorted by self-time with the top call chain shown.
#
#   julia --project=. scripts/profile.jl
# ──────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Profile, Random

include(joinpath(@__DIR__, "..", "src", "SelfFulfillingNKPC.jl"))
using .SelfFulfillingNKPC

const OUTFILE = joinpath(@__DIR__, "..", "results", "figures", "profile_simulate.txt")
mkpath(dirname(OUTFILE))

function inn(TT, seed)
    rng = MersenneTwister(seed)
    (inn_d=randn(rng, TT), inn_v=randn(rng, TT),
     inn_my=randn(rng, TT), inn_mpi=randn(rng, TT))
end

function profile_case(label, cfg; T_obs=166, N=30_000)
    θ = theta0_vec(cfg)
    TT = T_obs + cfg.burn_in
    kw = inn(TT, 42)

    simulate(θ, cfg, T_obs; kw...)        # warm up
    Profile.init(; n=10^7, delay=1e-4)    # sample every 0.1ms
    Profile.clear()
    @profile for _ in 1:N
        simulate(θ, cfg, T_obs; kw...)
    end

    open(OUTFILE, "a") do io
        println(io, "="^78)
        println(io, "  $label  (N=$N simulate calls, T_obs=$T_obs)")
        println(io, "="^78)
        println(io, "\n--- Flat profile (top by self-time) ---\n")
        Profile.print(IOContext(io, :displaysize => (60, 140));
                      format=:flat, sortedby=:count, mincount=20)
        println(io, "\n--- Call tree (top frames) ---\n")
        Profile.print(IOContext(io, :displaysize => (80, 140));
                      format=:tree, maxdepth=14, mincount=20)
        println(io)
    end
end

isfile(OUTFILE) && rm(OUTFILE)
profile_case("Case 1 — passive policy",  case_non_iid(intervention=false))
profile_case("Case 2 — active policy",   case_non_iid(intervention=true))
println("Profile written to: ", OUTFILE)
