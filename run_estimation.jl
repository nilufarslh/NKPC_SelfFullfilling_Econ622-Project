# ──────────────────────────────────────────────────────────────────────────────
# Self-Fulfilling NKPC — SMM estimation driver.
#
# Cases 1 & 3 first (faster), then Case 2 (intervention, slower).
# Bootstrap SE with B=20, parallel @threads.
# Detailed logging to timestamped logfile.
# ──────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include(joinpath(@__DIR__, "src", "SelfFulfillingNKPC.jl"))
using .SelfFulfillingNKPC
using Printf, Random, LinearAlgebra, Dates, CSV, DataFrames

# ── Paths ────────────────────────────────────────────────────────────────────

const PROJECT_ROOT = @__DIR__
const DATA_PATH    = joinpath(PROJECT_ROOT, "data", "targets.csv")
const RESULTS_DIR  = joinpath(PROJECT_ROOT, "results")

function format_elapsed(sec::Float64)
    sec < 60 ? @sprintf("%.1f s", sec) :
               @sprintf("%d min %.1f s", floor(Int, sec/60), sec - 60*floor(Int, sec/60))
end

# ── Main estimation driver ───────────────────────────────────────────────────

function estimate_case(label::String, cfg::EstimationConfig;
                       n_starts::Int=cfg.n_starts,
                       weighting::Symbol=:identity,
                       logfile::String="")
    t_total = time()

    function log_msg(msg)
        full = "[$(Dates.format(now(), "HH:MM:SS"))] $msg"
        println(full)
        if !isempty(logfile)
            open(logfile, "a") do io; println(io, full); end
        end
    end

    log_msg("="^70)
    log_msg("  Estimating: $label")
    log_msg("="^70)

    # Load data
    data = load_targets(DATA_PATH)
    T_obs = data.T
    m_data_mv = compute_moments(data.X; option=cfg.moment_option)
    m_data = m_data_mv.values
    n_mom  = length(m_data)

    log_msg("  Data: T=$T_obs, moments=$n_mom, params=$(length(cfg.estimated))")

    # Draw N(0,1) innovations — held fixed for all objective evaluations
    TT = T_obs + cfg.burn_in
    rng = MersenneTwister(cfg.rng_seed)
    inn_d   = randn(rng, TT)
    inn_v   = randn(rng, TT)
    inn_my  = randn(rng, TT)
    inn_mpi = randn(rng, TT)

    # Weighting matrix
    if weighting == :diagonal
        # W = diag(1/|m_data_i|) — moderate normalization by scale
        w_diag = [abs(m) > 1e-8 ? 1.0/abs(m) : 1.0 for m in m_data]
        W1 = Diagonal(w_diag) |> Matrix
        log_msg("  Stage 1: W = diagonal(1/|m|), n_starts=$n_starts")
    else
        W1 = Matrix{Float64}(I, n_mom, n_mom)
        log_msg("  Stage 1: W = I, n_starts=$n_starts")
    end
    t1 = time()
    res1 = estimate_smm(cfg, m_data, W1, T_obs;
                        inn_d=inn_d, inn_v=inn_v,
                        inn_my=inn_my, inn_mpi=inn_mpi,
                        n_starts=n_starts, verbose=true,
                        logfile=logfile)
    elapsed1 = time() - t1
    log_msg("  Stage 1 done in $(format_elapsed(elapsed1)): f=$(round(res1.fval, sigdigits=5))")

    theta_hat = res1.theta_hat
    fval = res1.fval

    # Bootstrap SE
    log_msg("  Bootstrap SE: B=$(cfg.bootstrap_reps), threads=$(Threads.nthreads())")
    t_se = time()
    se = bootstrap_se(theta_hat, cfg, m_data, W1, T_obs;
                      B=cfg.bootstrap_reps, logfile=logfile)
    elapsed_se = time() - t_se
    log_msg("  Bootstrap done in $(format_elapsed(elapsed_se))")

    # Simulated moments at θ̂
    sim_mv = moments_from_theta(theta_hat, cfg, T_obs;
                                inn_d=inn_d, inn_v=inn_v,
                                inn_my=inn_my, inn_mpi=inn_mpi)

    cr = CaseResult(
        cfg.environment, cfg.intervention,
        theta_hat, param_names(cfg), fval, se,
        m_data, sim_mv.values, sim_mv.names,
        W1, 1
    )

    print_results(cr)
    outdir = joinpath(RESULTS_DIR, label)
    save_results(cr, outdir)

    total = time() - t_total
    log_msg("  Total for $label: $(format_elapsed(total))")
    log_msg("")
    cr
end

# ── Entry point ──────────────────────────────────────────────────────────────

function main()
    mkpath(RESULTS_DIR)
    logfile = joinpath(RESULTS_DIR, "estimation_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")
    println("Logging to: $logfile")

    t_global = time()
    results = CaseResult[]

    # Case 1: No intervention (n_starts=20, diagonal weighting)
    cfg1 = case_non_iid(intervention=false)
    cr1 = estimate_case("case1_no_intervention", cfg1; n_starts=20, weighting=:diagonal, logfile=logfile)
    push!(results, cr1)

    # Case 2: With intervention (n_starts=10, identity weighting)
    cfg2 = case_non_iid(intervention=true)
    cr2 = estimate_case("case2_with_intervention", cfg2; n_starts=10, weighting=:identity, logfile=logfile)
    push!(results, cr2)

    elapsed = time() - t_global
    open(logfile, "a") do io
        println(io, "="^70)
        println(io, "All cases completed in $(format_elapsed(elapsed))")
        println(io, "="^70)
    end

    println("\n", "="^70)
    println("  All cases completed in $(format_elapsed(elapsed))")
    println("  Results: $RESULTS_DIR")
    println("  Log: $logfile")
    println("="^70)

    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
