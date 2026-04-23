# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap standard errors and result reporting.
#
# Bootstrap SE re-estimates from B independent innovation draws (parallel).
# ──────────────────────────────────────────────────────────────────────────────

"""
    CaseResult

Complete output container for one estimation case.
"""
struct CaseResult
    environment  ::Symbol
    intervention ::Bool
    theta_hat    ::Vector{Float64}
    param_names  ::Vector{Symbol}
    fval         ::Float64
    se           ::Vector{Float64}
    data_moments ::Vector{Float64}
    sim_moments  ::Vector{Float64}
    moment_names ::Vector{String}
    W            ::Matrix{Float64}
    stage        ::Int
end

# ── Bootstrap standard errors ────────────────────────────────────────────────

"""
    bootstrap_se(theta_hat, cfg, m_data, W, T_obs; B=cfg.bootstrap_reps,
                 logfile=nothing) -> Vector{Float64}

Nonparametric bootstrap standard errors. `B` independent innovation paths
are drawn at `theta_hat`, each is re-estimated with a single L-BFGS restart
warm-started at `theta_hat`, and the sample standard deviation of the
resulting parameter draws is returned. Runs in parallel across
`Threads.nthreads()` threads.
"""
function bootstrap_se(theta_hat::Vector{Float64}, cfg::EstimationConfig,
                      m_data::Vector{Float64}, W::Matrix{Float64},
                      T_obs::Int; B::Int=cfg.bootstrap_reps,
                      logfile::Union{Nothing,String}=nothing)

    TT = T_obs + cfg.burn_in
    theta_draws = zeros(length(theta_hat), B)

    function log_msg(msg)
        println(msg)
        if logfile !== nothing
            open(logfile, "a") do io; println(io, msg); end
        end
    end

    log_msg("[bootstrap] starting B=$B reps on $(Threads.nthreads()) threads")

    Threads.@threads for b in 1:B
        t_b = time()
        rng_b = MersenneTwister(10_000 + b)

        # Fresh N(0,1) innovations for this bootstrap rep
        inn_d_b  = randn(rng_b, TT)
        inn_v_b  = randn(rng_b, TT)
        inn_my_b = randn(rng_b, TT)
        inn_mpi_b = randn(rng_b, TT)

        res = Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Error)) do
            estimate_smm(cfg, m_data, W, T_obs;
                         inn_d=inn_d_b, inn_v=inn_v_b,
                         inn_my=inn_my_b, inn_mpi=inn_mpi_b,
                         n_starts=1, verbose=false,
                         warm_start=theta_hat, logfile=logfile)
        end
        theta_draws[:, b] = res.theta_hat
        elapsed = round(time() - t_b, digits=1)
        log_msg("[bootstrap] rep $b/$B done in $(elapsed)s  f=$(round(res.fval, sigdigits=4))")
    end

    log_msg("[bootstrap] all $B reps complete")
    vec(std(theta_draws, dims=2))
end

# ── Reporting ────────────────────────────────────────────────────────────────

"""
    print_results(cr)

Print a formatted parameter table (estimate and bootstrap SE) followed by
the moment-fit table (data, simulated, difference) to stdout.
"""
function print_results(cr::CaseResult)
    println("\n", "="^70)
    @printf("  Case: %s | intervention = %s\n",
            cr.environment, cr.intervention)
    println("="^70)

    println("\n  Parameter estimates:")
    println("  ", "-"^50)
    @printf("  %-12s  %10s  %10s\n", "Parameter", "Estimate", "SE")
    println("  ", "-"^50)
    for (i, nm) in enumerate(cr.param_names)
        se_str = isnan(cr.se[i]) || cr.se[i] <= 0 ? "   ---" :
                 @sprintf("%10.4f", cr.se[i])
        @printf("  %-12s  %10.4f  %s\n", nm, cr.theta_hat[i], se_str)
    end
    @printf("\n  Objective value: %.6e  (stage %d)\n", cr.fval, cr.stage)

    println("\n  Moment fit:")
    println("  ", "-"^60)
    @printf("  %-18s  %10s  %10s  %10s\n", "Moment", "Data", "Simulated", "Diff")
    println("  ", "-"^60)
    for (i, nm) in enumerate(cr.moment_names)
        @printf("  %-18s  %10.4f  %10.4f  %10.4f\n",
                nm, cr.data_moments[i], cr.sim_moments[i],
                cr.sim_moments[i] - cr.data_moments[i])
    end
    println()
end

"""
    save_results(cr, dir)

Write two CSVs into `dir` (creating it if missing): `params.csv` (parameter,
estimate, SE) and `moments.csv` (moment, data, simulated, difference).
"""
function save_results(cr::CaseResult, dir::AbstractString)
    mkpath(dir)

    param_df = DataFrame(
        Parameter = string.(cr.param_names),
        Estimate  = cr.theta_hat,
        SE        = cr.se,
    )
    CSV.write(joinpath(dir, "params.csv"), param_df)

    moment_df = DataFrame(
        Moment    = cr.moment_names,
        Data      = cr.data_moments,
        Simulated = cr.sim_moments,
        Difference = cr.sim_moments .- cr.data_moments,
    )
    CSV.write(joinpath(dir, "moments.csv"), moment_df)
end
