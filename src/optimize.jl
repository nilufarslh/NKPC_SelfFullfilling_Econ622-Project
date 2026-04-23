# ──────────────────────────────────────────────────────────────────────────────
# Multi-start Fminbox(L-BFGS) with ForwardDiff gradients
# ──────────────────────────────────────────────────────────────────────────────

struct OptimResult
    theta_hat ::Vector{Float64}
    fval      ::Float64
    converged ::Bool
    n_starts  ::Int
    stage     ::Int
end

function estimate_smm(cfg::EstimationConfig, m_data::Vector{Float64},
                      W::Matrix{Float64}, T_obs::Int;
                      inn_d::Vector{Float64}, inn_v::Vector{Float64},
                      inn_my::Vector{Float64}, inn_mpi::Vector{Float64},
                      n_starts::Int=cfg.n_starts,
                      verbose::Bool=true,
                      warm_start::Union{Nothing,Vector{Float64}}=nothing,
                      logfile::Union{Nothing,String}=nothing)
    lb = lower_bounds(cfg); ub = upper_bounds(cfg)
    x0 = warm_start === nothing ? theta0_vec(cfg) : warm_start

    obj = theta -> smm_objective(theta, cfg, m_data, W, T_obs;
                                  inn_d=inn_d, inn_v=inn_v,
                                  inn_my=inn_my, inn_mpi=inn_mpi)
    grad! = (G, theta) -> (G .= ForwardDiff.gradient(obj, theta); G)

    # Multiplier-based starts centered on x0 (keeps σ_mpi near initial value,
    # avoids degenerate κ≈0 basin that Sobol starts find)
    multipliers = [1.0, 1.5, 0.5, 2.0, 0.3, 0.7, 1.2, 0.4, 1.8, 0.6,
                   1.3, 0.8, 1.7, 0.35, 2.5, 0.25, 1.4, 0.9, 1.1, 0.45]
    S = min(n_starts, length(multipliers))
    starts = zeros(S, length(x0))
    for i in 1:S
        starts[i, :] = clamp.(x0 .* multipliers[i], lb .+ 1e-4, ub .- 1e-4)
    end

    opts = Optim.Options(
        iterations        = 10,
        f_reltol          = 1e-6,
        g_tol             = 1e-4,
        show_trace        = false,
        allow_f_increases = true,
        time_limit        = 120.0,    # 2 min cap per start
    )

    function log_msg(msg)
        println(msg)
        if logfile !== nothing
            open(logfile, "a") do io; println(io, msg); end
        end
    end

    quiet = Logging.ConsoleLogger(stderr, Logging.Error)

    fvals = fill(Inf, S)
    thetas = [copy(x0) for _ in 1:S]
    convs = falses(S)

    for s in 1:S
        t_s = time()
        x_init = clamp.(starts[s, :], lb .+ 1e-6, ub .- 1e-6)
        try
            r = Logging.with_logger(quiet) do
                optimize(obj, grad!, lb, ub, x_init, Fminbox(LBFGS()), opts)
            end
            fvals[s] = Optim.minimum(r)
            thetas[s] = Optim.minimizer(r)
            convs[s] = Optim.converged(r)
            elapsed = round(time() - t_s, digits=1)
            pnames = param_names(cfg)
            theta_str = join(["$(pnames[i])=$(round(thetas[s][i],sigdigits=4))" for i in eachindex(pnames)], ", ")
            log_msg("[estimate] start $s/$S done in $(elapsed)s  f=$(round(fvals[s],sigdigits=5))  $(convs[s] ? "conv" : "")  θ=[$theta_str]")
        catch e
            log_msg("[estimate] start $s/$S FAILED: $(sprint(showerror, e))")
        end
    end

    best = argmin(fvals)
    if verbose
        @printf("  Best objective: %.6e\n", fvals[best])
    end
    log_msg("[estimate] best: f=$(round(fvals[best],sigdigits=5)) (start $best)")

    OptimResult(thetas[best], fvals[best], convs[best], S, 1)
end
