# ──────────────────────────────────────────────────────────────────────────────
# Taylor rule, shock variance helper, and the CB's policy solve.
#
# The reduced form is inlined inside simulate() (it depends on the current
# belief k̂ and changes every period).  Here we keep the pieces that do not
# need to live in the hot loop: the Taylor rule, the unconditional variance
# formula used to initialise AR(1) shocks, and the CB's 1D Brent optimisation
# over φ_π.  The CB's subjective model omits θ_r and carries the two iid
# measurement shocks (m_y, m_π) separately, hence the four-shock sum.
# ──────────────────────────────────────────────────────────────────────────────

function taylor_rule(pi_t, y_t, u_t, phi_pi, phi_y)
    phi_pi * pi_t + phi_y * y_t + u_t
end

function uncond_var(sigma, rho)
    rho_c = clamp(rho, -0.9999, 0.9999)
    sigma^2 / (1 - rho_c^2)
end

# ── CB policy: numeric Brent, 4-shock subjective loss (no θ_r) ──────────────
function policy_numeric(k_hat, p::ModelParams, cfg::EstimationConfig)
    Vd  = uncond_var(p.sigma_d, p.rho_d)
    Vv  = uncond_var(p.sigma_v, p.rho_v)
    Vy  = max(p.sigma_my^2, 1e-10)
    Vpi = max(p.sigma_mpi^2, 1e-10)
    vars = (Vd, Vv, Vy, Vpi)
    rho = (p.rho_d, p.rho_v, 0.0, 0.0)
    phiy = p.phi_y

    function cb_loss(phipi)
        var_y = 0.0; var_pi = 0.0
        for j in 1:4
            Th = (1-rho[j])*(1-p.beta*rho[j])*p.gamma +
                 (1-p.beta*rho[j])*phiy + k_hat*phipi
            abs(Th) < 1e-10 && return 1e12
            iT = 1/Th
            if j == 1      # demand
                Ay = p.gamma*(1-p.beta*rho[j])*iT
                Ap = k_hat*p.gamma*iT
            elseif j == 2  # cost-push
                Ay = -phipi*iT
                Ap = ((1-rho[j])*p.gamma+phiy)*iT
            elseif j == 3  # m_y
                Ay = -(1-p.beta*rho[j])*phiy*iT
                Ap = -k_hat*phiy*iT
            else           # m_π
                Ay = -(1-p.beta*rho[j])*phipi*iT
                Ap = -k_hat*phipi*iT
            end
            var_y += Ay^2*vars[j]; var_pi += Ap^2*vars[j]
        end
        var_pi + p.lambda_y * var_y
    end

    result = optimize(cb_loss, cfg.phi_min, cfg.phi_max, Brent();
                      rel_tol=1e-8, abs_tol=1e-10)
    clamp(Optim.minimizer(result), cfg.phi_min, cfg.phi_max)
end

function compute_policy(k_hat, p::ModelParams, cfg::EstimationConfig; itp=nothing)
    if cfg.free_phi
        return (p.phi_pi, p.phi_y)
    end
    if itp !== nothing
        return evaluate_policy_itp(itp, k_hat, cfg)
    end
    phi_pi = policy_numeric(k_hat, p, cfg)
    (phi_pi, p.phi_y)
end
