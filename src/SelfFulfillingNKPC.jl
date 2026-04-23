module SelfFulfillingNKPC

using LinearAlgebra, Statistics, Random, Printf
using ForwardDiff, Optim, CSV, DataFrames
using Interpolations: linear_interpolation, Line
using Logging: Logging, ConsoleLogger, with_logger

include("config.jl")
include("data.jl")
include("model.jl")
include("policy_interp.jl")
include("simulation.jl")
include("moments.jl")
include("objective.jl")
include("optimize.jl")
include("inference.jl")

export
    # config / params
    ModelParams, ParamSpec, EstimationConfig,
    case_non_iid, case_non_iid_free_phi,
    param_names, theta0_vec, lower_bounds, upper_bounds, unpack,

    # data
    TargetData, load_targets,

    # model
    taylor_rule, uncond_var,
    policy_numeric, compute_policy,

    # policy interp
    PolicyInterp, build_policy_interp, evaluate_policy_itp,

    # simulation
    SimulationResult, simulate,

    # moments
    MomentVector, compute_moments, moments_from_theta,
    var1_ols, vec_mat, vech, cov_lag,

    # objective / estimation
    smm_objective,
    estimate_smm, OptimResult,

    # inference / output
    CaseResult, bootstrap_se,
    print_results, save_results

end
