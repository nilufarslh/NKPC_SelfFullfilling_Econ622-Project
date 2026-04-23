# ──────────────────────────────────────────────────────────────────────────────
# One-command reproduction of the project.
#
#   julia --project=. -t auto run_all.jl
#
# Steps:
#   1. Two-case SMM estimation  (run_estimation.jl)  → results/case1*/ , case2*/
#   2. Report figures           (scripts/fig_report.jl) → results/figures/
# ──────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

flush(stdout)

using Random
Random.seed!(123)

println("\n[1/2] SMM estimation (Case 1 and Case 2)")
include(joinpath(@__DIR__, "run_estimation.jl"))
main()

println("\n[2/2] Report figures")
include(joinpath(@__DIR__, "scripts", "fig_report.jl"))
plot_learning_paths()
plot_self_fulfilling_calibrated()
plot_policy_map()

println("\nPipeline complete. See results/ for CSVs and results/figures/ for PDFs.")
