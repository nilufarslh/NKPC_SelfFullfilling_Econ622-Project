# ──────────────────────────────────────────────────────────────────────────────
# Data loading and pre-processing
# ──────────────────────────────────────────────────────────────────────────────

struct TargetData
    X     ::Matrix{Float64}   # T×3 — [π  y  r], demeaned & annualised
    T     ::Int
    names ::Vector{String}
end

"""
    load_targets(path; annualize=false, demean=true) -> TargetData

Read quarterly US data from CSV. Expected columns: `date`, `pi`, `y`, `r`.
The default is NOT to annualise: the FRED/BHP series stored in
`data/targets.csv` are already in annual-percent units (π is the year-over-year
CPI inflation rate in %, r is the annualised Federal-Funds rate in %).
Set `annualize=true` only if the input series are quarterly rates.
Demeaning subtracts the column mean.
"""
function load_targets(path::AbstractString; annualize::Bool=false, demean::Bool=true)
    df = CSV.read(path, DataFrame)
    for col in ("pi", "y", "r")
        col in names(df) || error("Missing column \"$col\" in $path")
    end

    pi = Float64.(df.pi)
    y  = Float64.(df.y)
    r  = Float64.(df.r)

    if annualize
        pi .*= 4
        r  .*= 4
    end

    if demean
        pi .-= mean(filter(!isnan, pi))
        y  .-= mean(filter(!isnan, y))
        r  .-= mean(filter(!isnan, r))
    end

    X = hcat(pi, y, r)
    valid = .!any(isnan.(X), dims=2)[:]
    X = X[valid, :]

    TargetData(X, size(X, 1), ["pi", "y", "r"])
end
