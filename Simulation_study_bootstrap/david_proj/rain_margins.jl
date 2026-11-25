# # Rain amounts on a wet day
#
# Uniform on `[a, b]` with probability `p`, `b + EGPD` above `b`. Every parameter is a
# trigonometric polynomial of the day of the year, one per hidden state and site, stored as a
# named tuple `(; p, κ, σ, ξ)` of `K × D × (2degree + 1)` coefficient arrays. `p` and `ξ` go
# through a logistic, `κ` and `σ` through `exp`.

using Distributions, LogExpFunctions
using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve   # load `fit_loss_optim` of SmoothPeriodicStatsModels
using OptimizationOptimJL: NelderMead

const TBETA_SUPPORT = (1 / 32, 1 / 2)

"""
    EGPD(κ, σ, ξ)

Extended generalized Pareto law with a truncated-beta carrier, `F(y) = H(G(y))`.

`G` is the cdf of `GeneralizedPareto(0, σ, ξ)` and `H` that of `Beta(κ, κ)` truncated to
`[1/32, 1/2]` and rescaled to `[0, 1]`: the law of `ExtendedGeneralizedPareto(TBeta(κ),
GeneralizedPareto(0, σ, ξ))` in ExtendedExtremes.jl.
"""
struct EGPD{H<:ContinuousUnivariateDistribution,G<:GeneralizedPareto} <: ContinuousUnivariateDistribution
    carrier::H
    gpd::G
end

function EGPD(κ::Real, σ::Real, ξ::Real)
    l, u = TBETA_SUPPORT
    return EGPD((truncated(Beta(κ, κ), l, u) - l) / (u - l), GeneralizedPareto(zero(σ), σ, ξ))
end

Distributions.cdf(d::EGPD, y::Real) = cdf(d.carrier, cdf(d.gpd, y))
Distributions.logpdf(d::EGPD, y::Real) = logpdf(d.carrier, cdf(d.gpd, y)) + logpdf(d.gpd, y)
Distributions.quantile(d::EGPD, q::Real) = quantile(d.gpd, quantile(d.carrier, q))

"""
    MixedUniformEGPD(p, tail, a, b)

Law of a wet-day amount: uniform on `[a, b]` with probability `p`, `b + tail` otherwise.
"""
struct MixedUniformEGPD{P<:Real,E<:EGPD} <: ContinuousUnivariateDistribution
    p::P
    tail::E
    a::Float64
    b::Float64
end

Distributions.cdf(d::MixedUniformEGPD, r::Real) = r ≤ d.b ?
    d.p * clamp((r - d.a) / (d.b - d.a), 0, 1) : d.p + (1 - d.p) * cdf(d.tail, r - d.b)

Distributions.quantile(d::MixedUniformEGPD, q::Real) = q ≤ d.p ?
    d.a + (d.b - d.a) * q / d.p : d.b + quantile(d.tail, (q - d.p) / (1 - d.p))

"""
    egpd_family(T, degree) -> TimeVaryingFamily

The `EGPD` whose `κ`, `σ` and `ξ` are trigonometric polynomials of `degree` over a period `T`.
"""
egpd_family(T, degree) = TimeVaryingFamily(EGPD, ParamSpec(
    κ=TrigBasis(T, degree) => LogLink(),
    σ=TrigBasis(T, degree) => LogLink(),
    ξ=TrigBasis(T, degree) => LogitLink()))

"""
    rain_distributions(θ, T; a, b) -> Array{MixedUniformEGPD,3}

The `K × D × T` laws of the wet-day amount given by the coefficients `θ = (; p, κ, σ, ξ)`.
"""
function rain_distributions(θ, T; a, b)
    K, D, n = size(θ.κ)
    family = egpd_family(T, n ÷ 2)
    design = PeriodicDesign(family.spec, 1:T)
    X = design_matrix(TrigBasis(T, n ÷ 2), 1:T)
    tails = [distributions(family, design, [θ.κ[k, s, :]; θ.σ[k, s, :]; θ.ξ[k, s, :]]) for k in 1:K, s in 1:D]
    p = [logistic.(X * θ.p[k, s, :]) for k in 1:K, s in 1:D]
    return [MixedUniformEGPD(p[k, s][t], tails[k, s][t], a, b) for k in 1:K, s in 1:D, t in 1:T]
end

"""
    fit_small_rain_probability(r, t, T, degree; b) -> Vector

Coefficients of the probability that a wet-day amount is at most `b`, from amounts `r` seen on days `t`.

Least squares of the logit of the daily proportions; days whose proportion is 0, 1 or undefined
are left out.
"""
function fit_small_rain_probability(r, t, T, degree; b)
    p = [count(i -> r[i] ≤ b, days) / length(days) for days in group_by_time(t, T)]
    keep = findall(x -> 0 < x < 1, p)
    return design_matrix(TrigBasis(T, degree), keep) \ logit.(p[keep])
end

"""
    fit_egpd(y, t, θ0, T; maxiters, penalty = 0.1) -> Vector

Coefficients `[κ; σ; ξ]` of the periodic `EGPD` fitted to the excesses `y` seen on days `t`.

Minimizes the negative loglikelihood plus `Σₜ wₜ (κ(t) - 1)² / penalty`, `wₜ` the share of the
excesses on day `t`, by Nelder-Mead from `θ0`.
"""
function fit_egpd(y, t, θ0, T; maxiters, penalty=0.1)
    family = egpd_family(T, (length(θ0) ÷ 3 - 1) ÷ 2)
    design = PeriodicDesign(family.spec, 1:T)
    groups = group_by_time(t, T)
    nll = PeriodicNLL(family, design, y, groups)
    w = length.(groups) ./ max(length(y), 1)
    Xκ = design.designs[blockindex(family.spec, :κ)]
    loss(θ, _) = nll(θ) + sum(w .* (exp.(Xκ * coefs(family.spec, θ, :κ)) .- 1) .^ 2) / penalty
    # the carrier's normalization, `beta_inc` in κ, has no derivative: hence Nelder-Mead
    sol = fit_loss_optim(loss, y, θ0; solver=NelderMead(), adtype=NoAD(), solve_kwargs=(; maxiters))
    return sol.u
end

"""
    fit_rain_margins(R, z, n2t, θ0; T, a, b, maxiters) -> NamedTuple

Coefficients `(; p, κ, σ, ξ)` of the margins fitted to the amounts `R` (`D × N`) in each state of `z`.

A day is wet where `R ≥ a`. `θ0 = (; κ, σ, ξ)` starts the EGPD fits, and its size gives the
number of states and the degree.
"""
function fit_rain_margins(R, z, n2t, θ0; T, a, b, maxiters)
    K, D, n = size(θ0.κ)
    θ = (p=zeros(K, D, n), κ=zeros(K, D, n), σ=zeros(K, D, n), ξ=zeros(K, D, n))
    Threads.@threads for I in CartesianIndices((K, D))
        k, s = Tuple(I)
        wet = findall(i -> z[i] == k && R[s, i] ≥ a, eachindex(z))
        r, t = R[s, wet], n2t[wet]
        θ.p[k, s, :] = fit_small_rain_probability(r, t, T, n ÷ 2; b)
        above = r .> b
        u = fit_egpd(r[above] .- b, t[above], [θ0.κ[k, s, :]; θ0.σ[k, s, :]; θ0.ξ[k, s, :]], T; maxiters)
        θ.κ[k, s, :], θ.σ[k, s, :], θ.ξ[k, s, :] = u[1:n], u[n+1:2n], u[2n+1:3n]
    end
    return θ
end
