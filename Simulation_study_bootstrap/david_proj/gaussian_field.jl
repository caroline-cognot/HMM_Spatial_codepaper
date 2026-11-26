# # The latent Gaussian field behind the rain amounts
#
# Site `s` is wet on day `n` when `X[s, n] ≥ Φ⁻¹(pdry[s, n])`, and its amount is the quantile of
# its margin at the level `X[s, n]` has above that threshold.

using Distributions, LinearAlgebra, Random
using Optimization, OptimizationOptimJL
using BesselK: adbesselkxv
using SpecialFunctions: gamma
using TruncatedMVN: TruncatedMVN, TruncatedMVNormal
using SmoothPeriodicStatsModels: norm_cdf_2d_vfast

"""
    GneitingMatern(σ², c, a, α, β, δ, ν)

Gneiting space-time covariance with a Matérn spatial part of smoothness `ν`.

`C(h, u) = σ² ψ(u)^-(δ + β) M(h / (c ψ(u)^(β/2)))` at distance `h` (km) and lag `u` (days), with
`ψ(u) = 1 + (u / a)^(2α)` and `M` the Matérn correlation.
"""
struct GneitingMatern{T1,T<:Real}
    σ²::T1
    c::T
    a::T
    α::T
    β::T
    δ::T
    ν::T
end

matern_scaled(h, c, ν, scale) = iszero(h) ? one(c * ν) : scale * adbesselkxv(ν, h / c)
matern(h, c, ν) = matern_scaled(h, c, ν, 2^(1 - ν) / gamma(ν))

"""
    covariance(model::GneitingMatern, h, u)

Covariance of two sites at distance `h`, a number or an array, and `u` days apart.
"""
function covariance(m::GneitingMatern, h, u)
    ψ = 1 + (u / m.a)^(2m.α)
    return m.σ² .* matern.(h, m.c * ψ^(m.β / 2), m.ν) ./ ψ^(m.δ + m.β)
end

"""
    simulate_latent_field(model, distances, Y, pdry; lag = 10) -> Matrix

Latent field `X` (`D × N`) with `X[s, n] ≥ Φ⁻¹(pdry[s, n])` exactly where `Y[s, n]` is wet.

Day `n` is drawn from its truncated normal law given the `lag` previous days.
"""
function simulate_latent_field(model, distances, Y, pdry; lag=10)
    D, N = size(Y)
    C = [covariance(model, distances, u) for u in 0:lag]
    threshold = quantile.(Normal(), pdry)
    lower, upper = ifelse.(Y, threshold, -Inf), ifelse.(Y, Inf, threshold)
    # the regression of a day on its m previous days, most recent first, and its residual covariance
    regressions = map(0:lag) do m
        m == 0 && return zeros(D, 0), Matrix(Symmetric(C[1]))
        past = zeros(m * D, m * D)
        for i in 1:m, j in 1:m
            past[(i-1)*D+1:i*D, (j-1)*D+1:j*D] = C[abs(i - j)+1]
        end
        cross = reduce(hcat, C[2:m+1])
        W = cross / past
        return W, Matrix(Symmetric(C[1] - W * cross'))
    end
    X = zeros(D, N)
    for n in 1:N
        m = min(n - 1, lag)
        W, Σ = regressions[m+1]
        μ = W * vec(X[:, n-1:-1:n-m])
        X[:, n] = vec(TruncatedMVN.sample(TruncatedMVNormal(μ, Σ, lower[:, n], upper[:, n]), 1))
    end
    return X
end

"""
    rain_amounts(X, Y, z, n2t, pdry, margins) -> Matrix

Amount of the margin `margins[z[n], s, n2t[n]]` at the level of `X[s, n]` above its threshold, `0` where dry.
"""
rain_amounts(X, Y, z, n2t, pdry, margins) =
    [Y[s, n] ? quantile(margins[z[n], s, n2t[n]], (cdf(Normal(), X[s, n]) - pdry[s, n]) / (1 - pdry[s, n])) : 0.0
     for s in axes(Y, 1), n in axes(Y, 2)]

"""
    latent_field(R, Y, z, n2t, pdry, margins) -> Matrix

The inverse of [`rain_amounts`](@ref): the latent value of each wet amount, `NaN` where dry.

The level of the amount in its margin is clamped to `[1e-6, 1 - 1e-6]`.
"""
latent_field(R, Y, z, n2t, pdry, margins) =
    [Y[s, n] ? quantile(Normal(), pdry[s, n] + (1 - pdry[s, n]) * clamp(cdf(margins[z[n], s, n2t[n]], R[s, n]), 1e-6, 1 - 1e-6)) : NaN
     for s in axes(Y, 1), n in axes(Y, 2)]

function _near_pairs(distances, maxdist)
    D = size(distances, 1)
    pairs_u0 = Tuple{Int,Int,Float64}[]
    pairs_u = Tuple{Int,Int,Float64}[]
    for i in 1:D, j in 1:D
        h = Float64(distances[i, j])
        if h ≤ maxdist
            push!(pairs_u, (i, j, h))
            i < j && push!(pairs_u0, (i, j, h))
        end
    end
    return pairs_u0, pairs_u
end

function _lag_pair_params(m::GneitingMatern, pairs, u, scale)
    ψ = 1 + (u / m.a)^(2m.α)
    c_u = m.c * ψ^(m.β / 2)
    amp = m.σ² / ψ^(m.δ + m.β)
    return map(pairs) do (i, j, h)
        r = clamp(amp * matern_scaled(h, c_u, m.ν, scale), -0.9999, 0.9999)
        omr2 = 1 - r^2
        (i, j, r, sqrt(omr2), inv(2 * omr2), log(omr2) / 2)
    end
end

"""
    pairwise_loglikelihood(model, X, threshold, distances; maxdist, maxlag = 10)

Pairwise loglikelihood of the latent field `X` given which sites are wet, `X` being `NaN` where dry.

A dry site is censored below its `threshold`. Pairs are of sites at most `maxdist` apart (by
default two thirds of the largest distance) and at most `maxlag` days apart, in both orders.
"""
function pairwise_loglikelihood(model, X, threshold, distances; maxdist=2maximum(distances) / 3, maxlag=10)
    pairs_u0, pairs_u = _near_pairs(distances, maxdist)
    Φc = cdf.(Normal(), threshold)
    return _pairwise_loglikelihood_precomputed(model, X, threshold, Φc, pairs_u0, pairs_u; maxlag)
end

function _pairwise_loglikelihood_precomputed(model::GneitingMatern, X, c, Φc, pairs_u0, pairs_u; maxlag=10)
    N = size(X, 2)
    scale = 2^(1 - model.ν) / gamma(model.ν)
    params_u0 = _lag_pair_params(model, pairs_u0, 0, scale)
    params_u = [_lag_pair_params(model, pairs_u, u, scale) for u in 1:maxlag]
    tasks = map(Iterators.partition(1:N, cld(N, 4Threads.nthreads()))) do days
        Threads.@spawn _pairwise_loglikelihood(params_u0, params_u, X, c, Φc, days, zero(scale))
    end
    return sum(fetch, tasks)
end

function _pairwise_loglikelihood(params_u0, params_u, X, c, Φc, days, ℓ0)
    N = size(X, 2)
    maxlag = length(params_u)
    ℓ = ℓ0
    @inbounds for n₁ in days
        for (i, j, r, sqrt_omr2, inv_2omr2, half_log_omr2) in params_u0
            ℓ += pair_loglikelihood_both(X[i, n₁], X[j, n₁], c[i, n₁], c[j, n₁], Φc[i, n₁], Φc[j, n₁],
                r, sqrt_omr2, inv_2omr2, half_log_omr2)
        end
        for u in 1:min(maxlag, N - n₁)
            n₂ = n₁ + u
            for (i, j, r, sqrt_omr2, inv_2omr2, half_log_omr2) in params_u[u]
                ℓ += pair_loglikelihood_both(X[i, n₁], X[j, n₂], c[i, n₁], c[j, n₂], Φc[i, n₁], Φc[j, n₂],
                    r, sqrt_omr2, inv_2omr2, half_log_omr2)
            end
        end
    end
    return ℓ
end

# Sum of loglikelihoods of two standard latent values in both orders (to symmetrize `norm_cdf_2d_vfast`).
@inline function pair_loglikelihood_both(z₁, z₂, c₁, c₂, Φc₁, Φc₂, ρ, sqrt_omr2, inv_2omr2, half_log_omr2)
    isnan(z₁) && isnan(z₂) && return zero(ρ)
    Φ₂_12 = norm_cdf_2d_vfast(c₁, c₂, ρ)
    Φ₂_21 = norm_cdf_2d_vfast(c₂, c₁, ρ)
    if isnan(z₁)
        return 2 * logcdf(Normal(), (c₁ - ρ * z₂) / sqrt_omr2) -
               log(clamp(Φc₁ - Φ₂_12, 1e-10, 0.9999)) -
               log(clamp(Φc₁ - Φ₂_21, 1e-10, 0.9999))
    elseif isnan(z₂)
        return 2 * logcdf(Normal(), (c₂ - ρ * z₁) / sqrt_omr2) -
               log(clamp(Φc₂ - Φ₂_12, 1e-10, 0.9999)) -
               log(clamp(Φc₂ - Φ₂_21, 1e-10, 0.9999))
    end
    both_wet_12 = 1 - Φc₁ - Φc₂ + Φ₂_12
    both_wet_21 = 1 - Φc₂ - Φc₁ + Φ₂_21
    return 2 * (-log(2π) - half_log_omr2 - (z₁^2 + z₂^2 - 2ρ * z₁ * z₂) * inv_2omr2) -
           log(clamp(both_wet_12, 1e-10, 0.9999)) -
           log(clamp(both_wet_21, 1e-10, 0.9999))
end

# Loglikelihood of two standard latent values of correlation `ρ`, each censored below its
# threshold where `NaN`, given which of the two are wet.
function pair_loglikelihood(z₁, z₂, c₁, c₂, ρ)
    isnan(z₁) && isnan(z₂) && return zero(ρ)
    Φ₂ = norm_cdf_2d_vfast(c₁, c₂, ρ)
    if isnan(z₁)
        return logcdf(Normal(), (c₁ - ρ * z₂) / sqrt(1 - ρ^2)) - log(clamp(cdf(Normal(), c₁) - Φ₂, 1e-10, 0.9999))
    elseif isnan(z₂)
        return logcdf(Normal(), (c₂ - ρ * z₁) / sqrt(1 - ρ^2)) - log(clamp(cdf(Normal(), c₂) - Φ₂, 1e-10, 0.9999))
    end
    both_wet = 1 - cdf(Normal(), c₁) - cdf(Normal(), c₂) + Φ₂
    return -log(2π) - log(1 - ρ^2) / 2 - (z₁^2 + z₂^2 - 2ρ * z₁ * z₂) / (2(1 - ρ^2)) - log(clamp(both_wet, 1e-10, 0.9999))
end

"""
    fit_covariance(X, threshold, distances, θ0; lower, upper, maxiters) -> GneitingMatern

The `GneitingMatern` correlation, `σ² = 1`, maximizing [`pairwise_loglikelihood`](@ref).

`θ0`, `lower` and `upper` are `[c, a, α, β, δ, ν]`; LBFGS within the bounds. The thresholds
are clamped to `[-6, 6]`, so that a dry probability of 0 or 1 leaves the likelihood finite.
"""
function fit_covariance(X, threshold, distances, θ0; lower, upper, maxiters)
    c = clamp.(threshold, -6, 6)
    Φc = cdf.(Normal(), c)
    pairs_u0, pairs_u = _near_pairs(distances, 2maximum(distances) / 3)
    loss(θ, _) = -_pairwise_loglikelihood_precomputed(GneitingMatern(1.0, θ...), X, c, Φc, pairs_u0, pairs_u)
    problem = OptimizationProblem(OptimizationFunction(loss, AutoForwardDiff()), θ0; lb=lower, ub=upper)
    sol = solve(problem, LBFGS(); maxiters)
    return GneitingMatern(1.0, sol.u...)
end
