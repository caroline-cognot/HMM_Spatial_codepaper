#####################################################################################################################################""
#####              code from package ExpectationMaximisation ########################################################################
#####################################################################################################################################
# using ExpectationMaximization

abstract type AbstractEM end


####################################################################################################################################"
########### \src\ExpectationMaximisation.jl - commented all package - building lines ########################################################################
######################################################################################################################################

# module ExpectationMaximization

using ArgCheck
using Distributions
using Distributions: ArrayOfUnivariateDistribution, VectorOfUnivariateDistribution # for product distributions
using LogExpFunctions: logsumexp!, logsumexp
using StatsBase: weights
using Random # to add @kwdef 

# Extended functions
import Distributions: fit_mle, params

# export fit_mle, fit_mle!

abstract type AbstractEM end

# Utilities

size_sample(y::AbstractMatrix) = size(y, 2)
size_sample(y::AbstractVector) = length(y)

argmaxrow(M) = [argmax(r) for r in eachrow(M)]

"""
    predict(mix::MixtureModel, y::AbstractVector; robust=false)
Evaluate the most likely category for each observations given a `MixtureModel`.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
"""
function predict(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
    return argmaxrow(predict_proba(mix, y; robust=robust))
end

"""
    predict_proba(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
Evaluate the probability for each observations to belong to a category given a `MixtureModel`..
- `robust = true` will prevent the (log)likelihood to under(overflow)flow to `-∞` (or `∞`).
"""
function predict_proba(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
    # evaluate likelihood for each components k
    dists = mix.components
    α = probs(mix)
    K = length(dists)
    N = size_sample(y)
    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)
    E_step!(LL, c, γ, dists, α, y; robust=robust)
    return γ
end



####################################################################################################################################"
########### \src\fit_em.jl - david  ########################################################################
######################################################################################################################################

"""
    fit_mle(mix::MixtureModel, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, rtol=nothing, robust=false, infos=false)
Use the an Expectation Maximization (EM) algorithm to maximize the Loglikelihood (fit) the mixture with an i.i.d sample `y`.
The `mix` input is a mixture that is used to initilize the EM algorithm.
- `weights` when provided, it will compute a weighted version of the EM. (Useful for fitting mixture of mixtures)
- `method` determines the algorithm used.
- `infos = true` returns a `Dict` with informations on the algorithm (converged, iteration number, loglikelihood).
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
- `atol` criteria determining the convergence of the algorithm. If the Loglikelihood difference between two iteration `i` and `i+1` is smaller than `atol` i.e. `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<atol`, the algorithm stops.
- `rtol` relative tolerance for convergence, `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<rtol*(|ℓ⁽ⁱ⁺¹⁾| + |ℓ⁽ⁱ⁾|)/2` (does not check if `rtol` is `nothing`)
- `display` value can be `:none`, `:iter`, `:final` to display Loglikelihood evolution at each iterations `:iter` or just the final one `:final`
"""
function Distributions.fit_mle(
    mix::MixtureModel,
    y::AbstractVecOrMat,
    weights...;
    method=ClassicEM(),
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
    infos=false,
)

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

    #TODO is there a better way to do that when weight are not provided ? + avoid when infos = false allocating history?
    if isempty(weights)
        history = fit_mle!(
            α,
            dists,
            y,
            method;
            display=display,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
            robust=robust,
        )
    else
        history = fit_mle!(
            α,
            dists,
            y,
            weights...,
            method;
            display=display,
            maxiter=maxiter,
            atol=atol,
            rtol=rtol,
            robust=robust,
        )
    end

    return infos ? (MixtureModel(dists, α), history) : MixtureModel(dists, α)
end

"""
    fit_mle(mix::AbstractArray{<:MixtureModel}, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, rtol=nothing, robust=false, infos=false)

Do the same as `fit_mle` for each (initial) mixtures in the mix array. Then it selects the one with the largest loglikelihood.
Warning: It uses try and catch to avoid errors messages in case EM converges toward a singular solution (probably using robust should be enough in most case to avoid errors).
"""
function Distributions.fit_mle(
    mix::AbstractArray{<:MixtureModel},
    y::AbstractVecOrMat,
    weights...;
    method=ClassicEM(),
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
    infos=false,
)

    mx_max, history_max = fit_mle(
        mix[1],
        y,
        weights...;
        method=method,
        display=display,
        maxiter=maxiter,
        atol=atol,
        robust=robust,
        infos=true,
    )
    for j in eachindex(mix)[2:end]
        try
            mx_new, history_new = fit_mle(
                mix[j],
                y,
                weights...;
                method=method,
                display=display,
                maxiter=maxiter,
                atol=atol,
                rtol=rtol,
                robust=robust,
                infos=true,
            )
            if history_max["logtots"][end] < history_new["logtots"][end]
                mx_max = mx_new
                history_max = copy(history_new)
            end
        catch
            continue
        end
    end
    return infos ? (mx_max, history_max) : mx_max
end

# E-step methods

function E_step!(
    LL::AbstractMatrix{T},
    c::AbstractVector{T},
    γ::AbstractMatrix{T},
    dists::AbstractVector{F} where {F<:Distribution},
    α::AbstractVector,
    y::AbstractVector{<:Real};
    robust=false,
) where {T<:AbstractFloat}
    # evaluate likelihood for each type k
    for k in eachindex(dists)
        LL[:, k] .= log(α[k]) .+ logpdf.(dists[k], y)
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    logsumexp!(c, LL) # c[:] = logsumexp(LL, dims=2)
    γ[:, :] .= exp.(LL .- c)
end

function E_step!(
    LL::AbstractMatrix,
    c::AbstractVector,
    γ::AbstractMatrix,
    dists::AbstractVector{F} where {F<:Distribution},
    α::AbstractVector,
    y::AbstractMatrix;
    robust=false,
)
    # evaluate likelihood for each type k
    @views for k in eachindex(dists)
        LL[:, k] .= log(α[k])
        for n in axes(y, 2)
            LL[n, k] += logpdf(dists[k], y[:, n])
        end
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    logsumexp!(c, LL) # c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)
end


####################################################################################################################################"
########### \src\classic_em.jl - david ########################################################################
######################################################################################################################################

"""
    ClassicEM<:AbstractEM
The EM algorithm was introduced by A. P. Dempster, N. M. Laird and D. B. Rubin in 1977 in the reference paper [*Maximum Likelihood from Incomplete Data Via the EM Algorithm*](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1977.tb01600.x).
"""
struct ClassicEM <: AbstractEM end

struct PairwiseClassicEM <: AbstractEM end

"""
    fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, method::ClassicEM; display=:none, maxiter=1000, atol=1e-3, rtol=nothing, robust=false)
Use the EM algorithm to update the Distribution `dists` and weights `α` composing a mixture distribution.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
- `atol` criteria determining the convergence of the algorithm. If the Loglikelihood difference between two iteration `i` and `i+1` is smaller than `atol` i.e. `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<atol`, the algorithm stops.
- `rtol` relative tolerance for convergence, `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<rtol*(|ℓ⁽ⁱ⁺¹⁾| + |ℓ⁽ⁱ⁾|)/2` (does not check if `rtol` is `nothing`)
- `display` value can be `:none`, `:iter`, `:final` to display Loglikelihood evolution at each iterations `:iter` or just the final one `:final`
"""
function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    method::ClassicEM;
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size_sample(y), length(dists)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # E-step
    E_step!(LL, c, γ, dists, α, y; robust=robust)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter
        # M-step
        M_step!(α, dists, y, γ, method)

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust=robust)

        # Loglikelihood
        logtotp = sum(c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < atol || (rtol !== nothing && abs(logtotp - logtot) < rtol * (abs(logtot) + abs(logtotp)) / 2)
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println(
                "EM has not converged after $(history["iterations"]) iterations, final loglikelihood = $logtot",
            )
        end
    end

    return history
end

"""
    M_step!(α, dists, y, cat, method::StochasticEM)
For the `ClassicEM` the weigths `γ` computed at E-step for each observation in `y` are used to update `α` and `dists`.
"""
function M_step!(α, dists, y::AbstractVecOrMat, γ, method::ClassicEM)
    α[:] = mean(γ, dims=1)
    dists[:] = [fit_mle(dists[k], y, γₖ) for (k, γₖ) in enumerate(eachcol(γ))]
end

#TODO: could probably replace γ, w by γ*w,
function M_step!(α, dists, y::AbstractVecOrMat, γ, w, method::ClassicEM)
    α[:] = mean(γ, weights(w), dims=1)
    dists[:] = [fit_mle(dists[k], y, w[:] .* γₖ) for (k, γₖ) in enumerate(eachcol(γ))]
end

function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    w::AbstractVector,
    method::ClassicEM;
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    N, K = size_sample(y), length(dists)
    @argcheck length(w) == N
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # E-step
    E_step!(LL, c, γ, dists, α, y; robust=robust)

    # Loglikelihood
    logtot = sum(w[n] * c[n] for n = 1:N) #dot(w, c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter
        # M-step
        M_step!(α, dists, y, γ, w, method)

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust=robust)

        # Loglikelihood
        logtotp = sum(w[n] * c[n] for n in eachindex(c)) #dot(w, c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < atol || (rtol !== nothing && abs(logtotp - logtot) < rtol * (abs(logtot) + abs(logtotp)) / 2)
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println(
                "EM has not converged after $(history["iterations"]) iterations, final loglikelihood = $logtot",
            )
        end
    end

    return history
end
