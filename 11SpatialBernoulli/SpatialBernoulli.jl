# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
using Distributions
using LinearAlgebra
using Random
using MvNormalCDF
using Markdown
using BesselK
using BesselK: _gamma
import ForwardDiff # ForwardDiff is currently the only ad supported by `BesselK`
using Optimization
include("../utils/fast_bivariate_cdf.jl")
"""
    SpatialBernoulli{TR<:Real, TS<:Real, TO<:Real, AV<:AbstractVector, AM<:AbstractMatrix, AAM<:AbstractMatrix}
Defines a discrete multivariate distribution `SpatialBernoulli` using a latend Gaussian process. The latent covarience matrix is definied by a Matern covariance (range, sill, order). 
The marginal Bernoulli probabilities are given by `λ`.
"""
struct SpatialBernoulli{TR<:Real,TS<:Real,TO<:Real,AV<:AbstractVector,AM<:AbstractMatrix,AAM<:AbstractMatrix} <: DiscreteMultivariateDistribution
    range::TR
    sill::TS
    order::TO
    λ::AV
    h::AM
    ΣU::AAM
end

Base.length(d::SpatialBernoulli) = length(d.λ)

"""
    SpatialBernoulli(range, sill, order, λ, h)
Constructor for a discrete multivariate distribution `SpatialBernoulli` using a latend Gaussian process. The latent covarience matrix is definied by a Matern covariance (range, sill, order). 
The marginal Bernoulli probabilities are given by `λ`.
The constructor used the distance martix to compute the covariance matrix.
"""
function SpatialBernoulli(range, sill, order, λ, h)
    C_GS = matern.(h; range=range, sill=sill, order=order)
    return SpatialBernoulli(range, sill, order, λ, h, C_GS)
end

"""
matern(h; range, sill, order)
expkernel(h;range,sill)
Define the matérn and exponential kernel. 
"""
function matern(h; range, sill, order)
    order == 1 / 2 && return expkernel(h; range=range, sill=sill)
    iszero(h) && return sill * sill
    arg = sqrt(2 * order) * h / range
    (sill * sill * (2^(1 - order)) / _gamma(order)) * adbesselkxv(order, arg)
end


function expkernel(h; range, sill)
    iszero(h) && return sill * sill
    arg = h / range
    (sill * sill) * exp(-arg)
end


"""
Definition of random sampling of MultivariateBernoulli :
- Generate latent field U with covariance `ΣU` - can have a variance parameter != 1
- Threshold at ``\\sigma \\phi^{-1}``
```julia
dd = MultivariateBernoulli(Diagonal(ones(5)), rand(5))
rand(dd) # just a vector of length 5
rand(dd,2) # a matrix 5x2
rand(dd,2,5) # 
```
"""
function Distributions._rand!(rng::AbstractRNG, d::SpatialBernoulli, x::AbstractVector{T}) where {T<:Real}
    u = rand(rng, MvNormal(d.ΣU))
    thresholds = quantile.(Normal(), d.λ)
    x[:] .= u .< thresholds

end



"""

    pdf(d::SpatialBernoulli, y::AbstractVector{<:Real}; m=length(d) * 100, return_error=false)
pdf(y) = CDF(N(mu,sigma))(0,...,0) is all there is to compute !
``h(y) = \\int_{-infty}^0 \\dots \\int_{-infty}^0 pdf(\\mathcal{N}((-1)^{y_s}\\sigma_s \\phi^{-1}(\\lambda_s)),\\Sigma U)``
recommended to use m= d*1000 inside of MvNormalCDF, set to d*100 as a starting point for this work (is quite ok)
"""
function Distributions.pdf(d::SpatialBernoulli, y::AbstractVector{<:Real}; m=length(d) * 500, return_error=false, a=zeros(length(d)), b=zeros(length(d)), finite_bounds=zeros(length(d)), zerosvec=fill(0, length(d)))
    finite_bounds .= quantile.(Normal(), d.λ)
    a .= ifelse.(y .== 0, finite_bounds, -Inf)
    b .= ifelse.(y .== 1, finite_bounds, Inf)
    hy = mvnormcdf(zerosvec, d.ΣU, a, b; m=m)
    return return_error ? hy : hy[1]
end


"""
    logpdf(d::SpatialBernoulli, y::AbstractVector{<:Real}[, wp::AbstractMatrix{<:Real}]; m=length(d) * 100, return_error=false)
    - Computes logpdf(y) with model d.
    - If argument `wp` is used, pairwise log-likelihood is used instead of full log-likelihood, with weights matrix `wp`.

"""

Distributions.logpdf(d::SpatialBernoulli, y::AbstractVector{<:Real}; m=length(d) * 500, return_error=false, a=zeros(length(d)), b=zeros(length(d)), finite_bounds=zeros(length(d)), zerosvec=fill(0, length(d))) = log(Distributions.pdf(d, y; m=m, return_error=return_error, a=a, b=b, finite_bounds=finite_bounds, zerosvec=zerosvec))

function Distributions.logpdf(d::SpatialBernoulli, y::AbstractVector{<:Real}, wp::AbstractMatrix{<:Real}; m=2 * 100, return_error=false)
    # Pairwise indices where weights[i, j] > 0
    upper_triangle = triu(wp)
    pairwise_indices = findall(upper_triangle .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    pairwise_sum = sum(
        2 * wp[i, j] * logpdf(
            SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), y[[i, j]]; m=m
        ) for (i, j) in pairwise_indices2
    )

    diagonal_sum = sum(wp[i, i] * logpdf(SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, i]], d.h[[i, i], [i, i]]), y[[i, i]]; m=m) for i in 1:length(d))

    return pairwise_sum - diagonal_sum
end



"""
    loglikelihood(d::SpatialBernoulli, Y::AbstractArray{<:Real}[, wp::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}]; m=length(d) * 500)
    computes pairwise log-likelihood if wp is here.
Y is an array with length(d) lines and each column is a realization of d. w is an optional vector of weights to compute weighted log-likelihood as sum(w_i llh(yi)), used in EM algorithm.  
"""

Distributions.loglikelihood(d::SpatialBernoulli, Y::AbstractArray{<:Real}; m=length(d) * 500) = sum(logpdf(d, c; m=m) for c in eachcol(Y))


Distributions.loglikelihood(d::SpatialBernoulli, Y::AbstractArray{<:Real}, w::AbstractVector{<:Real}; m=length(d) * 500) = sum(w[i] * logpdf(d, c; m=m) for (i, c) in enumerate(eachcol(Y)))


# function Distributions.loglikelihood(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}; m=2 * 100)
#     ds, n = size(Y)
#     # Pairwise indices where weights[i, j] > 0

#     pairwise_indices = findall(wp .> 0)
#     pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

#     Iij1 = ones(eltype(d.ΣU), ds, ds)
#     for (i, j) in pairwise_indices2

#         Iij1[i, j] = pdf(
#             SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), [1, 1]; m=m)
#     end

#     Iij2 = ones(eltype(d.ΣU), ds, ds)

#     for (i, j) in pairwise_indices2
#         Iij2[i, j] = pdf(
#             SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), [1, 0]; m=m)
#     end

#     Iij3 = ones(eltype(d.ΣU), ds, ds)
#     for (i, j) in pairwise_indices2

#         Iij3[i, j] = ifelse(Iij2[j, i] != 1.0, Iij2[j, i], pdf(SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), [0, 1]; m=m))

#     end

#     Iij4 = ones(eltype(d.ΣU), ds, ds)
#     for (i, j) in pairwise_indices2
#         Iij4[i, j] = pdf(
#             SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), [0, 0]; m=m)
#     end

#     n1 = similar(wp)
#     n2 = similar(wp)
#     n3 = similar(wp)
#     for (i, j) in pairwise_indices2
#         n1[i, j] = sum(Y[i, k] == 1 && Y[j, k] == 1 for k in 1:n)

#     end

#     for (i, j) in pairwise_indices2
#         n2[i, j] = sum(Y[i, k] == 1 && Y[j, k] == 0 for k in 1:n)

#     end
#     for (i, j) in pairwise_indices2
#         n3[i, j] = sum(Y[i, k] == 0 && Y[j, k] == 1 for k in 1:n)

#     end
#     n4 = n .- (n1 .+ n2 .+ n3)


#     pairs = [i != j ?
#              wp[i, j] * n1[i, j] * log(Iij1[i, j]) +
#              wp[i, j] * n2[i, j] * log(Iij2[i, j]) +
#              wp[i, j] * n3[i, j] * log(Iij3[i, j]) +
#              wp[i, j] * n4[i, j] * log(Iij4[i, j]) :
#              wp[i, j] * n1[i, j] * log(Iij1[i, j]) +
#              wp[i, j] * n4[i, j] * log(Iij4[i, j])
#              for (i, j) in pairwise_indices2]

#     pairwise_sum = sum(pairs)

#     return pairwise_sum

# end

function Distributions.loglikelihood(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}; m=2 * 100)
    ds, n = size(Y)
    # Pairwise indices where weights[i, j] > 0

    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    eps = 1e-10
    Iij1 = ones(eltype(d.ΣU), ds, ds)
    Iij2 = ones(eltype(d.ΣU), ds, ds)
    Iij3 = ones(eltype(d.ΣU), ds, ds)
    Iij4 = ones(eltype(d.ΣU), ds, ds)

    for (i, j) in pairwise_indices2
        B_ij = @view d.λ[[i, j]]
        h_ij = @view d.h[[i, j], [i, j]]
        if i == j
            Iij1[i, j] = max(eps, B_ij[1])
            Iij4[i, j] = max(eps, 1 - B_ij[1])
        else
            Iij1[i, j] = max(eps, ifelse(Iij1[j, i] != 1.0, Iij1[j, i], pdf(
                SpatialBernoulli(d.range, d.sill, d.order, B_ij, h_ij), [1, 1]; m=m)
            ))
            Iij2[i, j] = max(eps, ifelse(Iij3[j, i] != 1.0, Iij3[j, i], B_ij[1] - Iij1[i, j]))
            Iij3[i, j] = max(eps, ifelse(Iij2[j, i] != 1.0, Iij2[j, i], B_ij[2] - Iij1[i, j]))
            Iij4[i, j] = max(eps, ifelse(i == j, 1.0 - Iij1[i, j], 1.0 - Iij1[i, j] - Iij2[i, j] - Iij3[i, j]))
        end

    end

    n1 = similar(wp)
    n2 = similar(wp)
    n3 = similar(wp)
    for (i, j) in pairwise_indices2
        n1[i, j] = sum(Y[i, k] == 1 && Y[j, k] == 1 for k in 1:n)

    end

    for (i, j) in pairwise_indices2
        n2[i, j] = sum(Y[i, k] == 1 && Y[j, k] == 0 for k in 1:n)

    end
    for (i, j) in pairwise_indices2
        n3[i, j] = sum(Y[i, k] == 0 && Y[j, k] == 1 for k in 1:n)

    end
    n4 = n .- (n1 .+ n2 .+ n3)


    pairs = [i != j ?
             wp[i, j] * n1[i, j] * log(Iij1[i, j]) +
             wp[i, j] * n2[i, j] * log(Iij2[i, j]) +
             wp[i, j] * n3[i, j] * log(Iij3[i, j]) +
             wp[i, j] * n4[i, j] * log(Iij4[i, j]) :
             wp[i, j] * n1[i, j] * log(Iij1[i, j]) +
             wp[i, j] * n4[i, j] * log(Iij4[i, j])
             for (i, j) in pairwise_indices2]

    pairwise_sum = sum(pairs)

    return pairwise_sum

end

function loglikelihood_vfast(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real})
    ds, n = size(Y)
    # Pairwise indices where weights[i, j] > 0

    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    eps = 1e-10
    Iij1 = ones(eltype(d.ΣU), ds, ds)
    Iij2 = ones(eltype(d.ΣU), ds, ds)
    Iij3 = ones(eltype(d.ΣU), ds, ds)
    Iij4 = ones(eltype(d.ΣU), ds, ds)

    for (i, j) in pairwise_indices2
        B_ij = @view d.λ[[i, j]]
        h_ij = @view d.h[[i, j], [i, j]]
        if i == j
            Iij1[i, j] = max(eps, B_ij[1])
            Iij4[i, j] = max(eps, 1 - B_ij[1])
        else
            Iij1[i, j] = max(eps, ifelse(Iij1[j, i] != 1.0, Iij1[j, i], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1]), quantile(Normal(), B_ij[2]), matern(h_ij[1, 2], range=d.range, sill=d.sill, order=d.order))))
            Iij2[i, j] = max(eps, ifelse(Iij3[j, i] != 1.0, Iij3[j, i], B_ij[1] - Iij1[i, j]))
            Iij3[i, j] = max(eps, ifelse(Iij2[j, i] != 1.0, Iij2[j, i], B_ij[2] - Iij1[i, j]))
            Iij4[i, j] = max(eps, ifelse(i == j, 1.0 - Iij1[i, j], 1.0 - Iij1[i, j] - Iij2[i, j] - Iij3[i, j]))
        end

    end

    n1 = similar(wp)
    n2 = similar(wp)
    n3 = similar(wp)
    for (i, j) in pairwise_indices2
        n1[i, j] = sum(Y[i, k] == 1 && Y[j, k] == 1 for k in 1:n)

    end

    for (i, j) in pairwise_indices2
        n2[i, j] = sum(Y[i, k] == 1 && Y[j, k] == 0 for k in 1:n)

    end
    for (i, j) in pairwise_indices2
        n3[i, j] = sum(Y[i, k] == 0 && Y[j, k] == 1 for k in 1:n)

    end
    n4 = n .- (n1 .+ n2 .+ n3)


    pairs = [i != j ?
             wp[i, j] * n1[i, j] * log(Iij1[i, j]) +
             wp[i, j] * n2[i, j] * log(Iij2[i, j]) +
             wp[i, j] * n3[i, j] * log(Iij3[i, j]) +
             wp[i, j] * n4[i, j] * log(Iij4[i, j]) :
             wp[i, j] * n1[i, j] * log(Iij1[i, j]) +
             wp[i, j] * n4[i, j] * log(Iij4[i, j])
             for (i, j) in pairwise_indices2]

    pairwise_sum = sum(pairs)

    return pairwise_sum

end


function loglikelihood_vfast(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real})
    ds, n = size(Y)

    # Pairwise indices where weights > 0
    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(i[1], i[2]) for i in pairwise_indices]

    eps = 1e-10
    Iij1 = ones(eltype(d.ΣU), ds, ds)
    Iij2 = ones(eltype(d.ΣU), ds, ds)
    Iij3 = ones(eltype(d.ΣU), ds, ds)
    Iij4 = ones(eltype(d.ΣU), ds, ds)

    # Compute pairwise joint probabilities safely
    for (i, j) in pairwise_indices2
        B_ij = @view d.λ[[i, j]]
        h_ij = @view d.h[[i, j], [i, j]]

        if i == j
            # Diagonal: probability of 1 and 0
            Iij1[i, j] = clamp(B_ij[1], eps, 1 - eps)
            Iij4[i, j] = clamp(1.0 - B_ij[1], eps, 1 - eps)
            Iij2[i, j] = eps
            Iij3[i, j] = eps
        else
            # Off-diagonal: compute correlation
            ρ = clamp(matern(h_ij[1, 2], range=d.range, sill=d.sill, order=d.order), -0.999999, 0.999999)

            # Clamp marginal probabilities to avoid Inf quantiles
            p1 = clamp(B_ij[1], eps, 1 - eps)
            p2 = clamp(B_ij[2], eps, 1 - eps)

            # Bivariate probability P(Y_i=1, Y_j=1)
            p11 = norm_cdf_2d_vfast(quantile(Normal(), p1), quantile(Normal(), p2), ρ)

            # Remaining probabilities
            p10 = max(eps, p1 - p11)
            p01 = max(eps, p2 - p11)
            p00 = max(eps, 1.0 - p11 - p10 - p01)

            # Renormalize to sum = 1
            s = p11 + p10 + p01 + p00
            p11 /= s
            p10 /= s
            p01 /= s
            p00 /= s

            # Assign
            Iij1[i, j] = p11
            Iij2[i, j] = p10
            Iij3[i, j] = p01
            Iij4[i, j] = p00
        end
    end

    # Count occurrences in Y
    n1 = similar(wp)
    n2 = similar(wp)
    n3 = similar(wp)

    for (i, j) in pairwise_indices2
        n1[i, j] = sum(Y[i, k] == 1 && Y[j, k] == 1 for k in 1:n)
        n2[i, j] = sum(Y[i, k] == 1 && Y[j, k] == 0 for k in 1:n)
        n3[i, j] = sum(Y[i, k] == 0 && Y[j, k] == 1 for k in 1:n)
    end
    n4 = n .- (n1 .+ n2 .+ n3)

    # Weighted log-likelihood
    pairs = [i != j ?
             wp[i, j] * (n1[i, j] * log(Iij1[i, j]) +
                         n2[i, j] * log(Iij2[i, j]) +
                         n3[i, j] * log(Iij3[i, j]) +
                         n4[i, j] * log(Iij4[i, j])) :
             wp[i, j] * (n1[i, j] * log(Iij1[i, j]) +
                         n4[i, j] * log(Iij4[i, j]))
             for (i, j) in pairwise_indices2]

    return sum(pairs)
end

function Distributions.loglikelihood(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; m=2 * 100)
    ds, n = size(Y)
    # Pairwise indices where weights[i, j] > 0

    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    Iij1 = ones(eltype(d.ΣU), ds, ds)
    for (i, j) in pairwise_indices2

        Iij1[i, j] = pdf(
            SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), [1, 1]; m=m)
    end

    Iij2 = ones(eltype(d.ΣU), ds, ds)

    for (i, j) in pairwise_indices2
        Iij2[i, j] = pdf(
            SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), [1, 0]; m=m)
    end

    Iij3 = ones(eltype(d.ΣU), ds, ds)
    for (i, j) in pairwise_indices2
        Iij3[i, j] = ifelse(Iij2[j, i] != 1.0, Iij2[j, i], pdf(SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), [0, 1]; m=m))


    end

    Iij4 = ones(eltype(d.ΣU), ds, ds)
    for (i, j) in pairwise_indices2
        Iij4[i, j] = pdf(
            SpatialBernoulli(d.range, d.sill, d.order, d.λ[[i, j]], d.h[[i, j], [i, j]]), [0, 0]; m=m)
    end

    n1 = similar(wp)
    n2 = similar(wp)
    n3 = similar(wp)
    n4 = similar(wp)

    for (i, j) in pairwise_indices2
        n1[i, j] = sum(w[k] .* (Y[i, k] == 1 && Y[j, k] == 1) for k in 1:n)

    end

    for (i, j) in pairwise_indices2
        n2[i, j] = sum(w[k] .* (Y[i, k] == 1 && Y[j, k] == 0) for k in 1:n)

    end
    for (i, j) in pairwise_indices2
        n3[i, j] = sum(w[k] .* (Y[i, k] == 0 && Y[j, k] == 1) for k in 1:n)

    end
    for (i, j) in pairwise_indices2
        n4[i, j] = sum(w[k] .* (Y[i, k] == 0 && Y[j, k] == 0) for k in 1:n)

    end


    pairs = [i != j ?
             wp[i, j] * n1[i, j] * log(Iij1[i, j]) +
             wp[i, j] * n2[i, j] * log(Iij2[i, j]) +
             wp[i, j] * n3[i, j] * log(Iij3[i, j]) +
             wp[i, j] * n4[i, j] * log(Iij4[i, j]) :
             wp[i, j] * n1[i, j] * log(Iij1[i, j]) +
             wp[i, j] * n4[i, j] * log(Iij4[i, j])
             for (i, j) in pairwise_indices2]

    pairwise_sum = sum(pairs)

    return pairwise_sum

end

function loglikelihood_vfast(d::SpatialBernoulli, Y::AbstractArray{<:Real},
    wp::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; m=2 * 100)
    ds, n = size(Y)

    # Pairwise indices where weights[i,j] > 0
    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(i[1], i[2]) for i in pairwise_indices]

    eps = 1e-10
    Iij1 = ones(eltype(d.ΣU), ds, ds)
    Iij2 = ones(eltype(d.ΣU), ds, ds)
    Iij3 = ones(eltype(d.ΣU), ds, ds)
    Iij4 = ones(eltype(d.ΣU), ds, ds)

    # Compute pairwise probabilities safely
    for (i, j) in pairwise_indices2
        B_ij = @view d.λ[[i, j]]
        h_ij = @view d.h[[i, j], [i, j]]

        if i == j
            # Diagonal: trivial
            Iij1[i, j] = clamp(B_ij[1], eps, 1 - eps)
            Iij4[i, j] = clamp(1.0 - B_ij[1], eps, 1 - eps)
            Iij2[i, j] = eps
            Iij3[i, j] = eps
        else
            # Off-diagonal: correlation
            ρ = clamp(matern(h_ij[1, 2], range=d.range, sill=d.sill, order=d.order),
                -0.999999, 0.999999)

            # Clamp marginals to avoid Inf quantiles
            p1 = clamp(B_ij[1], eps, 1 - eps)
            p2 = clamp(B_ij[2], eps, 1 - eps)

            # Joint probability P(Y_i=1,Y_j=1)
            p11 = norm_cdf_2d_vfast(quantile(Normal(), p1),
                quantile(Normal(), p2), ρ)

            # Remaining probabilities
            p10 = max(eps, p1 - p11)
            p01 = max(eps, p2 - p11)
            p00 = max(eps, 1.0 - p11 - p10 - p01)

            # Renormalize to sum = 1
            s = p11 + p10 + p01 + p00
            p11 /= s
            p10 /= s
            p01 /= s
            p00 /= s

            # Assign
            Iij1[i, j] = p11
            Iij2[i, j] = p10
            Iij3[i, j] = p01
            Iij4[i, j] = p00
        end
    end

    # Weighted counts
    n1 = similar(wp)
    n2 = similar(wp)
    n3 = similar(wp)
    n4 = similar(wp)

    for (i, j) in pairwise_indices2
        n1[i, j] = sum(w[k] * (Y[i, k] == 1 && Y[j, k] == 1) for k in 1:n)
        n2[i, j] = sum(w[k] * (Y[i, k] == 1 && Y[j, k] == 0) for k in 1:n)
        n3[i, j] = sum(w[k] * (Y[i, k] == 0 && Y[j, k] == 1) for k in 1:n)
        n4[i, j] = sum(w[k] * (Y[i, k] == 0 && Y[j, k] == 0) for k in 1:n)
    end

    # Weighted log-likelihood
    pairs = [i != j ?
             wp[i, j] * (n1[i, j] * log(Iij1[i, j]) +
                         n2[i, j] * log(Iij2[i, j]) +
                         n3[i, j] * log(Iij3[i, j]) +
                         n4[i, j] * log(Iij4[i, j])) :
             wp[i, j] * (n1[i, j] * log(Iij1[i, j]) +
                         n4[i, j] * log(Iij4[i, j]))
             for (i, j) in pairwise_indices2]

    return sum(pairs)
end



"""
    fit_mle(d::SpatialBernoulli, Y::AbstractArray{<:Real}[,wp::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}]; solver = Optimization.LBFGS(), m = 1000*length(d), return_sol = false, solkwargs...)
Return the (weigthed) MLE for the distribution `d::SpatialBernoulli`. 
- `solver` choice of solver
- `solkwargs` any keywords arguments that can be put inside the `solve` functions e.g. fit_mle(d, Y; maxiters = 2)
- `w` not necessary, used for using weighted loglikelihood instead of loglikelihood
- `wp` not technically  necessary, used for using pairwise loglikelihood instead of true loglikelihood. 

"""
function Distributions.fit_mle(d::SpatialBernoulli, Y::AbstractArray{<:Real}; solver=Optimization.LBFGS(), m=100 * length(d), return_sol=false, order=nothing, solkwargs...)
    λ_fitted = succprob.([fit_mle(Bernoulli, c) for c in eachrow(Y)])
    function optimfunction(u, p, m=500)
        if isnothing(order)
            dd = SpatialBernoulli(exp(u[1]), d.sill, exp(u[2]), λ_fitted, d.h)
        else
            dd = SpatialBernoulli(exp(u[1]), d.sill, order, λ_fitted, d.h)
        end
        return -loglikelihood(dd, p; m=m)
    end
    optf(m) = OptimizationFunction((u, p) -> optimfunction(u, p, m), AutoForwardDiff())
    u0 = log.([d.range, isnothing(order) ? d.order : order])
    prob = OptimizationProblem(optf(m), u0, Y)
    sol = solve(prob, solver; solkwargs...)
    if !SciMLBase.successful_retcode(sol.retcode) # not sucessful solve
        @warn "sol.retcode = $(sol.retcode)"
    end
    return return_sol ? (SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h), sol) : SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h)
end

function Distributions.fit_mle(d::SpatialBernoulli, Y::AbstractArray{<:Real}, w::AbstractVector{<:Real}; solver=Optimization.LBFGS(), m=100 * length(d), return_sol=false, order=nothing, solkwargs...)
    λ_fitted = succprob.([fit_mle(Bernoulli, c, w) for c in eachrow(Y)])
    function optimfunction(u, p, m=500)
        if isnothing(order)
            dd = SpatialBernoulli(exp(u[1]), d.sill, exp(u[2]), λ_fitted, d.h)
        else
            dd = SpatialBernoulli(exp(u[1]), d.sill, order, λ_fitted, d.h)
        end
        return -loglikelihood(dd, p[1], p[2]; m=m)
    end

    optf(m) = OptimizationFunction((u, p) -> optimfunction(u, p, m), AutoForwardDiff())
    u0 = log.([d.range, isnothing(order) ? d.order : order])
    prob = OptimizationProblem(optf(m), u0, [Y, w])
    sol = solve(prob, solver; solkwargs...)
    if !SciMLBase.successful_retcode(sol.retcode) # not sucessful solve
        @warn "sol.retcode = $(sol.retcode)"
    end
    return return_sol ? (SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h), sol) : SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h)
end




function Distributions.fit_mle(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real};
    solver=Optimization.LBFGS(), m=100 * 2, return_sol=false, order=nothing, solkwargs...)

    λ_fitted = succprob.([fit_mle(Bernoulli, c) for c in eachrow(Y)])

    if isnothing(order)
        # Branch: Optimize both `range` and `order`
        function optimfunction(u, p, m=500)
            dd = SpatialBernoulli(exp(u[1]), d.sill, exp(u[2]), λ_fitted, d.h)
            return -loglikelihood(dd, p, wp; m=m)
        end
        optf = m -> OptimizationFunction((u, p) -> optimfunction(u, p, m), AutoForwardDiff())
        u0 = log.([d.range, d.order])
        prob = OptimizationProblem(optf(m), u0, Y)

        # Solve the problem
        sol = solve(prob, solver; solkwargs...)

        # Check solution status
        if !SciMLBase.successful_retcode(sol.retcode)
            @warn "sol.retcode = $(sol.retcode)"
        end

        # Return the result
        return return_sol ?
               (SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h), sol) :
               SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h)


    else
        # Branch: Optimize only `range`, fixing `order`
        function optimfunction2(u, p, m=2 * 100)
            dd = SpatialBernoulli(exp(u[1]), d.sill, order, λ_fitted, d.h)
            return -loglikelihood(dd, p, wp; m=m)
        end
        optf2 = m -> OptimizationFunction((u, p) -> optimfunction2(u, p, m), AutoForwardDiff())
        u0 = log.([d.range])
        prob = OptimizationProblem(optf2(m), u0, Y)

        # Solve the problem
        sol = solve(prob, solver; solkwargs...)

        # Check solution status
        if !SciMLBase.successful_retcode(sol.retcode)
            @warn "sol.retcode = $(sol.retcode)"
        end

        # Return the result


        return return_sol ?
               (SpatialBernoulli(exp(sol.u[1]), d.sill, order, λ_fitted, d.h), sol) :
               SpatialBernoulli(exp(sol.u[1]), d.sill, order, λ_fitted, d.h)
    end

end


function Distributions.fit_mle(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; solver=Optimization.LBFGS(), m=1000 * 2, return_sol=false, order=nothing, solkwargs...)

    λ_fitted = succprob.([fit_mle(Bernoulli, c, w) for c in eachrow(Y)])

    if isnothing(order)
        # Branch: Optimize both `range` and `order`
        function optimfunction(u, p, m=500)
            dd = SpatialBernoulli(exp(u[1]), d.sill, exp(u[2]), λ_fitted, d.h)
            return -loglikelihood(dd, p[1], wp, p[2]; m=m)
        end
        optf = m -> OptimizationFunction((u, p) -> optimfunction(u, p, m), AutoForwardDiff())
        u0 = log.([d.range, d.order])
        prob = OptimizationProblem(optf(m), u0, [Y, w])

        # Solve the problem
        sol = solve(prob, solver; solkwargs...)

        # Check solution status
        if !SciMLBase.successful_retcode(sol.retcode)
            @warn "sol.retcode = $(sol.retcode)"
        end

        # Return the result
        return return_sol ?
               (SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h), sol) :
               SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h)


    else
        # Branch: Optimize only `range`, fixing `order`
        function optimfunction2(u, p, m=2 * 100)
            dd = SpatialBernoulli(exp(u[1]), d.sill, order, λ_fitted, d.h)
            return -loglikelihood(dd, p[1], wp, p[2]; m=m)
        end
        optf2 = m -> OptimizationFunction((u, p) -> optimfunction2(u, p, m), AutoForwardDiff())
        u0 = log.([d.range])
        prob = OptimizationProblem(optf2(m), u0, [Y, w])

        # Solve the problem
        sol = solve(prob, solver; solkwargs...)

        # Check solution status
        if !SciMLBase.successful_retcode(sol.retcode)
            @warn "sol.retcode = $(sol.retcode)"
        end

        # Return the result


        return return_sol ?
               (SpatialBernoulli(exp(sol.u[1]), d.sill, order, λ_fitted, d.h), sol) :
               SpatialBernoulli(exp(sol.u[1]), d.sill, order, λ_fitted, d.h)
    end
end


include("../utils/fast_bivariate_cdf.jl")

function fit_mle_vfast(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real};
    solver=Optimization.LBFGS(), return_sol=false, order=nothing, solkwargs...)

    λ_fitted = succprob.([fit_mle(Bernoulli, c) for c in eachrow(Y)])

    if isnothing(order)
        # Branch: Optimize both `range` and `order`
        function optimfunction(u, p)
            dd = SpatialBernoulli(exp(u[1]), d.sill, exp(u[2]), λ_fitted, d.h)
            return -loglikelihood_vfast(dd, p, wp)
        end
        optf2 = OptimizationFunction(
            (u, p) -> optimfunction(u, p),
            AutoForwardDiff()
        )
        u0 = log.([d.range, d.order])
        prob = OptimizationProblem(optf2, u0, Y)

        # Solve the problem
        sol = solve(prob, solver; solkwargs...)

        # Check solution status
        if !SciMLBase.successful_retcode(sol.retcode)
            @warn "sol.retcode = $(sol.retcode)"
        end

        # Return the result
        return return_sol ?
               (SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h), sol) :
               SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h)


    else
        # Branch: Optimize only `range`, fixing `order`
        function optimfunction2(u, p)
            dd = SpatialBernoulli(exp(u[1]), d.sill, order, λ_fitted, d.h)
            return -loglikelihood_vfast(dd, p, wp)
        end
        optf2 = OptimizationFunction(
            (u, p) -> optimfunction2(u, p),
            AutoForwardDiff()
        )
        u0 = log.([d.range])
        @show optimfunction2(u0, Y)
        prob = OptimizationProblem(optf2, u0, Y)

        # Solve the problem
        sol = solve(prob, solver; solkwargs...)

        # Check solution status
        if !SciMLBase.successful_retcode(sol.retcode)
            @warn "sol.retcode = $(sol.retcode)"
        end

        # Return the result


        return return_sol ?
               (SpatialBernoulli(exp(sol.u[1]), d.sill, order, λ_fitted, d.h), sol) :
               SpatialBernoulli(exp(sol.u[1]), d.sill, order, λ_fitted, d.h)
    end

end


function fit_mle_vfast(d::SpatialBernoulli, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, w::AbstractVector{<:Real}; solver=Optimization.LBFGS(), return_sol=false, order=nothing, solkwargs...)

    λ_fitted = succprob.([fit_mle(Bernoulli, c, w) for c in eachrow(Y)])

    if isnothing(order)
        # Branch: Optimize both `range` and `order`
        function optimfunction(u, p)
            dd = SpatialBernoulli(exp(u[1]), d.sill, exp(u[2]), λ_fitted, d.h)
            return -loglikelihood_vfast(dd, p[1], wp, p[2])
        end
        optf2 = OptimizationFunction(
            (u, p) -> optimfunction(u, p),
            AutoForwardDiff()
        )
        u0 = log.([d.range, d.order])
        prob = OptimizationProblem(optf2, u0, [Y, w])

        # Solve the problem
        sol = solve(prob, solver; solkwargs...)

        # Check solution status
        if !SciMLBase.successful_retcode(sol.retcode)
            @warn "sol.retcode = $(sol.retcode)"
        end

        # Return the result
        return return_sol ?
               (SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h), sol) :
               SpatialBernoulli(exp(sol.u[1]), d.sill, exp(sol.u[2]), λ_fitted, d.h)


    else
        # Branch: Optimize only `range`, fixing `order`
        function optimfunction2(u, p)
            dd = SpatialBernoulli(exp(u[1]), d.sill, order, λ_fitted, d.h)
            return -loglikelihood_vfast(dd, p[1], wp, p[2])
        end
        optf2 = OptimizationFunction(
            (u, p) -> optimfunction2(u, p),
            AutoForwardDiff()
        )
        u0 = log.([d.range])
        prob = OptimizationProblem(optf2, u0, [Y, w])

        # Solve the problem
        sol = solve(prob, solver; solkwargs...)

        # Check solution status
        if !SciMLBase.successful_retcode(sol.retcode)
            @warn "sol.retcode = $(sol.retcode)"
        end

        # Return the result


        return return_sol ?
               (SpatialBernoulli(exp(sol.u[1]), d.sill, order, λ_fitted, d.h), sol) :
               SpatialBernoulli(exp(sol.u[1]), d.sill, order, λ_fitted, d.h)
    end
end
