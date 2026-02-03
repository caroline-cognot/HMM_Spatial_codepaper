# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
# from include("SpatialBernoulli/SpatialBernoulli.jl")
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
function Distributions._rand!(rng::Random.AbstractRNG, d::SpatialBernoulli, x::AbstractVector{T}) where {T<:Real}
    u = rand(rng, MvNormal(d.ΣU))
    thresholds = quantile.(Normal(), d.λ)
    x[:] .= u .< thresholds
end

# =====================================================
# Covariance model definition
# =====================================================
abstract type CovarianceStructure end

struct ExpExp{T1,T2<:Real} <: CovarianceStructure
    sill::T1      # variance
    range_s::T2  # spatial range
    range_t::T2   # temporal range
end

# =====================================================
# Covariance computation
# =====================================================
function cov_spatiotemporal(model::ExpExp, h, u)
    return model.sill .* exp.(-h ./ model.range_s) .* exp(-u / model.range_t)
end

# =====================================================
# Gaussian Field definition
# =====================================================
struct GaussianField
    coords::Matrix{Float64}         # D × 2
    model::CovarianceStructure
    Mat_h::Matrix{Float64}
end


function GaussianField(coords, model)
    D = size(coords, 1)
    Mat_h = zeros((D, D))
    for j in 1:D
        for i in (j+1):D

            Mat_h[i, j] = Mat_h[j, i] = haversine(coordll[i, :], coordll[j, :]) / 1000
        end
    end
    return GaussianField(coords, model, Mat_h)
end

struct GneitingMatern{T1,T<:Real} <: CovarianceStructure
    σ²::T1 #sill
    c::T #spatial scale
    a::T #temporal scale
    α::T # temporal power
    β::T #sep
    δ::T #temporal covariance 
    ν::T # regularity
end

function matern0(h, c, ν)
    iszero(h) && return 1.0
    arg = h / c
    return ((2^(1 - ν)) / BesselK._gamma(ν)) * BesselK.adbesselkxv(ν, arg)
end

function cov_spatiotemporal(p::GneitingMatern, h, u)
    arg = 1 + (u / p.a)^(2p.α)
    argpow = arg^(p.β / 2)
    rho = matern0.(h, p.c * argpow, p.ν) / arg^(p.δ + p.β)
    return p.σ² .* rho
end


# from include("/home/caroline/Gitlab_SWG_Caro/hmmspa/TruncatedMVN/Covariance_models.jl")



"""
    spatiotemporal_correlation(data, mat_dist, max_lag_t, max_dist; min_pairs=30)

Compute empirical spatio-temporal correlation for a matrix `data` of size (D × N)
where D = number of locations, N = number of time points.

- `mat_dist` : D × D matrix of distances
- `max_lag_t` : maximum temporal lag (in days)
- `max_dist` : maximum spatial distance (in km)
- `min_pairs` : minimum number of point pairs required to compute correlation

Returns:
    corr_matrix : matrix of correlations indexed by (time_lag, spatial_bin)
    time_lags   : vector of time lags
    dist_bins   : vector of spatial bin centers
"""
function spatiotemporal_correlation(data, mat_dist, max_lag_t, max_dist; min_pairs=30, n_dist_bins=15
)
    D, N = size(data)
    @show D,N

    # Compute pairwise distances in km (Haversine)

    # Define spatial bins (uniform)
    dist_edges = range(0, max_dist; length=n_dist_bins + 1)
    dist_bins = 0.5 .* (dist_edges[1:end-1] .+ dist_edges[2:end])

    time_lags = 0:max_lag_t
    corr_matrix = fill(NaN, length(time_lags), length(dist_bins))

    for (li, lag) in enumerate(time_lags)
        # For each spatial bin
        for (bi, (dmin, dmax)) in enumerate(zip(dist_edges[1:end-1], dist_edges[2:end]))
            corrs = Float64[]

            # Loop over all station pairs
            for s1 in 1:D
                for s2 in (s1):D
                    dist = mat_dist[s1, s2]
                    if dmin <= dist < dmax
                        # Build paired time series for this lag
                        tmax = N - lag
                        x = data[s1, 1:tmax]
                        y = data[s2, 1+lag:tmax+lag]

                        # Only if both have enough non-missing datan
                        valid = .!isnan.(x) .& .!isnan.(y) 

                        if count(valid) >= min_pairs
                            xv, yv = x[valid], y[valid]
                            # if std(xv) > 0 && std(yv) > 0   # avoid NaN correlations
                                push!(corrs, cor(xv, yv))
                            # end
                        end
                    end
                end
            end

            if !isempty(corrs)
                corr_matrix[li, bi] = mean(corrs)
            end
        end
    end

    return corr_matrix, time_lags, dist_bins
end

## from PeriodicHMMSpa.jls


"""
    ARPeriodicHMMSpa([a, ]A, B) -> ARPeriodicHMMSpa

Build an Auto Regressive Periodic Hidden Markov Chain with Spatial Bernoulli emission `ARPeriodicHMMSpa` with transition matrix `A(t)` and observation distributions `B(t)`.  
If the initial state distribution `a` is not specified, it does not work. Please give initial state distribution.

Observations distributions can only be SpatialBernoulli.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractArray{T,3}`: transition matrix.
- `B`: rain probabilities
- `R` : range parameter
-  `h` distance matrix.
"""
struct PeriodicHMMSpaMemory{T}
    a::Vector{T}
    A::Array{T,3}
    R::Array{T,2}
    B::Array{T,4} 
    h::AbstractMatrix
end


#!!! ajout de Base.size sinon tu overload pas, tu remplaces!
Base.size(hmm::PeriodicHMMSpaMemory, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]
# K                # D             # T          # number of memory


#-----------------equivalent to trig_conversion.jl  ---------------------------#

function Trig2PeriodicHMMspaMemory(a::AbstractVector, my_trans_θ::AbstractArray{<:AbstractFloat,3}, Bernoulli_θ::AbstractArray{<:AbstractFloat,4}, Range_θ::AbstractArray{<:AbstractFloat,2}, my_T::Integer, my_h::AbstractMatrix)
    my_K, my_D, my_size_order = size(Bernoulli_θ)
    @assert my_K == size(my_trans_θ, 1)

    # make transition matrices as function of time
    if my_K == 1
        my_A = ones(my_K, my_K, my_T)
    else
        my_A = zeros(my_K, my_K, my_T)
        for k = 1:my_K, l = 1:my_K-1, t = 1:my_T
            my_A[k, l, t] = exp(polynomial_trigo(t, my_trans_θ[k, l, :], my_T))
        end
        for k = 1:my_K, t = 1:my_T
            my_A[k, my_K, t] = 1  # last colum is 1/normalization (one could do otherwise)
        end
        normalization_polynomial = [1 + sum(my_A[k, l, t] for l = 1:my_K-1) for k = 1:my_K, t = 1:my_T]
        for k = 1:my_K, l = 1:my_K, t = 1:my_T
            my_A[k, l, t] /= normalization_polynomial[k, t]
        end
    end
    my_A
    # A is a K*K* T matrix of transition.

    #make emission parameters
    my_p = [1 / (1 + exp(polynomial_trigo(t, Bernoulli_θ[k, s, h, :], my_T))) for k = 1:my_K, t = 1:my_T, s = 1:my_D, h = 1:my_size_order]
    # p is a K (states)* T(period) *  D (stations) * m+1 (memory) vector.
    my_range = [exp(polynomial_trigo(t, Range_θ[k, :], my_T)) for k = 1:my_K, t = 1:my_T]
    # range is a K (states)* T(period)  * m+1 (memory) vector.
    # return (my_A, p, range)


    model = PeriodicHMMSpaMemory(a, my_A, my_range, my_p, my_h)
    return model
end


## from egpd functions


struct MixedUniformTail{T1<:ContinuousUnivariateDistribution, T2<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    p::Float64 # probability of the left part
    uniform_part::T1 #left part
    tail_part::T2 # right part
    a::Float64 #minimum value, for precip it is 0.1
    b::Float64 #threshold between both part, 0.5 for precips (included in left part)
end

# PDF
function Distributions.pdf(d::MixedUniformTail, y::Real)
    if y < d.a
        return 0.0
    elseif y <= d.b
        return d.p * pdf(d.uniform_part, y)
    else
        return (1 - d.p) * pdf(d.tail_part, y - d.b)
    end
end

# CDF
function Distributions.cdf(d::MixedUniformTail, y::Real)
    if y < d.a
        return NaN #was NaN
    elseif y <= d.b
        return d.p * cdf(d.uniform_part, y)
    else
        return d.p + (1 - d.p) * cdf(d.tail_part, y - d.b)
    end
end

# Quantile function
function Distributions.quantile(d::MixedUniformTail, q::Real)
    if q < 0 || q > 1
        throw(DomainError(q, "Quantile outside [0,1]"))
    end
    if q <= d.p
        return quantile(d.uniform_part, q / d.p)
    else
        return d.b + quantile(d.tail_part, (q - d.p) / (1 - d.p))
    end
end

# Random sampling
function Base.rand(rng::Random.AbstractRNG, d::MixedUniformTail)
    if rand(rng) <= d.p
        return rand(rng, d.uniform_part)
    else
        return d.b + rand(rng, d.tail_part)
    end
end

function forced_by_normal_rand(d::MixedUniformTail,v::AbstractArray)
    Usims = cdf.(Normal(),v) #transform to uniform
    Usims .= quantile.(d,Usims)
return Usims
end

function forced_by_normal_rand(d::MixedUniformTail,v::Real)
    Usims = cdf(Normal(),v) #transform to uniform
    Usims = quantile(d,Usims)
return Usims
end


Base.rand(d::MixedUniformTail) = rand(Random.GLOBAL_RNG, d)

struct MixedUniformTailModel
    dists::Vector{MixedUniformTail}   # one distribution per class
    K::Int                           # number of classes
end



"""
    rand(model::MixedUniformTailModel, zt)

Draw one sample given class `zt`.
"""
function Base.rand(model::MixedUniformTailModel, zt::Int)
    return rand(model.dists[zt])
end

"""
    forced_by_normal_rand(model::MixedUniformTailModel, zt, v)

Transform latent Gaussian `v ~ N(0,1)` to rainfall given class `zt`.
"""
function forced_by_normal_rand(model::MixedUniformTailModel, zt::Int, v)
    d = model.dists[zt]
    return forced_by_normal_rand(d, v)
end
