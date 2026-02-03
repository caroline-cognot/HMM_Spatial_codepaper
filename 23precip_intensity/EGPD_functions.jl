# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
using Distributions, Random
using   ExtendedExtremes

import SmoothPeriodicStatsModels: polynomial_trigo 


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
function Base.rand(rng::AbstractRNG, d::MixedUniformTail)
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

# estimation


function fit_mix(::Type{MixedUniformTail}, data;left=0.1,middle=0.5,initial_values=[1 ,1, 0],left_censoring=0)
    u = middle
    prop_smallrain = sum(left .<= data .<= u) / sum(data .> 0)
    y = data[data.>u] .- u

    tail_part = fit_mle(ExtendedGeneralizedPareto{TBeta}, y,initial_values,leftcensoring = left_censoring)

    return MixedUniformTail(prop_smallrain, Uniform(left,middle), tail_part, left, middle)
end



######### try the new functions ###############################"
# Parameters
a, b = 0.1, 0.5
p = 0.3
uniform_part = Uniform(a, b)
tail_part = ExtendedGeneralizedPareto( TBeta(0.4), GeneralizedPareto(0.0,3, 0.1))   # Example distribution for y > b

# Create the mixed distribution
d = MixedUniformTail(p, uniform_part, tail_part, a, b)

# Example: use it like any Distributions.jl distribution
println(pdf(d, 0.2))      # PDF in uniform part
println(cdf(d, 0.2))      # CDF in uniform part
println(quantile(d, 0.8)) # Quantile in tail part

# Random draws
samples = rand(d, 1000)

using Plots
histogram(samples, bins=50, normalize=true, label="Sampled PDF")
Plots.plot!(x -> pdf(d, x), 0, 50, label="Theoretical PDF", lw=2)


dd = fit_mix(MixedUniformTail,samples,initial_values=[10 ,1 ,1])
samples2=rand(dd,10000)
histogram!(samples2 ,normalize=true)

samplesgaussian = rand(Normal(),1000)
samples3=forced_by_normal_rand(dd,samplesgaussian)
histogram!(samples2,normalize=true)


using Base.Threads
using Optim
# struct PeriodicDistributions{T1<:ContinuousUnivariateDistribution}
#     dists::Array{T1,2}
# end

function polynomial_trigo(t, β, T)
    d = (length(β) - 1) ÷ 2
    # println("in poly trigo : d = ",d)
    # println("in poly trigo : T = ",T)

    if d == 0
        return β[1]
    else
        f = 2π / T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] + sum(β[2*l] * cos(f * l * t) + β[2*l+1] * sin(f * l * t) for l = 1:d)
    end
end



function my_rand(dists::AbstractVector{<:AbstractVector{<:MixedUniformTail}}
    , n2t::AbstractVector)
    D = size(dists, 1)
    T = size(dists[1])
    N = length(n2t)
    y = Matrix{Float64}(undef, D, N)
    for s in 1:D
        for n in 1:N
            t = n2t[n]
            y[s, n] = rand(dists[s][t])
        end
    end
    return y
end

function my_rand(dists::AbstractVector{<:AbstractVector{<:MixedUniformTail}}
    , n2t::AbstractVector,occurence)
    D = size(dists, 1)
    T = size(dists[1])
    N = length(n2t)
    y = Matrix{Float64}(undef, D, N)
    for s in 1:D
        for n in 1:N
            t = n2t[n]
            y[s, n] = rand(dists[s][t])
        end
    end
    return y.*occurence
end

function my_logpdfs(dists, y::AbstractArray{<:AbstractFloat,2}; n2t=n_to_t(size(y, 2), size(dists, 1))::AbstractVector{<:Integer})
    D = size(dists, 1)
    T = size(dists[1])
    N = size(n2t, 1)
    logpdfs = fill(0., D)
    for s in 1:D
        for n in 1:N
            t = n2t[n]
            logpdfs[s] += logpdf(dists[s][t], y[s, n])
        end
    end
    return logpdfs

end




function my_rand_forced(dists, V)
    D = size(dists, 1)
    T = size(dists[1])
    N = length(n2t)
    y = Matrix{Float64}(undef, D, N)
    for s in 1:D
        for n in 1:N
            t = n2t[n]
            y[s, n] = quantile((dists[s][t]),cdf(Normal(),V[s,n]))
        end
    end
    return y
end

function my_rand_forced(dists, V,occurence)
    D = size(dists, 1)
    T = size(dists[1])
    N = length(n2t)
    y = Matrix{Float64}(undef, D, N)
    for s in 1:D
        for n in 1:N
            t = n2t[n]
            y[s, n] = quantile((dists[s][t]),cdf(Normal(),V[s,n]))
        end
    end
    return y.*occurence
end

function Trig2myEGPDPeriodicDistribution(param_sigma::AbstractArray{<:AbstractFloat,2}, param_xi::AbstractArray{<:AbstractFloat,2}, param_kappa::AbstractArray{<:AbstractFloat,2}, param_proba_lowrain::AbstractArray{<:AbstractFloat,2}, left_bound::AbstractFloat, middle_bound::AbstractFloat, T)
    D = size(param_sigma, 1)


    proba_lowrain = [1 / (1 + exp(polynomial_trigo(t, param_proba_lowrain[s, :], T))) for t = 1:T, s = 1:D]
    sigma = [exp(polynomial_trigo(t, param_sigma[s, :], T)) for t = 1:T, s = 1:D]
    xi = [1 / (1 + exp(polynomial_trigo(t, param_xi[s, :], T))) for t = 1:T, s = 1:D]
    kappa = [exp(polynomial_trigo(t, param_kappa[s, :], T)) for t = 1:T, s = 1:D]

    
    # return (my_A, p, range)


    uniform_part = Uniform(left_bound, middle_bound)
    tail_parts = [ExtendedGeneralizedPareto(TBeta(kappa[t, s]), GeneralizedPareto(0.0, sigma[t, s], xi[t, s])) for t in 1:T, s in 1:D]

    # Create the mixed distribution
    dists = [[MixedUniformTail(proba_lowrain[t, s], uniform_part, tail_parts[t, s], left_bound, middle_bound) for t in 1:T] for s in 1:D]

    return (dists)
end



function fit_EGPD_periodic(ytail_s, indicetail_s,param_kappas, param_sigmas, param_xis, ;maxiters=500)
    # give initial parameters
    param = [param_kappas; param_sigmas; param_xis]
    degp = Int((length(param_kappas) - 1) / 2)

    # ν₀, ϕ₀, ξ₀ = log(initialvalues[1]), log(initialvalues[2]), initialvalues[3]

    V = TBeta

    N = length(ytail_s)
    function loglike(param)
        n_in_tails = [findall(indicetail_s .== t) for t = 1:T]

        param_kappas = param[1:2*degp+1]
        param_sigmas = param[(2*degp+1)+1:2*(2*degp+1)]
        param_xis = param[2*(2*degp+1)+1:3*(2*degp+1)]


        sigma = [exp(polynomial_trigo(t, param_sigmas[:], T)) for t = 1:T]
        xi = [1 / (1 + exp(polynomial_trigo(t, param_xis[:], T))) for t = 1:T]
        kappa = [exp(polynomial_trigo(t, param_kappas[:], T)) for t = 1:T]

        pd = [ExtendedGeneralizedPareto(V(kappa[t]), GeneralizedPareto(0.0, sigma[t], xi[t])) for t in 1:T]
        loglik = 0
        for t in 1:T
            isempty(n_in_tails[t]) && continue
            loglik += sum(logpdf.(pd[t], ytail_s[n_in_tails[t]]))  - length(n_in_tails[t])/N*(kappa[t] - 1.)^2 / 0.1 # penalty to make sure kappa does not go to infinity

        end
       
        return loglik
        # --------------------------------------------------------

    end

    # 1st try
    fobj(θ) = -loglike(θ)
    res = optimize(fobj, param; iterations=maxiters)
    paramfit = Optim.minimizer(res)
    # 2nd pass 
    # res = optimize(fobj, paramfit)
    # paramfit = Optim.minimizer(res)


    # #2nd try
    # fobj(θ) = -loglike(θ)
    # optf = OptimizationFunction((θ,p) -> fobj(θ), Optimization.AutoForwardDiff())
    # prob = OptimizationProblem(optf, param)
    # sol = solve(prob, Optim.LBFGS(); maxiters=maxiters)
    # paramfit=sol.u

    # #3rd try

    # fobj(θ) = -loglike(θ)
    # res = Optim.optimize(fobj, param, LBFGS(), Optim.Options(;iterations=maxiters); autodiff = :forward)
    # paramfit = Optim.minimizer(res)


    param_kappas = paramfit[1:2*degp+1]
    param_sigmas = paramfit[(2*degp+1)+1:2*(2*degp+1)]
    param_xis = paramfit[2*(2*degp+1)+1:3*(2*degp+1)]


    sigma = [exp(polynomial_trigo(t, param_sigmas[:], T)) for t = 1:T]
    xi = [1 / (1 + exp(polynomial_trigo(t, param_xis[:], T))) for t = 1:T]
    kappa = [exp(polynomial_trigo(t, param_kappas[:], T)) for t = 1:T]


    fitted = [ExtendedGeneralizedPareto(V(kappa[t]), GeneralizedPareto(0.0, sigma[t], xi[t])) for t in 1:T]
    return fitted, param_kappas, param_sigmas, param_xis
end

function fit_dists(y::AbstractArray{<:AbstractFloat,2}, 
    param_kappa::AbstractArray{<:AbstractFloat,2},param_sigma::AbstractArray{<:AbstractFloat,2}, param_xi::AbstractArray{<:AbstractFloat,2},  param_proba_lowrain::AbstractArray{<:AbstractFloat,2}, left_bound::AbstractFloat, middle_bound::AbstractFloat, T, n2t::AbstractVector{<:Integer};maxiters=1000)
    # have to estimate for each location s dists[:,s]
    # dists[t,s] is the distribution for each time step
    deg = Int((size(param_sigma, 2) - 1) / 2)
    n_in_t = [findall(n2t .== t) for t = 1:T]

    # first to estimate : the parameter of low rain proba
    rain_occurences = (y .>= left_bound)
    low_rain_occurences = (y .>= left_bound) .&& (y .<= middle_bound)
    proba_low_rain_nosmooth = [sum(low_rain_occurences[s, n_in_t[t]])/sum(rain_occurences[s, n_in_t[t]]) for s in 1:D, t in 1:T]

    f = 2π / T
    X = ones(T, 1 + 2 * deg)
    for l in 1:deg
        X[:, 2l] = cos.(f * l .* collect(1:T))
        X[:, 2l+1] = sin.(f * l .* collect(1:T))
    end

    @threads for s in 1:D
        p = proba_low_rain_nosmooth[s, :]
        mask = .!iszero.(p) .& (p .< 1)  # exclude 0 and 1
        y_logit = log.((1 .- p[mask]) ./ p[mask])
        β = X[mask, :] \ y_logit
        param_proba_lowrain[s, :] .= β

        # # Logit transform the probabilities
        # y_logit = log.((1 .- proba_low_rain_nosmooth[s, :]) ./ proba_low_rain_nosmooth[s, :])

        # # Solve least squares
        # β = X \ y_logit
        # param_proba_lowrain[s, :] .= β
    end



    #Now on to estimating the tail EGPD part for each location
    @threads for s in 1:D
        ytail_s = y[s, :]
        indicetail_s = n2t[ytail_s.>middle_bound]
        ytail_s = ytail_s[ytail_s.>middle_bound] .- middle_bound# with probability defined above !




        param_sigmainit = param_sigma[s, :]
        param_xiinit = param_xi[s, :]
        param_kappainit = param_kappa[s, :]

        fitted, param_kappas, param_sigmas, param_xis = fit_EGPD_periodic(ytail_s, indicetail_s,  param_kappainit,param_sigmainit, param_xiinit,maxiters=maxiters)

        param_kappa[s, :] .= param_kappas
        param_sigma[s, :] .= param_sigmas
        param_xi[s, :] .= param_xis

    end


    dists = Trig2myEGPDPeriodicDistribution(param_sigma, param_xi, param_kappa, param_proba_lowrain, left_bound, middle_bound, T)
    return dists,param_kappa,param_sigma,param_xi,param_proba_lowrain

end


using Optimization
using OptimizationOptimJL
import OptimizationOptimJL: LBFGS
using ForwardDiff


# ##############################################################################################
# ####### add functions necessary for AD not defined in their original package ##########

# using SpecialFunctions
# import SpecialFunctions: _beta_inc
# import SpecialFunctions: beta_inc

# # === Public interface ===
# function  SpecialFunctions.beta_inc(a::Number, b::Number, x::Number)
#     return SpecialFunctions._beta_inc(a, b, x)
# end

# function  SpecialFunctions.beta_inc(a::Number, b::Number, x::Number, y::Number)
#     return SpecialFunctions._beta_inc(a, b, x, y)
# end

# # === Generic internal implementation ===
# function  SpecialFunctions._beta_inc(a::T, b::T, x::T, y::T=one(T)-x) where {T<:Float64}
#     # ERROR HANDLING REMOVED
#     # if a < zero(T) || b < zero(T)
#     #     throw(DomainError((a, b), "a or b is negative"))
#     # elseif a == zero(T) && b == zero(T)
#     #     throw(DomainError((a, b), "a and b are 0"))
#     # elseif x < zero(T) || x > one(T)
#     #     throw(DomainError(x, "x < 0 or x > 1"))
#     # elseif y < zero(T) || y > one(T)
#     #     throw(DomainError(y, "y < 0 or y > 1"))
#     # elseif abs(x + y - one(T)) > T(3)*eps(T)
#     #     throw(DomainError((x, y), "x + y != 1"))
#     # else
#     if isnan(x) || isnan(y) || isnan(a) || isnan(b)
#         return (T(NaN), T(NaN))
#     elseif x == zero(T)
#         return (zero(T), one(T))
#     elseif y == zero(T)
#         return (one(T), zero(T))
#     elseif a == zero(T)
#         return (one(T), zero(T))
#     elseif b == zero(T)
#         return (zero(T), one(T))
#     end

#     epps = max(eps(T), T(1e-15))
#     if max(a, b) < T(1e-3)*epps
#         return (b/(a+b), a/(a+b))
#     end

#     ind = false
#     a0, b0, x0, y0 = a, b, x, y

#     if min(a0, b0) > one(T)
#         lambda = a > b ? (a+b)*y - b : a - (a+b)*x
#         if lambda < zero(T)
#             ind = true
#             a0, b0, x0, y0 = b, a, y, x
#             lambda = abs(lambda)
#         end
#         if b0 < T(40) && b0*x0 <= T(0.7)
#             p = beta_inc_power_series(a0, b0, x0, epps)
#             q = one(T) - p
#         elseif b0 < T(40)
#             n = trunc(Int, b0)
#             b0 -= T(n)
#             if b0 == zero(T)
#                 n -= 1
#                 b0 = one(T)
#             end
#             p = beta_inc_diff(b0, a0, y0, x0, n, epps)
#             if x0 <= T(0.7)
#                 p += beta_inc_power_series(a0, b0, x0, epps)
#                 q = one(T) - p
#             else
#                 if a0 <= T(15)
#                     n = 20
#                     p += beta_inc_diff(a0, b0, x0, y0, n, epps)
#                     a0 += T(n)
#                 end
#                 p = beta_inc_asymptotic_asymmetric(a0, b0, x0, y0, p, T(15)*eps(T))
#                 q = one(T) - p
#             end
#         elseif a0 > b0
#             if b0 <= T(100) || lambda > T(0.03)*b0
#                 p = beta_inc_cont_fraction(a0, b0, x0, y0, lambda, T(15)*eps(T))
#                 q = one(T) - p
#             else
#                 p = beta_inc_asymptotic_symmetric(a0, b0, lambda, T(100)*eps(T))
#                 q = one(T) - p
#             end
#         elseif a0 <= T(100) || lambda > T(0.03)*a0
#             p = beta_inc_cont_fraction(a0, b0, x0, y0, lambda, T(15)*eps(T))
#             q = one(T) - p
#         else
#             p = beta_inc_asymptotic_symmetric(a0, b0, lambda, T(100)*eps(T))
#             q = one(T) - p
#         end
#         return ind ? (q, p) : (p, q)
#     end

#     # A0 <= 1 or B0 <= 1
#     if x > T(0.5)
#         ind = true
#         a0, b0, x0, y0 = b, a, y, x
#     end

#     if b0 < min(epps, epps*a0)
#         p = beta_inc_power_series2(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif a0 < min(epps, epps*b0) && b0*x0 <= one(T)
#         q = beta_inc_power_series1(a0, b0, x0, epps)
#         p = one(T) - q
#     elseif max(a0, b0) > one(T)
#         if b0 <= one(T)
#             p = beta_inc_power_series(a0, b0, x0, epps)
#             q = one(T) - p
#         elseif x0 >= T(0.3)
#             q = beta_inc_power_series(b0, a0, y0, epps)
#             p = one(T) - q
#         else
#             n = 20
#             q = beta_inc_diff(b0, a0, y0, x0, n, epps)
#             b0 += T(n)
#             q = beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, T(15)*eps(T))
#             p = one(T) - q
#         end
#     elseif a0 >= min(T(0.2), b0)
#         p = beta_inc_power_series(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif x0^a0 <= T(0.9)
#         p = beta_inc_power_series(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif x0 >= T(0.3)
#         q = beta_inc_power_series(b0, a0, y0, epps)
#         p = one(T) - q
#     else
#         n = 20
#         q = beta_inc_diff(b0, a0, y0, x0, n, epps)
#         b0 += T(n)
#         q = beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, T(15)*eps(T))
#         p = one(T) - q
#     end

#     return ind ? (q, p) : (p, q)
# end

# # === Generic internal implementation ===
# function  SpecialFunctions._beta_inc(a::T, b::T, x::T, y::T=one(T)-x) where {T<:Number}
#     # ERROR HANDLING REMOVED
#     # if a < zero(T) || b < zero(T)
#     #     throw(DomainError((a, b), "a or b is negative"))
#     # elseif a == zero(T) && b == zero(T)
#     #     throw(DomainError((a, b), "a and b are 0"))
#     # elseif x < zero(T) || x > one(T)
#     #     throw(DomainError(x, "x < 0 or x > 1"))
#     # elseif y < zero(T) || y > one(T)
#     #     throw(DomainError(y, "y < 0 or y > 1"))
#     # elseif abs(x + y - one(T)) > T(3)*eps(T)
#     #     throw(DomainError((x, y), "x + y != 1"))
#     # else
#     if isnan(x) || isnan(y) || isnan(a) || isnan(b)
#         return (T(NaN), T(NaN))
#     elseif x == zero(T)
#         return (zero(T), one(T))
#     elseif y == zero(T)
#         return (one(T), zero(T))
#     elseif a == zero(T)
#         return (one(T), zero(T))
#     elseif b == zero(T)
#         return (zero(T), one(T))
#     end

#     epps = max(eps(T), T(1e-15))
#     if max(a, b) < T(1e-3)*epps
#         return (b/(a+b), a/(a+b))
#     end

#     ind = false
#     a0, b0, x0, y0 = a, b, x, y

#     if min(a0, b0) > one(T)
#         lambda = a > b ? (a+b)*y - b : a - (a+b)*x
#         if lambda < zero(T)
#             ind = true
#             a0, b0, x0, y0 = b, a, y, x
#             lambda = abs(lambda)
#         end
#         if b0 < T(40) && b0*x0 <= T(0.7)
#             p = beta_inc_power_series(a0, b0, x0, epps)
#             q = one(T) - p
#         elseif b0 < T(40)
#             n = trunc(Int, b0)
#             b0 -= T(n)
#             if b0 == zero(T)
#                 n -= 1
#                 b0 = one(T)
#             end
#             p = beta_inc_diff(b0, a0, y0, x0, n, epps)
#             if x0 <= T(0.7)
#                 p += beta_inc_power_series(a0, b0, x0, epps)
#                 q = one(T) - p
#             else
#                 if a0 <= T(15)
#                     n = 20
#                     p += beta_inc_diff(a0, b0, x0, y0, n, epps)
#                     a0 += T(n)
#                 end
#                 p = beta_inc_asymptotic_asymmetric(a0, b0, x0, y0, p, T(15)*eps(T))
#                 q = one(T) - p
#             end
#         elseif a0 > b0
#             if b0 <= T(100) || lambda > T(0.03)*b0
#                 p = beta_inc_cont_fraction(a0, b0, x0, y0, lambda, T(15)*eps(T))
#                 q = one(T) - p
#             else
#                 p = beta_inc_asymptotic_symmetric(a0, b0, lambda, T(100)*eps(T))
#                 q = one(T) - p
#             end
#         elseif a0 <= T(100) || lambda > T(0.03)*a0
#             p = beta_inc_cont_fraction(a0, b0, x0, y0, lambda, T(15)*eps(T))
#             q = one(T) - p
#         else
#             p = beta_inc_asymptotic_symmetric(a0, b0, lambda, T(100)*eps(T))
#             q = one(T) - p
#         end
#         return ind ? (q, p) : (p, q)
#     end

#     # A0 <= 1 or B0 <= 1
#     if x > T(0.5)
#         ind = true
#         a0, b0, x0, y0 = b, a, y, x
#     end

#     if b0 < min(epps, epps*a0)
#         p = beta_inc_power_series2(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif a0 < min(epps, epps*b0) && b0*x0 <= one(T)
#         q = beta_inc_power_series1(a0, b0, x0, epps)
#         p = one(T) - q
#     elseif max(a0, b0) > one(T)
#         if b0 <= one(T)
#             p = beta_inc_power_series(a0, b0, x0, epps)
#             q = one(T) - p
#         elseif x0 >= T(0.3)
#             q = beta_inc_power_series(b0, a0, y0, epps)
#             p = one(T) - q
#         else
#             n = 20
#             q = beta_inc_diff(b0, a0, y0, x0, n, epps)
#             b0 += T(n)
#             q = beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, T(15)*eps(T))
#             p = one(T) - q
#         end
#     elseif a0 >= min(T(0.2), b0)
#         p = beta_inc_power_series(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif x0^a0 <= T(0.9)
#         p = beta_inc_power_series(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif x0 >= T(0.3)
#         q = beta_inc_power_series(b0, a0, y0, epps)
#         p = one(T) - q
#     else
#         n = 20
#         q = beta_inc_diff(b0, a0, y0, x0, n, epps)
#         b0 += T(n)
#         q = beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, T(15)*eps(T))
#         p = one(T) - q
#     end

#     return ind ? (q, p) : (p, q)
# end

# # === Public interface ===
# function  SpecialFunctions.beta_inc(a::Real, b::Real, x::Real)
#     return SpecialFunctions._beta_inc(a, b, x)
# end

# function  SpecialFunctions.beta_inc(a::Real, b::Real, x::Real, y::Real)
#     return SpecialFunctions._beta_inc(a, b, x, y)
# end

# # === Generic internal implementation ===
# function  SpecialFunctions._beta_inc(a::T, b::T, x::T, y::T=one(T)-x) where {T<:Real}
#     # ERROR HANDLING REMOVED
#     # if a < zero(T) || b < zero(T)
#     #     throw(DomainError((a, b), "a or b is negative"))
#     # elseif a == zero(T) && b == zero(T)
#     #     throw(DomainError((a, b), "a and b are 0"))
#     # elseif x < zero(T) || x > one(T)
#     #     throw(DomainError(x, "x < 0 or x > 1"))
#     # elseif y < zero(T) || y > one(T)
#     #     throw(DomainError(y, "y < 0 or y > 1"))
#     # elseif abs(x + y - one(T)) > T(3)*eps(T)
#     #     throw(DomainError((x, y), "x + y != 1"))
#     # else
#     if isnan(x) || isnan(y) || isnan(a) || isnan(b)
#         return (T(NaN), T(NaN))
#     elseif x == zero(T)
#         return (zero(T), one(T))
#     elseif y == zero(T)
#         return (one(T), zero(T))
#     elseif a == zero(T)
#         return (one(T), zero(T))
#     elseif b == zero(T)
#         return (zero(T), one(T))
#     end

#     epps = max(eps(T), T(1e-15))
#     if max(a, b) < T(1e-3)*epps
#         return (b/(a+b), a/(a+b))
#     end

#     ind = false
#     a0, b0, x0, y0 = a, b, x, y

#     if min(a0, b0) > one(T)
#         lambda = a > b ? (a+b)*y - b : a - (a+b)*x
#         if lambda < zero(T)
#             ind = true
#             a0, b0, x0, y0 = b, a, y, x
#             lambda = abs(lambda)
#         end
#         if b0 < T(40) && b0*x0 <= T(0.7)
#             p = beta_inc_power_series(a0, b0, x0, epps)
#             q = one(T) - p
#         elseif b0 < T(40)
#             n = trunc(Int, b0)
#             b0 -= T(n)
#             if b0 == zero(T)
#                 n -= 1
#                 b0 = one(T)
#             end
#             p = beta_inc_diff(b0, a0, y0, x0, n, epps)
#             if x0 <= T(0.7)
#                 p += beta_inc_power_series(a0, b0, x0, epps)
#                 q = one(T) - p
#             else
#                 if a0 <= T(15)
#                     n = 20
#                     p += beta_inc_diff(a0, b0, x0, y0, n, epps)
#                     a0 += T(n)
#                 end
#                 p = beta_inc_asymptotic_asymmetric(a0, b0, x0, y0, p, T(15)*eps(T))
#                 q = one(T) - p
#             end
#         elseif a0 > b0
#             if b0 <= T(100) || lambda > T(0.03)*b0
#                 p = beta_inc_cont_fraction(a0, b0, x0, y0, lambda, T(15)*eps(T))
#                 q = one(T) - p
#             else
#                 p = beta_inc_asymptotic_symmetric(a0, b0, lambda, T(100)*eps(T))
#                 q = one(T) - p
#             end
#         elseif a0 <= T(100) || lambda > T(0.03)*a0
#             p = beta_inc_cont_fraction(a0, b0, x0, y0, lambda, T(15)*eps(T))
#             q = one(T) - p
#         else
#             p = beta_inc_asymptotic_symmetric(a0, b0, lambda, T(100)*eps(T))
#             q = one(T) - p
#         end
#         return ind ? (q, p) : (p, q)
#     end

#     # A0 <= 1 or B0 <= 1
#     if x > T(0.5)
#         ind = true
#         a0, b0, x0, y0 = b, a, y, x
#     end

#     if b0 < min(epps, epps*a0)
#         p = beta_inc_power_series2(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif a0 < min(epps, epps*b0) && b0*x0 <= one(T)
#         q = beta_inc_power_series1(a0, b0, x0, epps)
#         p = one(T) - q
#     elseif max(a0, b0) > one(T)
#         if b0 <= one(T)
#             p = beta_inc_power_series(a0, b0, x0, epps)
#             q = one(T) - p
#         elseif x0 >= T(0.3)
#             q = beta_inc_power_series(b0, a0, y0, epps)
#             p = one(T) - q
#         else
#             n = 20
#             q = beta_inc_diff(b0, a0, y0, x0, n, epps)
#             b0 += T(n)
#             q = beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, T(15)*eps(T))
#             p = one(T) - q
#         end
#     elseif a0 >= min(T(0.2), b0)
#         p = beta_inc_power_series(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif x0^a0 <= T(0.9)
#         p = beta_inc_power_series(a0, b0, x0, epps)
#         q = one(T) - p
#     elseif x0 >= T(0.3)
#         q = beta_inc_power_series(b0, a0, y0, epps)
#         p = one(T) - q
#     else
#         n = 20
#         q = beta_inc_diff(b0, a0, y0, x0, n, epps)
#         b0 += T(n)
#         q = beta_inc_asymptotic_asymmetric(b0, a0, y0, x0, q, T(15)*eps(T))
#         p = one(T) - q
#     end

#     return ind ? (q, p) : (p, q)
# end

# using HypergeometricFunctions: _₂F₁

# using StatsFuns
# import StatsFuns: betalogcdf
# function StatsFuns.betalogcdf(α::T, β::T, x::T) where {T <: Real}
#     # Handle degenerate cases
#     if iszero(α) && β > zero(T)
#         return log(clamp(x, zero(T), one(T)))  # simpler degenerate handling
#     elseif iszero(β) && α > zero(T)
#         return log(clamp(x, zero(T), one(T)))
#     end

#     _x = clamp(x, zero(T), one(T))
#     p, q = beta_inc(α, β, _x)

#     tiny = eps(T)  # replaces floatmin(p)
#     if p < tiny
#         return -log(α) + xlogy(α, _x) + xlog1py(β, -_x) +
#                log(_₂F₁(promote(α + β, one(T), α + one(T), _x)...; method = :positive)) -
#                logbeta(α, β)
#     elseif p <= T(0.7)
#         return log(p)
#     else
#         return log1p(-q)
#     end
# end


# using Distributions
# import ExtendedExtremes: TBeta
# function ExtendedExtremes.TBeta(α::T; check_args=true) where {T <: Real}
#     return TBeta{T}(α)
# end

# import Distributions: GeneralizedPareto

# function Distributions.GeneralizedPareto(μ::T, σ::T, ξ::T; check_args::Bool=true) where {T <: Real}
#     return GeneralizedPareto{T}(μ, σ, ξ)
# end

# import Distributions: Beta
# function Distributions.Beta(α::T, β::T; check_args::Bool=true) where {T<:Real}
#     return Beta{T}(α, β)
# end

# import SpecialFunctions: beta_inc_power_series
# function SpecialFunctions.beta_inc_power_series(a::T, b::T, x::T, epps::T) where {T<:Real}
#     ans = zero(T)
#     if x == zero(T)
#         return zero(T)
#     end
#     a0 = min(a,b)
#     b0 = max(a,b)
#     if a0 >= one(T)
#         z = a*log(x) - logbeta(a,b)
#         ans = exp(z)/a
#     else

#         if b0 >= 8.0*one(T)
#             u = loggamma1p(a0) + loggammadiv(a0,b0)
#             z = a*log(x) - u
#             ans = (a0/a)*exp(z)
#             if ans == zero(T) || a <= 0.1*epps*one(T)
#                 return ans
#             end
#         elseif b0 > one(T)
#             u = loggamma1p(a0)
#             m = b0 - one(T)
#             if m >= one(T)
#                 c = one(T)
#                 for i = 1:m
#                     b0 -= one(T)
#                     c *= (b0/(a0+b0))
#                 end
#                 u += log(c)
#             end
#             z = a*log(x) - u
#             b0 -= one(T)
#             apb = a0 + b0
#             if apb > one(T)
#                 u = a0 + b0 - one(T)
#                 t = (one(T)+ rgamma1pm1(u))/apb
#             else
#                 t =one(T)+ rgamma1pm1(apb)
#             end
#             ans = exp(z)*(a0/a)*(one(T) + rgamma1pm1(b0))/t
#             if ans == zero(T) || a <= 0.1*epps*one(T)
#                 return ans
#             end
#         else
#         #PROCEDURE FOR A0 < 1 && B0 < 1
#             ans = x^a
#             if ans == zero(T)
#                 return ans
#             end
#             apb = a + b
#             if apb > one(T)
#                 u = a + b - one(T)
#                 z = (one(T) + rgamma1pm1(u))/apb
#             else
#                 z = one(T) + rgamma1pm1(apb)
#             end
#             c = (one(T) + rgamma1pm1(a))*(one(T) + rgamma1pm1(b))/z
#             ans *= c*(b/apb)
#             #label l70 start
#             if ans == zero(T) || a <= 0.1*epps*one(T)
#                 return ans
#             end
#         end
#     end
#     if ans == zero(T)|| a <= 0.1*epps*one(T)
#         return ans
#     end
#     # COMPUTE THE SERIES

#     sm = zero(T)
#     n = zero(T)
#     c = one(T)
#     tol = epps/a
#     n += one(T)
#     c *= x*(one(T) - b/n)
#     w = c/(a + n)
#     sm += w
#     while abs(w) > tol
#         n += one(T)
#         c *= x*(one(T) - b/n)
#         w = c/(a+n)
#         sm += w
#     end
#     return ans*(one(T) + a*sm)
# end

# using Base.Math: @horner

# import SpecialFunctions: rgamma1pm1
# function SpecialFunctions.rgamma1pm1(a::T)where {T <: Number}
#     @assert -0.5*one(T) <= a <= 1.5*one(T)
#     t = a
#     rangereduce = a > 0.5*one(T)
#     t = rangereduce ? a-one(T) : a #-0.5<= t <= 0.5
#     if t == zero(T)
#         return zero(T)
#     elseif t < zero(T)
#         top = @horner(t, -.422784335098468E+00, -.771330383816272E+00, -.244757765222226E+00, .118378989872749E+00, .930357293360349E-03, -.118290993445146E-01, .223047661158249E-02, .266505979058923E-03, -.132674909766242E-03)
#         bot = @horner(t, 1.0, .273076135303957E+00, .559398236957378E-01)
#         w = top/bot
#         return rangereduce ? t*w/a : a*(w + one(T))
#     else
#         top = @horner(t, .577215664901533E+00, -.409078193005776E+00, -.230975380857675E+00, .597275330452234E-01, .766968181649490E-02, -.514889771323592E-02, .589597428611429E-03)
#         bot = @horner(t, 1.0, .427569613095214E+00, .158451672430138E+00, .261132021441447E-01, .423244297896961E-02)
#         w = top/bot
#         return rangereduce ? (t/a)*(w - one(T)) : a*w
#     end
# end

# import SpecialFunctions: beta_inc_diff
# function SpecialFunctions.beta_inc_diff(a::T, b::T, x::T, y::T, n::Integer, epps::T)where {T <: Number}
#     apb = a + b
#     ap1 = a + one(T)
#     mu = zero(T)
#     d = one(T)
#     if n != 1 && a >= one(T) && apb >= 1.1*ap1
#         mu = abs(exparg_n)
#         k = exparg_p
#         if k < mu
#             mu = k
#         end
#         t = mu
#         d = exp(-t)
#     end

#     ans = beta_integrand(a, b, x, y, mu)/a
#     if n == 1 || ans == zero(T)
#         return ans
#     end
#     nm1 = n -1
#     w = d

#     k = 0
#     if b <= one(T)
#         kp1 = k + 1
#         for i = kp1:nm1
#             l = i - 1
#             d *= ((apb + l)/(ap1 + l))*x
#             w += d
#             if d <= epps*w
#                 break
#             end
#         end
#         return ans*w
#     elseif y > 1.0e-4*one(T)
#         r = trunc(Int,(b - one(T))*x/y - a)
#         if r < 1.0
#             kp1 = k + 1
#             for i = kp1:nm1
#                 l = i - 1
#                 d *= ((apb + l)/(ap1 + l))*x
#                 w += d
#                 if d <= epps*w
#                     break
#                 end
#             end
#             return ans*w
#         end
#         k = t = nm1
#         if r < t
#             k = r
#         end
#         # ADD INC TERMS OF SERIES
#         for i = 1:k
#             l = i -1
#             d *= ((apb + l)/(ap1 + l))*x
#             w += d
#         end
#         if k == nm1
#             return ans*w
#         end
#     else
#         k = nm1
#         for i = 1:k
#             l = i -1
#             d *= ((apb + l)/(ap1 + l))*x
#             w += d
#         end
#         if k == nm1
#             return ans*w
#         end
#     end

#     kp1 = k + 1
#     for i in kp1:nm1
#         l = i - 1
#         d *= ((apb + l)/(ap1 + l))*x
#         w += d
#         if d <= epps*w
#             break
#         end
#    end
#    return ans*w
# end

# import SpecialFunctions: beta_integrand
# function SpecialFunctions.beta_integrand(a::T, b::T, x::T, y::T, mu::T=zero(T))where {T <: Number}
#     a0, b0 = minmax(a,b)
#     if a0 >= 8.0*one(T)
#         if a > b
#             h = b/a
#             x0 = one(T)/(one(T) + h)
#             y0 = h/(one(T) + h)
#             lambda = (a+b)*y - b
#         else
#             h = a/b
#             x0 = h/(one(T) + h)
#             y0 = one(T)/(one(T) + h)
#             lambda = a - (a+b)*x
#         end
#         e = -lambda/a
#         u = abs(e) > 0.6*one(T) ? e - log(x/x0) : - LogExpFunctions.log1pmx(e)
#         e = lambda/b
#         v = abs(e) > 0.6*one(T) ? e - log(y/y0) : - LogExpFunctions.log1pmx(e)
#         z = esum(mu, -(a*u + b*v))
#         return sqrt(inv2π*b*x0)*z*exp(-stirling_corr(a,b))
#     elseif x > 0.375*one(T)
#         if y > 0.375*one(T)
#             lnx = log(x)
#             lny = log(y)
#         else
#             lnx = log1p(-y)
#             lny = log(y)
#         end
#     else
#         lnx = log(x)
#         lny = log1p(-x)
#     end
#     z = a*lnx + b*lny
#     if a0 < one(T)
#         b0 = max(a,b)
#         if b0 >= 8.0*one(T)
#             u = loggamma1p(a0) + loggammadiv(a0,b0)
#             return a0*(esum(mu, z-u))
#         elseif b0 > one(T)
#             u = loggamma1p(a0)
#             n = trunc(Int,b0 - one(T))
#             if n >= 1
#                 c = one(T)
#                 for i = 1:n
#                     b0 -= one(T)
#                     c *= (b0/(a0+b0))
#                 end
#                 u += log(c)
#             end
#             z -= u
#             b0 -= one(T)
#             apb = a0 + b0
#             if apb > one(T)
#                 u = a0 + b0 - one(T)
#                 t = (one(T) + rgamma1pm1(u))/apb
#             else
#                 t = one(T) + rgamma1pm1(apb)
#             end
#             return a0*(esum(mu,z))*(one(T) + rgamma1pm1(b0))/t
#         else
#             ans = esum(mu, z)
#             if ans == zero(T)
#                 return zero(T)
#             end
#             apb = a + b
#             if apb > one(T)
#                 z = (one(T) + rgamma1pm1(apb - one(T)))/apb
#             else
#                 z = one(T)+ rgamma1pm1(apb)
#             end
#             c = (one(T) + rgamma1pm1(a))*(one(T) + rgamma1pm1(b))/z
#             return ans*(a0*c)/(one(T) + a0/b0)
#         end
#     else
#         z -= logbeta(a, b)
#         ans = esum(mu, z)
#         return ans
#     end
# end

# import SpecialFunctions: auxgam
# function SpecialFunctions.auxgam(x::T) where {T <: Number}
#     @assert -one(T)<= x <= one(T)
#     if x < zero(T)
#         return -(one(T) + (one(T) + x)*(one(T) + x)*auxgam(one(T) + x))/(one(T) - x)
#     else
#         t = 2*x - one(T)
#         return chepolsum(t, auxgam_coef)
#     end
# end

# import SpecialFunctions: loggamma1p
# function loggamma1p(x::T) where {T <: Number}
#     @assert -one(T) < x <= one(T)
#     return -log1p(x*(x - one(T))*auxgam(x))
# end

# import SpecialFunctions: chepolsum
# function chepolsum(x::T, a::Array{Float64,1}) where {T <: Number}
#     n = length(a)
#     if n == 1
#         return a[1]/2.0
#     elseif n == 2
#         return a[1]/2.0 + a[2]*x
#     else
#         tx = 2*x
#         r = a[n]
#         h = a[n - 1] + r*tx
#         for k = n-2:-1:2
#             s = r
#             r = h
#             h = a[k] + r*tx - s
#         end
#         return a[1]/2.0 - r + h*x
#     end
# end
# const auxgam_coef = [-1.013609258009865776949, 0.784903531024782283535e-1, 0.67588668743258315530e-2, -0.12790434869623468120e-2, 0.462939838642739585e-4, 0.43381681744740352e-5, -0.5326872422618006e-6, 0.172233457410539e-7, 0.8300542107118e-9, -0.10553994239968e-9, 0.39415842851e-11, 0.362068537e-13, -0.107440229e-13, 0.5000413e-15, -0.62452e-17, -0.5185e-18, 0.347e-19, -0.9e-21]

# import SpecialFunctions: esum
# function SpecialFunctions.esum(mu::T, x::T) where {T <: Number}
#     if x > zero(T)
#         if mu > zero(T) || mu + x < zero(T)
#             return exp(mu)*exp(x)
#         else
#             return exp(mu + x)
#         end
#     elseif mu <zero(T) || mu + x > zero(T)
#         return exp(mu)*exp(x)
#     else
#         return exp(mu + x)
#     end
# end

# import SpecialFunctions: beta_inc_asymptotic_symmetric
# function beta_inc_asymptotic_symmetric(a::T, b::T, lambda::T, epps::T) where{T<: Number}
#     @assert a >= 15.0*one(T)
#     @assert b >= 15.0*one(T)
#     a0 =zeros(T,22)
#     b0 = zeros(T,22)
#     c = zeros(T,22)
#     d = zeros(T,22)
#     e0 = 2/sqrtπ * one(T)
#     e1 = 2^(-1.5)* one(T)
#     sm = zero(T)
#     ans =  zero(T)
#     if a > b
#         h = b/a
#         r0 = 1.0/(1.0 + h)
#         r1 = (b-a)/a
#         w0 = 1.0/sqrt(b*(1.0+h))
#     else
#         h = a/b
#         r0 = 1.0/(1.0 + h)
#         r1 = (b-a)/b
#         w0 = 1.0/sqrt(a*(1.0+h))
#     end
#     f = -a*LogExpFunctions.log1pmx(-(lambda/a)) - b*LogExpFunctions.log1pmx(lambda/b)
#     t = exp(-f)
#     if t == zero(T)
#         return ans
#     end
#     z0 = sqrt(f)
#     z = 0.5*(z0/e1)
#     z² = 2.0*f

#     a0[1] = (2.0/3.0)*r1
#     c[1] = -0.5*a0[1]
#     d[1] = - c[1]
#     j0 = (0.5/e0)*erfcx(z0)
#     j1 = e1
#     sm = j0 + d[1]*w0*j1

#     s = 1.0
#     h² = h*h
#     hn = 1.0
#     w = w0
#     znm1 = z
#     zn = z²

#     for n = 2: 2: 20
#         hn *= h²
#         a0[n] = 2.0*r0*(1.0 + h*hn)/(n + 2.0)
#         s += hn
#         a0[n+1] = 2.0*r1*s/(n+3.0)

#         for i = n: n+1
#             r = -0.5*(i + 1.0)
#             b0[1] = r*a0[1]
#             for m = 2:i
#                 bsum = 0.0
#                 for j =1: m-1
#                     bsum += (j*r - (m-j))*a0[j]*b0[m-j]
#                 end
#                 b0[m] = r*a0[m] + bsum/m
#             end
#             c[i] = b0[i]/(i+1.0)
#             dsum = 0.0
#             for j = 1: i-1
#                 imj = i - j
#                 dsum += d[imj]*c[j]
#             end
#             d[i] = -(dsum + c[i])
#         end

#         j0 = e1*znm1 + (n - 1)*j0
#         j1 = e1*zn + n*j1
#         znm1 *= z²
#         zn *= z²
#         w *= w0
#         t0 = d[n]*w*j0
#         w *= w0
#         t1 = d[n+1]*w*j1
#         sm += (t0 + t1)
#         if (abs(t0) + abs(t1)) <= epps*sm
#             break
#         end
#     end

#     u = exp(-stirling_corr(a,b))
#     return e0*t*u*sm
# end

# using LogExpFunctions
# import SpecialFunctions: stirling_corr
# function SpecialFunctions.stirling_corr(a0::T, b0::T) where {T <: Number}
#     a = min(a0, b0)
#     b = max(a0, b0)
#     @assert a >= 8.0*one(T)

#     h = a/b
#     c = h/(1.0 + h)
#     x = 1.0/(1.0 + h)
#     x² = x*x
#     #SET SN = (1-X^N)/(1-X)
#     s₃ = one(T) + (x + x²)
#     s₅ = one(T) + (x + x²*s₃)
#     s₇ = one(T) + (x + x²*s₅)
#     s₉ = one(T) + (x + x²*s₇)
#     s₁₁ = one(T) + (x + x²*s₉)
#     t = inv(b)^2
#     w = @horner(t, .833333333333333E-01, -.277777777760991E-02*s₃, .793650666825390E-03*s₅, -.595202931351870E-03*s₇, .837308034031215E-03*s₉, -.165322962780713E-02*s₁₁)
#     w *= c/b
#     # COMPUTE stirling(a) + w
#     t = inv(a)^2
#     return @horner(t, .833333333333333E-01, -.277777777760991E-02, .793650666825390E-03, -.595202931351870E-03, .837308034031215E-03, -.165322962780713E-02)/a + w
# end

# import SpecialFunctions: beta_inc_cont_fraction
# function beta_inc_cont_fraction(a::T, b::T, x::T, y::T, lambda::T, epps::T) where{T<:Number}
#     @assert a > one(T)
#     @assert b > one(T)
#     ans = beta_integrand(a,b,x,y)
#     if ans == zero(T)
#         return zero(T)
#     end
#     c = one(T) + lambda
#     c0 = b/a
#     c1 = one(T) + one(T)/a
#     yp1 = y + one(T)

#     n = zero(T)
#     p = one(T)
#     s = a + one(T)
#     an = zero(T)
#     bn = one(T)
#     anp1 = one(T)
#     bnp1 = c/c1
#     r = c1/c
#     #CONT FRACTION

#     while true
#         n += one(T)
#         t = n/a
#         w = n*(b - n)*x
#         e = a/s
#         alpha = (p*(p+c0)*e*e)*(w*x)
#         e = (one(T) + t)/(c1 + 2*t)
#         beta = n + w/s +e*(c + n*yp1)
#         p = one(T) + t
#         s += 2.0*one(T)

#         #update an, bn, anp1, bnp1
#         t = alpha*an  + beta*anp1
#         an = anp1
#         anp1 = t
#         t = alpha*bn + beta*bnp1
#         bn = bnp1
#         bnp1 = t

#         r0 = r
#         r = anp1/bnp1
#         if abs(r - r0) <= epps*r
#             break
#         end
#         #rescale
#         an /= bnp1
#         bn /= bnp1
#         anp1 = r
#         bnp1 = one(T)
#     end
#     return ans*r
# end

# import SpecialFunctions: loggammadiv

# function SpecialFunctions.loggammadiv(a::T, b::T) where{T<:Number}
#     @assert b >= 8*one(T)
#     if a > b
#         h = b/a
#         c = 1.0/(1.0 + h)
#         x = h/(1.0 + h)
#         d = a + (b - 0.5)
#     else
#         h = a/b
#         c = h/(1.0 + h)
#         x = 1.0/(1.0 + h)
#         d = b + a - 0.5
#     end
#     x² = x*x
#     s₃ = 1.0 + (x + x²)
#     s₅ = 1.0 + (x + x²*s₃)
#     s₇ = 1.0 + (x + x²*s₅)
#     s₉ = 1.0 + (x + x²*s₇)
#     s₁₁ = 1.0 + (x + x²*s₉)

#     # SET W = stirling(b) - stirling(a+b)
#     t = inv(b)^2
#     w = @horner(t, .833333333333333E-01, -.277777777760991E-02*s₃, .793650666825390E-03*s₅, -.595202931351870E-03*s₇, .837308034031215E-03*s₉, -.165322962780713E-02*s₁₁)
#     w *= c/b

#     #COMBINING
#     u = d*log1p(a/b)
#     v = a*(log(b) - 1.0)
#     return u <= v ? w - u - v : w - v - u
# end