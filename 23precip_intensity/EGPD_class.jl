struct MixedUniformTailModel
    dists::Vector{MixedUniformTail}   # one distribution per class
    K::Int                           # number of classes
end

"""
    fit_mix_model(::Type{MixedUniformTail}, data, z; left=0.1, middle=0.5, K=maximum(z))

Fit a `MixedUniformTailModel` with `K` classes.  
- `data`: observed rainfall amounts (vector).  
- `z`: integer class labels (vector, same length as `data`).  
- `left`, `middle`: thresholds for uniform/tail split.  
Returns a `MixedUniformTailModel`.
"""
function fit_mix_model(::Type{MixedUniformTail}, data::AbstractVector, z::AbstractVector; left=0.1, middle=0.5, K=length(unique(z)))
    dists = Vector{MixedUniformTail}(undef, K)
    for k in 1:K
        # Select data belonging to class k
        class_data = data[z .== k]
        dists[k] = fit_mix(MixedUniformTail, class_data; left=left, middle=middle)
    end
    return MixedUniformTailModel(dists, K)
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



# Example synthetic data
Random.seed!(1234)
N = 18600
z = rand(1:4, N)   # 3 classes
# --- Define three example MixedUniformTail distributions ---
a, b = 0.1, 0.5

d1 = MixedUniformTail(0.3, Uniform(a,b), ExtendedGeneralizedPareto(TBeta(0.4), GeneralizedPareto(0.0,2.0,0.1)), a, b)
d2 = MixedUniformTail(0.4, Uniform(a,b), ExtendedGeneralizedPareto(TBeta(0.5), GeneralizedPareto(0.0,3.0,0.2)), a, b)
d3 = MixedUniformTail(0.2, Uniform(a,b), ExtendedGeneralizedPareto(TBeta(0.3), GeneralizedPareto(0.0,1.5,0.05)), a, b)
d4 = MixedUniformTail(0.1, Uniform(a,b), ExtendedGeneralizedPareto(TBeta(0.3), GeneralizedPareto(0.0,1.5,0.05)), a, b)

# --- Build a model ---
model = MixedUniformTailModel([d1, d2, d3,d4], 4)
data = [ rand(model, z[t]) for t in 1:N ]

# Fit model
modelfit = fit_mix_model(MixedUniformTail, data, z; left=0.1, middle=0.5)


function Trig2myEGPDPeriodicDistribution(param_sigma::AbstractArray{<:AbstractFloat,3}, param_xi::AbstractArray{<:AbstractFloat,3}, param_kappa::AbstractArray{<:AbstractFloat,3}, param_proba_lowrain::AbstractArray{<:AbstractFloat,3}, left_bound::AbstractFloat, middle_bound::AbstractFloat, T)
    K,D,size_poly = size(param_sigma)


    proba_lowrain = [1 / (1 + exp(polynomial_trigo(t, param_proba_lowrain[k,s, : ], T))) for t = 1:T, s = 1:D,k =1:K]
    sigma = [exp(polynomial_trigo(t, param_sigma[k,s, :], T)) for t = 1:T, s = 1:D, k=1:K]
    xi = [1/(1+exp(polynomial_trigo(t, param_xi[k,s, :], T))) for t = 1:T, s = 1:D, k=1:K]
    kappa = [exp(polynomial_trigo(t, param_kappa[k,s, :], T)) for t = 1:T, s = 1:D, k=1:K]

    # return (my_A, p, range)


    uniform_part = Uniform(left_bound, middle_bound)
    tail_parts = [ExtendedGeneralizedPareto(TBeta(kappa[t, s,k]), GeneralizedPareto(0.0, sigma[t, s,k], xi[t, s,k])) for t in 1:T, s in 1:D,  k=1:K]

    # Create the mixed distribution
    dists = [[MixedUniformTail(proba_lowrain[t, s,k], uniform_part, tail_parts[t, s,k], left_bound, middle_bound) for k in 1:K,  t in 1:T ] for s in 1:D]

    return (dists)
end


function my_rand(dists::AbstractVector{<:AbstractMatrix{<:MixedUniformTail}},z , n2t::AbstractVector)
    D = size(dists, 1)
    K,T = size(dists[1])
    N = length(n2t)
    y = Matrix{Float64}(undef, D, N)
    for s in 1:D
        for n in 1:N
            t = n2t[n]
            y[s, n] = rand(dists[s][z[n],t])
        end
    end
    return y
end

function my_rand(dists::AbstractVector{<:AbstractMatrix{<:MixedUniformTail}},z
    , n2t::AbstractVector,occurence)
    D = size(dists, 1)
   K, T = size(dists[1])
    N = length(n2t)
    y = Matrix{Float64}(undef, D, N)
    for s in 1:D
        for n in 1:N
            t = n2t[n]
            y[s, n] = rand(dists[s][z[n],t])
        end
    end
    return y.*occurence
end

function fit_dists(y::AbstractArray{<:AbstractFloat,2}, param_kappa::AbstractArray{<:AbstractFloat,3}, param_sigma::AbstractArray{<:AbstractFloat,3}, param_xi::AbstractArray{<:AbstractFloat,3}, param_proba_lowrain::AbstractArray{<:AbstractFloat,3}, left_bound::AbstractFloat, middle_bound::AbstractFloat, T, z,n2t::AbstractVector{<:Integer};maxiters=1000)
    # have to estimate for each location s dists[:,s]
    # dists[t,s] is the distribution for each time step
    D=size(param_sigma,2)
    K= size(param_sigma,1)
    deg = Int((size(param_sigma, 3) - 1) / 2)
    n_in_t_inK = [findall((n2t .== t).&& (z .== k)) for k = 1:K, t = 1:T]

    # first to estimate : the parameter of low rain proba
    rain_occurences = (y .>= left_bound)
    low_rain_occurences = (y .>= left_bound) .&& (y .<= middle_bound)
    proba_low_rain_nosmooth = [sum(low_rain_occurences[s, n_in_t_inK[k,t]])/sum(rain_occurences[s, n_in_t_inK[k,t]]) for k= 1:K, s in 1:D, t in 1:T]

    f = 2π / T
    X = ones(T, 1 + 2 * deg)
    for l in 1:deg
        X[:, 2l] = cos.(f * l .* collect(1:T))
        X[:, 2l+1] = sin.(f * l .* collect(1:T))
    end

    for k in 1:K
    for s in 1:D
        p = proba_low_rain_nosmooth[k,s, :]
        mask = .!iszero.(p) .& (p .< 1)  # exclude 0 and 1
        y_logit = log.((1 .- p[mask]) ./ p[mask])
        β = X[mask, :] \ y_logit
        param_proba_lowrain[k,s, :] .= β

        # # Logit transform the probabilities
        # y_logit = log.((1 .- proba_low_rain_nosmooth[s, :]) ./ proba_low_rain_nosmooth[s, :])

        # # Solve least squares
        # β = X \ y_logit
        # param_proba_lowrain[s, :] .= β
    end
end


    #Now on to estimating the tail EGPD part for each location
    @threads for s in 1:D
        for k in 1:K
        ytail_s = y[s, :]
        indicetail_s = n2t[(ytail_s.>middle_bound) .&& (z .== k)]
        ytail_s = ytail_s[(ytail_s.>middle_bound) .&& (z .== k)] .- middle_bound# with probability defined above !




        param_sigmainit = param_sigma[k,s, :]
        param_xiinit = param_xi[k,s, :]
        param_kappainit = param_kappa[k,s, :]

        fitted, param_kappas, param_sigmas, param_xis = fit_EGPD_periodic(ytail_s, indicetail_s, param_kappainit, param_sigmainit, param_xiinit,maxiters=maxiters)

        param_kappa[k,s, :] .= param_kappas
        param_sigma[k,s, :] .= param_sigmas
        param_xi[k,s, :] .= param_xis
        end
    end


    dists = Trig2myEGPDPeriodicDistribution(param_sigma, param_xi, param_kappa, param_proba_lowrain, left_bound, middle_bound, T)
    return dists,param_kappa,param_sigma,param_xi,param_proba_lowrain

end


function my_rand_forced(dists::AbstractVector{<:AbstractMatrix{<:MixedUniformTail}},V,z , n2t::AbstractVector)
    D = size(dists, 1)
    T = size(dists[1])
    N = length(n2t)
    y = Matrix{Float64}(undef, D, N)
    for s in 1:D
        for n in 1:N
            t = n2t[n]
            y[s, n] = quantile((dists[s][z[n],t]),cdf(Normal(),V[s,n]))
        end
    end
    return y
end

