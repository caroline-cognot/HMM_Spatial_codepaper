# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
include("../utils/fast_bivariate_cdf.jl")

    # Random and Distributions
    using Distributions
    using Random: AbstractRNG, GLOBAL_RNG, rand!

    # ## Special function
    using LogExpFunctions: logsumexp!, logsumexp

    # ## HMM
    using PeriodicHiddenMarkovModels

    using PeriodicHiddenMarkovModels: viterbi, istransmat, update_a!, vec_maximum
    import PeriodicHiddenMarkovModels: forwardlog!, backwardlog!, viterbi, viterbi!, viterbilog!, posteriors!
    import PeriodicHiddenMarkovModels: fit_mle!, fit_mle

    # # Overloaded functions
    import Distributions: fit_mle
    import Base: rand
    import Base: ==, copy, size
    using Base.Threads
    
interleave2(args...) = collect(Iterators.flatten(zip(args...)))

remaining(N::Int) = N > 0 ? range(1, length=N) : Int64[]

argmaxrow(A::AbstractMatrix{<:Real}) = [argmax(A[i, :]) for i = axes(A, 1)]


n_per_category(s, h, t, y, n_in_t, n_occurence_history) = (n_in_t[t] ∩ n_occurence_history[s, h, y])

bin2digit(x) = sum(x[length(x)-i+1] * 2^(i - 1) for i = 1:length(x)) + 1
bin2digit(x::Tuple) = bin2digit([x...])

function dayx(lag_obs::AbstractArray)
    order = length(lag_obs)
    t = tuple.([lag_obs[m] for m = 1:order]...)
    bin2digit.(t)
end

function conditional_to(Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool})
    order = size(Y_past, 1)
    if order == 0
        return ones(Int, size(Y))
    else
        lag_obs = [copy(lag(Y, m)) for m = 1:order]  # transform dry=0 into 1 and wet=1 into 2 for array indexing
        for m = 1:order
            lag_obs[m][1:m, :] .= reverse(Y_past[1:m, :], dims=1) # avoid the missing first row
        end
        return dayx(lag_obs)
    end
end

function idx_observation_of_past_cat(lag_cat, n2t, T, size_order)
    # Matrix(T,D) of vector that give the index of data of same Y_past.
    # ie. size_order = 1 (no order) -> every data is in category 1
    # ie size_order = 2 (order on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_tj = Matrix{Vector{Vector{Int}}}(undef, T, D)
    n_in_t = [findall(n2t .== t) for t = 1:T] # could probably be speeded up e.g. recusivly suppressing already classified label with like sortperm
    for t in OneTo(T)
        n_t = n_in_t[t]
        for j = 1:D
            n_tm = [findall(lag_cat[n_t, j] .== m) for m = 1:size_order]
            idx_tj[t, j] = [n_t[n_tm[m]] for m = 1:size_order]
            ##
        end
    end
    return idx_tj
end

function idx_observation_of_past_cat(lag_cat, size_order)
    # Matrix(T,D) of vector that give the index of data of same past.
    # ie. size_order = 1 (no order) -> every data is in category 1
    # ie size_order = 2 (order on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_j = Vector{Vector{Vector{Int}}}(undef, D)
    for j = 1:D
        idx_j[j] = [findall(lag_cat[:, j] .== m) for m = 1:size_order]
    end
    return idx_j
end




function γₛ!(γₛ, γ, n_all)
    K, D, size_order, T, rain_cat = size(γₛ)
    for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)
        for k = 1:K
            γₛ[k, tup...] = sum(γ[n, k] for n in n_all[tup...]; init=0)
        end
    end
end

function s_ξ!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ)
    for t = 1:T
        for (k, l) in Iterators.product(1:K, 1:K)
            s_ξ[t, k, l] = sum(ξ[n, k, l] for n in n_in_t[t])
        end
    end
    # * We add ξ[N, k, l] but it should be zeros
end





# JuMP model use to increase R(θ,θ^i) for the Q(t) matrix
function model_for_A(s_ξ::AbstractArray, d::Int; silence=true)
    T, K = size(s_ξ)
    @assert K > 1 "To define a transition matrix K ≥ 2, here K = $K"
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 200)
    silence && set_silent(model)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]

    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    @variable(model, pklj_jump[l=1:(K-1), j=1:(2d+1)], start = 0.01)
    # Polynomial P_kl
    @NLexpression(model, Pkl[t=1:T, l=1:K-1], sum(trig[t][j] * pklj_jump[l, j] for j = 1:length(trig[t])))

    @NLparameter(model, s_πkl[t=1:T, l=1:K-1] == s_ξ[t, l])
    #TODO? is it useful to define the extra parameter for the sum?
    @NLparameter(model, s_πk[t=1:T] == sum(s_ξ[t, l] for l = 1:K))

    @NLobjective(
        model,
        Max,
        sum(sum(s_πkl[t, l] * Pkl[t, l] for l = 1:K-1) - s_πk[t] * log1p(sum(exp(Pkl[t, l]) for l = 1:K-1)) for t = 1:T)
    )
    # To add NL parameters to the model for later use https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:s_πkl] = s_πkl
    model[:s_πk] = s_πk
    return model
end

function update_A!(
    A::AbstractArray{<:AbstractFloat,3},
    θᴬ::AbstractArray{<:AbstractFloat,3},
    ξ::AbstractArray,
    s_ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
    n2t::AbstractArray{Int},
    n_in_t,
    model_A::Model;
    warm_start=true
)
    @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
    @argcheck size(α, 2) ==
              size(β, 2) ==
              size(LL, 2) ==
              size(A, 1) ==
              size(A, 2) ==
              size(ξ, 2) ==
              size(ξ, 3)

    N, K = size(LL)
    T = size(A, 3)
    @inbounds for n in OneTo(N - 1)
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))
        c = 0

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] = α[n, i] * A[i, j, t] * exp(LL[n+1, j] - m) * β[n+1, j]
            c += ξ[n, i, j]
        end

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] /= c
        end
    end
    ## 
    # ξ are the filtering probablies
    s_ξ!(s_ξ, ξ, n_in_t)

    θ_res = pmap(k -> fit_mle_one_A(θᴬ[k, :, :], model_A, s_ξ[:, k, :]; warm_start=warm_start), 1:K)

    for k = 1:K
        θᴬ[k, :, :] = θ_res[k][:, :]
    end

    for k = 1:K, l = 1:K-1, t = 1:T
        A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
    end
    for k = 1:K, t = 1:T
        A[k, K, t] = 1  # last colum is 1/normalization (one could do otherwise)
    end
    normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
    for k = 1:K, l = 1:K, t = 1:T
        A[k, l, t] /= normalization_polynomial[k, t]
    end
end

update_A!(
    A::AbstractArray{<:AbstractFloat,3},
    θᴬ::AbstractArray{<:AbstractFloat,3},
    ξ::AbstractArray,
    s_ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
    n2t::AbstractArray{Int},
    n_in_t,
    model_A::Nothing;
    warm_start=true
) = nothing

function fit_mle_one_A(θᴬ, model, s_ξ; warm_start=true)
    T, K = size(s_ξ)
    pklj_jump = model[:pklj_jump]
    s_πk = model[:s_πk]
    s_πkl = model[:s_πkl]
    ## Update the smoothing parameters in the JuMP model
    for t = 1:T
        set_value(s_πk[t], sum(s_ξ[t, l] for l = 1:K))
        for l = 1:K-1
            set_value(s_πkl[t, l], s_ξ[t, l])
        end
    end
    warm_start && set_start_value.(pklj_jump, θᴬ[:, :])
    # Optimize the updated model
    optimize!(model)
    # Obtained the new parameters
    return value.(pklj_jump)
end


function update_RB!(hmm::PeriodicHMMSpaMemory,
    Range_θ::AbstractArray{N,2} where {N}, theta_B::AbstractArray{N,4} where {N},
    γ::AbstractArray, wp, Y, Situations::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, maxiters=10)
    @argcheck size(γ, 1) == size(Y, 1)
    N = size(γ, 1)
    R = hmm.R
    size_order=size(hmm,4)
    K = size(R, 1)
    T = size(R, 2)
    D = size(hmm, 2)
    @show K, T
    # println("inside updateR! - before fit: ", Range_θ)
    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    # Parallelized loop
     @threads for k in 1:K

            @show k
            h = hmm.h
            w = γ[:, k]
            # println("B,h,w ok")
            n_pair = zeros(eltype(R), 4, D, D, T)
    
            @inbounds for tk in 1:N
                t = n2t[tk]
                for (i, j) in pairwise_indices2
                    w_k = w[tk]
                    @views begin
                        n_pair[1, i, j, t] += w_k * Situations[1, tk, i, j]
                        n_pair[2, i, j, t] += w_k * Situations[2, tk, i, j]
                        n_pair[3, i, j, t] += w_k * Situations[3, tk, i, j]
                        n_pair[4, i, j, t] += w_k* Situations[4, tk, i, j]
                    end
                end
            end
        # println("weight pairs ok")
        # Fix: Use `view` to pass mutable references
        # @show (Range_θ[ k, :])
        fit_mle_one_RB!(view(Range_θ, k, :), view(theta_B, k, :,:,:), h, Y, wp, n_pair,T,size_order,D; n2t=n2t, maxiters=maxiters)
        # @show (Range_θ[ k, :])
    end
    # println("inside updateR! - after fit: ", Range_θ)


    # Ensure in-place modification of R
    for k in 1:K
        R[k, :] .= [exp(mypolynomial_trigo(t, view(Range_θ, k, :), T)) for t = 1:T]
    end
    p = [1 / (1 + exp(polynomial_trigo(t, theta_B[k, s, h, :], T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    hmm.B[:, :, :, :] .= p
end




### try with maximum likelihood estim using optim .
using Optimization




function mypolynomial_trigo(t, β, T)
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


function likelihoods!(L::AbstractMatrix, hmm::PeriodicHMMSpaMemory, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(L, 1), size(model, 3))::AbstractVector{<:Integer}, QMC_m=30)
    N, K, D = size(Y, 1), size(model, 1), size(model, 2)
    @argcheck size(L) == (N, K)

    for i in 1:K
        @show i
        for n in 1:N
            t = n2t[n] # periodic t
            modelit = SpatialBernoulli(hmm.R[i, t], 1.0, 1 / 2, hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])], hmm.h)

            L[n, i] = pdf(modelit, Y[n, :]; m=D * QMC_m)
        end
    end
end



# try to save some time by building the matrices in advance, does not work....
function loglikelihoods!(LL::AbstractMatrix, hmm::PeriodicHMMSpaMemory, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer}, QMC_m=30)
    N, K, D, T = size(Y, 1), size(hmm, 1), size(hmm, 2), size(hmm, 3)
    @argcheck size(LL) == (N, K)

    Sigmat = zeros(D, D, T, K)
    @threads for t in 1:T
        for k in 1:K
            Sigmat[:, :, t, k] = matern.(hmm.h; range=hmm.R[k, t], sill=1, order=0.5)
        end
    end

    @threads for n in 1:N
        # Preallocate vectors outside loops to avoid repeated allocations
        a = fill(-Inf, D)
        b = fill(Inf, D)
        finite_bounds = zeros(D)
        zerosvec = zeros(D)
        for i in 1:K
            # @show n
            t = n2t[n] # periodic t
            finite_bounds .= quantile.(Normal(), hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])])
            a .= ifelse.(Y[n, :] .== 0, finite_bounds, -Inf)
            b .= ifelse.(Y[n, :] .== 1, finite_bounds, Inf)
            hy = mvnormcdf(zerosvec, Sigmat[:, :, t, i], a, b; m=D * QMC_m)
            LL[n, i] = log(hy[1])
        end
    end
end



function SmoothPeriodicStatsModels.loglikelihoods(hmm::PeriodicHMMSpaMemory, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust=false, n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=30)
    N, K = size(Y, 1), size(hmm, 1)
    LL = Matrix{Float64}(undef, N, K)

    lag_cat = conditional_to(Y, Y_past)

    loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t, QMC_m=QMC_m)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    return LL
end




function complete_loglikelihood(hmm::PeriodicHMMSpaMemory, y::AbstractArray, y_past::AbstractArray, z::AbstractVector; n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=30)
    N, D = size(y)
    lag_cat = conditional_to(y, y_past)
    return sum(log(hmm.A[z[n], z[n+1], n2t[n]]) for n = 1:N-1) + sum(logpdf(SpatialBernoulli(hmm.R[z[n], n2t[n]], 1.0, 1 / 2, hmm.B[CartesianIndex.(z[n], n2t[n], 1:D, lag_cat[n, :])], hmm.h), y[n, :]; m=D * QMC_m) for n = 1:N)
end

nb_param_HMMSpa(K, memory, d, D) = (2d + 1) * (K * 2^memory * D + K * (K - 1) + K)


function complete_loglikelihood(hmm::PeriodicHMMSpaMemory, y::AbstractArray, y_past::AbstractArray, z::AbstractVector; n2t=n_to_t(size(y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=30)
    N, D = size(y)
    lag_cat = conditional_to(y, y_past)

    # Parallelizing the two summations
    log_A_sum_threads = zeros(Float64, Threads.nthreads())
    @threads for n = 1:N-1
        tid = Threads.threadid()
        log_A_sum_threads[tid] += log(hmm.A[z[n], z[n+1], n2t[n]])
    end
    log_A_sum = sum(log_A_sum_threads)


    logpdf_sum_threads = zeros(Float64, Threads.nthreads())
    @threads for n = 1:N
        tid = Threads.threadid()
        logpdf_sum_threads[tid] += logpdf(
            SpatialBernoulli(hmm.R[z[n], n2t[n]], 1.0, 1 / 2,
                hmm.B[CartesianIndex.(z[n], n2t[n], 1:D, lag_cat[n, :])],
                hmm.h),
            y[n, :]; m=D * QMC_m
        )
    end
    logpdf_sum = sum(logpdf_sum_threads)


    return log_A_sum + logpdf_sum
end

function fit_mle!(
    hmm::PeriodicHMMSpaMemory,
    thetaA::AbstractArray{<:AbstractFloat,3},
    thetaB::AbstractArray{<:AbstractFloat,4},
    thetaR::AbstractArray{<:AbstractFloat,2}, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer},
    display=:none,
    maxiter=100,
    tol=1e-3,
    robust=false,
    silence=true,
    warm_start=true,
    tdist=1,
    QMC_m=30,
    maxiters_R=10, QMC_E=1, wp=1.0 .* (hmm.h .< maximum(hmm.h) * tdist)
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    # println("tdist = ",tdist)
    N, K, T, size_order, D = size(Y, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4), size(hmm, 2)
    @show N, K, T, size_order, D

    deg_A = (size(thetaA, 3) - 1) ÷ 2
    deg_B = (size(thetaB, 4) - 1) ÷ 2
    # println("wp= ",wp)
    rain_cat = 2 # dry or wet
    @argcheck T == size(hmm.B, 2)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => Float64[])

    all_thetaA_iterations = [copy(thetaA)]
    all_thetaB_iterations = [copy(thetaB)]

    # add new param : range !

    all_thetaR_iterations = [copy(thetaR)]

    # Allocate order for in-place updates
    c = zeros(N)
    α = zeros(N, K) #forward ?
    β = zeros(N, K) #backward ?
    γ = zeros(N, K) # regular smoothing proba
    γₛ = zeros(K, D, size_order, T, rain_cat) # summed smoothing proba
    ξ = zeros(N, K, K) # pi_kl(t) ?
    s_ξ = zeros(T, K, K) #? somme pi_kl(t) pour t de même périodicité
    LL = zeros(N, K) # stock the loglikelihoods for each state at each time ? (completely unwanted in the M step)

    # assign category for observation depending in the Y_past Y
    order = Int(log2(size_order))
    lag_cat = conditional_to(Y, Y_past)

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(Y[:, j] .== y, lag_cat[:, j] .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]

    model_A = K ≥ 2 ? model_for_A(s_ξ[:, 1, :], deg_A, silence=silence) : nothing # JuMP Model for transition matrix

    #Done check  model_B


    # generate situations
    if size_order == 1
        Situations = zeros(Int, 4, N, D, D)

        for k in 1:N
            for i in 1:D
                for j in 1:D
                    Situations[1, k, i, j] = (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[2, k, i, j] = (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[3, k, i, j] = (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[4, k, i, j] = (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0
                end
            end
        end
    elseif size_order == 2
        # generate situations
        Situations = zeros(Int, 16, N, D, D)

        for k in 2:N
            for i in 1:D
                for j in 1:D
                    Situations[1, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[2, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[3, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[4, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[5, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[6, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[7, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[8, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[9, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[10, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[11, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[12, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[13, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[14, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[15, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[16, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0
                end
            end
        end
        for i in 1:D
            for j in 1:D
                Situations[1, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[2, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[3, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[4, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[5, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[6, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[7, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[8, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[9, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[10, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[11, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[12, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[13, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[14, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[15, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[16, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0
            end
        end
    elseif size_order > 2
        println("memory of more than 2 not yet implemented for the mle estimation")
        return

    end
    println("Situations generated")

    loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t, QMC_m=QMC_m * QMC_E)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")


    for it = 1:maxiter
        println(it)
        update_a!(hmm.a, α, β)

        # DONE :need to check update_A
        update_A!(hmm.A, thetaA, ξ, s_ξ, α, β, LL, n2t, n_in_t, model_A; warm_start=warm_start)
        println("done updating A")



       # my_update_B!(hmm.B, thetaB, γ, γₛ, Y, n_all, model_B; warm_start=warm_start)
      #  println("done updating B")

        if size_order == 1
            update_RB!(hmm, thetaR,thetaB, γ, wp, Y, Situations; n2t=n2t, maxiters=maxiters_R)
            
         elseif size_order == 2
            update_RB_memory1!(hmm, thetaR,thetaB, γ, wp, Y, Situations; n2t=n2t, maxiters=maxiters_R)

            # update_R_memory1!(hmm, thetaR, γ, wp, Y, Situations; n2t=n2t, maxiters=maxiters_R)
        end
        println("done updating R")


        push!(all_thetaA_iterations, copy(thetaA))
        push!(all_thetaR_iterations, copy(thetaR))
        push!(all_thetaB_iterations, copy(thetaB))   
        # @show thetaR
        # @show hmm.R
        # update_R!(hmm, thetaR, γ, wp, Y; QMC_m=QMC_m, maxiters=maxiters_R)
        # @show thetaR
        # @show hmm.R
        # println("Before update_R!: ", thetaR)


        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely Y.
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check all(t -> istransmat(hmm.A[:, :, t]), 1:T)


        push!(all_thetaR_iterations, copy(thetaR))

        # loglikelihoods!(LL, hmm, Y, n2t)
        loglikelihoods!(LL, hmm, Y, lag_cat; n2t=n2t, QMC_m=QMC_m * QMC_E)

        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

        forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
        backwardlog!(β, c, hmm.a, hmm.A, LL; n2t=n2t)
        posteriors!(γ, α, β)

        logtotp = sum(c)


        if display == :iter
            ΔmaxA = round(maximum(abs, (all_thetaA_iterations[it+1] - all_thetaA_iterations[it]) ./ all_thetaA_iterations[it]), digits=5)
            ΔmaxB = round(maximum(abs, (all_thetaB_iterations[it+1] - all_thetaB_iterations[it]) ./ all_thetaB_iterations[it]), digits=5)
            ΔmaxR = round(maximum(abs, (all_thetaR_iterations[it+1] - all_thetaR_iterations[it]) ./ all_thetaR_iterations[it]), digits=5)
            println("Iteration $it: logtot = $(round(logtotp, digits = 6)), max(|θᴬᵢ-θᴬᵢ₋₁|/|θᴬᵢ₋₁|) = ", ΔmaxA, " & max(|θᴮᵢ-θᴮᵢ₋₁|/|θᴮᵢ₋₁|) = ", ΔmaxB, " & max(|θRᵢ-θRᵢ₋₁|/|θRᵢ₋₁|) = ", ΔmaxR)
            # flush(stdout)
        end

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if (ΔmaxA < tol && ΔmaxB < tol && ΔmaxR < tol)
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end
        # if abs((logtotp - logtot) / logtotp) > tol && logtotp < logtot
        #     (display in [:iter, :final]) &&
        #         println("stop the loglikelihood has deacreased dramatically")
        #     history["converged"] = false
        #     break
        # end
        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println("EM has not converged after $(history["iterations"]) iterations, logtot = $logtot")
        end
    end
    history, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations

end
function my_loglikelihood(R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, eps=1e-10, pairwise_indices2=Tuple.(findall(wp .> 0))
)
    N, D = size(Y)
    T = size(R, 1)
    # println("T = size(R,1)",T)
    # @show R

    Iij = ones(eltype(R), 4, D, D, T)
    @inbounds for t in 1:T



        for (i, j) in pairwise_indices2
            # @show (i,j)
            B_ij = @view B[t, [i, j]]
            h_ij = @view h[[i, j], [i, j]]
            if i == j
                Iij[1, i, j, t] = B_ij[1]
                Iij[4, i, j, t] = 1 - B_ij[1]
            else
                Iij[1, i, j, t] = ifelse(Iij[1, j, i, t] != 1.0, Iij[1, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1]), quantile(Normal(), B_ij[2]), exp(-h_ij[1, 2] / R[t])))
                Iij[2, i, j, t] = ifelse(Iij[3, j, i, t] != 1.0, Iij[3, j, i, t], B_ij[1] - Iij[1, i, j, t])
                Iij[3, i, j, t] = ifelse(Iij[2, j, i, t] != 1.0, Iij[2, j, i, t], B_ij[2] - Iij[1, i, j, t])
                Iij[4, i, j, t] = ifelse(i == j, 1.0 - Iij[1, i, j, t], 1.0 - Iij[1, i, j, t] - Iij[2, i, j, t] - Iij[3, i, j, t])
            end
        end
    end

    Iij .= max.(Iij, eps)  # Replace elements < eps with eps

    pairwise_sum = 0.0
    @inbounds for (i, j) in pairwise_indices2
        for t in 1:T
            if i != j
                pairwise_sum += wp[i, j] * n_pair[1, i, j, t] * log(Iij[1, i, j, t]) +
                                wp[i, j] * n_pair[2, i, j, t] * log(Iij[2, i, j, t]) +
                                wp[i, j] * n_pair[3, i, j, t] * log(Iij[3, i, j, t]) +
                                wp[i, j] * n_pair[4, i, j, t] * log(Iij[4, i, j, t])
            else
                pairwise_sum += wp[i, j] * n_pair[1, i, j, t] * log(Iij[1, i, j, t]) +
                                wp[i, j] * n_pair[4, i, j, t] * log(Iij[4, i, j, t])
            end
        end
    end

    return (pairwise_sum)
end






function fit_mle_one_RB!(theta_R, theta_B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real},T,size_order,D; n2t=n_to_t(size(Y, 1), T)::AbstractVector{<:Integer}, solver=Optimization.LBFGS(), return_sol=false, solkwargs...)
   
    # println("size(B,1) = T ? =",T)
    # println("inside updateR! - inside fit - before estim: ", theta_R)
    pairwise_indices2 = Tuple.(findall(wp .> 0))

    # Branch: Optimize only `range`, fixing `order`
    function optimfunction2(u, p)
        degP = Int((size(u,1)/(D*size_order + 1)-1)/2)
        uR=u[1:2*degP + 1]
        uB = reshape(u[2*degP + 1+1:end], D, size_order, 2*degP + 1)  # adjust reshape according to your actual shape


        Rt = ones(eltype(u), T)
        for t in 1:T
            Rt[t] = exp(mypolynomial_trigo(t, uR, T)) #u[1]= param for R
        end  
              # println("u inside optimfun",u)
        B = [1 / (1 + exp(polynomial_trigo(t, uB[ s, h, :], T))) for  t = 1:T, s = 1:D, h = 1:size_order]

        return -my_loglikelihood(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2
        )
    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), AutoForwardDiff())
    u0 = collect(vcat(theta_R, vec(theta_B)))
     
    optimfunction2(u0,[Y, n_pair, h, wp])

    # println("inside updateR! - inside fit - u0: ", u0)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    # Solve the problem
    sol = solve(prob, solver ; solkwargs...)

    # Check solution status
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end
    # println("inside updateR! - inside fit - after estim sol.u: ", sol.u)

    # Return the result
    degP = Int((size(sol.u,1)/(D*size_order + 1)-1)/2)

    theta_R[:] .=sol.u[1:2*degP + 1]
    theta_B[:,:,:] .= reshape(sol.u[2*degP + 1+1:end], D, 1, 2*degP + 1)  # adjust reshape according to your actual shape

    # @show theta_R
end





function my_loglikelihood_memory1(R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, eps=1e-10, pairwise_indices2=Tuple.(findall(wp .> 0)))
    N, D = size(Y)
    T = size(R, 1)
    # println("T = size(R,1)",T)
    # @show R


    Iij = ones(eltype(R), 16, D, D, T)
    @inbounds for t in 1:T
        for (i, j) in pairwise_indices2
            # @show (i,j)
            h_ij = @view h[[i, j], [i, j]]

            B_ij = @view B[t, [i, j], :]

            if i == j
                Iij[1, i, j, t] = B_ij[1, 2]
                Iij[4, i, j, t] = 1 - B_ij[1, 2]
                Iij[13, i, j, t] = B_ij[1, 1]
                Iij[16, i, j, t] = 1 - B_ij[1, 1]
            end
            if i != j
                Iij[1, i, j, t] = ifelse(Iij[1, j, i, t] != 1.0, Iij[1, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1, 2]), quantile(Normal(), B_ij[2, 2]), exp(-h_ij[1, 2] / R[t])))
                Iij[2, i, j, t] = ifelse(Iij[3, j, i, t] != 1.0, Iij[3, j, i, t], B_ij[1, 2] - Iij[1, i, j, t])
                Iij[3, i, j, t] = ifelse(Iij[2, j, i, t] != 1.0, Iij[2, j, i, t], B_ij[2, 2] - Iij[1, i, j, t])
                Iij[4, i, j, t] = ifelse(i == j, 1.0 - Iij[1, i, j, t], 1.0 - Iij[1, i, j, t] - Iij[2, i, j, t] - Iij[3, i, j, t])

                Iij[5, i, j, t] = ifelse(Iij[9, j, i, t] != 1.0, Iij[9, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1, 2]), quantile(Normal(), B_ij[2, 1]), exp(-h_ij[1, 2] / R[t])))
                Iij[6, i, j, t] = ifelse(Iij[11, j, i, t] != 1.0, Iij[11, j, i, t], B_ij[1, 2] - Iij[5, i, j, t])
                Iij[7, i, j, t] = ifelse(Iij[10, j, i, t] != 1.0, Iij[10, j, i, t], B_ij[2, 1] - Iij[5, i, j, t])
                Iij[8, i, j, t] = 1.0 - Iij[5, i, j, t] - Iij[6, i, j, t] - Iij[7, i, j, t]


                Iij[9, i, j, t] = ifelse(Iij[5, j, i, t] != 1.0, Iij[5, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1, 1]), quantile(Normal(), B_ij[2, 2]), exp(-h_ij[1, 2] / R[t])))
                Iij[10, i, j, t] = ifelse(Iij[7, j, i, t] != 1.0, Iij[7, j, i, t], B_ij[1, 1] - Iij[9, i, j, t])
                Iij[11, i, j, t] = ifelse(Iij[6, j, i, t] != 1.0, Iij[6, j, i, t], B_ij[2, 2] - Iij[9, i, j, t])
                Iij[12, i, j, t] = 1.0 - Iij[9, i, j, t] - Iij[10, i, j, t] - Iij[11, i, j, t]

                Iij[13, i, j, t] = ifelse(Iij[13, j, i, t] != 1.0, Iij[13, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1, 1]), quantile(Normal(), B_ij[2, 1]), exp(-h_ij[1, 2] / R[t])))
                Iij[14, i, j, t] = ifelse(Iij[15, j, i, t] != 1.0, Iij[15, j, i, t], B_ij[1, 1] - Iij[13, i, j, t])
                Iij[15, i, j, t] = ifelse(Iij[14, j, i, t] != 1.0, Iij[14, j, i, t], B_ij[2, 1] - Iij[13, i, j, t])
                Iij[16, i, j, t] = ifelse(i == j, 1.0 - Iij[13, i, j, t], 1.0 - Iij[13, i, j, t] - Iij[14, i, j, t] - Iij[15, i, j, t])
            end
        end
    end
    Iij .= max.(Iij, eps)  # Replace elements < eps with eps

    # bad_indices = findall(Iij .< 0)
    # println(bad_indices)




    pairwise_sum = 0.0
    @inbounds for (i, j) in pairwise_indices2
        for t in 1:T
            if i != j
                pairwise_sum += wp[i, j] * sum(n_pair[k, i, j, t] * log(Iij[k, i, j, t]) for k in 1:16)
            else
                pairwise_sum += wp[i, j] * n_pair[1, i, j, t] * log(Iij[1, i, j, t]) +
                                wp[i, j] * n_pair[4, i, j, t] * log(Iij[4, i, j, t]) +
                                wp[i, j] * n_pair[13, i, j, t] * log(Iij[13, i, j, t]) +
                                wp[i, j] * n_pair[16, i, j, t] * log(Iij[16, i, j, t])
            end
        end
    end
    return (pairwise_sum)
end



function fit_mle_one_RB_memory1!(theta_R, theta_B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real},T,size_order,D; n2t=n_to_t(size(Y, 1), T)::AbstractVector{<:Integer}, solver=Optimization.LBFGS(), return_sol=false, solkwargs...)
    
    # println("size(B,1) = T ? =",T)
    # println("inside updateR! - inside fit - before estim: ", theta_R)
    pairwise_indices2 = Tuple.(findall(wp .> 0))

    # Branch: Optimize only `range`, fixing `order`
    function optimfunction2(u, p)
        degP = Int((size(u,1)/(D*size_order + 1)-1)/2)
        uR=u[1:2*degP + 1]
        uB = reshape(u[2*degP + 1+1:end], D, size_order, 2*degP + 1)  # adjust reshape according to your actual shape


        Rt = ones(eltype(u), T)
        for t in 1:T
            Rt[t] = exp(mypolynomial_trigo(t, uR, T)) #u[1]= param for R
        end  
              # println("u inside optimfun",u)
        B = [1 / (1 + exp(polynomial_trigo(t, uB[ s, h, :], T))) for  t = 1:T, s = 1:D, h = 1:size_order]

        return -my_loglikelihood_memory1(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)

    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), AutoForwardDiff())
    u0 = vcat(theta_R, vec(theta_B))
    optimfunction2(u0,[Y, n_pair, h, wp])

    # println("inside updateR! - inside fit - u0: ", u0)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    # Solve the problem
    sol = solve(prob, solver ; solkwargs...)

    # Check solution status
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end
    # println("inside updateR! - inside fit - after estim sol.u: ", sol.u)

    # Return the result
    degP = Int((size(sol.u,1)/(D*size_order + 1)-1)/2)

    theta_R[:] .=sol.u[1:2*degP + 1]
    theta_B[:,:,:] .= reshape(sol.u[2*degP + 1+1:end], D,size_order, 2*degP + 1)  # adjust reshape according to your actual shape

    # @show theta_R
end




function update_RB_memory1!(hmm::PeriodicHMMSpaMemory,
    Range_θ::AbstractArray{N,2} where {N}, theta_B::AbstractArray{N,4} where {N},
    γ::AbstractMatrix, wp, Y, Situations::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, maxiters=10)
    @argcheck size(γ, 1) == size(Y, 1)
    N = size(γ, 1)
    R = hmm.R
    D = size(hmm, 2)
    size_order=size(hmm,4)
    K = size(R, 1)
    T = size(R, 2)
    # println("inside updateR! - before fit: ", Range_θ)
    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    # Parallelized loop
    @threads for k in 1:K
        B = hmm.B[k, :, :, :]  # B[k,t,h]
        h = hmm.h
        w = γ[:, k]
        n_pair = zeros(eltype(R), 16, D, D, T)

        @inbounds for tk in 1:N
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                w_k = w[tk]
                @views begin
                    for s in 1:16
                        n_pair[s, i, j, t] += w_k * Situations[s, tk, i, j]

                    end

                end
            end
        end

        # Fix: Use `view` to pass mutable references
        # @show (Range_θ[ k, :])
        fit_mle_one_RB_memory1!(view(Range_θ, k, :), view(theta_B, k, :,:,:), h, Y, wp, n_pair,T,size_order,D; n2t=n2t, maxiters=maxiters)

        # @show (Range_θ[ k, :])
    end
    # println("inside updateR! - after fit: ", Range_θ)


    # Ensure in-place modification of R
    for k in 1:K
        R[k, :] .= [exp(mypolynomial_trigo(t, view(Range_θ, k, :), T)) for t = 1:T]
    end

 
    p = [1 / (1 + exp(polynomial_trigo(t, theta_B[k, s, h, :], T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    hmm.B[:, :, :, :] .= p
end


function fit_mle_one_RB!(theta_R, theta_B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real},T,size_order,D; n2t=n_to_t(size(Y, 1), T)::AbstractVector{<:Integer}, solver=Optimization.LBFGS(), return_sol=false, solkwargs...)
   
    # println("size(B,1) = T ? =",T)
    # println("inside updateR! - inside fit - before estim: ", theta_R)
    pairwise_indices2 = Tuple.(findall(wp .> 0))
    degP = Int((size(theta_R,1)-1)/2)
    # Branch: Optimize only `range`, fixing `order`
    function optimfunction2(u, p)
        uR=u[1:2*degP + 1]
        uB = reshape(u[2*degP + 1+1:end], D, size_order, 2*degP + 1)  # adjust reshape according to your actual shape


        Rt = ones(eltype(u), T)
        for t in 1:T
            Rt[t] = exp(mypolynomial_trigo(t, uR, T)) #u[1]= param for R
        end  
              # println("u inside optimfun",u)
        B = [1 / (1 + exp(polynomial_trigo(t, uB[ s, h, :], T))) for  t = 1:T, s = 1:D, h = 1:size_order]

        return -my_loglikelihood(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2
        )
    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), AutoForwardDiff())
    u0 = vcat(theta_R, vec(theta_B))
    optimfunction2(u0,[Y, n_pair, h, wp])

    # println("inside updateR! - inside fit - u0: ", u0)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    # Solve the problem
    sol = solve(prob, solver ; solkwargs...)

    # Check solution status
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end
    # println("inside updateR! - inside fit - after estim sol.u: ", sol.u)

    # Return the result
    degP = Int((size(sol.u,1)/(D*size_order + 1)-1)/2)

    theta_R[:] .=sol.u[1:2*degP + 1]
    theta_B[:,:,:] .= reshape(sol.u[2*degP + 1+1:end], D, 1, 2*degP + 1)  # adjust reshape according to your actual shape

    # @show theta_R
end