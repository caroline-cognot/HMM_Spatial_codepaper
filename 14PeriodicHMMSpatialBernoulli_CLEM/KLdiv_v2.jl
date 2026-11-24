# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr>

begin
    seed=3
    include("../13PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
    Random.seed!(seed)
    using ArgCheck
    using Base: OneTo
    using ShiftedArrays: lead, lag
    using Distributed
    using JuMP, Ipopt
    using Optimization, OptimizationMOI
    using LsqFit
    using Dates
    using Distributions
    using Random: AbstractRNG, GLOBAL_RNG, rand!
    using LogExpFunctions: logsumexp!, logsumexp
    using PeriodicHiddenMarkovModels
    using PeriodicHiddenMarkovModels: viterbi, istransmat, update_a!, vec_maximum
    import PeriodicHiddenMarkovModels: forwardlog!, backwardlog!, viterbi, viterbi!, viterbilog!, posteriors!
    import PeriodicHiddenMarkovModels: fit_mle!, fit_mle
    import Distributions: fit_mle
    import Base: rand
    import Base: ==, copy, size
    using Base.Threads
    using CSV
    using DataFrames
    using Statistics: mean
end

tdist = 1.2

my_K = 2
my_T = 15
my_N = my_T * 100
n2t = n_to_t(my_N, my_T)


    my_distance = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
    my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
    my_D = length(my_locations[:, 1])

    begin
        my_autoregressive_order = 0
        my_size_order = 2^my_autoregressive_order
        my_degree_of_P = 0
        my_size_degree_of_P = 2 * my_degree_of_P + 1

        my_trans_θ = 4 * (rand(my_K, my_K - 1, my_size_degree_of_P) .- 1 / 2)

        if my_autoregressive_order == 0
            my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
            if my_K > 1
                my_Bernoulli_θ[2, :, :, :] .+= 0.5
            end
        else
            my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
            my_Bernoulli_θ[:, :, 1, :] .= my_Bernoulli_θ[:, :, 2, :]
            my_Bernoulli_θ[:, :, 1, 1] .= min.(my_Bernoulli_θ[:, :, 2, 1] .+ 1, 1)
        end

        my_Range_θ = (rand(my_K, my_size_degree_of_P) .- 1 / 2)
        my_Range_θ[:, 1] = log.(300 .* (1:my_K))

        my_a = fill(1 / my_K, my_K)
        model = Trig2PeriodicHMMspaMemory(my_a, my_trans_θ, my_Bernoulli_θ, my_Range_θ, my_T, my_distance)

        size(model)
        model.B
    end

    z, Y = my_rand(model, n2t; seq=true)
    Y = convert(Array{Bool}, Y)
    Y_past = rand(Bool, my_autoregressive_order, my_D)

    N, K, T, size_order, D = size(Y, 1), size(model, 1), size(model, 3), size(model, 4), size(model, 2)

    # ── Pairwise forward-backward ─────────────────────────────────────────────

    include("../14PeriodicHMMSpatialBernoulli_CLEM/estimation_functions_BandR_v3.jl")

    wp = 1.0 .* (model.h .< maximum(model.h) * tdist)
    cij = zeros(N, D, D)
    αij = zeros(N, K, D, D)
    βij = zeros(N, K, D, D)
    γij = zeros(N, K, D, D)
    ξij = zeros(N, K, K, D, D)
    LLij = zeros(N, K, D, D)
    Iij = zeros(4, K, D, D, T)

    pairwise_indices2 = Tuple.(findall(wp .> 0))

    if size_order == 1
        Situations = zeros(Int, 4, N, D, D)
        for k in 1:N, i in 1:D, j in 1:D
            Situations[1, k, i, j] = (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
            Situations[2, k, i, j] = (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
            Situations[3, k, i, j] = (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
            Situations[4, k, i, j] = (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0
        end
    elseif size_order == 2
        Situations = zeros(Int, 16, N, D, D)
        for k in 2:N, i in 1:D, j in 1:D
            Situations[1,  k, i, j] = (Y[k-1,i]==1 && Y[k-1,j]==1) && (Y[k,i]==1 && Y[k,j]==1) ? 1 : 0
            Situations[2,  k, i, j] = (Y[k-1,i]==1 && Y[k-1,j]==1) && (Y[k,i]==1 && Y[k,j]==0) ? 1 : 0
            Situations[3,  k, i, j] = (Y[k-1,i]==1 && Y[k-1,j]==1) && (Y[k,i]==0 && Y[k,j]==1) ? 1 : 0
            Situations[4,  k, i, j] = (Y[k-1,i]==1 && Y[k-1,j]==1) && (Y[k,i]==0 && Y[k,j]==0) ? 1 : 0
            Situations[5,  k, i, j] = (Y[k-1,i]==1 && Y[k-1,j]==0) && (Y[k,i]==1 && Y[k,j]==1) ? 1 : 0
            Situations[6,  k, i, j] = (Y[k-1,i]==1 && Y[k-1,j]==0) && (Y[k,i]==1 && Y[k,j]==0) ? 1 : 0
            Situations[7,  k, i, j] = (Y[k-1,i]==1 && Y[k-1,j]==0) && (Y[k,i]==0 && Y[k,j]==1) ? 1 : 0
            Situations[8,  k, i, j] = (Y[k-1,i]==1 && Y[k-1,j]==0) && (Y[k,i]==0 && Y[k,j]==0) ? 1 : 0
            Situations[9,  k, i, j] = (Y[k-1,i]==0 && Y[k-1,j]==1) && (Y[k,i]==1 && Y[k,j]==1) ? 1 : 0
            Situations[10, k, i, j] = (Y[k-1,i]==0 && Y[k-1,j]==1) && (Y[k,i]==1 && Y[k,j]==0) ? 1 : 0
            Situations[11, k, i, j] = (Y[k-1,i]==0 && Y[k-1,j]==1) && (Y[k,i]==0 && Y[k,j]==1) ? 1 : 0
            Situations[12, k, i, j] = (Y[k-1,i]==0 && Y[k-1,j]==1) && (Y[k,i]==0 && Y[k,j]==0) ? 1 : 0
            Situations[13, k, i, j] = (Y[k-1,i]==0 && Y[k-1,j]==0) && (Y[k,i]==1 && Y[k,j]==1) ? 1 : 0
            Situations[14, k, i, j] = (Y[k-1,i]==0 && Y[k-1,j]==0) && (Y[k,i]==1 && Y[k,j]==0) ? 1 : 0
            Situations[15, k, i, j] = (Y[k-1,i]==0 && Y[k-1,j]==0) && (Y[k,i]==0 && Y[k,j]==1) ? 1 : 0
            Situations[16, k, i, j] = (Y[k-1,i]==0 && Y[k-1,j]==0) && (Y[k,i]==0 && Y[k,j]==0) ? 1 : 0
        end
        for i in 1:D, j in 1:D
            Situations[1,  1, i, j] = (Y_past[i]==1 && Y_past[j]==1) && (Y[1,i]==1 && Y[1,j]==1) ? 1 : 0
            Situations[2,  1, i, j] = (Y_past[i]==1 && Y_past[j]==1) && (Y[1,i]==1 && Y[1,j]==0) ? 1 : 0
            Situations[3,  1, i, j] = (Y_past[i]==1 && Y_past[j]==1) && (Y[1,i]==0 && Y[1,j]==1) ? 1 : 0
            Situations[4,  1, i, j] = (Y_past[i]==1 && Y_past[j]==1) && (Y[1,i]==0 && Y[1,j]==0) ? 1 : 0
            Situations[5,  1, i, j] = (Y_past[i]==1 && Y_past[j]==0) && (Y[1,i]==1 && Y[1,j]==1) ? 1 : 0
            Situations[6,  1, i, j] = (Y_past[i]==1 && Y_past[j]==0) && (Y[1,i]==1 && Y[1,j]==0) ? 1 : 0
            Situations[7,  1, i, j] = (Y_past[i]==1 && Y_past[j]==0) && (Y[1,i]==0 && Y[1,j]==1) ? 1 : 0
            Situations[8,  1, i, j] = (Y_past[i]==1 && Y_past[j]==0) && (Y[1,i]==0 && Y[1,j]==0) ? 1 : 0
            Situations[9,  1, i, j] = (Y_past[i]==0 && Y_past[j]==1) && (Y[1,i]==1 && Y[1,j]==1) ? 1 : 0
            Situations[10, 1, i, j] = (Y_past[i]==0 && Y_past[j]==1) && (Y[1,i]==1 && Y[1,j]==0) ? 1 : 0
            Situations[11, 1, i, j] = (Y_past[i]==0 && Y_past[j]==1) && (Y[1,i]==0 && Y[1,j]==1) ? 1 : 0
            Situations[12, 1, i, j] = (Y_past[i]==0 && Y_past[j]==1) && (Y[1,i]==0 && Y[1,j]==0) ? 1 : 0
            Situations[13, 1, i, j] = (Y_past[i]==0 && Y_past[j]==0) && (Y[1,i]==1 && Y[1,j]==1) ? 1 : 0
            Situations[14, 1, i, j] = (Y_past[i]==0 && Y_past[j]==0) && (Y[1,i]==1 && Y[1,j]==0) ? 1 : 0
            Situations[15, 1, i, j] = (Y_past[i]==0 && Y_past[j]==0) && (Y[1,i]==0 && Y[1,j]==1) ? 1 : 0
            Situations[16, 1, i, j] = (Y_past[i]==0 && Y_past[j]==0) && (Y[1,i]==0 && Y[1,j]==0) ? 1 : 0
        end
    elseif size_order > 2
        println("memory of more than 2 not yet implemented")
        return
    end

    loglikelihoods!(LLij, Iij, model, Situations, pairwise_indices2, QMC_m=1000; n2t)
    forwardlog!(αij, cij, model.a, model.A, LLij, pairwise_indices2; n2t=n2t)
    backwardlog!(βij, cij, model.a, model.A, LLij, pairwise_indices2; n2t=n2t)
    posteriors!(γij, αij, βij, pairwise_indices2)

    Threads.@threads for n in 1:(N-1)
        t = n2t[n]
        At = @view model.A[:, :, t]
        for (ii, jj) in pairwise_indices2
            w = wp[ii, jj]
            ξ_pair = zeros(K, K)
            m = maximum(@view LLij[n+1, :, ii, jj])
            for i in 1:K, j in 1:K
                ξ_pair[i,j] = αij[n,i,ii,jj] * At[i,j] * exp(LLij[n+1,j,ii,jj]-m) * βij[n+1,j,ii,jj]
            end
            Z = sum(ξ_pair)
            if Z == 0.0 || !isfinite(Z)
                continue
            end
            invZ = w / Z
            for i in 1:K, j in 1:K
                ξij[n,i,j,ii,jj] = ξ_pair[i,j] * invZ
            end
        end
    end

    # ── Full forward-backward ─────────────────────────────────────────────────

    c  = zeros(N)
    α  = zeros(N, K)
    β  = zeros(N, K)
    γ  = zeros(N, K)
    ξ  = zeros(N, K, K)
    LL = zeros(N, K)

    lag_cat = conditional_to(Y, Y_past)

    function loglikelihoodshere!(LL, hmm, Y, lag_cat; n2t=n_to_t(size(LL,1), size(hmm,3)), QMC_m=30)
        N, K, D, T = size(Y,1), size(hmm,1), size(hmm,2), size(hmm,3)
        @argcheck size(LL) == (N, K)
        Sigmat = zeros(D, D, T, K)
        @threads for t in 1:T
            for k in 1:K
                Sigmat[:,:,t,k] = matern.(hmm.h; range=hmm.R[k,t], sill=1, order=0.5)
            end
        end
        @threads for n in 1:N
            a = fill(-Inf, D)
            b = fill(Inf, D)
            finite_bounds = zeros(D)
            zerosvec = zeros(D)
            for i in 1:K
                t = n2t[n]
                finite_bounds .= quantile.(Normal(), hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n,:])])
                a .= ifelse.(Y[n,:] .== 0, finite_bounds, -Inf)
                b .= ifelse.(Y[n,:] .== 1, finite_bounds, Inf)
                hy = mvnormcdf(zerosvec, Sigmat[:,:,t,i], a, b; m=D*QMC_m)
                LL[n, i] = log(hy[1])
            end
        end
    end

    loglikelihoodshere!(LL, model, Y, lag_cat; n2t=n2t, QMC_m=100)
    forwardlog!(α, c, model.a, model.A, LL; n2t=n2t)
    backwardlog!(β, c, model.a, model.A, LL; n2t=n2t)
    posteriors!(γ, α, β)

    @inbounds for n in OneTo(N-1)
        t = n2t[n]
        m = vec_maximum(view(LL, n+1, :))
        cc = 0.0
        for i in OneTo(K), j in OneTo(K)
            ξ[n,i,j] = α[n,i] * model.A[i,j,t] * exp(LL[n+1,j]-m) * β[n+1,j]
            cc += ξ[n,i,j]
        end
        for i in OneTo(K), j in OneTo(K)
            ξ[n,i,j] /= cc
        end
    end

    # ── KL(p(Z|Y) || p(Z|Y_ij)) — emission route ─────────────────────────────

    # log p(Y): sum of per-step log normalisation constants from full forward
    log_pY = sum(c)

    # log p(Y_ij): sum of per-step log normalisation constants from pairwise forward
    log_pYij = zeros(D, D)
    for (ii, jj) in pairwise_indices2
        log_pYij[ii, jj] = sum(cij[n, ii, jj] for n in 1:N)
    end

    # E_{Z|Y}[log p(Y|Z) - log p(Y_ij|Z)]
    term1 = zeros(D, D)
    for (ii, jj) in pairwise_indices2
        for n in 1:N, k in 1:K
            term1[ii, jj] += γ[n, k] * (LL[n, k] - LLij[n, k, ii, jj])
        end
    end

    # Full KL
    KLij_emission = zeros(D, D)
    for (ii, jj) in pairwise_indices2
        KLij_emission[ii, jj] = term1[ii, jj] + log_pYij[ii, jj] - log_pY
    end

    # ── Sanity checks ─────────────────────────────────────────────────────────
    @assert all(KLij_emission[ii,jj] >= -1e-6 for (ii,jj) in pairwise_indices2) "KL is negative"
    @assert all(term1[ii,jj] <= 1e-6 for (ii,jj) in pairwise_indices2) "term1 is positive — unexpected"
    @assert all(log_pYij[ii,jj] >= log_pY - 1e-6 for (ii,jj) in pairwise_indices2) "log p(Y_ij) < log p(Y)"

    Lij_theta = sum(sum(cij[:, i, j] .* wp[i,j]) for (i,j) in pairwise_indices2)

    println("#############################################")
    println("Case: K=$K, wp = 1_{dij < dmax*$(tdist)}, N=$N")
    println("log p(Y)              = ", log_pY)
    println("mean log p(Y_ij)      = ", mean(log_pYij[ii,jj] for (ii,jj) in pairwise_indices2))
    println("mean KL(p(Z|Y)||p(Z|Y_ij)) = ", mean(KLij_emission[ii,jj] for (ii,jj) in pairwise_indices2))
    println("Log PL(Y,theta)       = ", Lij_theta)
    println("#############################################")
