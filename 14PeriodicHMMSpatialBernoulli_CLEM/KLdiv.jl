# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr

begin
    seed=3
    include("../13PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
    Random.seed!(seed)
    # ## Utilities
    using ArgCheck
    using Base: OneTo
    using ShiftedArrays: lead, lag
    using Distributed
    # ## Optimization
    using JuMP, Ipopt
    using Optimization, OptimizationMOI
    using LsqFit
    using Dates

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
    using CSV
    using DataFrames


end
# include("updateHMMSpa_functions_CLEM.jl")

# test --------------------------------------#
tdist = 1.2

my_K = 2# Number of Hidden states
my_T = 3 # Period
my_N = my_T * 10
n2t = n_to_t(my_N, my_T)


begin
my_distance = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
my_D = length(my_locations[:, 1])



begin #parameters true
my_autoregressive_order = 0

my_size_order = 2^my_autoregressive_order
my_degree_of_P = 0
my_size_degree_of_P = 2 * my_degree_of_P + 1

my_trans_θ = 4 * (rand(my_K, my_K - 1, my_size_degree_of_P) .- 1 / 2)
# parameters of the transition matrix. TO KEEP
# matrix of size K * (K-1) * (2degP+1) : parameters of the transition matrix for the K(K-1), for each of the trigo param.

if my_autoregressive_order == 0
    my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
    if my_K>1
    my_Bernoulli_θ[2, :, :, :] .+= 0.5
    end
else

    my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
    my_Bernoulli_θ[:, :, 1, :] .= my_Bernoulli_θ[:, :, 2, :]

    my_Bernoulli_θ[:, :, 1, 1] .= min.(my_Bernoulli_θ[:, :, 2, 1] .+ 1, 1)
end
# rain probability parameters of the bernoulli Emissions
# K (states) * D (stations)  * 1+AR order (memory in the HMM) * (2degP+1) (each of the trigo param)
my_Range_θ = (rand(my_K, my_size_degree_of_P) .- 1 / 2)
my_Range_θ[:, 1] = log.(300 .* (1:my_K))
# range  parameters of the bernoulli Emissions 
# K (states)   * 1+AR order (memory in the HMM) * (2degP+1) (each of the trigo param) - range par

my_a = fill(1 / my_K, my_K)
model = Trig2PeriodicHMMspaMemory(my_a, my_trans_θ, my_Bernoulli_θ, my_Range_θ, my_T, my_distance);

# model of emission at state 1, time 1, and it rained before (?)

size(model)
model.B

end

z, Y = my_rand(model, n2t; seq=true)
Y = convert(Array{Bool}, Y)
Y_past = rand(Bool, my_autoregressive_order, my_D)


N, K, T, size_order, D = size(Y, 1), size(model, 1), size(model, 3), size(model, 4), size(model, 2)


# compute p(z|Ya,theta)
include("../PeriodicHMMSpatialBernoulli_CLEM/estimation_functions_BandR_v2.jl")

wp=1.0 .* (model.h .< maximum(model.h) * tdist)
    cij = zeros(N, D, D)
    αij = zeros(N, K, D, D) #forward ? but ij
    βij = zeros(N, K, D, D) #backward ?
    γij = zeros(N, K, D, D) # regular smoothing proba

    # ξ = zeros(N, K, K) # sum_ wij pi_kl(t)^ij ?
    # s_ξ = zeros(T, K, K) #? somme pi_kl(t) pour t de même périodicité
    ξij = zeros(N, K, K,D,D) #  pi_kl(t)^ij ?
    s_ξ = zeros(T, K, K) #? somme sur ij wp_ij pi_kl(t)^ij pour t de même périodicité

    LLij = zeros(N, K, D, D) # value of pariwise likelihood for each time step
    Iij = zeros(4, K, D, D, T) #possible values of pairwise for all times in periodic

    pairwise_indices2 = Tuple.(findall(wp .> 0))






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



    loglikelihoods!(LLij, Iij, model, Situations, pairwise_indices2, QMC_m=1000; n2t)

    forwardlog!(αij, cij, model.a, model.A, LLij, pairwise_indices2; n2t=n2t)
    backwardlog!(βij, cij, model.a, model.A, LLij, pairwise_indices2; n2t=n2t)

    posteriors!(γij, αij, βij, pairwise_indices2)
    Threads.@threads for n in 1:(N-1)
        t = n2t[n]
        At = @view model.A[:, :, t]

        for (ii,jj) in pairwise_indices2
            w = wp[ii,jj]

            # compute unnormalized ξ_pair (K×K)
            ξ_pair = zeros(K,K)
            m = maximum(@view LLij[n+1, :, ii, jj])
            for i in 1:K, j in 1:K
                ξ_pair[i,j] = αij[n,i,ii,jj] * At[i,j] * exp(LLij[n+1,j,ii,jj]-m) * βij[n+1,j,ii,jj]
            end

            Z = sum(ξ_pair)
            if Z == 0.0 || !isfinite(Z)
                continue
            end

            # normalize per pair and store
            invZ = w / Z
            for i in 1:K, j in 1:K
                ξij[n,i,j,ii,jj] = ξ_pair[i,j] * invZ
            end
        end
    end




# compute p(z|Y,theta)

    c= zeros(N)
    α= zeros(N, K) #forward ? 
    β = zeros(N, K) #backward ?
    γ = zeros(N, K) # regular smoothing proba

    ξ = zeros(N, K, K,) #  pi_kl(t)^ij ?

    LL = zeros(N, K) # value of pariwise likelihood for each time step







    lag_cat = conditional_to(Y, Y_past)

    function loglikelihoodshere!(LL::AbstractMatrix, hmm::PeriodicHMMSpaMemory, Y::AbstractMatrix, lag_cat::AbstractMatrix{<:Integer}; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer}, QMC_m=30)
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
    
    loglikelihoodshere!(LL, model, Y, lag_cat; n2t=n2t, QMC_m=100)
    forwardlog!(α, c, model.a, model.A, LL; n2t=n2t)
    backwardlog!(β, c, model.a, model.A, LL; n2t=n2t)
    posteriors!(γ, α, β)
    @inbounds for n in OneTo(N - 1)
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))
        c = 0

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] = α[n, i] * model.A[i, j, t] * exp(LL[n+1, j] - m) * β[n+1, j]
            c += ξ[n, i, j]
        end

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] /= c
        end
    end
    



#compute KL divergence

KLij = zeros(D,D)
for (i,j) in pairwise_indices2
KLij[i,j] += sum((γ[1,k]*log(γ[1,k]/γij[1,k,i,j])) for k in 1:K)
for n in 1:N-1
    KLij[i,j] += sum((γ[n,k]*ξ[n, k, l]*log(ξ[n, k, l]/ξij[n, k, l,i,j])) for k in 1:K,l in 1:K)
end
end
KLijsum= sum(sum(wp.*KLij))
Lij_theta  = sum(sum(cij[i, :, :] .* wp) for i in 1:N)
end
println("#############################################")
println("Case :K=$K, wp = 1_{dij<dmax* $(tdist)} , N = $N")
println("Log PL(Y,theta) = ",Lij_theta)
println("lp2(theta, theta) - Log PL(Y,theta) = ", -KLijsum)
println("lp2(theta, theta)  = ",-KLijsum+Lij_theta, " supposed to be close to logPL(Y, theta)")
println("logPL > lp2 is supposed to be true. Really :", (-KLijsum < 0) )
println("#############################################")
