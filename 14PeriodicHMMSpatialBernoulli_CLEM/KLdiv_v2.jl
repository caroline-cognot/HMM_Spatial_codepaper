# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr>
# Study of KL(p(Z|Y) || p(Z|Y_ij)) — numerically stable chain-rule formulation
# KL is computed as sum of KL on marginals and conditional transitions,
# avoiding catastrophic cancellation from LL vs LLij scale mismatch.

begin
    seed = 3
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
    using Statistics: mean, std
    using Plots
    using LaTeXStrings
end

include("../14PeriodicHMMSpatialBernoulli_CLEM/estimation_functions_BandR_v3.jl")

# ── Parameters ────────────────────────────────────────────────────────────────
tdist  = 1.2
my_K   = 2
my_T   = 15
my_N   = my_T * 1000
n2t    = n_to_t(my_N, my_T)
my_autoregressive_order = 0
my_size_order       = 2^my_autoregressive_order
my_degree_of_P      = 0
my_size_degree_of_P = 2 * my_degree_of_P + 1
outdir = "./14PeriodicHMMSpatialBernoulli_CLEM/res_sim_data/"

# ── Data ──────────────────────────────────────────────────────────────────────
my_distance  = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
my_D = length(my_locations[:, 1])

Random.seed!(seed)
my_trans_θ     = 4 * (rand(my_K, my_K - 1, my_size_degree_of_P) .- 1 / 2)
my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
my_Bernoulli_θ[2, :, :, :] .+= 0.5
my_Range_θ     = (rand(my_K, my_size_degree_of_P) .- 1 / 2)
my_Range_θ[:, 1] = log.(100 .* (1:my_K))
my_a = fill(1 / my_K, my_K)

model  = Trig2PeriodicHMMspaMemory(my_a, my_trans_θ, my_Bernoulli_θ, my_Range_θ, my_T, my_distance)
z, Y   = my_rand(model, n2t; seq=true)
Y      = convert(Array{Bool}, Y)
Y_past = rand(Bool, my_autoregressive_order, my_D)

N, K, T, size_order, D = size(Y,1), size(model,1), size(model,3), size(model,4), size(model,2)
wp = 1.0 .* (model.h .< maximum(model.h) * tdist)
pairwise_indices2 = Tuple.(findall(wp .> 0))

println("Setup: K=$K, T=$T, N=$N, D=$D, npairs=$(length(pairwise_indices2))")

# ── compute_KL: numerically stable chain-rule formulation ─────────────────────
#
# KL(p(Z|Y) || p(Z|Y_ij)) via chain rule on the HMM joint:
#
#   = KL(p(Z_1|Y) || p(Z_1|Y_ij))
#   + sum_{n=1}^{N-1} E_{Z_n|Y}[ KL(p(Z_{n+1}|Z_n,Y) || p(Z_{n+1}|Z_n,Y_ij)) ]
#
#   = sum_k γ_1(k) log(γ_1(k)/γij_1(k))
#   + sum_{n,k,l} ξ_n(k,l) log( [ξ_n(k,l)/γ_n(k)] / [ξij_n(k,l)/γij_n(k)] )
#
# All quantities are probabilities over {1,...,K} → bounded, no cancellation.
#
function compute_KL(hmm, Y, Y_past, wp, pairwise_indices2;
                    n2t, QMC_m_full=100, QMC_m_pair=100)

    N, K, D, T = size(Y,1), size(hmm,1), size(hmm,2), size(hmm,3)
    lag_cat = conditional_to(Y, Y_past)

    # ── Situations (size_order==1 only here; extend if needed) ──────────────
    Situations = zeros(Int, 4, N, D, D)
    for n in 1:N, i in 1:D, j in 1:D
        Situations[1,n,i,j] = (Y[n,i]==1 && Y[n,j]==1) ? 1 : 0
        Situations[2,n,i,j] = (Y[n,i]==1 && Y[n,j]==0) ? 1 : 0
        Situations[3,n,i,j] = (Y[n,i]==0 && Y[n,j]==1) ? 1 : 0
        Situations[4,n,i,j] = (Y[n,i]==0 && Y[n,j]==0) ? 1 : 0
    end

    # ── Pairwise forward-backward → γij, ξij ────────────────────────────────
    LLij  = zeros(N, K, D, D)
    Iij   = zeros(4, K, D, D, T)
    αij   = zeros(N, K, D, D)
    βij   = zeros(N, K, D, D)
    γij   = zeros(N, K, D, D)
    ξij   = zeros(N, K, K, D, D)
    cij_f = zeros(N, D, D)   # forward only — not overwritten by backward
    cij_b = zeros(N, D, D)   # backward — separate

    loglikelihoods!(LLij, Iij, hmm, Situations, pairwise_indices2, QMC_m=QMC_m_pair; n2t)
    forwardlog!(αij, cij_f, hmm.a, hmm.A, LLij, pairwise_indices2; n2t=n2t)
    backwardlog!(βij, cij_b, hmm.a, hmm.A, LLij, pairwise_indices2; n2t=n2t)
    posteriors!(γij, αij, βij, pairwise_indices2)

    # ξij: joint smoothing probabilities for each pair
    Threads.@threads for n in 1:(N-1)
        t  = n2t[n]
        At = @view hmm.A[:, :, t]
        for (ii, jj) in pairwise_indices2
            ξ_pair = zeros(K, K)
            m = maximum(@view LLij[n+1, :, ii, jj])
            for i in 1:K, j in 1:K
                ξ_pair[i,j] = αij[n,i,ii,jj] * At[i,j] *
                               exp(LLij[n+1,j,ii,jj]-m) * βij[n+1,j,ii,jj]
            end
            Z = sum(ξ_pair)
            (Z == 0.0 || !isfinite(Z)) && continue
            for i in 1:K, j in 1:K
                ξij[n,i,j,ii,jj] = ξ_pair[i,j] / Z
            end
        end
    end

    # ── Full forward-backward → γ, ξ ────────────────────────────────────────
    LL   = zeros(N, K)
    Sigmat = zeros(D, D, T, K)
    @threads for t in 1:T
        for k in 1:K
            Sigmat[:,:,t,k] = matern.(hmm.h; range=hmm.R[k,t], sill=1, order=0.5)
        end
    end
    @threads for n in 1:N
        a_b = fill(-Inf, D); b_b = fill(Inf, D)
        fb  = zeros(D);      zv  = zeros(D)
        for k in 1:K
            t = n2t[n]
            fb .= quantile.(Normal(), hmm.B[CartesianIndex.(k, t, 1:D, lag_cat[n,:])])
            a_b .= ifelse.(Y[n,:] .== 0, fb, -Inf)
            b_b .= ifelse.(Y[n,:] .== 1, fb,  Inf)
            hy = mvnormcdf(zv, Sigmat[:,:,t,k], a_b, b_b; m=D*QMC_m_full)
            LL[n,k] = log(hy[1])
        end
    end

    α = zeros(N, K); β  = zeros(N, K); γ  = zeros(N, K)
    c = zeros(N);    cb = zeros(N)
    forwardlog!(α, c, hmm.a, hmm.A, LL; n2t=n2t)
    log_pY = sum(c)                          # save before backward can touch anything
    backwardlog!(β, cb, hmm.a, hmm.A, LL; n2t=n2t)
    posteriors!(γ, α, β)

    # ξ: joint smoothing probabilities for full model
    ξ = zeros(N, K, K)
    @inbounds for n in OneTo(N-1)
        t  = n2t[n]
        cc = 0.0
        for i in OneTo(K), j in OneTo(K)
            ξ[n,i,j] = α[n,i] * hmm.A[i,j,t] *
                        exp(LL[n+1,j] - vec_maximum(view(LL,n+1,:))) * β[n+1,j]
            cc += ξ[n,i,j]
        end
        for i in OneTo(K), j in OneTo(K)
            ξ[n,i,j] /= cc
        end
    end

    # ── KL via chain rule (numerically stable) ────────────────────────────────
    # KL(p(Z|Y) || p(Z|Y_ij))
    # = sum_k γ_1(k) log(γ_1(k)/γij_1(k))                        [Z_1 marginal]
    # + sum_{n,k,l} ξ_n(k,l) log([ξ_n(k,l)/γ_n(k)]/[ξij_n(k,l)/γij_n(k)])
    #                                                     [conditional transitions]
    #
    # All ratios are over {1,...,K} → no large-number cancellation.

    KLij = zeros(D, D)

    for (ii, jj) in pairwise_indices2

        # Z_1 marginal term
        for k in 1:K
            if γ[1,k] > 0 && γij[1,k,ii,jj] > 0
                KLij[ii,jj] += γ[1,k] * log(γ[1,k] / γij[1,k,ii,jj])
            end
        end

        # conditional transition terms
        for n in 1:N-1
            for k in 1:K, l in 1:K
                xf = ξ[n,k,l]
                xp = ξij[n,k,l,ii,jj]
                gf = γ[n,k]
                gp = γij[n,k,ii,jj]
                if xf > 0 && xp > 0 && gf > 0 && gp > 0
                    KLij[ii,jj] += xf * log((xf / gf) / (xp / gp))
                end
            end
        end
    end

    return KLij, γ, γij, ξ, ξij, log_pY
end

# ── Run ───────────────────────────────────────────────────────────────────────
println("\nComputing KL on true model...")
@time KLij, γ_full, γij_full, ξ_full, ξij_full, log_pY = compute_KL(
    model, Y, Y_past, wp, pairwise_indices2;
    n2t=n2t, QMC_m_full=100, QMC_m_pair=100)

kl_vals   = [KLij[ii,jj] for (ii,jj) in pairwise_indices2]
distances = [model.h[ii,jj] for (ii,jj) in pairwise_indices2]

println("\n=== KL Summary ===")
println("log(K)     = $(round(log(K),     digits=4))  [single-step max]")
println("N*log(K)   = $(round(N*log(K),   digits=4))  [sequence max]")
println("KL range   : [$(round(minimum(kl_vals),digits=4)), $(round(maximum(kl_vals),digits=4))]")
println("KL mean    : $(round(mean(kl_vals),digits=4))")
println("KL std     : $(round(std(kl_vals), digits=4))")
println("KL/N*log(K): $(round(mean(kl_vals)/(N*log(K))*100, digits=1))% of max on average")
println("Pairs with KL > 0.5*N*log(K): $(sum(kl_vals .> 0.5*N*log(K))) / $(length(kl_vals))")
println("Pairs with KL < 0.1*N*log(K): $(sum(kl_vals .< 0.1*N*log(K))) / $(length(kl_vals))")

# ── Plots ──────────────────────────────────────────────────────────────────────
tag = "K$(K)_T$(T)_N$(N)_D$(D)_seed$(seed)"

# Plot 1: KL vs distance
p1 = scatter(distances, kl_vals;
    xlabel = "Distance "*L"h_{ij}",
    ylabel = L"\mathrm{KL}(p(Z|Y)\,\|\,p(Z|Y_{ij}))",
    title  = "Information loss vs pair distance",
    label  = "", alpha=0.5, markersize=3)
hline!(p1, [N*log(K)]; linestyle=:dash, c=:red,  label="N·log(K) upper bound")

