# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr>
# Pipeline: fit EM first, then use as starting point for CLEM
# Comparison: true model vs EM vs CLEM

begin
    seed = 13
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
    using Optim, LineSearches
    using OptimizationOptimJL
    using Plots
    using LaTeXStrings
    using Statistics: mean
end

# ── Hyperparameters ───────────────────────────────────────────────────────────
tdist        = 1.2
maxiter_EM   = 10      # max iterations for EM
maxiter_CLEM = 10    # max iterations for CLEM
maxiters_R   = 100       # inner optimiser iterations for R/B update
tol_EM       = 5e-3
tol_CLEM     = 1e-5
QMC_m_EM     = 100
QMC_m_CLEM   = 1000

my_K                  = 2
my_T                  = 3
my_N                  = my_T * 3000
my_autoregressive_order = 0
my_degree_of_P        = 0

n2t               = n_to_t(my_N, my_T)
my_size_order     = 2^my_autoregressive_order
my_size_degree_of_P = 2 * my_degree_of_P + 1

solver = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking())

# ── Data ──────────────────────────────────────────────────────────────────────
my_distance = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
my_D = length(my_locations[:, 1])
wp = 1.0 .* (my_distance .< tdist * maximum(my_distance))

# ── True model ────────────────────────────────────────────────────────────────
Random.seed!(seed)
my_trans_θ    = 4 * (rand(my_K, my_K - 1, my_size_degree_of_P) .- 1 / 2)
my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
my_Bernoulli_θ[2, :, :, :] .+= 0.5
my_Range_θ    = (rand(my_K, my_size_degree_of_P) .- 1 / 2)
my_Range_θ[:, 1] = log.(300 .* (1:my_K))
my_a          = fill(1 / my_K, my_K)

model_true = Trig2PeriodicHMMspaMemory(my_a, my_trans_θ, my_Bernoulli_θ, my_Range_θ, my_T, my_distance)
z, Y = my_rand(model_true, n2t; seq=true)
Y = convert(Array{Bool}, Y)
Y_past = rand(Bool, my_autoregressive_order, my_D)

println("Data generated: N=$(my_N), D=$(my_D), K=$(my_K), T=$(my_T)")

# ── Includes ──────────────────────────────────────────────────────────────────
include("../13PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl")
include("../14PeriodicHMMSpatialBernoulli_CLEM/estimation_functions_BandR_v3.jl")

# ── Step 1: smart initialisation via independent HMM (slice) ─────────────────
println("\n=== Step 1: Slice HMM initialisation ===")
ξ_init       = fill(1 / my_K, my_K)
ref_station  = 1
hmm_random   = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order; ξ=ξ_init, ref_station=ref_station)

@time "Slice fit" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past;
    n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1)

θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)

@time "Baum-Welch" hmm_bw, θq_bw, θy_bw, hist_bw, histo_A_bw, histo_B_bw = fit_mle(
    hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past;
    maxiter=10000, robust=true, display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)

# ── Step 2: EM fit (spatial, ) ─────────────────────────────────
println("\n=== Step 2: modifed EM fit  ===")
thetaA_em = copy(θq_bw)
thetaB_em = copy(θy_bw)
thetaR_em = zeros(my_K, my_size_degree_of_P)
thetaR_em[:, 1] .= log.(300 .* (1:my_K))

hmmEM = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA_em, thetaB_em, thetaR_em, my_T, my_distance)

@time "EM fit" history_EM, all_thetaA_EM, all_thetaB_EM, all_thetaR_EM = fit_mle!(
    hmmEM, thetaA_em, thetaB_em, thetaR_em, Y, Y_past;
    solver=solver, n2t=n2t, maxiter=maxiter_EM, tol=tol_EM,
    maxiters_R=maxiters_R, display=:iter, tdist=tdist, wp=wp, QMC_m=QMC_m_EM)

# reconstruct best EM model from iteration with highest logtot
iter_EM = argmax(history_EM["logtots"])
println("Best EM iteration: $iter_EM / $(length(history_EM["logtots"]))")

R_EM = [exp(mypolynomial_trigo(t, all_thetaR_EM[iter_EM+1][k, :], my_T)) for k in 1:my_K, t in 1:my_T]
B_EM = [1 / (1 + exp(polynomial_trigo(t, all_thetaB_EM[iter_EM+1][k, s, h, :], my_T)))
        for k in 1:my_K, t in 1:my_T, s in 1:my_D, h in 1:my_size_order]
if my_K == 1
    A_EM = ones(my_K, my_K, my_T)
else
    A_EM = zeros(my_K, my_K, my_T)
    for k in 1:my_K, l in 1:my_K-1, t in 1:my_T
        A_EM[k, l, t] = exp(polynomial_trigo(t, all_thetaA_EM[iter_EM+1][k, l, :], my_T))
    end
    for k in 1:my_K, t in 1:my_T
        A_EM[k, my_K, t] = 1
    end
    norm_A_EM = [1 + sum(A_EM[k, l, t] for l in 1:my_K-1) for k in 1:my_K, t in 1:my_T]
    for k in 1:my_K, l in 1:my_K, t in 1:my_T
        A_EM[k, l, t] /= norm_A_EM[k, t]
    end
end
hmmEM_best = PeriodicHMMSpaMemory(fill(1/my_K, my_K), A_EM, R_EM, B_EM, my_distance)

# ── Step 3: CLEM fit, starting from best EM parameters ───────────────────────
println("\n=== Step 3: CLEM fit (starting from best EM) ===")

# use best EM iteration parameters as starting point for CLEM
thetaA_clem = copy(all_thetaA_EM[iter_EM+1])
thetaB_clem = copy(all_thetaB_EM[iter_EM+1])
thetaR_clem = copy(all_thetaR_EM[iter_EM+1])

hmmCLEM = Trig2PeriodicHMMspaMemory(fill(1/my_K, my_K), thetaA_clem, thetaB_clem, thetaR_clem, my_T, my_distance)

@time "CLEM fit" history_CLEM, all_thetaA_CLEM, all_thetaB_CLEM, all_thetaR_CLEM = fit_mle_CLEM!(
    hmmCLEM, thetaA_clem, thetaB_clem, thetaR_clem, Y, Y_past;
    solver=solver, n2t=n2t, maxiter=maxiter_CLEM, tol=tol_CLEM,
    maxiters_R=maxiters_R, display=:iter, tdist=tdist, wp=wp, QMC_m=QMC_m_CLEM)

# reconstruct best CLEM model
iter_CLEM = argmax(history_CLEM["logtots"])
println("Best CLEM iteration: $iter_CLEM / $(length(history_CLEM["logtots"]))")

R_CLEM = [exp(mypolynomial_trigo(t, all_thetaR_CLEM[iter_CLEM+1][k, :], my_T)) for k in 1:my_K, t in 1:my_T]
B_CLEM = [1 / (1 + exp(polynomial_trigo(t, all_thetaB_CLEM[iter_CLEM+1][k, s, h, :], my_T)))
          for k in 1:my_K, t in 1:my_T, s in 1:my_D, h in 1:my_size_order]
if my_K == 1
    A_CLEM = ones(my_K, my_K, my_T)
else
    A_CLEM = zeros(my_K, my_K, my_T)
    for k in 1:my_K, l in 1:my_K-1, t in 1:my_T
        A_CLEM[k, l, t] = exp(polynomial_trigo(t, all_thetaA_CLEM[iter_CLEM+1][k, l, :], my_T))
    end
    for k in 1:my_K, t in 1:my_T
        A_CLEM[k, my_K, t] = 1
    end
    norm_A_CLEM = [1 + sum(A_CLEM[k, l, t] for l in 1:my_K-1) for k in 1:my_K, t in 1:my_T]
    for k in 1:my_K, l in 1:my_K, t in 1:my_T
        A_CLEM[k, l, t] /= norm_A_CLEM[k, t]
    end
end
hmmCLEM_best = PeriodicHMMSpaMemory(fill(1/my_K, my_K), A_CLEM, R_CLEM, B_CLEM, my_distance)

# ── Step 4: Plots ─────────────────────────────────────────────────────────────
println("\n=== Step 4: Plots ===")

tag = "K$(my_K)_mem$(my_autoregressive_order)_T$(my_T)_N$(my_N)_D$(my_D)_tdist$(tdist)_seed$(seed)"
outdir = "./14PeriodicHMMSpatialBernoulli_CLEM/res_sim_data/"

# -- convergence curves
pp_conv = plot(
    plot(history_EM["logtots"],   title="EM logtots",   xlabel="iteration", ylabel="logtot", legend=false),
    plot(history_CLEM["logtots"], title="CLEM logtots", xlabel="iteration", ylabel="logtot", legend=false);
    layout=(1,2), size=(1000, 400)
)
savefig(pp_conv, outdir * "convergence_" * tag * ".png")

# -- comparison function: true / EM / CLEM
function plot_comparison(hmm_true, hmm_em, hmm_clem; indices_sta=1:min(4, size(hmm_true, 2)))
    K = size(hmm_true, 1)
    T = size(hmm_true, 3)
    nsta = length(indices_sta)

    # Range R
    p_R = plot(title=L"\rho_{CY,k}^{(t)}", xlabel="t", legend=:topright)
    for k in 1:K
        plot!(p_R, 1:T, hmm_true.R[k, :], c=k, label=k==1 ? "True" : "")
        plot!(p_R, 1:T, hmm_em.R[k, :],   c=k, linestyle=:dash,   label=k==1 ? "EM" : "")
        plot!(p_R, 1:T, hmm_clem.R[k, :], c=k, linestyle=:dashdot, label=k==1 ? "CLEM" : "")
    end
    # dummy series for legend
    plot!(p_R, 1:T, fill(NaN,T), c=:black, label="True")
    plot!(p_R, 1:T, fill(NaN,T), c=:black, linestyle=:dash,    label="EM")
    plot!(p_R, 1:T, fill(NaN,T), c=:black, linestyle=:dashdot, label="CLEM")

    # Transition A
    pA = [plot(xlabel="t", title="State $k transitions") for k in 1:K]
    for k in 1:K
        for l in 1:K
            plot!(pA[k], 1:T, hmm_true.A[k,l,:], c=l, label=L"Q_{%$(k)\to%$(l)}")
            plot!(pA[k], 1:T, hmm_em.A[k,l,:],   c=l, linestyle=:dash,    label="")
            plot!(pA[k], 1:T, hmm_clem.A[k,l,:], c=l, linestyle=:dashdot, label="")
        end
        hline!(pA[k], [0.5], c=:black, s=:dot, label="")
        ylims!(pA[k], 0, 1)
    end
    pall_A = plot(pA..., size=(1000, 400))

    # Rain probability B for selected stations
    pB = [plot(xlabel="t", title=latexstring("\\lambda_{$(indices_sta[j])}^{(t)}")) for j in 1:nsta]
    for j in 1:nsta
        for k in 1:K
            plot!(pB[j], 1:T, [hmm_true.B[k,t,indices_sta[j],1] for t in 1:T], c=k, label="")
            plot!(pB[j], 1:T, [hmm_em.B[k,t,indices_sta[j],1]   for t in 1:T], c=k, linestyle=:dash,    label="")
            plot!(pB[j], 1:T, [hmm_clem.B[k,t,indices_sta[j],1] for t in 1:T], c=k, linestyle=:dashdot, label="")
        end
        hline!(pB[j], [0.5], c=:black, s=:dot, label="")
        ylims!(pB[j], 0, 1)
    end
    pall_B = plot(pB...)

    return p_R, pall_A, pall_B
end

p_R, pall_A, pall_B = plot_comparison(model_true, hmmEM_best, hmmCLEM_best; indices_sta=1:min(4, my_D))

savefig(plot(p_R;    size=(600, 400)), outdir * "range_comparison_"      * tag * ".png")
savefig(plot(pall_A; size=(1000,400)), outdir * "transition_comparison_" * tag * ".png")
savefig(plot(pall_B; size=(1000,400)), outdir * "rain_proba_comparison_" * tag * ".png")

pp_all = plot(p_R, pall_A, pall_B, layout=(1,3), size=(1500, 400))
savefig(pp_all, outdir * "full_comparison_" * tag * ".png")

println("\nDone. Figures saved to $outdir")
println("Tag: $tag")