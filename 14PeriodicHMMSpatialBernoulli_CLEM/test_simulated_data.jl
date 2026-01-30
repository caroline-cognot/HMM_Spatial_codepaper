
begin
    seed=12
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
pert = 0.5

# test --------------------------------------#
tdist = 1.2
maxiter = 1000
maxiterEM = 40
my_K = 2# Number of Hidden states
my_T = 3 # Period

my_N = my_T * 100
n2t = n_to_t(my_N, my_T)



my_distance = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
my_D = length(my_locations[:, 1])


# # with triangulation weigths
# using DelaunayTriangulation

# points = [
#     (my_locations[:, 1][j], my_locations[:, 2][j]) for j in 1:size(my_locations)[1]
# ];
# tri = triangulate(points);

# wp = zeros(size(my_locations)[1], size(my_locations)[1]);
# for i in 1:size(my_locations)[1]
    
#     set = get_neighbours(tri, i)
#     for j in set
#         if j > 0
#             wp[i, j] = 1.0;
#         end
#     end
# end
# wp;

wp = 1.0 .* (my_distance .< tdist*maximum(my_distance))
p = scatter(my_locations[:, 1], my_locations[:, 2]);

my_autoregressive_order = 0

my_size_order = 2^my_autoregressive_order
my_degree_of_P = 0
my_size_degree_of_P = 2 * my_degree_of_P + 1

my_trans_θ = 4 * (rand(my_K, my_K - 1, my_size_degree_of_P) .- 1 / 2)
# parameters of the transition matrix. TO KEEP
# matrix of size K * (K-1) * (2degP+1) : parameters of the transition matrix for the K(K-1), for each of the trigo param.

if my_autoregressive_order == 0
    my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
    my_Bernoulli_θ[2, :, :, :] .+= 0.5
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



z, Y = my_rand(model, n2t; seq=true)
Y = convert(Array{Bool}, Y)
Y_past = rand(Bool, my_autoregressive_order, my_D)








include("../PeriodicHMMSpatialBernoulli_CLEM/estimation_functions_BandR_v2.jl")

# test --------------------------------------#


# # make initial parameters set cleverly using independent HMM
begin
# ξ = [1; zeros(my_K - 1)]
ξ = fill(1/my_K,my_K)

ref_station = 1
    hmm_random = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order; ξ=ξ, ref_station=ref_station)
    @time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1,
    #  Yₜ_extanted=[-12, -7, 0, 6, 13]
     )
    θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)
    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
        maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)
    thetaA = θq_fit
    thetaB = θy_fit
    # now on to the next part : spatial model ----------------------------------------------------
    thetaR = zeros(my_K, my_size_degree_of_P)
    thetaR[:, 1] .= (log.(300 .* (1:my_K)))
    hmmCLEM = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)

end


thetaA_init=copy(thetaA)
thetaB_init=copy(thetaB)
thetaR_init=copy(thetaR)

hmm_init= Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA_init, thetaB_init, thetaR_init, my_T, my_distance)
# thetaA = copy(my_trans_θ).+(rand(size(my_trans_θ))*pert)
# thetaB= copy(my_Bernoulli_θ).+(rand(size(my_Bernoulli_θ))*pert)
#    thetaR = zeros(my_K, my_size_degree_of_P)
#  thetaR[:, 1] .= (log.(300 .* (1:my_K))).+(rand(size(thetaR[:,1]))*pert)

#  hmmCLEM = Trig2PeriodicHMMspaMemory(copy(my_a),thetaA, thetaB, thetaR, my_T, my_distance);
# hmm_init = Trig2PeriodicHMMspaMemory(copy(my_a), copy(thetaA), copy(thetaB), copy(thetaR), my_T, my_distance);

@time begin
    history2, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations = fit_mle_CLEM!(hmmCLEM, thetaA, thetaB, thetaR, Y, Y_past; n2t=n2t, maxiter=maxiter, tol=1e-5, maxiters_R=100, display=:iter, tdist=tdist,wp=wp,QMC_m=1000)
end
pp1 = plot(history2["logtots"])
savefig(pp1, "./PeriodicHMMSpatialBernoulli_CLEM/res_sim_data/logtotspairwise_K" * string(my_K) * "_memory" * string(my_autoregressive_order) * "_T" * string(my_T) * "_Neq" * string(my_N) * "_D" * string(my_D) * "_t" * string(tdist) * "maxiterCLEM_is_"*string(maxiter)*"seed"*string(seed)*".png")

logtots = history2["logtots"]

                iter=argmax(logtots)

                    R_all = [exp(mypolynomial_trigo(t, all_thetaR_iterations[iter+1][k, :]', my_T)) for k in 1:my_K, t = 1:my_T]
                    B_all = [1 / (1 + exp(polynomial_trigo(t, all_thetaB_iterations[iter+1][k, s, h, :], my_T))) for k = 1:my_K, t = 1:my_T, s = 1:my_D, h = 1:my_size_order] 
                    
                    if my_K == 1
                        all_A = ones(my_K, my_K, my_T)
                    else
                        all_A = zeros(my_K, my_K, my_T)
                            for k = 1:my_K, l = 1:my_K-1, t = 1:my_T
                                all_A[ k, l, t] = exp(polynomial_trigo(t, all_thetaA_iterations[iter+1][k, l, :], my_T))
                            end
                            for k = 1:my_K, t = 1:my_T
                                all_A[ k, my_K, t] = 1  # last colum is 1/normalization (one could do otherwise)
                            end
                            normalization_polynomial = [1 + sum(all_A[k, l, t] for l = 1:my_K-1) for k = 1:my_K, t = 1:my_T]
                            for k = 1:my_K, l = 1:my_K, t = 1:my_T
                                all_A[ k, l, t] /= normalization_polynomial[k, t]
                            end
                    end
                    hmmCLEM2= PeriodicHMMSpaMemory(fill(1/my_K,my_K), all_A[ :, :, :], R_all, B_all, my_distance)
                 

## compare with non-pairwise 

include("../13PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl")
begin
    thetaA = copy(thetaA_init)
    thetaB = copy(thetaB_init)


    # now on to the next part : spatial model ----------------------------------------------------
    thetaR = zeros(my_K, my_size_degree_of_P)

    thetaR[:, 1] .=reverse(log.(300 .* (1:my_K)))
    hmmEM = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)
end

# thetaA = copy(my_trans_θ).+(rand(size(my_trans_θ))*pert)
# thetaB= copy(my_Bernoulli_θ).+(rand(size(my_Bernoulli_θ))*pert)
#    thetaR = zeros(my_K, my_size_degree_of_P)
#  thetaR[:, 1] .= (log.(300 .* (1:my_K))).+(rand(size(thetaR[:,1]))*pert)

#  hmmEM = Trig2PeriodicHMMspaMemory(copy(my_a),thetaA, thetaB, thetaR, my_T, my_distance);

@time begin
    history2, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations = fit_mle!(hmmEM, thetaA, thetaB, thetaR, Y, Y_past; n2t=n2t, maxiter=minimum([100,maxiter,maxiterEM]), tol=5*1e-3, maxiters_R=100, display=:iter, tdist=tdist, wp=wp,QMC_m=100)
end
pp1 = plot(history2["logtots"])
savefig(pp1, "./PeriodicHMMSpatialBernoulli_CLEM/res_sim_data/logtotsfull_K" * string(my_K) * "_memory" * string(my_autoregressive_order) * "_T" * string(my_T) * "_Neq" * string(my_N) * "_D" * string(my_D) * "_t" * string(tdist) * ".png")

function PlotCLEMandEMandreal(hmmCLEM::PeriodicHMMSpaMemory,hmmEM::PeriodicHMMSpaMemory,model; indices_sta=1:size(hmm, 2))
    K = size(hmmCLEM, 1)
    d = size(hmmCLEM, 2)
    T = size(hmmCLEM, 3)
    nsta = length(indices_sta)
    # plot the ranges
    p1 = plot(title=L" \rho_{CY,k}^{(t)}",xlabel="t")
    for k in 1:K
        plot!(p1, 1:T, hmmCLEM.R[k, :], linestyle=:dash, label=:none, c=k)
        plot!(p1, 1:T, hmmEM.R[k, :], linestyle=:dashdot,label=:none,c=k)
        plot!(p1, 1:T, model.R[k, :],label=:none,c=k)

    end
    plot!(p1,1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
    plot!(p1,1:T,fill(NaN,T),label="EM",linestyle=:dashdot,c=:black)
    plot!(p1,1:T,fill(NaN,T),label="Real",c=:black)

    # plot the transotion parameters
    begin
        pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing,xlabel="t") for k in 1:K]
        for k in 1:K
            [plot!(pA[k], hmmCLEM.A[k, l, :],linestyle=:dash, c=l, label=L"Q_{%$(k)\to %$(l)}^{(t)}", legend=:topleft) for l in 1:K]
            [plot!(pA[k], hmmEM.A[k, l, :],linestyle=:dashdot, c=l, label="", legend=:topleft) for l in 1:K]
            [plot!(pA[k], model.A[k, l, :], c=l, label="", legend=:topleft) for l in 1:K]
            hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
            ylims!(0, 1)
            # plot!(pA[k],1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
            # plot!(pA[k],1:T,fill(NaN,T),label="EM",linestyle=:dot,c=:black)
            # plot!(pA[k],1:T,fill(NaN,T),label="Real",c=:black)
        
          
        end
        pallA = plot(pA..., size=(1000, 500))
    end

    # plot the proba of rain for all stations

    begin
        pB = [plot(xlabel="t") for j in 1:nsta]
        for j in 1:nsta
            [plot!(pB[j], [hmmCLEM.B[k, t, indices_sta[j], 1] for t in 1:T],linestyle=:dash, c=k, label=:none) for k in 1:K]
            [plot!(pB[j], [hmmEM.B[k, t, indices_sta[j], 1] for t in 1:T],linestyle=:dashdot, c=k, label=:none) for k in 1:K]
            [plot!(pB[j], [model.B[k, t, indices_sta[j], 1] for t in 1:T], c=k, label=:none) for k in 1:K]


            hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
            ylims!(pB[j], (0, 1))
            title!(pB[j], latexstring("\\lambda_{$(indices_sta[j])}^{(t)}"))
            # # Add dummy plots just for the legend
            # plot!(pB[j],1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
            # plot!(pB[j],1:T,fill(NaN,T),label="EM",linestyle=:dot,c=:black)
            # plot!(pB[j],1:T,fill(NaN,T),label="Real",c=:black)
        
    
        end

       
        pallB = plot(pB...)
    end


    return p1, pallA, pallB


end

p1, p2, p3 = PlotCLEMandEMandreal(hmmCLEM,hmmEM, model; indices_sta=1:4);
pp = plot(p1, p2, p3, layout=(1,3); size=(1000, 400))
savefig(pp, "./PeriodicHMMSpatialBernoulli_CLEM/res_sim_data/CLEMBR_VS_EM"* string(my_degree_of_P) *"deg" * string(my_K) * "_memory" * string(my_autoregressive_order) * "_T" * string(my_T) * "_Neq" * string(my_N) * "_D" * string(my_D) * "_t" * string(tdist) * "maxiterCLEM_is_"*string(maxiter)*"seed"*string(seed)*".png")

function PlotCLEMandCLEMandreal(hmmCLEM::PeriodicHMMSpaMemory,hmmEM::PeriodicHMMSpaMemory,model; indices_sta=1:size(hmm, 2))
    K = size(hmmCLEM, 1)
    d = size(hmmCLEM, 2)
    T = size(hmmCLEM, 3)
    nsta = length(indices_sta)
    # plot the ranges
    p1 = plot(title=L" \rho_{CY,k}^{(t)}",xlabel="t")
    for k in 1:K
        plot!(p1, 1:T, hmmCLEM.R[k, :], linestyle=:dash, label=:none, c=k)
        plot!(p1, 1:T, hmmEM.R[k, :], linestyle=:dashdot,label=:none,c=k)
        plot!(p1, 1:T, model.R[k, :],label=:none,c=k)

    end
    plot!(p1,1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
    plot!(p1,1:T,fill(NaN,T),label="best iteration in CLEM",linestyle=:dashdot,c=:black)
    plot!(p1,1:T,fill(NaN,T),label="Real",c=:black)

    # plot the transotion parameters
    begin
        pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing,xlabel="t") for k in 1:K]
        for k in 1:K
            [plot!(pA[k], hmmCLEM.A[k, l, :],linestyle=:dash, c=l, label=L"Q_{%$(k)\to %$(l)}^{(t)}", legend=:topleft) for l in 1:K]
            [plot!(pA[k], hmmEM.A[k, l, :],linestyle=:dashdot, c=l, label="", legend=:topleft) for l in 1:K]
            [plot!(pA[k], model.A[k, l, :], c=l, label="", legend=:topleft) for l in 1:K]
            hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
            ylims!(0, 1)
            # plot!(pA[k],1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
            # plot!(pA[k],1:T,fill(NaN,T),label="EM",linestyle=:dot,c=:black)
            # plot!(pA[k],1:T,fill(NaN,T),label="Real",c=:black)
        
          
        end
        pallA = plot(pA..., size=(1000, 500))
    end

    # plot the proba of rain for all stations

    begin
        pB = [plot(xlabel="t") for j in 1:nsta]
        for j in 1:nsta
            [plot!(pB[j], [hmmCLEM.B[k, t, indices_sta[j], 1] for t in 1:T],linestyle=:dash, c=k, label=:none) for k in 1:K]
            [plot!(pB[j], [hmmEM.B[k, t, indices_sta[j], 1] for t in 1:T],linestyle=:dashdot, c=k, label=:none) for k in 1:K]
            [plot!(pB[j], [model.B[k, t, indices_sta[j], 1] for t in 1:T], c=k, label=:none) for k in 1:K]


            hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
            ylims!(pB[j], (0, 1))
            title!(pB[j], latexstring("\\lambda_{$(indices_sta[j])}^{(t)}"))
            # # Add dummy plots just for the legend
            # plot!(pB[j],1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
            # plot!(pB[j],1:T,fill(NaN,T),label="EM",linestyle=:dot,c=:black)
            # plot!(pB[j],1:T,fill(NaN,T),label="Real",c=:black)
        
    
        end

       
        pallB = plot(pB...)
    end


    return p1, pallA, pallB


end
p1, p2, p3 = PlotCLEMandCLEMandreal(hmmCLEM,hmmCLEM2, model; indices_sta=1:4);
pp = plot(p1, p2, p3, layout=(1,3); size=(1000, 400))
savefig(pp, "./PeriodicHMMSpatialBernoulli_CLEM/res_sim_data/CLEMBR_VS_CLEMBRbest"* string(my_degree_of_P) *"deg" * string(my_K) * "_memory" * string(my_autoregressive_order) * "_T" * string(my_T) * "_Neq" * string(my_N) * "_D" * string(my_D) * "_t" * string(tdist) * "maxiterCLEM_is_"*string(maxiter)*"seed"*string(seed)*".png")

function PlotCLEMandinitandreal(hmmCLEM::PeriodicHMMSpaMemory,hmmEM::PeriodicHMMSpaMemory,model; indices_sta=1:size(hmm, 2))
    K = size(hmmCLEM, 1)
    d = size(hmmCLEM, 2)
    T = size(hmmCLEM, 3)
    nsta = length(indices_sta)
    # plot the ranges
    p1 = plot(title=L" \rho_{CY,k}^{(t)}",xlabel="t")
    for k in 1:K
        plot!(p1, 1:T, hmmCLEM.R[k, :], linestyle=:dash, label=:none, c=k)
        plot!(p1, 1:T, hmmEM.R[k, :], linestyle=:dashdot,label=:none,c=k)
        plot!(p1, 1:T, model.R[k, :],label=:none,c=k)

    end
    plot!(p1,1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
    plot!(p1,1:T,fill(NaN,T),label="intial guess",linestyle=:dashdot,c=:black)
    plot!(p1,1:T,fill(NaN,T),label="Real",c=:black)

    # plot the transotion parameters
    begin
        pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing,xlabel="t") for k in 1:K]
        for k in 1:K
            [plot!(pA[k], hmmCLEM.A[k, l, :],linestyle=:dash, c=l, label=L"Q_{%$(k)\to %$(l)}^{(t)}", legend=:topleft) for l in 1:K]
            [plot!(pA[k], hmmEM.A[k, l, :],linestyle=:dashdot, c=l, label="", legend=:topleft) for l in 1:K]
            [plot!(pA[k], model.A[k, l, :], c=l, label="", legend=:topleft) for l in 1:K]
            hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
            ylims!(0, 1)
            # plot!(pA[k],1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
            # plot!(pA[k],1:T,fill(NaN,T),label="EM",linestyle=:dot,c=:black)
            # plot!(pA[k],1:T,fill(NaN,T),label="Real",c=:black)
        
          
        end
        pallA = plot(pA..., size=(1000, 500))
    end

    # plot the proba of rain for all stations

    begin
        pB = [plot(xlabel="t") for j in 1:nsta]
        for j in 1:nsta
            [plot!(pB[j], [hmmCLEM.B[k, t, indices_sta[j], 1] for t in 1:T],linestyle=:dash, c=k, label=:none) for k in 1:K]
            [plot!(pB[j], [hmmEM.B[k, t, indices_sta[j], 1] for t in 1:T],linestyle=:dashdot, c=k, label=:none) for k in 1:K]
            [plot!(pB[j], [model.B[k, t, indices_sta[j], 1] for t in 1:T], c=k, label=:none) for k in 1:K]


            hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
            ylims!(pB[j], (0, 1))
            title!(pB[j], latexstring("\\lambda_{$(indices_sta[j])}^{(t)}"))
            # # Add dummy plots just for the legend
            # plot!(pB[j],1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
            # plot!(pB[j],1:T,fill(NaN,T),label="EM",linestyle=:dot,c=:black)
            # plot!(pB[j],1:T,fill(NaN,T),label="Real",c=:black)
        
    
        end

       
        pallB = plot(pB...)
    end


    return p1, pallA, pallB


end
p1, p2, p3 = PlotCLEMandinitandreal(hmmCLEM,hmm_init, model; indices_sta=1:4);
pp = plot(p1, p2, p3, layout=(1,3); size=(1000, 400))
savefig(pp, "./PeriodicHMMSpatialBernoulli_CLEM/res_sim_data/CLEMBR_VS_CLEMInit"* string(my_degree_of_P) *"deg" * string(my_K) * "_memory" * string(my_autoregressive_order) * "_T" * string(my_T) * "_Neq" * string(my_N) * "_D" * string(my_D) * "_t" * string(tdist) * "maxiterCLEM_is_"*string(maxiter)*"seed"*string(seed)*".png")

function PlotCLEMandreal(hmmCLEM::PeriodicHMMSpaMemory,model; indices_sta=1:size(hmm, 2))
    K = size(hmmCLEM, 1)
    d = size(hmmCLEM, 2)
    T = size(hmmCLEM, 3)
    nsta = length(indices_sta)
    # plot the ranges
    p1 = plot(title=L" \rho_{CY,k}^{(t)}",xlabel="t")
    for k in 1:K
        plot!(p1, 1:T, hmmCLEM.R[k, :], linestyle=:dash, label=:none, c=k)
        plot!(p1, 1:T, model.R[k, :],label=:none,c=k)

    end
    plot!(p1,1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
    plot!(p1,1:T,fill(NaN,T),label="Real",c=:black)

    # plot the transotion parameters
    begin
        pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing,xlabel="t") for k in 1:K]
        for k in 1:K
            [plot!(pA[k], hmmCLEM.A[k, l, :],linestyle=:dash, c=l, label=L"Q_{%$(k)\to %$(l)}^{(t)}", legend=:topleft) for l in 1:K]
            [plot!(pA[k], model.A[k, l, :], c=l, label="", legend=:topleft) for l in 1:K]
            hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
            ylims!(0, 1)
            # plot!(pA[k],1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
            # plot!(pA[k],1:T,fill(NaN,T),label="EM",linestyle=:dot,c=:black)
            # plot!(pA[k],1:T,fill(NaN,T),label="Real",c=:black)
        
          
        end
        pallA = plot(pA..., size=(1000, 500))
    end

    # plot the proba of rain for all stations

    begin
        pB = [plot(xlabel="t") for j in 1:nsta]
        for j in 1:nsta
            [plot!(pB[j], [hmmCLEM.B[k, t, indices_sta[j], 1] for t in 1:T],linestyle=:dash, c=k, label=:none) for k in 1:K]
            [plot!(pB[j], [model.B[k, t, indices_sta[j], 1] for t in 1:T], c=k, label=:none) for k in 1:K]


            hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
            ylims!(pB[j], (0, 1))
            title!(pB[j], latexstring("\\lambda_{$(indices_sta[j])}^{(t)}"))
            # # Add dummy plots just for the legend
            # plot!(pB[j],1:T,fill(NaN,T),label="CLEM",linestyle=:dash,c=:black)
            # plot!(pB[j],1:T,fill(NaN,T),label="EM",linestyle=:dot,c=:black)
            # plot!(pB[j],1:T,fill(NaN,T),label="Real",c=:black)
        
    
        end

       
        pallB = plot(pB...)
    end


    return p1, pallA, pallB


end
p1, p2, p3 = PlotCLEMandreal(hmmCLEM, model; indices_sta=1:4);
pp = plot(p1, p2, p3, layout=(1,3); size=(1000, 400))
savefig(pp, "./PeriodicHMMSpatialBernoulli_CLEM/res_sim_data/CLEM_vs_REAL"* string(my_degree_of_P) *"deg" * string(my_K) * "_memory" * string(my_autoregressive_order) * "_T" * string(my_T) * "_Neq" * string(my_N) * "_D" * string(my_D) * "_t" * string(tdist) * "maxiterCLEM_is_"*string(maxiter)*"seed"*string(seed)*".png")
