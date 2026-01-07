using Pkg
Pkg.activate("HMMSPAcodepaper")
Pkg.instantiate()

begin
    include("../PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
    Random.seed!(0)
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
    using LaTeXStrings
    using Profile
    using BenchmarkTools
    using CSV
    using DataFrames
    include("../PeriodicHMMSpatialBernoulli/estimation_functions_BandR.jl")
end
# begin
#     include("/home/caroline/Gitlab_SWG_Caro/hmmspa/PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
#     Random.seed!(0)
#     # ## Utilities
#     using ArgCheck
#     using Base: OneTo
#     using ShiftedArrays: lead, lag
#     using Distributed
#     # ## Optimization
#     using JuMP, Ipopt
#     using Optimization, OptimizationMOI
#     using LsqFit
#     using Dates

#     # Random and Distributions
#     using Distributions
#     using Random: AbstractRNG, GLOBAL_RNG, rand!

#     # ## Special function
#     using LogExpFunctions: logsumexp!, logsumexp

#     # ## HMM
#     using PeriodicHiddenMarkovModels

#     using PeriodicHiddenMarkovModels: viterbi, istransmat, update_a!, vec_maximum
#     import PeriodicHiddenMarkovModels: forwardlog!, backwardlog!, viterbi, viterbi!, viterbilog!, posteriors!
#     import PeriodicHiddenMarkovModels: fit_mle!, fit_mle

#     # # Overloaded functions
#     import Distributions: fit_mle
#     import Base: rand
#     import Base: ==, copy, size
#     using Base.Threads

# using LaTeXStrings
# using Profile
# using BenchmarkTools
# using CSV
# using DataFrames
#     include("/home/caroline/Gitlab_SWG_Caro/hmmspa/PeriodicHMMSpatialBernoulli/estimation_functions_BandR.jl")
# end


# test --------------------------------------#
tdist = 0.3

my_K = 2# Number of Hidden states
my_T = 36 # Period

my_N = my_T * 10
n2t = n_to_t(my_N, my_T)




my_distance = Matrix(CSV.read("./data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./data/transformedECAD_locs.csv", DataFrame, header=false))
my_D = length(my_locations[:, 1])



p = scatter(my_locations[:, 1], my_locations[:, 2]);

my_autoregressive_order = 0

my_size_order = 2^my_autoregressive_order
my_degree_of_P = 1
my_size_degree_of_P = 2 * my_degree_of_P + 1

my_trans_θ = 4 * (rand(my_K, my_K - 1, my_size_degree_of_P) .- 1 / 2)
# parameters of the transition matrix. TO KEEP
# matrix of size K * (K-1) * (2degP+1) : parameters of the transition matrix for the K(K-1), for each of the trigo param.

if my_autoregressive_order == 0
my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
else

my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 1 / 2)
my_Bernoulli_θ[:, :, 1, :] .= my_Bernoulli_θ[:, :, 2, :]

my_Bernoulli_θ[:, :, 1, 1] .= min.(my_Bernoulli_θ[:, :, 2, 1] .+ 1, 1)
end
# rain probability parameters of the bernoulli Emissions
# K (states) * D (stations)  * 1+AR order (memory in the HMM) * (2degP+1) (each of the trigo param)
my_Range_θ = (rand(my_K, my_size_degree_of_P) .- 1 / 2)
my_Range_θ[:, 1] = log.(300 .*(1:my_K))
# range  parameters of the bernoulli Emissions 
# K (states)   * 1+AR order (memory in the HMM) * (2degP+1) (each of the trigo param) - range par

my_a = fill(1 / my_K, my_K)
model = Trig2PeriodicHMMspaMemory(my_a, my_trans_θ, my_Bernoulli_θ, my_Range_θ, my_T, my_distance);

# model of emission at state 1, time 1, and it rained before (?)

size(model)
model.B



z, Y = my_rand(model, n2t; seq=true)
Y = convert(Array{Bool}, Y)


#initial parameters in the choux
thetaA = rand(my_K, my_K - 1, my_size_degree_of_P)
thetaB = zeros(my_K, my_D, my_size_order, my_size_degree_of_P)
thetaR = zeros(my_K, my_size_degree_of_P)
thetaR = zeros(my_K, my_size_degree_of_P)

thetaA[:, :, 1] .= copy(my_trans_θ[:, :, 1]) # cheating on initial guess to recover very good mle maxima
thetaB[:, :, :, 1] = copy(my_Bernoulli_θ[:, :, :, 1])
thetaR[:, 1] .= copy(my_Range_θ[:, 1]) .-log(10)# cheating on initial guess to recover very good mle maxima
hmm = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)
p1, p2, p3 = PlotFitAndReal(model, hmm; indices_sta=1:2);
pp = plot(p1, p2, p3, layout=@layout [a b; c]; size=(1000, 1000))
savefig(pp,"./PeriodicHMMSpatialBernoulli/res_sim_data_BandR/pairwise_K"*string(my_K)*"_memory"*string(my_autoregressive_order)*"_T"*string(my_T)*"_Neq"*string(my_N)*"_D"*string(my_D)*"_t"*string(tdist)*"beforefit.png")


tol = 1e-4

D = size(my_locations, 1)
Y_past = rand(Bool, my_autoregressive_order, D)

using LaTeXStrings, Plots
println("Before estimation: ", thetaR)
using Profile
using BenchmarkTools


# with usual weights
@time begin
    history2, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations = fit_mle!(hmm, thetaA, thetaB, thetaR, Y, Y_past; n2t=n2t, maxiter=5, tol=tol, maxiters_R=100, display=:iter, tdist=tdist, QMC_m=30)
end
pp1 = plot(history2["logtots"])
p1, p2, p3 = PlotFitAndReal(hmm, model; indices_sta=1:2);


pp = plot(p1, p2, p3, layout=@layout [a b; c]; size=(1000, 1000))
savefig(pp,"./PeriodicHMMSpatialBernoulli/res_sim_data_BandR/pairwise_K"*string(my_K)*"_memory"*string(my_autoregressive_order)*"_T"*string(my_T)*"_Neq"*string(my_N)*"_D"*string(my_D)*"_t"*string(tdist)*".png")
