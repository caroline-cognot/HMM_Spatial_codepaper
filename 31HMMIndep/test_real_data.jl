# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
# This code tries to fit a HMM-indep model with K=4 for many stations, more than David's paper.
# The goal is to show that when looking at less stations, it works better. 
# And that adding spatial dependence is key when adding more stations.


# ## Utilities
using ArgCheck
using Base: OneTo
using ShiftedArrays: lead, lag
using Distributed
# ## Optimization
using JuMP, Ipopt
using Optimization, OptimizationMOI
using LsqFit


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
using SmoothPeriodicStatsModels # Name might change. Small collection of smooth periodic models e.g. AR, HMM
using StochasticWeatherGenerators # interface to use with SmoothPeriodicStatsModels.jl

# # Overloaded functions
import Distributions: fit_mle
import Base: rand
import Base: ==, copy, size
using Base.Threads

import StochasticWeatherGenerators.dayofyear_Leap

include("../13PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
include("../13PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl")

Random.seed!(0)

# test --------------------------------------#
tdist = 0.3

my_K = 4 # Number of Hidden states
my_T = 366 # Period

my_autoregressive_order = 0

my_size_order = 2^my_autoregressive_order
my_degree_of_P = 0
my_size_degree_of_P = 2 * my_degree_of_P + 1

maxiter = 15
using LaTeXStrings, Plots
using CSV
using DataFrames
using JLD2
using Dates


########################################
# real data #
########################################


md"""
Compare real data to simulated data : get the real data.
"""


d = CSV.read("00data/transformedECAD_Yobs.csv", DataFrame, header=false)
Yobs = transpose(Matrix(d))
my_distance = Matrix(CSV.read("00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("00data/transformedECAD_locs.csv", DataFrame, header=false))
my_D = size(my_locations, 1)
select_month = function (m::Int64, dates, Y::AbstractMatrix)
    indicesm = findall(month.(dates) .== m)
    return Y[:, indicesm]
end



date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end
my_N = length(every_year)

n2t = dayofyear_Leap.(every_year)
pp = [plot() for k in 1:6]

for my_K in 1:6

    Yall = convert(Array{Bool}, Yobs)
    Y_past = rand(Bool, my_autoregressive_order, my_D)
    ξ = [1; zeros(my_K - 1)]  # 1 jan 1956 was most likely a type Z = 1 wet day all over France
    Y = Yall[1+my_autoregressive_order:end, :]
    ref_station = 1

    hmm_random = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order; ξ=ξ, ref_station=ref_station)

    @time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13])

    θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
        maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)



    Nb = 100
    nlocs = length(my_locations[:, 1])
    begin
        Ys = zeros(Bool, nlocs, my_N, Nb)
        @time "Simulations  Y" for i in 1:Nb
            z, Y = rand(hmm_fit, n2t; seq=true)
            Ys[:, :, i] = Y'
        end

    end


    p1 = compare_ROR_histogram(Yobs', Ys)
    plot!(p1, title="HMM 37")

    # less stations 
    indices_david = [1, 2, 19, 6, 14, 17, 34, 32, 37]
    D2 = length(indices_david)
    Yall2 = Yall[:, indices_david]
    Y_past = rand(Bool, my_autoregressive_order, D2)
    ξ = [1; zeros(my_K - 1)]  # 1 jan 1956 was most likely a type Z = 1 wet day all over France
    Y = Yall2[1+my_autoregressive_order:end, :]
    ref_station = 1

    hmm_random = randARPeriodicHMM(my_K, my_T, D2, my_autoregressive_order; ξ=ξ, ref_station=ref_station)

    @time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13])

    θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
        maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)



    Nb = 100
    nlocs = D2
    begin
        Ys = zeros(Bool, nlocs, my_N, Nb)
        @time "Simulations  Y" for i in 1:Nb
            z, Y = rand(hmm_fit, n2t; seq=true)
            Ys[:, :, i] = Y'
        end

    end


    p2 = compare_ROR_histogram(Yall2', Ys)
    plot!(p2, title="HMM - $D2 stations proches david")

    plot(p1, p2)

    # less stations 
    indices_david_augment = [1, 2, 19, 6, 14, 17, 34, 32, 37, 3, 4, 5, 8]
    D2 = length(indices_david_augment)
    Yall2 = Yall[:, indices_david_augment]
    Y_past = rand(Bool, my_autoregressive_order, D2)
    ξ = [1; zeros(my_K - 1)]  # 1 jan 1956 was most likely a type Z = 1 wet day all over France
    Y = Yall2[1+my_autoregressive_order:end, :]
    ref_station = 1

    hmm_random = randARPeriodicHMM(my_K, my_T, D2, my_autoregressive_order; ξ=ξ, ref_station=ref_station)

    @time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13])

    θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
        maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)



    Nb = 100
    nlocs = D2
    begin
        Ys = zeros(Bool, nlocs, my_N, Nb)
        @time "Simulations  Y" for i in 1:Nb
            z, Y = rand(hmm_fit, n2t; seq=true)
            Ys[:, :, i] = Y'
        end

    end


    p3 = compare_ROR_histogram(Yall2', Ys)
    plot!(p3, title="HMM - $D2 ")


    # less stations 
    indices_david_augment = [1, 19, 6, 14, 34, 37]
    D2 = length(indices_david_augment)
    Yall2 = Yall[:, indices_david_augment]
    Y_past = rand(Bool, my_autoregressive_order, D2)
    ξ = [1; zeros(my_K - 1)]  # 1 jan 1956 was most likely a type Z = 1 wet day all over France
    Y = Yall2[1+my_autoregressive_order:end, :]
    ref_station = 1

    hmm_random = randARPeriodicHMM(my_K, my_T, D2, my_autoregressive_order; ξ=ξ, ref_station=ref_station)

    @time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13])

    θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
        maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)



    Nb = 100
    nlocs = D2
    begin
        Ys = zeros(Bool, nlocs, my_N, Nb)
        @time "Simulations  Y" for i in 1:Nb
            z, Y = rand(hmm_fit, n2t; seq=true)
            Ys[:, :, i] = Y'
        end

    end


    p4 = compare_ROR_histogram(Yall2', Ys)
    plot!(p4, title="HMM - $D2 ")






    pp[my_K] = plot(p1, p2, p3, p4, size=(1000, 1000))
end

for k in 1:6
    plot!(pp[k], suptitle="K=$k")
end

import Measures: mm
# Combine with margins
finalplot = plot(pp...,
    layout=(2, 3),
    size=(6000, 6000),
    left_margin=10mm,
    right_margin=10mm,
    top_margin=15mm,
    bottom_margin=10mm,
    plot_spacing=5mm,   # spacing between subplots
)

savefig(finalplot, "./31HMMIndep/trysomeKandD.png")