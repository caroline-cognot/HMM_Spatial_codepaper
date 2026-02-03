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

using LaTeXStrings
using CSV, DataFrames
import Random
using CairoMakie
CairoMakie.activate!()
using NaNMath, StatsBase
using GeoMakie
using NaturalEarth
using Distributions, LinearAlgebra
using TruncatedMVN
using Base: OneTo
using Printf
using JLD2
using Dates
# import StochasticWeatherGenerators.dayofyear_Leap
# using StochasticWeatherGenerators
using RollingFunctions
using MvNormalCDF
using BesselK
using ExtendedExtremes
using Distances: haversine

# test --------------------------------------#


using Colors
mycolors = RGB{Float64}[RGB(0.0, 0.6056031704619725, 0.9786801190138923), RGB(0.24222393333911896, 0.6432750821113586, 0.304448664188385), RGB(0.7644400000572205, 0.4441118538379669, 0.8242975473403931), RGB(0.8888735440600661, 0.435649148506399, 0.2781230452972766), RGB(0.6755439043045044, 0.5556622743606567, 0.09423444420099258), RGB(0.0, 0.6657590270042419, 0.6809969544410706), RGB(0.9307674765586853, 0.3674771189689636, 0.5757699012756348), RGB(0.776981770992279, 0.5097429752349854, 0.14642538130283356), RGB(5.29969987894674e-8, 0.6642677187919617, 0.5529508590698242), RGB(0.558464765548706, 0.59348464012146, 0.11748137325048447), RGB(0.0, 0.6608786582946777, 0.7981787919998169), RGB(0.609670877456665, 0.49918484687805176, 0.9117812514305115), RGB(0.38000133633613586, 0.5510532855987549, 0.9665056467056274), RGB(0.9421815872192383, 0.3751642107963562, 0.4518167972564697), RGB(0.8684020638465881, 0.39598923921585083, 0.7135148048400879), RGB(0.4231467843055725, 0.6224954128265381, 0.19877080619335175)];

my_T = 366 # Period

my_autoregressive_order = 0 

my_size_order = 2^my_autoregressive_order
my_degree_of_P = 0
my_size_degree_of_P = 2 * my_degree_of_P + 1

maxiter  = 15
using LaTeXStrings, Plots
using CSV
using DataFrames
using JLD2
using Dates



d = CSV.read("./00data/transformedECAD_Yobs.csv", DataFrame, header=false)
Yobs = transpose(Matrix(d))
my_distance = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
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


for my_K in [1,2,4,9]

    Yall = convert(Array{Bool}, Yobs)
    Y_past = rand(Bool, my_autoregressive_order, my_D)
    ξ = [1; zeros(my_K - 1)]  # 1 jan 1956 was most likely a type Z = 1 wet day all over France
    Y = Yall[1+my_autoregressive_order:end, :]
    ref_station = 1

    hmm_random = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order; ξ=ξ, ref_station=ref_station)

    @time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13])

    θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, history, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
        maxiter=1000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)

    save("./31HMMIndep/D37_"*string(my_K)*".jld2", Dict("hmm" => hmm_fit))


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

    @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, history, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
        maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)
        save("./31HMMIndep/D9_"*string(my_K)*".jld2", Dict("hmm" => hmm_fit))



end

function simulate_model(path, Nb, n2t)
    data = JLD2.load(path)
    hmm = data["hmm"]

    D = size(hmm, 2)   # number of locations
    T = length(n2t)

    Ys = zeros(Bool, D, T, Nb)
    for i in 1:Nb
        z, Y = rand(hmm, n2t; seq=true)
        Ys[:, :, i] = Y'
    end
    return Ys
end
indices_david = [1, 2, 19, 6, 14, 17, 34, 32, 37]

Yobs9 = Yobs[:, indices_david]

paths_37 = sort(filter(p -> occursin("D37", p), readdir("./31HMMIndep"; join=true)))
paths_9  = sort(filter(p -> occursin("D9",  p), readdir("./31HMMIndep"; join=true)))

Ys_37 = Dict()
Ys_9  = Dict()

Nb = 100

for p in paths_37
    Ys_37[p] = simulate_model(p, Nb, n2t)
end

for p in paths_9
    Ys_9[p] = simulate_model(p, Nb, n2t)
end
Ys_37[paths_37[4]]

function ror_stats(Ys, RRmax)
    Nb = size(Ys,3)
    return [ [mean(col .> RRmax) for col in eachcol(Ys[:,:,b])] for b in 1:Nb ]
end

RRmax = 0.
RORo37 = [mean(r .> RRmax) for r in eachrow(Yobs)]
RORo9 = [mean(r .> RRmax) for r in eachrow(Yobs9)]

ROR_37 = Dict()
for p in keys(Ys_37)
    ROR_37[p] = ror_stats(Ys_37[p], RRmax)
end

ROR_9 = Dict()
for p in keys(Ys_9)
    ROR_9[p] = ror_stats(Ys_9[p], RRmax)
end

ROR_37[paths_37[1]][1]


using CairoMakie
include("../41Plots_for_paper/utilities.jl")
begin
    # Makie ROR distribution and autocorrelation (2×4 grid)

    fig_ROR = Figure(fontsize=19)
    wwww = 220
    hhhh = 150
    # Row 1: Distribution plots
        row = 1
        col=1
        # Distribution subplot
        ax_dist = Axis(fig_ROR[row,col],
        xlabel = L"\mathrm{ROR}",
        ylabel=col == 1 ? "Distribution" : "",
            title="D=37",
            width=wwww,
            height=hhhh)
        xax = 0:(1/37):1.0
        xaxbin = vcat(xax, [1.01])
      
        errorlinehist!(ax_dist, [ROR_37[paths_37[1]][i] for i in 1:Nb];
            label="",
            color=mycolors[1],
            secondarycolor=mycolors[1], normalization=:probability,
            bins=xaxbin,
            errortype=:percentile,
            percentiles=[0, 100],
            alpha=0.5,
            secondaryalpha=0.2,
            centertype=:median)
        errorlinehist!(ax_dist, [ROR_37[paths_37[2]][i]  for i in 1:Nb];
            label="",
            color=mycolors[2],
            secondarycolor=mycolors[2],
            normalization=:probability,
            bins=xaxbin,
            errortype=:percentile,
            percentiles=[0, 100],
            alpha=0.5,
            secondaryalpha=0.2,
            centertype=:median)
 
        errorlinehist!(ax_dist, [ROR_37[paths_37[3]][i] for i in 1:Nb];
            label="",
            color=mycolors[3],
            secondarycolor=mycolors[3],
            normalization=:probability,
            bins=xaxbin,
            errortype=:percentile,
            percentiles=[0, 100],
            alpha=0.5,
            secondaryalpha=0.2,
            centertype=:median)
            errorlinehist!(ax_dist, [ROR_37[paths_37[4]][i] for i in 1:Nb];
            label="",
            color=mycolors[4],
            secondarycolor=mycolors[4],
            normalization=:probability,
            bins=xaxbin,
            errortype=:percentile,
            percentiles=[0, 100],
            alpha=0.5,
            secondaryalpha=0.2,
            centertype=:median)
        Makie.scatter!(ax_dist, xax, [mean(RORo37 .== x) for x in xax], color=:blue, markersize=6, label= "Observations")

 # Row 1: Distribution plots
 row = 1
 col=2
 # Distribution subplot
 ax_dist = Axis(fig_ROR[row,col],
 xlabel = L"\mathrm{ROR}",
 ylabel=col == 1 ? "Distribution" : "",
     title="D=9",
     width=wwww,
     height=hhhh)
 xax = 0:(1/9):1.0
 xaxbin = vcat(xax, [1.01])

 errorlinehist!(ax_dist, [ROR_9[paths_9[1]][i] for i in 1:Nb];
     label="",
     color=mycolors[1],
     secondarycolor=mycolors[1], normalization=:probability,
     bins=xaxbin,
     errortype=:percentile,
     percentiles=[0, 100],
     alpha=0.5,
     secondaryalpha=0.2,
     centertype=:median)
 errorlinehist!(ax_dist, [ROR_9[paths_9[2]][i]  for i in 1:Nb];
     label="",
     color=mycolors[2],
     secondarycolor=mycolors[2],
     normalization=:probability,
     bins=xaxbin,
     errortype=:percentile,
     percentiles=[0, 100],
     alpha=0.5,
     secondaryalpha=0.2,
     centertype=:median)

 errorlinehist!(ax_dist, [ROR_9[paths_9[3]][i] for i in 1:Nb];
     label="",
     color=mycolors[3],
     secondarycolor=mycolors[3],
     normalization=:probability,
     bins=xaxbin,
     errortype=:percentile,
     percentiles=[0, 100],
     alpha=0.5,
     secondaryalpha=0.2,
     centertype=:median)
     errorlinehist!(ax_dist, [ROR_9[paths_9[4]][i] for i in 1:Nb];
     label="",
     color=mycolors[4],
     secondarycolor=mycolors[4],
     normalization=:probability,
     bins=xaxbin,
     errortype=:percentile,
     percentiles=[0, 100],
     alpha=0.5,
     secondaryalpha=0.2,
     centertype=:median)
 Makie.scatter!(ax_dist, xax, [mean(RORo9 .== x) for x in xax], color=:blue, markersize=6, label= "Observations")

       
    
    Legend(fig_ROR[:, 3],
        [
            [
                [LineElement(color=mycolors[1]), PolyElement(color=mycolors[1], alpha=0.2)],
                [LineElement(color=mycolors[2]), PolyElement(color=mycolors[2], alpha=0.2)],
                [LineElement(color=mycolors[3]), PolyElement(color=mycolors[3], alpha=0.2)],
                [LineElement(color=mycolors[4]), PolyElement(color=mycolors[4], alpha=0.2)]
            ],
            [MarkerElement(color=:blue, marker=:circle, markersize=8)]
        ],
        [
            [
                L"Ind $K = 1$",
                L"Ind $K = 2$",
                L"Ind $K = 4$",
                L"Ind $K = 9$"
            ],
            ["Observations"]
        ],
        [" ", " "]
    )
    resize_to_layout!(fig_ROR)
    fig_ROR
end
savefigcrop(fig_ROR, "./31HMMIndep/plotK")
