# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
using Pkg
Pkg.activate("HMMSPAcodepaper")
Pkg.instantiate()

begin
    include("../13PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
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
    include("../13PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl")
end
# test --------------------------------------#
import Distributions.Categorical
tdist = 0.3

my_K = 2# Number of Hidden states
my_T = 366 # Period
my_degree_of_P = 1

my_autoregressive_order = 0 #

my_N = my_T * 25
n2t = n_to_t(my_N, my_T)




my_distance = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))

my_D = length(my_locations[:, 1])
my_index = sample(1:my_D, my_D, replace=false)

my_distance = my_distance[my_index, my_index]
my_locations = my_locations[my_index, :]
my_D = length(my_index)

p = Plots.scatter(my_locations[:, 1], my_locations[:, 2]);

# choose real set of parameters


my_size_order = 2^my_autoregressive_order
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



#initial parameters in the choux
thetaA = rand(my_K, my_K - 1, my_size_degree_of_P)
thetaB = zeros(my_K, my_D, my_size_order, my_size_degree_of_P)
thetaR = zeros(my_K, my_size_degree_of_P)
thetaR = zeros(my_K, my_size_degree_of_P)

thetaA[:, :, :] .= my_trans_θ[:, :, 1] # cheating on initial guess to recover very good mle maxima
thetaB[:, :, :, 1] .= my_Bernoulli_θ[:, :, :, 1]
thetaR[:, 1] .= my_Range_θ[:, 1] .- log(10)# cheating on initial guess to recover very good mle maxima
hmm = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)
start_model = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)

tol = 1e-4

D = size(my_locations, 1)
Y_past = rand(Bool, my_autoregressive_order, D)

println("Before estimation: ", thetaR)


# # with triangulation weigths : add argument wp =wp in fit_mle if this is to be used instead of basic distance pair thresholding.
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
# with usual weights
@time begin
    history2, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations = fit_mle!(hmm, thetaA, thetaB, thetaR, Y, Y_past; n2t=n2t, maxiter=50, tol=tol, maxiters_R=100, display=:iter, tdist=tdist, QMC_m=30)
end
pp1 = Plots.plot(history2["logtots"])



#############################################################################
######## get rain proba plot       ##########################################
#############################################################################
month_days = repeat(1:my_T, 50)

include("../utils/maps.jl")
include("../utils/seasons_and_other_dates.jl")



using CairoMakie # or GLMakie if you prefer interactive
using Statistics

# --- Utilities ---
rmse(x, y) = sqrt(mean((x .- y) .^ 2))

# function to get the last fitted thetas from your optimization history
function extract_last_theta(last_iter_array)
    # If stored as vector of arrays, take last:
    return last_iter_array[end]
end

# Convenience: pick fitted thetas from history
fitted_thetaA = extract_last_theta(all_thetaA_iterations)
fitted_thetaB = extract_last_theta(all_thetaB_iterations)
fitted_thetaR = extract_last_theta(all_thetaR_iterations)

# Create three models: true_model, start_model, fitted_model
fitted_model = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), fitted_thetaA, fitted_thetaB, fitted_thetaR, my_T, my_distance)
true_model = model  # assuming `model` is your true object from above

ndays = size(true_model.B, 2)  

# selected stations to demonstrate (you used select_plot earlier; choose some indexes)
select_demo = 1:min(6, size(my_locations, 1))  # first 6 stations or fewer

# colors
mycolors = [:red, :blue, :green, :orange, :purple, :brown][1:my_K]


fig_Bcompare = Figure()
chosen_stations = 1:6
for (idx, j) in enumerate(chosen_stations)
    row = (idx - 1) ÷ 3 + 1
    col = (idx - 1) % 3 + 1

    if col > 1
        ax = Axis(fig_Bcompare[row, col],
            title="Location $j",
            limits=(0, ndays + 1, 0, 1), yticks=0:0.2:1,
            width=200, height=150,
            ylabel=L"\lambda_{k,s}^{(t)}")
        hideydecorations!(ax; label=true, ticklabels=true, ticks=false,
            grid=false, minorgrid=true, minorticks=false)
    else
        ax = Axis(fig_Bcompare[row, col],
            title="Location $j",
            limits=(0, ndays + 1, 0, 1), yticks=0:0.2:1,
            width=200, height=150,
            ylabel=L"\lambda_{k,s}^{(t)}")
    end

    # ---- plot curves for each state k ----
    for k in 1:my_K
        color = mycolors[k]

        # TRUE model = solid
        lines!(ax, 1:ndays, [true_model.B[k, t, j, 1] for t in 1:ndays],
            color=color, linewidth=2)

        # START model = dashed
        lines!(ax, 1:ndays, [start_model.B[k, t, j, 1] for t in 1:ndays],
            color=color, linewidth=1.5, linestyle=:dash)

        # FITTED model = dotted
        lines!(ax, 1:ndays, [fitted_model.B[k, t, j, 1] for t in 1:ndays],
            color=color, linewidth=2, linestyle=:dot)

        # If AR order is present and h = 2 exists → also plot h=2 curves
        if size(true_model.B, 4) == 2
            # TRUE dashed for h=1
            lines!(ax, 1:ndays, [true_model.B[k, t, j, 2] for t in 1:ndays],
                color=color, linewidth=1, linestyle=:dashdot)
            # FITTED dashed for h=1
            lines!(ax, 1:ndays, [fitted_model.B[k, t, j, 2] for t in 1:ndays],
                color=color, linewidth=1, linestyle=:dash)
        end
    end

    # horizontal midline
    hlines!(ax, [0.5], color=:black, linestyle=:dot)

    # seasonal ticks
end

# ---- legend on the right ----
Legend(fig_Bcompare[1:2, 4],
    [
        LineElement(color=:black, linestyle=:solid),
        LineElement(color=:black, linestyle=:dash),
        LineElement(color=:black, linestyle=:dot),
        [LineElement(color=mycolors[k], linestyle=:solid) for k in 1:my_K]...
    ],
    [
        L"true",
        L"start",
        L"fitted",
        [L"k=%$k" for k in 1:my_K]...
    ]
)

resize_to_layout!(fig_Bcompare)

fig_Bcompare

# savefigcrop
savefigcrop(fig_Bcompare, "./13PeriodicHMMSpatialBernoulli/res_sim_data/B_true_start_fitted_compare")

begin
    fig_R = Figure()

    ax = Axis(fig_R[1, 1],
        ylabel=L"\rho_{CY,k}^{(t)}\ (km)",
        width=400, height=250)

    for k in 1:my_K
        color = mycolors[k]

        # --- Real series (solid) ---
        lines!(ax, 1:366, true_model.R[k, :],
            color=color, linewidth=2)

        # --- Starting model (dashed) ---
        lines!(ax, 1:366, start_model.R[k, :],
            color=color, linewidth=2, linestyle=:dash)

        # --- Fitted model (thicker solid or dotted) ---
        lines!(ax, 1:366, fitted_model.R[k, :],
            color=color, linewidth=3, linestyle=:dot)
    end



    Legend(fig_R[1, 2],
        [
            LineElement(color=:black, linestyle=:solid),
            LineElement(color=:black, linestyle=:dash),
            LineElement(color=:black, linestyle=:dot),
            [LineElement(color=mycolors[k], linestyle=:solid) for k in 1:my_K]...
        ],
        [
            L"true",
            L"start",
            L"fitted",
            [L"k=%$k" for k in 1:my_K]...
        ])

    fig_R
end
savefigcrop(fig_R, "./13PeriodicHMMSpatialBernoulli/res_sim_data/Compare_R_true_start_fitted")

begin
    fig_Q = Figure()

    for k in 1:my_K
        row = (k - 1) ÷ 2 + 1
        col = (k - 1) % 2 + 1

        ax = Axis(fig_Q[row, col],
            limits=(0, 367, 0, 1), yticks=0:0.2:1,
            width=300, height=250)

        if col > 1
            hideydecorations!(ax; label=true, ticklabels=true, ticks=false,
                grid=false, minorgrid=true, minorticks=false)
        end

        for l in 1:my_K
            color = mycolors[l]

            # --- REAL series ---
            lines!(ax, 1:366, true_model.A[k, l, :],
                color=color, linewidth=2)

            # --- STARTING model ---
            lines!(ax, 1:366, start_model.A[k, l, :],
                color=color, linewidth=2, linestyle=:dash)

            # --- FITTED model ---
            lines!(ax, 1:366, fitted_model.A[k, l, :],
                color=color, linewidth=3, linestyle=:dot)
        end

        # Horizontal reference line
        hlines!(ax, [0.5], color=:black, linestyle=:dot)

        # Month ticks identical to B-plot

        # Legend for each panel
        axislegend(ax,
            [
                LineElement(color=:black, linestyle=:solid),
                LineElement(color=:black, linestyle=:dash),
                LineElement(color=:black, linestyle=:dot),
                [LineElement(color=mycolors[l], linestyle=:solid) for l in 1:my_K]...
            ],
            [
                L"Real",
                L"Starting",
                L"Fitted",
                [L"Q^{(t)}(%$k, %$l)" for l in 1:my_K]...
            ],
            position=:ct, nbanks=3,
            labelsize=14.7, patchlabelgap=0,
            colgap=6, framevisible=false)
    end

    resize_to_layout!(fig_Q)
    fig_Q
end
savefigcrop(fig_Q, "./13PeriodicHMMSpatialBernoulli/res_sim_data/Compare_Q_true_start_fitted")

#################### m=1 ################"

tdist = 0.3

my_K = 2# Number of Hidden states
my_T = 366 # Period
my_degree_of_P = 1

my_autoregressive_order = 1 #

my_N = my_T * 50
n2t = n_to_t(my_N, my_T)





my_D = length(my_locations[:, 1])
my_index = sample(1:my_D, my_D, replace=false)

my_distance = my_distance[my_index, my_index]
my_locations = my_locations[my_index, :]
my_D = length(my_index)


# choose real set of parameters


my_size_order = 2^my_autoregressive_order
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



#initial parameters in the choux
thetaA = rand(my_K, my_K - 1, my_size_degree_of_P)
thetaB = zeros(my_K, my_D, my_size_order, my_size_degree_of_P)
thetaR = zeros(my_K, my_size_degree_of_P)
thetaR = zeros(my_K, my_size_degree_of_P)

thetaA[:, :, :] .= my_trans_θ[:, :, 1] # cheating on initial guess to recover very good mle maxima
thetaB[:, :, :, 1] .= my_Bernoulli_θ[:, :, :, 1]
thetaR[:, 1] .= my_Range_θ[:, 1] .- log(10)# cheating on initial guess to recover very good mle maxima
hmm = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)
start_model = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)

tol = 1e-4

D = size(my_locations, 1)
Y_past = rand(Bool, my_autoregressive_order, D)

println("Before estimation: ", thetaR)


# with usual weights
@time begin
    history2, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations = fit_mle!(hmm, thetaA, thetaB, thetaR, Y, Y_past; n2t=n2t, maxiter=50, tol=tol, maxiters_R=100, display=:iter, tdist=tdist, QMC_m=30)
end
pp1 = Plots.plot(history2["logtots"])



#############################################################################
######## get rain proba plot       ##########################################
#############################################################################
month_days = repeat(1:my_T, 50)


fitted_thetaA = extract_last_theta(all_thetaA_iterations)
fitted_thetaB = extract_last_theta(all_thetaB_iterations)
fitted_thetaR = extract_last_theta(all_thetaR_iterations)

# Create three models: true_model (you already have as `model`), start_model, fitted_model
fitted_model = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), fitted_thetaA, fitted_thetaB, fitted_thetaR, my_T, my_distance)
true_model = model  # assuming `model` is your true object from above

# daily sequence length (assumed 366 in your plotting code)
ndays = size(true_model.B, 2)  # keep general, but likely 366

# selected stations to demonstrate (you used select_plot earlier; choose some indexes)
select_demo = 1:min(6, size(my_locations, 1))  # first 6 stations or fewer

# colors
mycolors = [:red, :blue, :green, :orange, :purple, :brown][1:my_K]
mycolors_bis = [:pink, :cyan]

fig_Bcompare = Figure()
chosen_stations = 1:6
for (idx, j) in enumerate(chosen_stations)
    row = (idx - 1) ÷ 3 + 1
    col = (idx - 1) % 3 + 1

    if col > 1
        ax = Axis(fig_Bcompare[row, col],
            title="Location $j",
            limits=(0, ndays + 1, 0, 1), yticks=0:0.2:1,
            width=200, height=150,
            ylabel=L"\lambda_{k,s}^{(t)}")
        hideydecorations!(ax; label=true, ticklabels=true, ticks=false,
            grid=false, minorgrid=true, minorticks=false)
    else
        ax = Axis(fig_Bcompare[row, col],
            title="Location $j",
            limits=(0, ndays + 1, 0, 1), yticks=0:0.2:1,
            width=200, height=150,
            ylabel=L"\lambda_{k,s}^{(t)}")
    end

    # ---- plot curves for each state k ----
    for k in 1:my_K
        color = mycolors[k]
        colorbis=mycolors_bis[k]

        # TRUE model = solid
        lines!(ax, 1:ndays, [true_model.B[k, t, j, 1] for t in 1:ndays],
            color=color, linewidth=2)

        # START model = dashed
        lines!(ax, 1:ndays, [start_model.B[k, t, j, 1] for t in 1:ndays],
            color=color, linewidth=1.5, linestyle=:dash)

        # FITTED model = dotted
        lines!(ax, 1:ndays, [fitted_model.B[k, t, j, 1] for t in 1:ndays],
            color=color, linewidth=2, linestyle=:dot)

        # If AR order is present and h = 2 exists → also plot h=2 curves
        if size(true_model.B, 4) == 2
            # TRUE model = solid
            lines!(ax, 1:ndays, [true_model.B[k, t, j, 2] for t in 1:ndays],
                color=colorbis, linewidth=2)

            # START model = dashed
            lines!(ax, 1:ndays, [start_model.B[k, t, j, 2] for t in 1:ndays],
                color=colorbis, linewidth=1.5, linestyle=:dash)

            # FITTED model = dotted
            lines!(ax, 1:ndays, [fitted_model.B[k, t, j, 2] for t in 1:ndays],
                color=colorbis, linewidth=2, linestyle=:dot)
        end
    end

    # horizontal midline
    hlines!(ax, [0.5], color=:black, linestyle=:dot)

    # seasonal ticks
end

# ---- legend on the right ----
Legend(fig_Bcompare[1:2, 4],
    [
        LineElement(color=:black, linestyle=:solid),
        LineElement(color=:black, linestyle=:dash),
        LineElement(color=:black, linestyle=:dot),
        [LineElement(color=mycolors[k], linestyle=:solid) for k in 1:my_K]...,
        [LineElement(color=mycolors_bis[k], linestyle=:solid) for k in 1:my_K]...
    ],
    [
        L"true",
        L"start",
        L"fitted",
        [L"k=%$k - y^{(n-1)}_s = 0" for k in 1:my_K]...,
        [L"k=%$k - y^{(n-1)}_s = 1" for k in 1:my_K]...
    ]
)

resize_to_layout!(fig_Bcompare)

fig_Bcompare

# savefigcrop
savefigcrop(fig_Bcompare, "./13PeriodicHMMSpatialBernoulli/res_sim_data/B_true_start_fitted_compare1")

begin
    fig_R = Figure()

    ax = Axis(fig_R[1, 1],
        ylabel=L"\rho_{CY,k}^{(t)}\ (km)",
        width=400, height=250)

    for k in 1:my_K
        color = mycolors[k]

        # --- Real series (solid) ---
        lines!(ax, 1:366, true_model.R[k, :],
            color=color, linewidth=2)

        # --- Starting model (dashed) ---
        lines!(ax, 1:366, start_model.R[k, :],
            color=color, linewidth=2, linestyle=:dash)

        # --- Fitted model (thicker solid or dotted) ---
        lines!(ax, 1:366, fitted_model.R[k, :],
            color=color, linewidth=3, linestyle=:dot)
    end



    Legend(fig_R[1, 2],
        [
            LineElement(color=:black, linestyle=:solid),
            LineElement(color=:black, linestyle=:dash),
            LineElement(color=:black, linestyle=:dot),
            [LineElement(color=mycolors[k], linestyle=:solid) for k in 1:my_K]...
        ],
        [
            L"true",
            L"start",
            L"fitted",
            [L"k=%$k" for k in 1:my_K]...
        ])

    fig_R
end
savefigcrop(fig_R, "./13PeriodicHMMSpatialBernoulli/res_sim_data/Compare_R_true_start_fitted1")

begin
    fig_Q = Figure()

    for k in 1:my_K
        row = (k - 1) ÷ 2 + 1
        col = (k - 1) % 2 + 1

        ax = Axis(fig_Q[row, col],
            limits=(0, 367, 0, 1), yticks=0:0.2:1,
            width=300, height=250)

        if col > 1
            hideydecorations!(ax; label=true, ticklabels=true, ticks=false,
                grid=false, minorgrid=true, minorticks=false)
        end

        for l in 1:my_K
            color = mycolors[l]

            # --- REAL series ---
            lines!(ax, 1:366, true_model.A[k, l, :],
                color=color, linewidth=2)

            # --- STARTING model ---
            lines!(ax, 1:366, start_model.A[k, l, :],
                color=color, linewidth=2, linestyle=:dash)

            # --- FITTED model ---
            lines!(ax, 1:366, fitted_model.A[k, l, :],
                color=color, linewidth=3, linestyle=:dot)
        end

        # Horizontal reference line
        hlines!(ax, [0.5], color=:black, linestyle=:dot)

        # Month ticks identical to B-plot

        # Legend for each panel
        axislegend(ax,
            [
                LineElement(color=:black, linestyle=:solid),
                LineElement(color=:black, linestyle=:dash),
                LineElement(color=:black, linestyle=:dot),
                [LineElement(color=mycolors[l], linestyle=:solid) for l in 1:my_K]...
            ],
            [
                L"Real",
                L"Starting",
                L"Fitted",
                [L"Q^{(t)}(%$k, %$l)" for l in 1:my_K]...
            ],
            position=:ct, nbanks=3,
            labelsize=14.7, patchlabelgap=0,
            colgap=6, framevisible=false)
    end

    resize_to_layout!(fig_Q)
    fig_Q
end
savefigcrop(fig_Q, "./13PeriodicHMMSpatialBernoulli/res_sim_data/Compare_Q_true_start_fitted1")

