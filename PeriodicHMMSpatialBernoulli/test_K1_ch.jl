using Pkg
Pkg.activate("HMMSPAcodepaper")
Pkg.instantiate()

using Random, CSV, DataFrames, Statistics
using JuMP, Ipopt
using LaTeXStrings
using Distributed

include("../PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
include("../PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl")
include("../SpatialBernoulli/SpatialBernoulli.jl")

Random.seed!(0)

###############################################################################
# SETTINGS
###############################################################################

Ds = [ 5, 10,35]         # VALUES OF my_D YOU WANT TO TEST
Nrep = 100               # repeats for each D
tol = 1e-4
tdist = 1

my_T = 5
my_N = 10* my_T # = 50
n2t = n_to_t(my_N, my_T)

# Load distance matrix & locations
full_distance = Matrix(CSV.read("./data/transformedECAD_locsdistances.csv", DataFrame, header=false))
full_locations = Matrix(CSV.read("./data/transformedECAD_locs.csv", DataFrame, header=false))
full_D = size(full_locations, 1)

###############################################################################
# Containers to store RESULTS for all Ds
###############################################################################

results = Dict()

###############################################################################
# Helper
###############################################################################
flatten_params(arrs) = hcat([vec(a) for a in arrs]...)'

###############################################################################
# MAIN LOOP OVER D
###############################################################################

for my_D in Ds
    println("===================================================")
    println("Running experiments for D = $my_D")
    println("===================================================")

    # Random subset of locations
    idx = sample(1:full_D, my_D, replace=false)
    my_distance = full_distance[idx, idx]
    my_locations = full_locations[idx, :]

    # True parameters (same structure as you had)
    my_K = 1
    my_autoregressive_order = 0
    my_size_order = 2^my_autoregressive_order
    my_degree_of_P = 0
    my_size_degree_of_P = 2 * my_degree_of_P + 1

    my_trans_θ = 4 * (rand(my_K, my_K - 1, my_size_degree_of_P) .- 0.5)
    my_Bernoulli_θ = 2 * (rand(my_K, my_D, my_size_order, my_size_degree_of_P) .- 0.5)
    my_Range_θ = rand(my_K, my_size_degree_of_P) .- 0.5
    my_Range_θ[:, 1] .= log.(300 .* (1:my_K))

    my_a = fill(1 / my_K, my_K)

    model_true = Trig2PeriodicHMMspaMemory(my_a, my_trans_θ, my_Bernoulli_θ, my_Range_θ, my_T, my_distance)
    trueB = vec(model_true.B[1, 1, :, 1])
    trueR = model_true.R[1, 1]

    # storage per D
    est_thetaB = Vector{Array{Float64}}(undef, Nrep)
    est_thetaR = Vector{Float64}(undef, Nrep)
    est_lambda_full = Vector{Array{Float64}}(undef, Nrep)
    est_R_full = Vector{Float64}(undef, Nrep)
    time_pairwise = zeros(Nrep)
    time_full     = zeros(Nrep)

    ############################################################################
    # repetitions
    ############################################################################
    for rep in 1:Nrep
        println("  Rep $rep")

        z, Y = my_rand(model_true, n2t; seq=true)
        Y = convert(Array{Bool}, Y)
        D = my_D

        # initial guess
        thetaA = rand(my_K, my_K - 1, my_size_degree_of_P)
        thetaB = zeros(my_K, my_D, my_size_order, my_size_degree_of_P)
        thetaR = zeros(my_K, my_size_degree_of_P)
        Y_past = rand(Bool, my_autoregressive_order, D)

        thetaA[:, :, :] .= my_trans_θ[:, :, 1]
        thetaB[:, :, :, 1] .= my_Bernoulli_θ[:, :, :, 1]
        thetaR[:, 1] .= my_Range_θ[:, 1] .- log(10)

        hmm = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K),
            thetaA, thetaB, thetaR, my_T, my_distance)

        ########################################################################
        # pairwise fit
        ########################################################################
        time_pairwise[rep] = @elapsed begin
            history2, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations =
                fit_mle!(hmm, thetaA, thetaB, thetaR, Y, Y_past;
                    n2t=n2t, maxiter=30, tol=tol,
                    maxiters_R=100, display=:none, tdist=tdist, QMC_m=100)
        end

        est_thetaB[rep] = deepcopy(hmm.B[1, 1, :, 1])
        est_thetaR[rep] = deepcopy(hmm.R[1, 1])

        ########################################################################
        # full-likelihood fit
        ########################################################################
        init_range = 400.
        init_order = 0.5
        init_lambda = fill(0.4, my_D)
        init_d = SpatialBernoulli(init_range, 1.0, init_order, init_lambda, my_distance)

        time_full[rep] = @elapsed begin
            sol = fit_mle(init_d, Y';
                m=100 * length(init_d), return_sol=false, order=init_order)
            est_lambda_full[rep] = sol.λ
            est_R_full[rep]      = sol.range
        end
    end

    ###########################################################################
    # store in results dict
    ###########################################################################
    results[my_D] = (
        B_mat       = flatten_params(est_thetaB),
        B_full      = flatten_params(est_lambda_full),
        R_mat       = est_thetaR,
        R_full      = est_R_full,
        trueB       = trueB,
        trueR       = trueR,
        t_pair      = time_pairwise,
        t_full      = time_full
    )
end




using CairoMakie, LaTeXStrings


function plot_results_compact(results, Ds)

    fig = Figure(
        resolution = (1800, 350*length(Ds)),
        fontsize = 22
    )

    # Row/column gaps (BRF-style)
    rowgap!(fig.layout, 25)
    colgap!(fig.layout, 25)

    # Legend elements
    legend_items = [
        MarkerElement(color = :red, marker = :star5, markersize = 15),
        PolyElement(color = (:lightgray, 0.9)),
        PolyElement(color = (:lightblue, 0.9))
    ]
    legend_labels = ["True value", "Pairwise", "Full"]

    # Loop over D values
    for (row, my_D) in enumerate(Ds)

       
        Rmat      = results[my_D].R_mat
        Rmat_full = results[my_D].R_full
        trueR     = results[my_D].trueR
        tpair     = results[my_D].t_pair
        tfull     = results[my_D].t_full

        Nrep = size(Bmat, 1)
        D = length(trueB)

        ########################################################################
        ### Left "D = ..." label (exactly like your BRF lambda labels)
        ########################################################################
        Label(fig[row, 0], "D = $my_D", tellheight=false, fontsize=26)


        ########################################################################
        ### PANEL 2 : ρ
        ########################################################################

        axR = Axis(
            fig[row, 1],
            title = (row == 1 ? L"\textbf{Spatial parameter }(\rho)" : ""),
            ylabel = L"\rho",
            xticks = ([1], [" "]),
            titlegap = 2
        )

        times = vcat(Rmat,Rmat_full)
        dodge= vcat(fill(0,length(Rmat)),fill(1,length(Rmat_full)))
        Makie.boxplot!(axR, fill(1, length(times)), times,dodge=dodge,color=dodge)
        hlines!(axR, [trueR], color=:red, linewidth=3)
        Makie.ylims!(axR, 0, 600)


        ########################################################################
        ### PANEL 3 : computation times
        ########################################################################

        axT = Axis(
            fig[row, 2],
            title = (row == 1 ? L"\textbf{Computation time (s)}" : ""),
            ylabel = "Seconds",
            titlegap = 2
        )
        times = vcat(tpair,tfull)
        dodge= vcat(fill(0,length(tpair)),fill(1,length(tfull)))
        Makie.boxplot!(axT, fill(1, length(times)), times,dodge=dodge,color=dodge)
    end


    ########################################################################
    ### Shared legend (BRF-style placement)
    ########################################################################
    Legend(
        fig[length(Ds)+1, 1:2],
        legend_items,
        legend_labels,
        orientation=:horizontal,
        halign=:center,
        valign=:top,
        framecolor=:transparent,
        padding = (5,5,5,5),
        labelsize = 22
    )

    fig
end

function plot_R_and_time(results, Ds)

    fig = Figure(resolution=(1400, 600), fontsize=22)
    rowgap!(fig.layout, 30)
    colgap!(fig.layout, 40)

    # COLORS
    color_pair = :lightgray
    color_full = :lightblue

    ########################################################################
    ### PANEL 1 — R estimates vs D
    ########################################################################
    axR = Axis(
        fig[1,1],
        title = L"\textbf{Estimated spatial scale}(\rho)",
        xlabel = "Dimension D",
        ylabel = L"\rho (\text{km})",
        xticks = (1:length(Ds), string.(Ds))
    )

    for (i, my_D) in enumerate(Ds)
        Rmat      = results[my_D].R_mat
        Rmat_full = results[my_D].R_full
        trueR     = results[my_D].trueR

        times = vcat(Rmat, Rmat_full)
        dodge = vcat(fill(1,length(Rmat)), fill(2,length(Rmat_full)))
        xvals = fill(i, length(times))

        colors = ifelse.(dodge .== 1, color_pair, color_full)

        Makie.boxplot!(axR, xvals, times; dodge=dodge, color=colors)

        # True value as horizontal line
        Makie.hlines!(axR, [trueR], color=:red, linewidth=3)
    end

    ########################################################################
    ### PANEL 2 — Computation time vs D
    ########################################################################
    axT = Axis(
        fig[1,2],
        title = L"\textbf{Computation time (s)}",
        xlabel = "Dimension D",
        ylabel = "Seconds",
        xticks = (1:length(Ds), string.(Ds)), yscale = log10
    )

    for (i, my_D) in enumerate(Ds)
        tpair = results[my_D].t_pair
        tfull = results[my_D].t_full

        times = vcat(tpair, tfull)
        dodge = vcat(fill(1, length(tpair)), fill(2, length(tfull)))
        xvals = fill(i, length(times))

        colors = ifelse.(dodge .== 1, color_pair, color_full)

        Makie.boxplot!(axT, xvals, times; dodge=dodge, color=colors)
    end

    ########################################################################
    ### Shared legend
    ########################################################################
    legend_items = [
        MarkerElement(color=:red, marker=:hline, markersize=15),
        PolyElement(color=color_pair),
        PolyElement(color=color_full)
    ]
    legend_labels = ["True value", "Pairwise", "Full"]

    Legend(
        fig[2,1:2],
        legend_items,
        legend_labels,
        orientation=:horizontal,
        halign=:center,
        framecolor=:transparent
    )

    return fig
end



fig = plot_R_and_time(results, Ds)

save("./PeriodicHMMSpatialBernoulli/estim_parameters_by_D_compact_noprob_N"*string(my_N)*"400.pdf", fig)
