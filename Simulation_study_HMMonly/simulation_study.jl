
# Simulation study proposée :

# - Récupérer les paramètres "réels" de pluie : HMM, puis precip;  ce sont des params trigos ! donc on verra si ça a un sens ou pas 
# - générer des données complètes : Z,Y,R à partir de celui-là pour <51 ans, peut-être seulement l'équivalent de 10 ans (pour que ce soit manageable).
# - Sur chacune de ces séries : 
# 	- Estimer les params du HMM; 
# 	- Inférer Zhat 
# 	- Estimer les params EGPD
# 	- Estimer les params de la covariance (nouveau modèle).
# - Mettre dans une table, pour chaque paramètre HMM : vrai param - moyenne param estimé - mediane - quelques quantiles. on peut éventuellement faire une table Lambda, une table Q, une table R.
# - Comparer les classifications de Z/Zhat (je ne sais pas encore comment)
# - Mettre dans une table, pour chaque param de precip, la même chose que pour HMM. Il y a bcp de stations : cette table sera extrêmement grande ! Mais je ne vois pas trop quoi faire d'autre. Faire une table pour chaque station et les laisser sur le git ? on peut en garder simplement une.
# - Mettre dans une (petite) table les params de covariance GM : mm chose que pour HMM mais en sacrément plus simple.  trop facile !



##############################################################################
# Strategy (single replicate, to be run many times on cluster):
#   1. Load parameters estimated on real data ("true" parameters)
#   2. Simulate one replicate (Z, Y, R) from the true model
#   3. Fit the HMM on (Y) → get Ẑ via Viterbi
#   4. Fit intensity margins twice: once with Z, once with Ẑ
#   5. Fit covariance parameters twice: once with Z, once with Ẑ
#   6. Save all results for comparison
##############################################################################
cd(@__DIR__)
using Pkg
Pkg.activate(".")


using SmoothPeriodicStatsModels
using Distributions
using Optimization, OptimizationOptimJL, LineSearches
using ForwardDiff
using Random
using StochasticWeatherGenerators
using CSV, DataFrames, Dates, JLD2
using Base.Threads
number=Threads.nthreads()



##############################################################################
# 0. Global settings
##############################################################################

Random.seed!(2024)  # for reproducibility; change per replicate on cluster

const my_K  = 4
const my_T  = 366
const my_autoregressive_order = 1
const degree_of_P = 1
const tdist   = 0.3
const QMC_m   = 30


##############################################################################
# 1. Load spatial metadata
##############################################################################

my_distance = Matrix(CSV.read("transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("transformedECAD_locs.csv", DataFrame, header=false))
my_D = size(my_locations, 1)
locsdata = CSV.read("transformedECAD_stations.csv", DataFrame, header=true)
locsdata.LON = my_locations[:, 1]
locsdata.LAT = my_locations[:, 2]
coordll  = [locsdata.LON locsdata.LAT]


##############################################################################
# 2. Define time axis
##############################################################################

date_start = Date(1973)
date_end   = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end
my_N       = length(every_year)
n2t        = dayofyear_Leap.(every_year)

##############################################################################
# 3. Load true parameters (estimated on real data)
##############################################################################

hmmspadata = load("K4_resu_18627days_degree1_maxiter100_m1_R0500_QMCm30.jld2")

iter_true = argmax(hmmspadata["logtots"])
thetaA_true = hmmspadata["all_thetaA_iterations"][iter_true + 1]
thetaB_true = hmmspadata["all_thetaB_iterations"][iter_true + 1]
thetaR_true = hmmspadata["all_thetaR_iterations"][iter_true + 1]
a_true      = hmmspadata["hmm"].a

modeltrue = Trig2ARPeriodicHMMSpatial(a_true, thetaA_true, thetaB_true, thetaR_true, my_T, my_distance)


##############################################################################
# 4. Simulate one replicate (Z, Y, R) from the true model
##############################################################################
for replicate in 1:2
seed = 2026 + replicate
Random.seed!(seed)

Z, Yall = rand(modeltrue, my_N, seq=true)

# Y_past: random initial condition 
Y_past = BitMatrix(rand(Bool, my_D, my_autoregressive_order))
Y = Yall  # (my_N × my_D), aligned with n2t

# Lag matrix for R simulation (prepend one dry day as initial condition)
Yprevious = BitMatrix(fill(1, my_D, 1))
Ymoins1   = hcat(Yprevious, Yall')  # (my_D × (my_N + 1)), Ymoins1[:, n] is Y at n-1



##############################################################################
# 5. Fit HMM on simulated Y → get Ẑ
##############################################################################

# 5a. Independent HMM initialisation (slice + Baum-Welch)
hmm_random = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order;
                                ξ=[1; zeros(my_K - 1)], ref_station=1)

@time "Slice init" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past';
    n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1)

    
θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, 1)

@time "Baum-Welch" hmm_ind, θq_fit, θy_fit, hist_ind, _, _ = fit_mle(
    hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past';
    maxiter=100, robust=true, display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)

# 5b. Spatial HMM (EM with LBFGS for R update)
θᴿ_init        = zeros(my_K, 2 * degree_of_P + 1)
θᴿ_init[:, 1] .= log(500.0)

hmm_spa_init = Trig2ARPeriodicHMMSpatial(
    fill(1 / my_K, my_K), θq_fit, θy_fit, θᴿ_init, my_T, my_distance)

solver = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking())

@time "Spatial EM" history_spa, all_θᴬ, all_θᴮ, all_θᴿ = fit_mle!(
    hmm_spa_init, θq_fit, θy_fit, θᴿ_init, Y, Y_past';
    solver=solver, n2t=n2t, maxiter=5, tol=1e-3,
    maxiters_R=100, display=:iter, tdist=tdist, QMC_m=QMC_m)

iter_best  = argmax(history_spa["logtots"])
thetaA_fit = all_θᴬ[iter_best + 1]
thetaB_fit = all_θᴮ[iter_best + 1]
thetaR_fit = all_θᴿ[iter_best + 1]

hmm_spa_fit = Trig2ARPeriodicHMMSpatial(
    fill(1 / my_K, my_K), thetaA_fit, thetaB_fit, thetaR_fit, my_T, my_distance)


save("./simulation_study/replicate"*string(replicate)*".jld2",
    # True parameters
    "thetaA_true",   thetaA_true,
    "thetaB_true",   thetaB_true,
    "thetaR_true",   thetaR_true,
    # Fitted HMM parameters
    "thetaA_fit",    thetaA_fit,
    "thetaB_fit",    thetaB_fit,
    "thetaR_fit",    thetaR_fit,
   
)

end


using CairoMakie
using Statistics

outdir = "./simulation_study/"

mycolors     = [:red, :blue, :green, :orange, :purple, :brown, :teal, :magenta][1:my_K]
mycolors_bis = [:pink, :cyan, :lightgreen, :lightyellow][1:my_K]  # for h=2 (AR)


##############################################################################
#  Load replicate results and reconstruct parameter curves
##############################################################################
chosen_stations = 1:6   # stations to plot for B
replicate_files = sort(filter(f -> startswith(f, "replicate") && endswith(f, ".jld2"),
                               readdir(outdir)))
n_rep = length(replicate_files)
println("Found $n_rep replicates")

# Helper: reconstruct model from saved thetas
function model_from_jld(d)
    Trig2ARPeriodicHMMSpatial(
        fill(1/my_K, my_K),
        d["thetaA_fit"], d["thetaB_fit"], d["thetaR_fit"],
        my_T, my_distance)
end

# Collect curves across replicates
# B_reps[rep, k, t, s, h] — rain probability
# R_reps[rep, k, t]        — range
# A_reps[rep, k, l, t]     — transition

B_size_order = size(modeltrue.B, 4)   # 1 (AR=0) or 2 (AR=1)

B_reps = zeros(n_rep, my_K, my_T, my_D, B_size_order)
R_reps = zeros(n_rep, my_K, my_T)
A_reps = zeros(n_rep, my_K, my_K, my_T)

for (r, fname) in enumerate(replicate_files)
    d = load(joinpath(outdir, fname))
    m = model_from_jld(d)
    B_reps[r, :, :, :, :] .= m.B
    R_reps[r, :, :]        .= m.R
    A_reps[r, :, :, :]     .= m.A
end

# Summary statistics across replicates at each (k,t,...):
# Returns (median, q025, q975, min, max) arrays of same shape as input[1,...]
function rep_stats(arr; dims=1)
    med  = mapslices(x -> quantile(x, 0.50), arr, dims=dims)
    q025 = mapslices(x -> quantile(x, 0.025), arr, dims=dims)
    q975 = mapslices(x -> quantile(x, 0.975), arr, dims=dims)
    mn   = minimum(arr, dims=dims)
    mx   = maximum(arr, dims=dims)
    return dropdims.((med, q025, q975, mn, mx); dims=dims)
end

B_med, B_q025, B_q975, B_min, B_max = rep_stats(B_reps)
R_med, R_q025, R_q975, R_min, R_max = rep_stats(R_reps)
A_med, A_q025, A_q975, A_min, A_max = rep_stats(A_reps)

tt = 1:my_T

##############################################################################
# Helper: draw one parameter curve with envelope
##############################################################################

function draw_curve_with_band!(ax, tt, true_vals, med, q025, q975, mn, mx, color;
                                linewidth=2)
    # min/max band (very transparent)
    band!(ax, tt, mn, mx; color=(color, 0.10))
    # 95% CI band
    band!(ax, tt, q025, q975; color=(color, 0.25))
    # median
    lines!(ax, tt, med;  color=color, linewidth=linewidth, linestyle=:dash)
    # true
    lines!(ax, tt, true_vals; color=color, linewidth=linewidth, linestyle=:solid)
end

##############################################################################
# 4. Plot B — rain probabilities (6 stations, 2 rows × 3 cols)
##############################################################################

for h in 1:B_size_order
    h_label = B_size_order == 1 ? "" : (h == 1 ? "_dry" : "_wet")

    fig_B = Figure(size=(900, 500))

    for (idx, j) in enumerate(chosen_stations)
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1

        ax = Axis(fig_B[row, col],
            title   = "Station $j",
            limits  = (0, my_T + 1, 0, 1),
            yticks  = 0:0.2:1,
            width   = 200, height = 150,
            ylabel  = L"\lambda_{k,s}^{(t)}")

        if col > 1
            hideydecorations!(ax; label=true, ticklabels=true,
                              ticks=false, grid=false, minorgrid=true, minorticks=false)
        end

        for k in 1:my_K
            draw_curve_with_band!(ax, tt,
                [modeltrue.B[k, t, j, h] for t in tt],
                B_med[k, :, j, h], B_q025[k, :, j, h],
                B_q975[k, :, j, h], B_min[k, :, j, h], B_max[k, :, j, h],
                mycolors[k])
        end

        hlines!(ax, [0.5]; color=:black, linestyle=:dot)
    end

    # Legend
    legend_elements = vcat(
        [LineElement(color=:black, linestyle=:solid, linewidth=2)],
        [LineElement(color=:black, linestyle=:dash,  linewidth=2)],
        [PolyElement(color=(:black, 0.25))],
        [PolyElement(color=(:black, 0.10))],
        [LineElement(color=mycolors[k], linestyle=:solid, linewidth=2) for k in 1:my_K]
    )
    legend_labels = vcat(
        ["True"],
        ["Median"],
        ["95% CI"],
        ["Min/Max"],
        ["k=$k" for k in 1:my_K]
    )
    Legend(fig_B[1:2, 4], legend_elements, legend_labels)

    resize_to_layout!(fig_B)
    save(outdir * "sim_study_B$(h_label)_$(n_rep)rep.pdf", fig_B)
    println("Saved B$(h_label) figure")
end

##############################################################################
# 5. Plot R — spatial correlation range
##############################################################################

fig_R = Figure(size=(550, 300))
ax_R  = Axis(fig_R[1, 1],
    ylabel = L"\rho_{CY,k}^{(t)}\ \mathrm{(km)}",
    width  = 400, height = 250)

for k in 1:my_K
    draw_curve_with_band!(ax_R, tt,
        modeltrue.R[k, :],
        R_med[k, :], R_q025[k, :], R_q975[k, :],
        R_min[k, :], R_max[k, :],
        mycolors[k])
end

legend_elements_R = vcat(
    [LineElement(color=:black, linestyle=:solid, linewidth=2)],
    [LineElement(color=:black, linestyle=:dash,  linewidth=2)],
    [PolyElement(color=(:black, 0.25))],
    [PolyElement(color=(:black, 0.10))],
    [LineElement(color=mycolors[k], linestyle=:solid, linewidth=2) for k in 1:my_K]
)
legend_labels_R = vcat(["True"], ["Median"], ["95% CI"], ["Min/Max"],
                        ["k=$k" for k in 1:my_K])
Legend(fig_R[1, 2], legend_elements_R, legend_labels_R)

resize_to_layout!(fig_R)
save(outdir * "sim_study_R_$(n_rep)rep.pdf", fig_R)
println("Saved R figure")

##############################################################################
# 6. Plot A — transition matrix (one panel per row k, same layout as original)
##############################################################################

fig_Q = Figure(size=(700, 350 * ceil(Int, my_K / 2)))

for k in 1:my_K
    row = (k - 1) ÷ 2 + 1
    col = (k - 1) % 2 + 1

    ax = Axis(fig_Q[row, col],
        title  = "From state $k",
        limits = (0, my_T + 1, 0, 1),
        yticks = 0:0.2:1,
        width  = 300, height = 250,
        ylabel = L"Q_{k\to l}^{(t)}")

    if col > 1
        hideydecorations!(ax; label=true, ticklabels=true,
                          ticks=false, grid=false, minorgrid=true, minorticks=false)
    end

    for l in 1:my_K
        draw_curve_with_band!(ax, tt,
            modeltrue.A[k, l, :],
            A_med[k, l, :], A_q025[k, l, :], A_q975[k, l, :],
            A_min[k, l, :], A_max[k, l, :],
            mycolors[l])
    end

    hlines!(ax, [0.5]; color=:black, linestyle=:dot)

    axislegend(ax,
        vcat(
            [LineElement(color=:black, linestyle=:solid, linewidth=2)],
            [LineElement(color=:black, linestyle=:dash,  linewidth=2)],
            [PolyElement(color=(:black, 0.25))],
            [PolyElement(color=(:black, 0.10))],
            [LineElement(color=mycolors[l], linestyle=:solid, linewidth=2) for l in 1:my_K]
        ),
        vcat(["True"], ["Median"], ["95% CI"], ["Min/Max"],
             [L"Q_{%$k\to%$l}" for l in 1:my_K]),
        position=:ct, nbanks=3,
        labelsize=12, patchlabelgap=0,
        colgap=6, framevisible=false)
end

resize_to_layout!(fig_Q)
save(outdir * "sim_study_A_$(n_rep)rep.pdf", fig_Q)
println("Saved A figure")

println("\nDone. All figures saved to $outdir")