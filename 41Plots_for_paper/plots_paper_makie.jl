cd(@__DIR__)
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

include("function.jl")
include("utilities.jl")
include("../utils/maps.jl")
include("../utils/seasons_and_other_dates.jl")

using Colors
mycolors = RGB{Float64}[RGB(0.0, 0.6056031704619725, 0.9786801190138923), RGB(0.24222393333911896, 0.6432750821113586, 0.304448664188385), RGB(0.7644400000572205, 0.4441118538379669, 0.8242975473403931), RGB(0.8888735440600661, 0.435649148506399, 0.2781230452972766), RGB(0.6755439043045044, 0.5556622743606567, 0.09423444420099258), RGB(0.0, 0.6657590270042419, 0.6809969544410706), RGB(0.9307674765586853, 0.3674771189689636, 0.5757699012756348), RGB(0.776981770992279, 0.5097429752349854, 0.14642538130283356), RGB(5.29969987894674e-8, 0.6642677187919617, 0.5529508590698242), RGB(0.558464765548706, 0.59348464012146, 0.11748137325048447), RGB(0.0, 0.6608786582946777, 0.7981787919998169), RGB(0.609670877456665, 0.49918484687805176, 0.9117812514305115), RGB(0.38000133633613586, 0.5510532855987549, 0.9665056467056274), RGB(0.9421815872192383, 0.3751642107963562, 0.4518167972564697), RGB(0.8684020638465881, 0.39598923921585083, 0.7135148048400879), RGB(0.4231467843055725, 0.6224954128265381, 0.19877080619335175)];

# mycolors = RGB{Float64}[RGB(0,0.447,0.698),
#  RGB(0,0.620,0.451),
#   RGB(0.8,0.475,0.655), 
#   RGB(0.835,0.369,0)];

#   mycolors= RGB{Float64}[RGB(43/255,131/255,186/255),RGB(171/255,221/255,164/255),RGB(253/255,174/255,97/255),RGB(215/255,25/255,28/255)]

# # select stations to plot when looking at marginals
# select_plot = [6, 1, 16, 14, 21, 3]

# fontsize = 16
# update_theme!(fontsize=fontsize, theme_latexfonts())



#############################################################################
######## explanation : spatial model ########################################
#############################################################################


Random.seed!(124)




# select stations to plot when looking at marginals
select_plot = [6, 1, 16, 14, 21, 3]
width_plot = 1000

fontsize = 15
update_theme!(fontsize=fontsize, theme_latexfonts())



#############################################################################
######## explanation : spatial model ########################################
#############################################################################



# Function to generate binary field given λ and range parameter
function generate_binary_field(my_λ, ρ)
    my_sill, my_order = 1.0, 0.5
    d = SpatialBernoulli(ρ, my_sill, my_order, my_λ, my_distance)
    u = rand(MvNormal(d.ΣU))
    thresholds = quantile.(Normal(), d.λ)
    ys = u .< thresholds
    return reshape(ys, dy, dx)
end


"""
    make_squares(λs, ρs, fields)
Binary fields for different ρ, λ values and fields
"""
function make_square(λs, ρs, fields, xg, yg)
    fig = Figure()
    axs = [Axis(fig[j, i], width=120, height=120, limits=(0, 2, 0, 2)) for (j, λ) in enumerate(λs), (i, ρ) in enumerate(ρs)]
    for (j, λ) in enumerate(λs)
        for (i, ρ) in enumerate(ρs)
            heatmap!(axs[j, i], xg, yg, fields[j, i], colormap=:binary, colorrange=(0, 1))
            if j < length(λs)
                hidexdecorations!(axs[j, i])
            end
            if i > 1
                hideydecorations!(axs[j, i])
            end
        end
    end
    for (j, λ) in enumerate(λs)
        Label(fig[j, 0], L"\lambda = %$(λ[1])", tellheight=false)
    end
    for (i, ρ) in enumerate(ρs)
        Label(fig[0, i], L"\rho = %$(ρ)", tellwidth=false)
    end
    Label(fig, bbox=BBox(-115, 200, 50, 300), L"$Y=1: \blacksquare$\n$Y=0: □$")
    colgap!(fig.layout, (25))
    resize_to_layout!(fig)
    # end
    return fig
end

# Grid
xg = 0:0.03:2
yg = 0:0.03:2
dx, dy = length(xg), length(yg)
my_locations = vcat(([xx yy] for xx in xg for yy in yg)...)
my_distance = [sqrt(sum(abs2, my_locations[i, :] - my_locations[j, :]))
               for i in axes(my_locations, 1), j in axes(my_locations, 1)]
nlocs = size(my_locations, 1)

λs = [0.2, 0.5]
ρs = [0.01, 0.1, 0.5]
fields = [generate_binary_field(fill(λ, nlocs), ρ) for λ in λs, ρ in ρs]
fig_row = make_square(λs, ρs, fields, xg, yg)

savefigcrop("./plots_paper/Lambda_BRF_grid2.pdf", fig_row)


######## explanation : EGPD    model ########################################
#############################################################################
## Section intentionally removed during cleanup (was unrelated and corrupted).

# -----------------------------
# Truncated Beta transform
# -----------------------------
function G_truncbeta(v, κ; a=1 / 32)
    if v < 0
        return 0.0
    elseif v > 1
        return 1.0
    else
        B = Beta(κ, κ)
        numerator = cdf(B, (1 / 2 - a) * v + a) - cdf(B, a)
        denominator = cdf(B, 1 / 2) - cdf(B, a)
        return numerator / denominator
    end
end

# -----------------------------
# Power transform
# -----------------------------
G_power(v, κ) = v < 0 ? 0.0 : (v > 1 ? 1.0 : v^κ)
# -----------------------------
# Mixture transform
# -----------------------------
function G_mixture(v, ξ1, ξ2, p)
    if v < 0
        return 0.0
    elseif v > 1
        return 1.0
    else
        return p * v^ξ1 + (1 - p) * v^ξ2
    end
end

function plot_transforms(; vmin=0.0, vmax=1.0, n=400)
    v = range(vmin, vmax; length=n)

    fig = Figure()

    # Power transform
    ax1 = Axis(fig[1, 1], xlabel=L"v", ylabel=L"G(v)", title="Power", limits=(vmin, vmax, 0, 1), width=120 * 1.2, height=120 * 1.5 * 1.2)
    κ_power = 2.0
    gvals_power = [G_power(x, κ_power) for x in v]
    plt1 = lines!(ax1, v, v, color=:red, linewidth=1.5, linestyle=:dash, label=L"y=v")
    plt2 = lines!(ax1, v, gvals_power, linewidth=2, color=mycolors[1], label=L"v^{%$κ_power}")
    # axislegend(ax1, position=:lt)

    # Mixture transform
    ax2 = Axis(fig[1, 2], xlabel=L"v", ylabel=L"G(v)", title="Mixture power", limits=(vmin, vmax, 0, 1), width=120 * 1.2, height=120 * 1.5 * 1.2)
    hideydecorations!(ax2)
    ξ1, ξ2, p = 0.5, 2.0, 0.3
    gvals_mixture = [G_mixture(x, ξ1, ξ2, p) for x in v]
    lines!(ax2, v, v, color=:red, linewidth=1.5, linestyle=:dash)
    plt_3 = lines!(ax2, v, gvals_mixture, linewidth=2, color=mycolors[2], label=L"$p v^{%$ξ1} + (1-p) v^{%$ξ2}$ ($p=%$p$)")
    # axislegend(ax2, position=:lt)

    # Truncated Beta transform
    ax3 = Axis(fig[1, 3], xlabel=L"v", ylabel=L"G(v)", title="Truncated Beta", limits=(vmin, vmax, 0, 1), width=120 * 1.2, height=120 * 1.5 * 1.2)
    κ_beta = 2
    gvals_beta = [G_truncbeta(x, κ_beta) for x in v]
    lines!(ax3, v, v, color=:red, linewidth=1.5, linestyle=:dash)
    plt_4 = lines!(ax3, v, gvals_beta, linewidth=2, color=mycolors[3], label=L"G(v;\, \kappa=%$κ_beta)")
    # axislegend(ax3, position=:lt)
    hideydecorations!(ax3)
    colgap!(fig.layout, 1, 22)
    colgap!(fig.layout, 2, 22)

    Legend(fig[1, 4], [plt1, plt2, plt_3, plt_4], [L"y=v", L"v^{%$κ_power}", L"$p v^{%$ξ1} + (1-p) v^{%$ξ2}$ ($p=%$p$)", L"G(v;\, \kappa=%$κ_beta)"])
    resize_to_layout!(fig)

    return fig
end
fontsize = 16
update_theme!(fontsize=fontsize, theme_latexfonts())
fig_EGPD = plot_transforms()
savefigcrop("./plots_paper/Transform_comparison_with_identity.pdf", fig_EGPD)

#############################################################################
######## explanation : cond sim      ########################################
#############################################################################

####################################################################################################

Random.seed!(123)
x = 0:0.01:1
y = 0
dx, dy = length(x), length(y)
locations = vcat(([xx yy] for xx in x for yy in y)...)
Mat_h = [sqrt(sum(abs2, locations[i, :] - locations[j, :]))
         for i in axes(locations, 1), j in axes(locations, 1)]
nlocs = size(locations, 1)
coordll = locations
D = size(coordll, 1)

# Create GaussianField with ExpExp model
gf = GaussianField(coordll, ExpExp(1.0, 0.5, 1.0))

#make a similar truncated, 
Psec = (fill(0.4, D) .+ 0.2) ./ 2
Psect = quantile(Normal(), Psec)
my_λ = 1 .- Psec# proba de pluie !
my_range = 0.2
my_sill = 1.0
my_order = 1 / 2
d = SpatialBernoulli(my_range, my_sill, my_order, my_λ, Mat_h)

Pluie_en_station = rand(d)

# Bounds
lb = ifelse.(Pluie_en_station .== 1, Psect, -Inf)
ub = ifelse.(Pluie_en_station .== 0, Psect, Inf)

# Sample 10 truncated MVN vectors from R
gf_mvnorma_trunc = TruncatedMVNormal(fill(0., size(gf.coords, 1)), cov_spatiotemporal(gf.model, gf.Mat_h, 0), lb, ub)
X = TruncatedMVN.sample(gf_mvnorma_trunc, 1)
#this verion is a lot better. takes 0.01s for one sim, so 0.05 hour for 18000 samples  !
begin
    markers = [p == 1 ? :circle : :cross for p in Pluie_en_station]

    fig_exemple = Figure()

    ax1 = Axis(
        fig_exemple[1, 1],
        ylabel="Multivariate Normal Value",
        #title="Rainfall generated conditionaly to rain occurrence"
    )
ylims!(ax1,-4.5,4.5)
    lines!(ax1, x, Psect, color=:red, label=L"\Phi^{-1}(\mathbb{P}(Y_s = 0))")
    scatter!(ax1, x[(X.<Psect[1])], X[(X.<Psect[1])][:, 1], color=:blue, marker=:cross, label=L"X_R(s) \mid Y_s = 0")
    scatter!(ax1, x[(X.>Psect[1])],X[(X.>Psect[1])][:, 1], color=:blue, marker=:circle, label=L"X_R(s) \mid Y_s = 1")

    # Add legend entries for markers
    axislegend(ax1, position=:rb,orientation=:horizontal)
    # Legend(fig_exemple, bbox=BBox(650, 220, 650, 35),
    #     [MarkerElement(color=:black, marker=:circle), MarkerElement(color=:black, marker=:cross)],
    #     [L"Y_s =1", L"Y_s =0"])
    D = length(Psect)
    precip_amount = similar(X)
    for i in 1:D
        # @show Pluie_en_station[i]
        if Pluie_en_station[i] == 1
            # Gaussian truncated below Psect[i] (rain)
            d_trunc = truncated(Normal(), Psect[i], Inf)
            u = cdf(d_trunc, X[i])
            precip_amount[i] = quantile(Exponential(10), u)  # mean=10
        else
            precip_amount[i] = 0.0   # no rain
        end
    end
    mycolors
    ax2 = Axis(fig_exemple[2, 1], xlabel=L"s", ylabel="Precip amount (mm)")
    markers2 = [p == 1 ? :circle : :cross for p in Pluie_en_station]
    scatter!(ax2, x, precip_amount[:, 1], color=:green, marker=markers2, label=L"R_s")
    # Legend(fig_exemple, bbox=BBox(650, 220, 35, 350),
        # [MarkerElement(color=:black, marker=:circle), MarkerElement(color=:black, marker=:cross)],
        # [L"Y_s =1", L"Y_s =0"])
    axislegend(ax2, position=:rt)

    fig_exemple
end


savefigcrop("./plots_paper/Example_conditional_generation_dim1.pdf", fig_exemple)

#############################################################################
######## get data + make maps data ##########################################
#############################################################################


precision_scale = 50 # meter
LON_min = -6 # West
LON_max = 10 # East
LAT_min = 41 # South
LAT_max = 52 # North
station_50Q = CSV.read("../00data/transformedECAD_stations.csv", DataFrame)
station_50Q.STANAME = station_50Q.STANAME .|> shortname
Yobs = Matrix(CSV.read("../00data/transformedECAD_Yobs.csv", header=false, DataFrame))
Robs = Matrix(CSV.read("../00data/transformedECAD_Robs.csv", DataFrame, header=false)[:, :])'

station_name = station_50Q.STANAME
STAID = station_50Q.STAID
D = size(Yobs, 1)
LAT_idx = station_50Q.LAT_idx

LON_idx = station_50Q.LON_idx

locsdata = CSV.read("../00data/transformedECAD_stations.csv", DataFrame, header=true)
locations = Matrix(CSV.read("../00data/transformedECAD_locs.csv", DataFrame, header=false))
locsdata.LON = locations[:, 1]
locsdata.LAT = locations[:, 2]
value = fill(0, length(LON_idx))

value[select_plot] .= 1
FR_map_spell = map_with_stations(LON_idx, LAT_idx, value; station_name=string.(STAID), colorbar_label="")

savefigcrop(FR_map_spell, "./plots_paper/map_rien")


my_distance = Matrix(CSV.read("../00data/transformedECAD_locsdistances.csv", header=false, DataFrame))
my_locations = hcat(station_50Q.LON_idx, station_50Q.LAT_idx)

prettytable= station_50Q[:,[1,2,4,5,6]]
using DataFrames

D = size(Yobs, 1)

mean_proba = round.((mean(Yobs[s, :]) for s in 1:D), digits=2)

# Function for consecutive zeros
function max_consecutive_zeros(v)
    m = 0
    c = 0
    for x in v
        if x == 0
            c += 1
            m = max(m, c)
        else
            c = 0
        end
    end
    return m
end

max_dry = round.((max_consecutive_zeros(Yobs[s, :]) for s in 1:D),digits=2)

max_rain = round.([maximum(Robs[s, :]) for s in 1:D],digits=2)

# Mean rain (only when raining)
mean_rain = round.([mean(filter(!=(0), Robs[s, :])) for s in 1:D],digits=2)

using PrettyTables
column_labels=["ID","Name","Lat","Lon","Elevation (m)","Rain frequency","Max dry spell duration (days)","Max daily rainfall","Mean rainfall"]
prettytable=Matrix(prettytable)
prettytable=hcat(prettytable,mean_proba,max_dry,max_rain,mean_rain)
pretty_table(prettytable; header=column_labels,backend = Val(:latex))

###############################################################################################################
#get fitted HMMSPA model
date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end

my_N = length(every_year)
n2t = dayofyear_Leap.(every_year)

doss_save = "../13PeriodicHMMSpatialBernoulli/res_real_data/"

begin
    Mat_h = Matrix(CSV.read("../00data/transformedECAD_locsdistances.csv", DataFrame, header=false))


    my_K = 4
    my_degree_of_P = 1
    maxiter = 100
    my_autoregressive_order = 1
    R0 = 500
    QMC_m = 30
    datafile = doss_save * "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"


    hmmspa = load(datafile)["hmm"]
    datafile = doss_save * "/parameters/K" * string(1) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"
    hmmspa1 = load(datafile)["hmm"]

end

#############################################################################
######## get ICL plot              ##########################################
#############################################################################


# using SwarmMakie # not yet updated with latest Makie -> cause downgrading versions
 dficl = CSV.read(doss_save * "/ICLmemory01_m30_endll.csv", DataFrame)

difcl2 = copy(dficl)

begin
    fig_ICL = Figure()
    ax = Axis(fig_ICL[1, 1], xlabel=L"$K$", ylabel="ICL",
        xticks=minimum(difcl2.K):maximum(difcl2.K))
    x = difcl2.K
    y = difcl2.ICL
    scatter!(ax, x, y, marker=ifelse.(difcl2.memory .== 0, :circle, :utriangle), color=mycolors[difcl2.deg.+1], markersize=14)
    Legend(fig_ICL, bbox=BBox(600, 200, 50, 300),
        vec([MarkerElement(color=mycolors[d+1], marker=m, markersize=14) for m in (:circle, :utriangle), d in unique(difcl2.deg)]),
        vec([L"m = %$(ifelse.(m.==1, 0, 1)),\, \text{deg} = %$(d)" for m in eachindex([:circle, :utriangle]), d in unique(difcl2.deg)]), nbanks=2)
    fig_ICL
end

# savefigcrop("plots_paper/ICLmemory01_m30_endll.pdf", fig_ICL)


# Filter data
difcl2 = difcl2[difcl2.memory .== 1, :]

# Define marker per degree (extend if needed)
deg_vals = sort(unique(difcl2.deg))
markers = Dict(d => m for (d, m) in zip(deg_vals, (:circle, :utriangle, :diamond, :rect)))

fig_ICL = Figure()
ax = Axis(
    fig_ICL[1, 1],
    xlabel = L"$K$",
    ylabel = "ICL",
    xticks = minimum(difcl2.K):maximum(difcl2.K)
)

# Plot by degree
for d in deg_vals
    idx = difcl2.deg .== d
    scatter!(
        ax,
        difcl2.K[idx],
        difcl2.ICL[idx];
        marker = markers[d],
        color = mycolors[d + 1],
        markersize = 14,
        label = L"\text{deg} = %$d"
    )
end

axislegend(ax; nbanks=2, position=:rb)
fig_ICL
savefigcrop("./plots_paper/ICLmemory01_m30_endll.pdf", fig_ICL)

using PrettyTables
# Filter
df = dficl[dficl.memory .== 1, :]

# Unique sorted values
Ks   = sort(unique(df.K))
degs = sort(unique(df.deg))

# Initialize matrix
ICLmat = Matrix{Float64}(undef, length(degs), length(Ks))

# Fill matrix
for (i, d) in enumerate(degs)
    for (j, k) in enumerate(Ks)
        idx = (df.deg .== d) .& (df.K .== k)
        ICLmat[i, j] = df.ICL[idx][1]
    end
end

# Row names
row_labels = ["deg = $d" for d in degs]

# Column names
col_labels = ["K = $k" for k in Ks]

# Print LaTeX table
pretty_table(ICLmat; header=col_labels,backend = Val(:latex))

#############################################################################
######## get rain info             ##################################
#############################################################################
idx_days = [findall(x -> dayofyear_Leap.(x) == m, every_year) for m in 1:366]

daily_rain_histo = [monthly_agg(Robs[j, :], idx_days) for j in 1:D]
daily_rain_prob = [monthly_agg((Robs[j, :] .> 0), idx_days) .* 1.0 for j in 1:D]
for j in 1:D
    for i in 1:366
        daily_rain_prob[j][i] = daily_rain_prob[j][i] / length(idx_days[i])
    end
end

# daily_rain_prob[1]
# @time "Plot daily mean" begin
#     fig_month_RR = Figure(fontsize=19)

#     for (idx, j) in enumerate(select_plot[[1, 5, 2, 6, 4, 3]])
#         row = (idx - 1) ÷ 3 + 1
#         col = (idx - 1) % 3 + 1

#         ax = Axis(fig_month_RR[row, col],
#             title=station_50Q.STANAME[j],
#             xlabel="",
#             ylabel=(col == 1 ? "Rain (mm)" : ""),
#             width=170 * 1.25,
#             height=170)
#  # Set x-axis ticks to month abbreviations
#  month_days = dayofyear_Leap.(Date.(2000, 1:12))
#  month_labels = [string(monthabbr(m)[1]) for m in 1:12]
#  ax.xticks = (vcat(month_days, 366), vcat(month_labels, ""))



#         # Plot observations
#             obs_data = [mean(daily_rain_histo[j][ m]) for m in 1:366]
#             scatterlines!(ax, 1:366, obs_data;
#                 color=:blue,
#                 linewidth=1.75)

#         ylims!(ax, 0, 410)
#     end
#     resize_to_layout!(fig_month_RR)
#     fig_month_RR
# end
# savefigcrop("plots_paper/raincharac.pdf", fig_month_RR)

@time "Plot daily mean and rain probability" begin
    fig_month_RR = Figure(fontsize=19)

    for (idx, j) in enumerate(select_plot[[1, 5, 2, 6, 4, 3]])
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1

        # Main axis for rain amount
        ax = Axis(fig_month_RR[row, col],
            title = "$(station_50Q.STANAME[j]) - $(station_50Q.STAID[j])",
            xlabel="",
            ylabel=(col == 1 ? "Rain (mm)" : ""),
            width=170 * 1.25,
            height=170)

        # Month ticks
        month_days = dayofyear_Leap.(Date.(2000, 1:12))
        month_labels = [string(monthabbr(m)[1]) for m in 1:12]
        ax.xticks = (vcat(month_days, 366), vcat(month_labels, ""))

        # Observed mean rainfall
        obs_data = [mean(daily_rain_histo[j][m]) for m in 1:366]
        lines!(ax, 1:366, obs_data; color=mycolors[1], linewidth=1.5, label="Mean rain")

        # --- Second y-axis for rain probability ---
        ax2 = Axis(fig_month_RR[row, col],
            yaxisposition=:right,
            ylabel=(col == 3 ? L"\mathbb{P}(R>0)" : ""),
            yticklabelcolor=mycolors[4],
            ylabelcolor=mycolors[4],
            ytickcolor=mycolors[4])
        hidespines!(ax2)
        hidexdecorations!(ax2)
        linkxaxes!(ax, ax2)

        # Compute daily probability of rain
        prob_rain = [mean(daily_rain_prob[j][m]) for m in 1:366]

        lines!(ax2, 1:366, prob_rain; color=mycolors[4], linewidth=1.5, label="Rain prob.")

        linkxaxes!(ax, ax2)
        ylims!(ax, 0, 400 * (1 + 0.05))
        ylims!(ax2, 0, 1.05)
    end

    resize_to_layout!(fig_month_RR)
    fig_month_RR
end

savefigcrop("./plots_paper/raincharac.pdf", fig_month_RR)


#############################################################################
######## get rain proba map        ##########################################
#############################################################################


 p_FR_map_mean_prob1 = map_with_stations(LON_idx, LAT_idx, [[mean((hmmspa.B[k, :, j, 1])) for j in 1:D] for k in 1:my_K], colorbar_show=true, colorbar_label=L"\mathbb{P}(Y_s^{(n)} = 1\mid Z = k,Y_s^{(n-1)} =0)", precision_scale=50)
savefigcrop("./plots_paper" * "/RainProbaMap_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * "previousdry.pdf", p_FR_map_mean_prob1)

p_FR_map_mean_prob2 = map_with_stations(LON_idx, LAT_idx, [[mean((hmmspa.B[k, :, j, 2])) for j in 1:D] for k in 1:my_K], colorbar_show=true, colorbar_label=L"\mathbb{P}(Y_s^{(n)} = 1\mid Z = k,Y_s^{(n-1)}=1)", precision_scale=50)
savefigcrop("./plots_paper" * "/RainProbaMap_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * "previouswet.pdf", p_FR_map_mean_prob2)



#############################################################################
######## get rain proba plot       ##########################################
#############################################################################
month_days = dayofyear_Leap.(Date.(2000, 1:12))
month_labels = [string(monthabbr(m)[1]) for m in 1:12]


# Generate one plot per station (only first selected plot gets legend)
begin
    fig_proba_rain = Figure()

    for (idx, j) in enumerate(select_plot[[1, 5, 2, 6, 4, 3]])
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1

        if col > 1
            ax = Axis(fig_proba_rain[row, col],
                title="$(station_50Q.STANAME[j])",
                limits=(0, 367, 0, 1), yticks=0:0.2:1, width=200, height=200,
                ylabel=L"\lambda_{k,s,h}^{(t)}")
            hideydecorations!(ax; label=true, ticklabels=true, ticks=false, grid=false, minorgrid=true, minorticks=false)
        else
            ax = Axis(fig_proba_rain[row, col], width=200, height=200,
                title="$(station_50Q.STANAME[j])",
                limits=(0, 367, 0, 1), yticks=0:0.2:1,
                ylabel=L"\lambda_{k,s,h}^{(t)}")
        end
        for k in 1:my_K
            color = mycolors[k]

            # Plot solid line (h=0)
            lines!(ax, 1:366, [hmmspa.B[k, t, j, 1] for t in 1:366],
                color=color, linewidth=3)

            # Plot dashed line (h=1) if applicable
            if size(hmmspa.B, 4) == 2
                lines!(ax, 1:366, [hmmspa.B[k, t, j, 2] for t in 1:366],
                    color=color, linewidth=3, linestyle=:dash)
            end
        end

        hlines!(ax, [0.5], color=:black, linestyle=:dot)

        # Set x-axis ticks to month abbreviations
        ax.xticks = (vcat(month_days, 366), vcat(month_labels, ""))
    end

    # Create legend in the right margin
    if my_autoregressive_order > 0
        Legend(fig_proba_rain[1:2, 4],
            [LineElement(color=:black, linestyle=:solid),
                LineElement(color=:black, linestyle=:dash),
                [LineElement(color=mycolors[k], linestyle=:solid) for k in 1:my_K]...],
            [L"h=0", L"h=1", [L"k=%$k" for k in 1:my_K]...])
    else
        Legend(fig_proba_rain[1:2, 4],
            [[LineElement(color=mycolors[k], linestyle=:solid) for k in 1:my_K]...],
            [[L"k=%$k" for k in 1:my_K]...])
    end
    resize_to_layout!(fig_proba_rain)
    fig_proba_rain
end

# Save the figure
savefigcrop("./plots_paper" * "/RainProba_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_proba_rain)

#############################################################################
######## get rain occurrence spatial scale ##################################
#############################################################################
month_days = dayofyear_Leap.(Date.(2000, 1:12))
month_labels = [string(monthabbr(m)[1]) for m in 1:12]
begin
    fig_ρ = Figure()
    ax = Axis(fig_ρ[1, 1],
        # title=L"Spatial scale parameter ",
        ylabel=L"Spatial scale $\rho_{CY,k}^{(t)}$ (km)",
        limits=(0, 367, 100, 500))

    for k in 1:my_K
        lines!(ax, 1:366, hmmspa.R[k, :], color=mycolors[k], label=L"k= %$k",linewidth=3)
    end

    lines!(ax, 1:366, hmmspa1.R[1, :], color=:black, label=L"$K=1$", linestyle=:dash,linewidth=3)

    # Set x-axis ticks to month abbreviations

    ax.xticks = (vcat(month_days, 366), vcat(month_labels, ""))

    axislegend(ax, position=:lt, nbanks=4)
    fig_ρ
end
savefigcrop("./plots_paper" * "/R_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_ρ)







#############################################################################
######## get rain occurrence transitions   ##################################
#############################################################################

begin
    fig_Q = Figure()

    for k in 1:my_K
        row = (k - 1) ÷ 2 + 1
        col = (k - 1) % 2 + 1

        ax = Axis(fig_Q[row, col],
            # title=L"Transition parameters $Q^{(t)}(%$k, x)$",
            limits=(0, 367, 0, 1), yticks=0:0.2:1,
            width=310)
        if col > 1
            hideydecorations!(ax; label=true, ticklabels=true, ticks=false, grid=false, minorgrid=true, minorticks=false)
        end
        for l in 1:my_K
            label_text = L"Q^{(t)}(%$(k), %$(l))"
            lines!(ax, 1:366, hmmspa.A[k, l, :], color=mycolors[l], label=label_text)
        end

        hlines!(ax, [0.5], color=:black, linestyle=:dot)

        # Set x-axis ticks to month abbreviations
        month_days = dayofyear_Leap.(Date.(2000, 1:12))
        month_labels = [string(monthabbr(m)[1]) for m in 1:12]
        ax.xticks = (vcat(month_days, 366), vcat(month_labels, ""))
        # ax.xticklabelsize = 10
        # ax.yticklabelsize = 10
        resize_to_layout!(fig_Q)

        axislegend(ax, position=:ct, nbanks=4, labelsize=14.7, patchlabelgap=0, colgap=6, framevisible=false)
    end

    fig_Q
end
savefigcrop("./plots_paper" * "/Qt_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_Q)

################################################################################################################"
################



# get seasonal info
year_range = unique(year.(every_year));
idx_year = [findall(x -> year.(x) == m, every_year) for m in year_range];
select_month = 1:12
idx_months = [findall(x -> month.(x) == m, every_year) for m in 1:12]
idx_month_vcat = vcat(idx_months[select_month]...)
idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_months];

#############################################################################
######## get pretty viterbi plot           ##################################
#############################################################################

select_year = [51, 33, 31, 27, 4]

ẑ = CSV.read("../00data/transformedECAD_zhatbis.csv", DataFrame,header=false).Column1
select_years = unique(year.(every_year))[select_year]
ẑ_per_cat = [findall(ẑ .== k) for k in 1:my_K]
begin
    year_nb = length(select_year)
    z_hat_mat = zeros(year_nb, 366)

    for (i, y) in enumerate(select_year)
        println(i, y)
        if isleapyear(year_range[y])
            z_hat_mat[i, :] = ẑ[idx_year[y]]
        else
            z_hat_mat[i, :] = [ẑ[idx_year[y]]; 0]
        end
    end
    thick = 1
    fig_viterbi = Figure()
    ax = Axis(fig_viterbi[1, 1], xticks=(vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), "")), yticks=(1:5, string.(select_years)), yreversed=true)

    heatmap!(ax, 1:366, 1:year_nb, permutedims(z_hat_mat), colormap=mycolors[1:my_K], colorrange=(1, my_K))
    hlines!(ax, (1:(year_nb-1)) .+ 0.5, color=:black, linewidth=4)
    xlims!(ax, 0, 367)
    fig_viterbi
end
savefigcrop(fig_viterbi, "./plots_paper" * "/Viterbi_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf")


#############################################################################
######## get intensity marginals           ##################################
#############################################################################
K = my_K
degree_of_P = 2 # degree of trigo in rain intensity
T = 366


@load "../precip_censored_gaussian/res_real_data/periodicEGPD_K" * string(K) * string(degree_of_P) * ".jld2" di

# plot the parameters

modellist_noK = JLD2.load("../precip_amount_withclassinmarginal_boundxi/res_real_data/EGPD_constant_noK.jld2", "modellist")
modellist = JLD2.load("../precip_amount_noclass_boundxi/res_real_data/periodicEGPD" * string(degree_of_P) * ".jld2", "di")

function fig_marginal_param(stations)
    fig_marginals = Figure(fontsize=17)
    ww = 185
    hh = 150
    xticks = (vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], ""))
    for (idx, i) in enumerate(stations)
        # Each station gets its own row
        row = idx

        # Subplot 1: p parameter
        ax1 = Axis(fig_marginals[row, 1], title=idx == 1 ? L"p_{k,s}^{(t)}" : "",
            limits=(nothing, nothing, 0, 1), width=ww, height=hh, xticks=xticks, yticks=0:0.2:1, titlesize=22)
        lines!(ax1, 1:T, [modellist[i][t].p for t in 1:T], color=:black)
        for k in 1:K
            lines!(ax1, 1:T, [di[i][k, t].p for t in 1:T], color=mycolors[k],
                label=idx == 1 ? L"k= %$k" : nothing)
        end

        # Subplot 2: σ parameter
        ax2 = Axis(fig_marginals[row, 2], title=idx == 1 ? L"\sigma_{k,s}^{(t)}" : "", width=ww, height=hh, xticks=xticks, titlesize=22)
        lines!(ax2, 1:T, [modellist[i][t].tail_part.G.σ for t in 1:T], color=:black)
        for k in 1:K
            lines!(ax2, 1:T, [di[i][k, t].tail_part.G.σ for t in 1:T], color=mycolors[k])
        end
        ylims!(ax2, 0, 20)

        # Subplot 3: ξ parameter
        ax3 = Axis(fig_marginals[row, 3], title=idx == 1 ? L"\xi_{k,s}^{(t)}" : "", width=ww, height=hh, xticks=xticks, yticks=0:0.2:1, titlesize=22)
        lines!(ax3, 1:T, [modellist[i][t].tail_part.G.ξ for t in 1:T], color=:black)
        for k in 1:K
            lines!(ax3, 1:T, [di[i][k, t].tail_part.G.ξ for t in 1:T], color=mycolors[k])
        end

        # Subplot 4: κ parameter
        ax4 = Axis(fig_marginals[row, 4], title=idx == 1 ? L"\kappa_{k,s}^{(t)}" : "", width=ww, height=hh, xticks=xticks, yticks=0:0.5:2, titlesize=22)
        lines!(ax4, 1:T, [modellist[i][t].tail_part.V.α for t in 1:T], color=:black)
        for k in 1:K
            lines!(ax4, 1:T, [di[i][k, t].tail_part.V.α for t in 1:T], color=mycolors[k])
        end
        ylims!(ax4, 0, 2.25)

        # Add station name as label on the left
        Label(fig_marginals[row, 0], station_50Q.STANAME[i],
            tellheight=false, rotation=0)
    end

    Legend(fig_marginals[:, 0],
        [LineElement(color=:black)]
        ∪
        [LineElement(color=mycolors[k]) for k in 1:K],
        vcat([L"K=1"], [L"k= %$k" for k in 1:K]))
    resize_to_layout!(fig_marginals)
    fig_marginals
end

fig_marginals_all = fig_marginal_param(select_plot[[1, 5, 2, 6, 4, 3]])
fig_marginals_lille_nice = fig_marginal_param(select_plot[[1, 3]])
fig_marginals_other = fig_marginal_param(select_plot[[5, 2, 6, 4]])

savefigcrop("./plots_paper/" * string(K) * "trydeg" * string(degree_of_P) * "sim.pdf", fig_marginals_all)
savefigcrop("./plots_paper/4trydeg2sim_Lille_Nice.pdf", fig_marginals_lille_nice)
savefigcrop("./plots_paper/4trydeg2sim_other.pdf", fig_marginals_other)

#############################################################################
######## get latent covariance model       ##################################
#############################################################################

covmodel = load("../precip_censored_gaussian/res_real_data/GMcov_withmarginalsdeg" * string(degree_of_P) * ".jld2")["fitted"]
covmodel = load("../precip_censored_gaussian/res_real_data/GMcov_withmarginalsdeg" * string(degree_of_P) * ".jld2")["fitted"]

################# plot the covariance model ###################################
coordll = [locsdata.LON locsdata.LAT]
gf = GaussianField(coordll, covmodel)
u_lags = 0:3
h_vals = 0:50:900
begin
    fig_cov = Figure()
    ax = Axis(fig_cov[1, 1],
        xlabel="Distance h (km)",
        ylabel="Covariance",
        # title="Spatio-temporal fitted covariance"
    )

    for u in u_lags
        covs = cov_spatiotemporal.(Ref(covmodel), h_vals, u)
        lines!(ax, h_vals, covs, linewidth=2, label=L"u = %$u")
    end

    axislegend(ax, position=:rt)

    # # Add formula and parameter annotations using text!
    # text!(ax, 150, 0.7, 
    #     text=L"C(h,u) = \frac{\sigma^2}{\left(\left(\frac{u}{a}\right)^{2\alpha} + 1\right)^{b+\delta}}\mathcal{M}\!\left(\frac{h}{\sqrt{\left(\left(\frac{u}{a}\right)^{2\alpha} + 1\right)^b}}; r; \nu\right)",
    #     color=:black, align=(:left, :center))

    # text!(ax, 400, 0.55,
    #     text=L"\sigma = %$(round(covmodel.σ², digits = 2)),\, b = %$(round(covmodel.β, digits = 2))",
    #   color=:black, align=(:left, :center))

    # text!(ax, 400, 0.5,
    #     text=L"r = %$(round(covmodel.c, digits = 2)),\, \nu = %$(round(covmodel.ν, digits = 2))", color=:black, align=(:left, :center))

    # text!(ax, 400, 0.45,
    #     text=L"a = %$(round(covmodel.a, digits = 2)),\, \alpha =  %$(round(covmodel.α, digits = 2)),\, \delta  = %$(round(covmodel.δ, digits = 2))",
    #     color=:black, align=(:left, :center))

    fig_cov
end

savefigcrop("./plots_paper/covariance_fitted.pdf", fig_cov)

######################################################################################################################
data = JLD2.load("../precip_censored_gaussian/res_real_data/periodicEGPD_K$(my_K)$(degree_of_P)_Sim_ZYR.jld2")
Rs = data["Rs"]
Ys = data["Ys"]
Zs = data["Zs"]
Nb = size(Zs, 2)



#############################################################################
######## get dry and wet spells            ##################################
#############################################################################
len_spell_hist = [pmf_spell(Yobs[j, idx_month_vcat], dw) for j in 1:D, dw in 0:1];

len_spell_simu = [pmf_spell(Ys[j, idx_month_vcat, i], dw) for i in 1:Nb, j in 1:D, dw in 0:1]

make_range(y, step=1) = range(extrema(y)..., step=step)
begin #spells (Makie)
    # -------------------- DRY SPELLS --------------------
    dry_or_wet = 1 # dry
    fig_spell_dry = Figure()
    for (idx, j) in enumerate(select_plot[[1, 5, 2, 6, 4, 3]])
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1

        ax = Axis(fig_spell_dry[row, col],
            title=station_50Q.STANAME[j],
            xlabel=(row == 2 ? "Nb of days" : ""),
            ylabel=(col == 1 ? "Distribution" : ""),
            yscale=log10)

        all_spells = len_spell_simu[:, j, dry_or_wet]
        spell_range = 1:1:(1+maximum(vcat(reduce(vcat, all_spells), len_spell_hist[j, dry_or_wet])))

        # Enveloppe 0-100
        errorlinehist!(ax, all_spells;
            color=:red,
            secondarycolor=:grey,
            label=(idx == 1 ? "Simu q_{0,100}" : nothing),
            normalization=:probability,
            bins=spell_range,
            errortype=:percentile,
            percentiles=[0, 100],
            secondaryalpha=0.4,
            centertype=:median)

        # Interquartile 25-75
        errorlinehist!(ax, all_spells;
            color=:red,
            secondarycolor=:red,
            label=(idx == 1 ? "Simu q_{25,75}" : nothing),
            normalization=:probability,
            bins=make_range(reduce(vcat, all_spells)),
            errortype=:percentile,
            percentiles=[25, 75],
            secondaryalpha=0.5,
            centertype=:median)

        # Observations
        histo_spell = len_spell_hist[j, dry_or_wet]
        errorlinehist!(ax, [histo_spell];
            label=(idx == 1 ? "Obs" : nothing),
            color=:blue,
            linewidth=1.5,
            normalization=:probability,
            bins=spell_range,
            errortype=:percentile,
            alpha=0.8)

        xlims!(ax, 0, maximum(spell_range) + 2)
        ylims!(ax, 1e-4, 1)
        if col > 1
            hideydecorations!(ax; label=true, ticklabels=true, ticks=false, grid=false, minorgrid=true, minorticks=false)
        end
        if idx == 2
            #     axislegend(ax, position=:rt)
            axislegend(ax,
                # fig_spell_dry, bbox=BBox(550, 220, 55, 350),
                [PolyElement(color=:grey, alpha=0.5), [PolyElement(color=:red, alpha=0.5),
                        LineElement(color=:red)], LineElement(color=:blue)],
                [L"Simu $q_{0,100}$", L"Simu $q_{25,75}$", "Obs"], position=:rt)
        end
    end
    rowgap!(fig_spell_dry.layout, 2)
    resize_to_layout!(fig_spell_dry)
    fig_spell_dry
end

savefigcrop("./plots_paper" * "/dry_spells_few_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_spell_dry)

begin #spells (Makie)
    # -------------------- WET SPELLS --------------------
    dry_or_wet = 2 # wet
    fig_spell_wet = Figure()
    for (idx, j) in enumerate(select_plot[[1, 5, 2, 6, 4, 3]])
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1

        ax = Axis(fig_spell_wet[row, col],
            title=station_50Q.STANAME[j],
            xlabel=(row == 2 ? "Nb of days" : ""),
            ylabel=(col == 1 ? "Distribution" : ""),
            yscale=log10, xticks=0:10:50)

        all_spells = len_spell_simu[:, j, dry_or_wet]
        spell_range = 1:1:(1+maximum(vcat(reduce(vcat, all_spells), len_spell_hist[j, dry_or_wet])))

        # Enveloppe 0-100
        errorlinehist!(ax, all_spells;
            color=:red,
            secondarycolor=:grey,
            label=(idx == 1 ? "Simu q_{0,100}" : nothing),
            normalization=:probability,
            bins=spell_range,
            errortype=:percentile,
            percentiles=[0, 100],
            secondaryalpha=0.4,
            centertype=:median)

        # Interquartile 25-75
        errorlinehist!(ax, all_spells;
            color=:red,
            secondarycolor=:red,
            label=(idx == 1 ? "Simu q_{25,75}" : nothing),
            normalization=:probability,
            bins=make_range(reduce(vcat, all_spells)),
            errortype=:percentile,
            percentiles=[25, 75],
            secondaryalpha=0.5,
            centertype=:median)

        # Observations
        histo_spell = len_spell_hist[j, dry_or_wet]
        errorlinehist!(ax, [histo_spell];
            label=(idx == 1 ? "Obs" : nothing),
            color=:blue,
            linewidth=1.5,
            normalization=:probability,
            bins=spell_range,
            errortype=:percentile,
            alpha=0.8)

        xlims!(ax, 0, maximum(spell_range) + 2)
        ylims!(ax, 1e-4, 1)
        if col > 1
            hideydecorations!(ax; label=true, ticklabels=true, ticks=false, grid=false, minorgrid=true, minorticks=false)
        end
        if idx == 2
            #     axislegend(ax, position=:rt)
            axislegend(ax,
                # fig_spell_wet, bbox=BBox(550, 220, 55, 350),
                [PolyElement(color=:grey, alpha=0.5), [PolyElement(color=:red, alpha=0.5),
                        LineElement(color=:red)], LineElement(color=:blue)],
                [L"Simu $q_{0,100}$", L"Simu $q_{25,75}$", "Obs"], position=:rt)
        end
    end
    rowgap!(fig_spell_wet.layout, 2)
    resize_to_layout!(fig_spell_wet)
    fig_spell_wet
end
savefigcrop("./plots_paper" * "/wet_spells_few_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_spell_wet)


#############################################################################
######## get cumulative precip             ##################################
#############################################################################
month_rain_simu = [monthly_agg(Rs[j, :, i], idx_all) for j in 1:D, i in 1:Nb]
month_rain_histo = [monthly_agg(Robs[j, :], idx_all) for j in 1:D]

popo(color) = [PolyElement(color=color, alpha=0.18 * 1^2), PolyElement(color=color, alpha=0.18 * 2^2, points=Point2f[(0, 1 / 3), (1, 1 / 3), (1, 2 / 3), (0, 2 / 3)]), LineElement(color=color, alpha=0.5)]

qs = [0.9, 0.5, 0.1]
@time "Plot monthly quantile" begin
    fig_month_RR = Figure(fontsize=19)

    for (idx, j) in enumerate(select_plot[[1, 5, 2, 6, 4, 3]])
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1

        ax = Axis(fig_month_RR[row, col],
            title=station_50Q.STANAME[j],
            xlabel="",
            ylabel=(col == 1 ? "Rain (mm)" : ""),
            xticks=(1:12, string.(first.(monthabbr.(1:12)))),
            width=170 * 1.25,
            height=170)

        # Plot simulation quantiles with errorline!
        for (α, per) in enumerate([[0, 100], [25, 75]])
            for (cc, q) in enumerate(qs)
                data_matrix = [quantile(month_rain_simu[j, i][:, m], q) for m in 1:12, i in 1:Nb]
                scc = cc == 3 ? 4 : cc
                errorline!(ax, 1:12, data_matrix;
                    label=(α == 1 && idx == 1 ? L"Simu  $q_{%$(Int(q*100))}$" : nothing),
                    secondaryalpha=0.18 * α^2,
                    centertype=:median,
                    errortype=:percentile,
                    percentiles=per,
                    color=mycolors[scc],
                    secondarycolor=mycolors[scc])
            end
        end

        # Plot observations
        for (cc, q) in enumerate(qs)
            obs_data = [quantile(month_rain_histo[j][:, m], q) for m in 1:12]
            scatterlines!(ax, 1:12, obs_data;
                label=(q == qs[1] && idx == 1 ? "Obs" : nothing),
                color=:blue,
                linewidth=1.75)
        end
        ylims!(ax, 0, nothing)
    end
    Legend(fig_month_RR[:, 4],
        [
            popo(mycolors[1]), popo(mycolors[2]), popo(mycolors[4]), [LineElement(color=:blue),
                MarkerElement(color=:blue, marker=:circle, markersize=10)]
        ],
        [L"Simu $q_{90}$", L"Simu $q_{50}$", L"Simu $q_{10}$", "Obs"])
    resize_to_layout!(fig_month_RR)
    fig_month_RR
end
savefigcrop("./plots_paper/SpaTgen_PeriodicEGPDS_deg" * string(degree_of_P) * ".pdf", fig_month_RR)

ndays = 5
Robs3 = hcat([vcat(fill(0, ndays), rolling(sum, Robs[s, :], ndays)) for s in 1:D]...)'
Rs3 = cat(
    [hcat([vcat(fill(0, ndays), rolling(sum, Rs[s, :, i], ndays)) for s in 1:D]...)'
     for i in 1:Nb]...;
    dims=3
)

idx_day = [findall(x -> dayofyear.(x) == m, every_year) for m in 1:366]

idx_all_day = [intersect(yea, mon) for yea in idx_year, mon in idx_day]

day5_rain_simu = [monthly_agg(Rs3[j, :, i], idx_all_day) for j in 1:D, i in 1:Nb]
day5_rain_histo = [monthly_agg(Robs3[j, :], idx_all_day) for j in 1:D]
qs = [0.9, 0.5, 0.1]
@time "Plot 5day quantile" begin
    fig_month_RR_5day = Figure(fontsize=19)

    for (idx, j) in enumerate(select_plot[[1, 5, 2, 6, 4, 3]])
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1

        ax = Axis(fig_month_RR_5day[row, col],
            title=station_50Q.STANAME[j],
            xlabel="",
            ylabel=(col == 1 ? "Rain (mm)" : ""),
            xticks=(vcat(month_days, 366), vcat(month_labels, "")),
            width=170 * 1.25,
            height=170)
        xlims!(ax, 1, 366)
        # Plot simulation quantiles with errorline!
        for (α, per) in enumerate([[0, 100], [25, 75]])
            for (cc, q) in enumerate(qs)
                data_matrix = [quantile(day5_rain_simu[j, i][:, m], q) for m in 1:366, i in 1:Nb]
                scc = cc == 3 ? 4 : cc
                errorline!(ax, 1:366, data_matrix;
                    label=(α == 1 && idx == 1 ? L"Simu  $q_{%$(Int(q*100))}$" : nothing),
                    secondaryalpha=0.18 * α^2,
                    centertype=:median,
                    errortype=:percentile,
                    percentiles=per,
                    color=mycolors[scc],
                    secondarycolor=mycolors[scc])
            end
        end

        # Plot observations
        for (cc, q) in enumerate(qs)
            obs_data = [quantile(day5_rain_histo[j][:, m], q) for m in 1:366]
            # scatter!(ax, 1:366, obs_data;
            #     color=:blue)
            lines!(ax, 1:366, obs_data;
                label=(q == qs[1] && idx == 1 ? "Obs" : nothing),
                color=:blue,
                linewidth=1.75, alpha=0.8)
        end
        ylims!(ax, 0, nothing)
    end
    Legend(fig_month_RR_5day[:, 4],
        [
            popo(mycolors[1]), popo(mycolors[2]), popo(mycolors[4]), [LineElement(color=:blue)]
        ],
        [L"Simu $q_{90}$", L"Simu $q_{50}$", L"Simu $q_{10}$", "Obs"])
    resize_to_layout!(fig_month_RR_5day)
    fig_month_RR_5day
end
savefigcrop("./plots_paper/5daycumulSpaTgen_PeriodicEGPDS_deg" * string(degree_of_P) * ".pdf", fig_month_RR_5day)

#############################################################################
######## get autocor from precip           ##################################
#############################################################################

acfrange = 0:9
@views aa = [autocor(Rs[j, :, i], acfrange) for j in 1:D, i in 1:Nb]

begin
    fig_ACF = Figure()

    for (idx, j) in enumerate(select_plot[[1, 5, 2, 6, 4, 3]])
        row = (idx - 1) ÷ 3 + 1
        col = (idx - 1) % 3 + 1

        ax = Axis(fig_ACF[row, col],
            title=station_50Q.STANAME[j],
            xlabel=(row == 2 ? "Lag" : ""),
            ylabel=(col == 1 ? "ACF" : ""))

        # Plot observations
        obs_acf = autocor(Robs[j, :], acfrange)
        errorline!(ax, acfrange, stack(aa[j, :], dims=1)';
            secondarycolor=mycolors[4],
            color=mycolors[4],
            label=(idx == 1 ? L"Simu $q_{0,100}$" : nothing),
            errortype=:percentile,
            percentiles=[0, 100],
            alpha=1,
            secondaryalpha=0.4,
            linewidth=2,
            centertype=:median)
        scatterlines!(ax, acfrange, obs_acf;
            label=(idx == 1 ? "Obs" : nothing),
            linewidth=1.5,
            color=:blue, marker=:circle, alpha=0.95)
        # Plot simulation ACF with errorline!

        if idx == 1
            axislegend(ax, position=:rt)
        end
    end

    resize_to_layout!(fig_ACF)
    fig_ACF
end
savefigcrop("./plots_paper/acfrr.pdf", fig_ACF)

#############################################################################
######## get pairwise occurrence proba     ##################################
#############################################################################

JJA = [6, 7, 8]
MAM = [3, 4, 5]
SON = [9, 10, 11]
DJF = [12, 1, 2]
SEASONS = [DJF, MAM, JJA, SON]
seasonname = ["DJF", "MAM", "JJA", "SON"]
idx_seasons = [findall(month.(every_year) .∈ tuple(season)) for season in SEASONS]

f00(a, b) = (a .== 0) .& (b .== 0)
f01(a, b) = (a .== 0) .& (b .== 1)
f00lag(a, b) = (a[2:end] .== 0) .& (b[1:end-1] .== 0)
f01lag(a, b) = (a[2:end] .== 0) .& (b[1:end-1] .== 1)

function pairwise_prob(Y, f)
    D = size(Y, 1)
    P = zeros(D, D)
    for i in 1:D, j in 1:D
        P[i, j] = mean(f(Y[i, :], Y[j, :]))
    end
    return P
end

# Function to compute seasonal pairwise probability
function seasonal_pairwise(Yobs, Ysim, idx_seasons, f)
    D, Nt = size(Yobs)
    Nb = size(Ysim, 3)
    nseasons = length(idx_seasons)
    Pobs = Vector{Matrix{Float64}}(undef, nseasons)
    Psim = Vector{Matrix{Float64}}(undef, nseasons)

    for s in 1:nseasons
        idx = idx_seasons[s]
        # slice time indices for the season
        Yobs_s = Yobs[:, idx]
        Ysim_s = Ysim[:, idx, :]

        # observed
        Pobs[s] = pairwise_prob(Yobs_s, f)
        # simulated (average over Nb)
        Psim[s] = mean([pairwise_prob(Ysim_s[:, :, b], f) for b in 1:Nb])
    end
    return Pobs, Psim
end

# --- Compute seasonal probabilities for each event
Pobs_00, Psim_00 = seasonal_pairwise(Yobs, Ys, idx_seasons, f00)
Pobs_01, Psim_01 = seasonal_pairwise(Yobs, Ys, idx_seasons, f01)
Pobs_00lag, Psim_00lag = seasonal_pairwise(Yobs, Ys, idx_seasons, f00lag)
Pobs_01lag, Psim_01lag = seasonal_pairwise(Yobs, Ys, idx_seasons, f01lag)

# --- Build 4×5 grid (first column season labels)
titles = [
    L"\mathbb{P}(Y_i=0, Y_j=0)",
    L"\mathbb{P}(Y_i=0, Y_j=1)",
    L"\mathbb{P}(Y_i=0, Y_j=0)",
    L"\mathbb{P}(Y_i=0, Y_j=1)"
]

begin
    fig_pairwise = Figure(fontsize=20)

    for s in 1:4  # season row
        # First column: season label
        Label(fig_pairwise[s, 1], seasonname[s], rotation=0, tellheight=false)

        for i in 1:4  # probability type column
            Pobs_s = [Pobs_00, Pobs_01, Pobs_00lag, Pobs_01lag][i][s]
            Psim_s = [Psim_00, Psim_01, Psim_00lag, Psim_01lag][i][s]

            ax = Axis(fig_pairwise[s, i+1],
                title=(s == 1 ? titles[i] : ""),
                xlabel=(s == 4 ? "Observed" : ""),
                ylabel=(i == 1 ? "Simulated" : ""),
                limits=(0, 1, 0, 1),
                aspect=DataAspect(), xminorticks=IntervalsBetween(2), yminorticks=IntervalsBetween(2), xminorticksvisible=true, xminorgridvisible=true, yminorticksvisible=true, yminorgridvisible=true,
                width=165, height=165, xticks=([0, 0.5, 1], ["0", "0.5", "1"]), yticks=([0, 0.5, 1], ["0", "0.5", "1"]))

            # Scatter plot
            scatter!(ax, vec(Pobs_s), vec(Psim_s), markersize=8, color=mycolors[1])

            # Diagonal reference line
            lines!(ax, [0, 1], [0, 1], color=:black, linestyle=:dash, linewidth=1)
            # text!(ax, 0.05, 0.85, text="RMSE = " * string(round(sqrt(mean(abs2, (vec(Pobs_s) .- vec(Psim_s)))), digits=3)))
            if i > 1
                hideydecorations!(ax, grid=false, ticks=false,
                    minorgrid=false, minorticks=false)
            end
            if s < 4
                hidexdecorations!(ax, grid=false, ticks=false,
                    minorgrid=false, minorticks=false)
            end
        end
    end
    # colgap!(fig_pairwise.layout, Relative(-0.05))

    # colgap!(fig_pairwise.layout, 1, Relative(-0.0))
    resize_to_layout!(fig_pairwise)
    fig_pairwise
end
savefigcrop("./plots_paper/probs01.pdf", fig_pairwise)

#############################################################################
######## pairwise cor, pairwise corTail, CR  ################################
#############################################################################
cor_hist = cor(Robs');
qtail = 0.95
corT_hist = corTail(Robs', qtail);

cor_mean_simu = mean(cor(Rs[:, :, i]') for i in 1:Nb);

corT_mean_simu = mean(corTail(Rs[:, :, i]', qtail) for i in 1:Nb);


CRobs = continuity_ratio(Robs)
CRsim = continuity_ratio(Rs)

threshold = 1.5
dmax = 800

mask = .!(isnan.(CRobs) .| isnan.(CRsim) .| (Mat_h .> dmax))
x = CRobs[mask]
y = CRsim[mask]

begin
    fig_cor = Figure(fontsize=18)
    www = 225
    hhh = 225
    # Subplot 1: Correlations
    ax1 = Axis(fig_cor[1, 1], xlabel="Observations", ylabel="Simulations", aspect=DataAspect(), width=www, height=hhh, xminorticks=IntervalsBetween(2), yminorticks=IntervalsBetween(2))
    scatter!(ax1, vec_triu(cor_hist), vec_triu(cor_mean_simu), color=mycolors[1], label="Correlations")
    ablines!(ax1, 0, 1, linestyle=:dash, color=:black)
    xlims!(ax1, -0.1, 1)
    ylims!(ax1, -0.1, 1)
    # text!(ax1, (0.372, 0.175), text="RMSE = " * string(round(sqrt(mean(abs2, vec_triu(cor_hist) - vec_triu(cor_mean_simu))), digits=3)), align=(:left, :top), space=:relative)
    Legend(fig_cor[0, 1], ax1, framevisible=false)

    # Subplot 2: Tail correlations
    ax2 = Axis(fig_cor[1, 2], xlabel="Observations", aspect=DataAspect(), width=www, height=hhh, xminorticks=IntervalsBetween(2), yminorticks=IntervalsBetween(2),)
    scatter!(ax2, vec_triu(corT_hist), vec_triu(corT_mean_simu), color=mycolors[2], label=L"Tail index $q=%$qtail$")
    ablines!(ax2, 0, 1, linestyle=:dash, color=:black)
    xlims!(ax2, -0.1, 1)
    ylims!(ax2, -0.1, 1)
    # text!(ax2, (0.372, 0.175), text="RMSE = " * string(round(sqrt(mean(abs2, vec_triu(corT_hist) .- vec_triu(corT_mean_simu))), digits=3)), align=(:left, :top), space=:relative)
    Legend(fig_cor[0, 2], ax2, framevisible=false)

    # Subplot 3: Continuity Ratio
    ax3 = Axis(fig_cor[1, 3], xlabel="Observations", aspect=DataAspect(), width=www, height=hhh)
    scatter!(ax3, x, y, color=mycolors[3], label="CR")
    ablines!(ax3, 0, 1, linestyle=:dash, color=:black)
    xlims!(ax3, 0, threshold)
    ylims!(ax3, 0, threshold)
    Legend(fig_cor[0, 3], ax3, framevisible=false)
    Label(fig_cor, bbox=BBox(1465, 0.5, 595, 0.15), "Pairs with distance < $dmax km", fontsize=17)
    # text!(ax3, (0.372, 0.175), text="RMSE = " * string(round(sqrt(mean(abs2, x .- y)), digits=3)), align=(:left, :top), space=:relative)
    rowgap!(fig_cor.layout, -90)
    resize_to_layout!(fig_cor)
    fig_cor
end
savefigcrop("./plots_paper/corplot.pdf", fig_cor)

#############################################################################
######## ROR  autocor and distribution      ################################
############################################################################
## to do the actual simulations, uncomment the following lines
# import Distributions: Categorical
# Random.seed!(12345)
# Ysnostate = zeros(Bool, D, length(n2t), Nb)
# @time "Simulations  Y" Threads.@threads for i in 1:Nb
#     z, Y = my_rand(hmmspa1, n2t; seq=true)
#     Ysnostate[:, :, i] = Y'
# end
# #get Ysindep from other_models.jl
# hmm_fit = load("HMMIndep/K" * string(my_K) * "DegP" * string(my_degree_of_P) * "memory" * string(my_autoregressive_order) * ".jld2")["hmm"]
# begin
#     zs = zeros(Int, length(n2t), Nb)
#     ys = zeros(Bool, length(n2t), D, Nb)
#     @time "Simulations Z, Y" for i in 1:Nb
#         zs[:, i], ys[:, :, i] = rand(hmm_fit, n2t; y_ini=Yobs[:, 1:my_autoregressive_order]', z_ini=1, seq=true)
#     end
# end

Ysnostate = JLD2.load("../precip_censored_gaussian/res_real_data/K1_Y.jld2")["Ysnostate"]
Ysindep = JLD2.load("../precip_censored_gaussian/res_real_data/K4_indep_Y.jld2")["Ysindep"]
Ysm0 = JLD2.load("../precip_censored_gaussian/res_real_data/K4_m0_Y.jld2")["Ys"]

RRmax = 0
RORo = [mean(r .> RRmax) for r in eachcol(Yobs)]
RORs = [[mean(r .> RRmax) for r in eachcol(rr)] for rr in eachslice(Ys, dims=3)]
RORsnostate = [[mean(r .> RRmax) for r in eachcol(rr)] for rr in eachslice(Ysnostate, dims=3)]
RORsindep = [[mean(r .> RRmax) for r in eachrow(rr)] for rr in eachslice(Ysindep, dims=3)]
RORsm0 = [[mean(r .> RRmax) for r in eachcol(rr)] for rr in eachslice(Ysm0, dims=3)]

## ROR density
maxlag = 10


begin
    # Makie ROR distribution and autocorrelation (2×4 grid)

    fig_ROR = Figure(fontsize=19)
    wwww = 200
    hhhh = 150
    # Row 1: Distribution plots
    for m in eachindex(idx_seasons)
        row = 1
        col = m

        # Distribution subplot
        ax_dist = Axis(fig_ROR[row, col],
            xlabel="ROR",
            ylabel=col == 1 ? "Distribution" : "",
            title=seasonname[m],
            width=wwww,
            height=hhhh)
        xax = 0:(1/D):1.0
        xaxbin = vcat(xax, [1.01])
        errorlinehist!(ax_dist, [RORsindep[i][idx_seasons[m]] for i in 1:Nb];
            label="",
            color=:gray,
            secondarycolor=:gray, normalization=:probability,
            bins=xaxbin,
            errortype=:percentile,
            percentiles=[0, 100],
            alpha=0.5,
            secondaryalpha=0.2,
            centertype=:median)
        errorlinehist!(ax_dist, [RORsnostate[i][idx_seasons[m]] for i in 1:Nb];
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
        # errorlinehist!(ax_dist, [RORsm0[i][idx_seasons[m]] for i in 1:Nb];
        #     label="",
        #     color=mycolors[5],
        #     secondarycolor=mycolors[5],
        #     normalization=:probability,
        #     bins=xaxbin,
        #     errortype=:percentile,
        #     percentiles=[0, 100],
        #     alpha=0.5,
        #     secondaryalpha=0.2,
        #     centertype=:median)
        errorlinehist!(ax_dist, [RORs[i][idx_seasons[m]] for i in 1:Nb];
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
        scatter!(ax_dist, xax, [mean(RORo[idx_seasons[m]] .== x) for x in xax], color=:blue, markersize=6, label=m == 1 ? "Observations" : nothing)

        ylims!(ax_dist, 0, 0.06)
        col > 1 && hideydecorations!(ax_dist, grid=false)

        # Autocorrelation subplot
        row = 2
        ax_acf = Axis(fig_ROR[row, col],
            xlabel="Lag",
            ylabel=col == 1 ? "ACF" : "",
            width=wwww,
            height=hhhh,
        )

        # # Model 1: K = 1
        rorsim = [RORsnostate[i][idx_seasons[m]] for i in 1:Nb]
        acf_sim = [autocor(rorsim[i], 0:maxlag) for i in 1:length(rorsim)]
        errorline!(ax_acf, 0:maxlag, stack(acf_sim, dims=1)',
            color=mycolors[2],
            secondarycolor=mycolors[2],
            errortype=:percentile,
            percentiles=[0, 100],
            secondaryalpha=0.2,
            centertype=:median)
        # # Model 2: Indep
        rorsim = [RORsindep[i][idx_seasons[m]] for i in 1:Nb]
        acf_sim = [autocor(rorsim[i], 0:maxlag) for i in 1:length(rorsim)]
        errorline!(ax_acf, 0:maxlag, stack(acf_sim, dims=1)',
            color=:gray,
            secondarycolor=:gray,
            errortype=:percentile,
            percentiles=[0, 100],
            secondaryalpha=0.2,
            centertype=:median)
        # # # Model 3: K = 4, m=0               
        # rorsim = [RORsm0[i][idx_seasons[m]] for i in 1:Nb]
        # acf_sim = [autocor(rorsim[i], 0:maxlag) for i in 1:length(rorsim)]
        # errorline!(ax_acf, 0:maxlag, stack(acf_sim, dims=1)',
        #     color=mycolors[5],
        #     secondarycolor=mycolors[5],
        #     errortype=:percentile,
        #     percentiles=[0, 100],
        #     secondaryalpha=0.2,
        #     centertype=:median)
        # Model 4: K = my_K
        rorsim = [RORs[i][idx_seasons[m]] for i in 1:Nb]
        acf_sim = [autocor(rorsim[i], 0:maxlag) for i in 1:length(rorsim)]
        errorline!(ax_acf, 0:maxlag, stack(acf_sim, dims=1)',
            color=mycolors[4],
            secondarycolor=mycolors[4],
            errortype=:percentile,
            percentiles=[0, 100],
            secondaryalpha=0.2,
            centertype=:median)


        # Observations
        acf_obs = autocor(RORo, 0:maxlag)
        scatter!(ax_acf, 0:maxlag, acf_obs, color=:blue, markersize=7)
        col > 1 && hideydecorations!(ax_acf, grid=false)
    end
    Legend(fig_ROR[:, 5],
        [
            [
                [LineElement(color=:gray), PolyElement(color=:gray, alpha=0.2)],
                [LineElement(color=mycolors[2]), PolyElement(color=mycolors[2], alpha=0.2)],
                [LineElement(color=mycolors[4]), PolyElement(color=mycolors[4], alpha=0.2)]
            ],
            [MarkerElement(color=:blue, marker=:circle, markersize=8)]
        ],
        [
            [
                L"Ind $K = 4$",
                L"SPA $K = 1$",
                L"SPA $K = %$my_K$"
            ],
            ["Observations"]
        ],
        [L"HMM $m=1$", " "]
    )
    resize_to_layout!(fig_ROR)
    fig_ROR
end

savefigcrop("./plots_paper" * "/RORplots_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_ROR)

#############################################################################
######## ROR  dry spell      ################################
############################################################################
# for perc = [0.1, 0.2, 0.5]

QQ = [5, 95]
function fig_ror_spell(perc)
    fig_spell = Figure(fontsize=17)
    wwwww = 220
    hhhhh = 150
    for m in eachindex(idx_seasons)
        row, col = (m - 1) ÷ 2 + 1, (m - 1) % 2 + 1
        ax = Axis(fig_spell[row, col], yscale=log10,
            xlabel=row == 2 ? "Nb of days" : "",
            ylabel=col == 1 ? "Probability" : "",
            title=seasonname[m],
            xticks=(0:3:15),
            width=wwwww,
            height=hhhhh)

        len_ror_hist = pmf_spell(RORo[idx_seasons[m]] .≤ perc, 1)
        len_ror_simu = [pmf_spell(RORs[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]
        len_ror_simum0 = [pmf_spell(RORsm0[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]
        len_ror_simunos = [pmf_spell(RORsnostate[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]
        len_ror_simuindep = [pmf_spell(RORsindep[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]

        # # Observations
        errorlinehist!(ax, [len_ror_hist], secondarycolor=:black, color=:blue, linewidth=2,
            normalization=:probability, bins=make_range(len_ror_hist),
            errortype=:percentile, label=(m == 1 ? "Obs" : nothing))

        # # HMM-SPA K=1
        sim_range = make_range(reduce(vcat, len_ror_simunos))
        errorlinehist!(ax, len_ror_simunos,
            color=mycolors[2],
            secondarycolor=mycolors[2],
            normalization=:probability, bins=sim_range,
            errortype=:percentile, percentiles=QQ, secondaryalpha=0.2,
            centertype=:median, alpha=0.6, linewidth=1.5
        )

        # # HMM-Indep K=4
        sim_range = make_range(reduce(vcat, len_ror_simuindep))
        errorlinehist!(ax, len_ror_simuindep,
            secondarycolor=:gray,
            color=:gray,
            normalization=:probability, bins=sim_range,
            errortype=:percentile, percentiles=QQ, secondaryalpha=0.2,
            centertype=:median, alpha=0.6, linewidth=1.5)

        # # HMM-SPA K=4, m=0
        sim_range = make_range(reduce(vcat, len_ror_simum0))
        errorlinehist!(ax, len_ror_simum0,
            secondarycolor=mycolors[6],
            color=mycolors[6],
            normalization=:probability, bins=sim_range, linestyle=:dash,
            errortype=:percentile, percentiles=QQ, secondaryalpha=0.2,
            centertype=:median, alpha=0.6, linewidth=1.5)

        # # HMM-SPA K=my_K
        sim_range = make_range(reduce(vcat, len_ror_simu))
        errorlinehist!(ax, len_ror_simu,
            secondarycolor=mycolors[4],
            color=mycolors[4],
            normalization=:probability, bins=sim_range,
            errortype=:percentile, percentiles=QQ, secondaryalpha=0.2,
            centertype=:median, alpha=0.6, linewidth=1.5)
        ylims!(ax, 9e-4, 1)
        xlims!(ax, 0.9, 15.5)
        if col > 1
            hideydecorations!(ax, grid=false, ticks=false,
                minorgrid=false, minorticks=false)
        end

    end
    Legend(fig_spell[:, 3],
        [
            [
                [LineElement(color=mycolors[6], linestyle=:dash), PolyElement(color=mycolors[6], alpha=0.2)]
            ],
            [
                [LineElement(color=:gray), PolyElement(color=:gray, alpha=0.2)],
                [LineElement(color=mycolors[2]), PolyElement(color=mycolors[2], alpha=0.2)],
                [LineElement(color=mycolors[4]), PolyElement(color=mycolors[4], alpha=0.2)]
            ],
            [MarkerElement(color=:blue, marker=:circle, markersize=8)]
        ],
        [
            [L"SPA $K = %$my_K$"],
            [
                L"Ind $K = 4$",
                L"SPA $K = 1$",
                L"SPA $K = %$my_K$"
            ],
            ["Observations"]
        ],
        [L"HMM $m=0$", L"HMM $m=1$", " "]
    )
    # colsize!(fig_spell.layout, 3, Relative(1/6))
    resize_to_layout!(fig_spell)
    fig_spell
end


fig_spell0p1 = fig_ror_spell(0.1)
fig_spell0p2 = fig_ror_spell(0.2)

savefigcrop("./plots_paper" * "/dry_spells_ROR" * string(0.1) * "_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_spell0p1)

savefigcrop("./plots_paper" * "/dry_spells_ROR" * string(0.2) * "_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_spell0p2)

#############################################################################
######## QER   autocor and distribution      ################################
############################################################################

# xmaxli = [1., 1., 1., 0.5, 0.4, 0.25]
# ymaxli = [0.23, 0.27, 0.35, 0.41, 0.6, 0.86]
# qmaxli = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
# nli = 6
##QER density

# for i in 1:nli
# qmax = qmaxli[i]
function fig_QER(qmax)
    RRmaxq = [quantile(r, qmax) for r in eachrow(Robs)]
    RORoq = mean(Robs .> RRmaxq, dims=1)
    RORsq = [mean(rr .> RRmaxq, dims=1) for rr in eachslice(Rs, dims=3)]

    ## ROR density
    maxlag = 4
    K = 4
    m = 1
    maxlag = 10

    fig_QER = Figure(fontsize=18)
    wwww_qer = 200
    hhhh_qer = 150

    # Row 1: Distribution plots
    for m in eachindex(idx_seasons)
        xax = 0:(1/D):1.0
        xaxbin = vcat(xax, [1.01])

        row = 1
        col = m

        ax_dist = Axis(fig_QER[row, col],
            xlabel=L"\mathrm{QER}_{%$qmax}",
            ylabel=col == 1 ? "Distribution" : "",
            title=seasonname[m],
            width=wwww_qer,
            height=hhhh_qer,
            yticks=0:0.2:0.8)

        errorlinehist!(ax_dist, [RORsq[i][idx_seasons[m]] for i in 1:Nb];
            label=label = m == 1 ? "HMM-SPA" : nothing,
            color=mycolors[4],
            secondarycolor=mycolors[4],
            normalization=:probability,
            bins=xaxbin,
            errortype=:percentile,
            percentiles=[0, 100],
            alpha=1,
            secondaryalpha=0.4,
            centertype=:median)
        scatter!(ax_dist, xax, [mean(RORoq[idx_seasons[m]] .== x) for x in xax], color=:blue, markersize=7, label=m == 1 ? "Observations" : nothing)


        # ylims!(ax_dist, 0, ymaxli[i])
        # xlims!(ax_dist, -0.05 * xmaxli[i], xmaxli[i])
        col > 1 && hideydecorations!(ax_dist, grid=false, minorgrid=false)
        ylims!(ax_dist, -0.05, 0.6)
        xlims!(ax_dist, -0.01, 0.4)
        if col == 1
            axislegend(ax_dist, position=:rt)
        end
    end

    # Row 2: ACF plots
    for m in eachindex(idx_seasons)
        rorsim = [RORsq[i][idx_seasons[m]] for i in 1:Nb]
        acf_sim = [autocor(rorsim[i], 0:maxlag) for i in 1:length(rorsim)]

        row = 2
        col = m

        ax_acf = Axis(fig_QER[row, col],
            xlabel="Lag",
            ylabel=col == 1 ? "ACF" : "",
            width=wwww_qer,
            height=hhhh_qer)

        errorline!(ax_acf, 0:maxlag, stack(acf_sim, dims=1)',
            color=mycolors[4],
            secondarycolor=mycolors[4],
            errortype=:percentile,
            percentiles=[0, 100],
            secondaryalpha=0.4,
            centertype=:median)

        acf_obs = autocor(RORoq[1, :], 0:maxlag)
        scatter!(ax_acf, 0:maxlag, acf_obs, color=:blue, markersize=7)
        col > 1 && hideydecorations!(ax_acf, grid=false)


    end
    resize_to_layout!(fig_QER)
    fig_QER
end

fig_QER0p95 = fig_QER(0.95)
savefigcrop("./plots_paper/QER" * string(0.95) * ".pdf", fig_QER0p95)

for qmax in [0.75, 0.8, 0.85, 0.9, 0.99]
    fig_QER_q = fig_QER(qmax)
    savefigcrop("./plots_paper/QER" * string(qmax) * ".pdf", fig_QER_q)
end

#############################################################################
######## ROR 1mm   autocor and distribution      ################################
############################################################################

##QER density

RRmax1mm = 1.0
RORo1mm = mean(Robs .> RRmax1mm, dims=1)[1, :]
RORs1mm = [mean(rr .> RRmax1mm, dims=1)[1, :] for rr in eachslice(Rs, dims=3)]


## ROR density
maxlag = 4
K = 4
m = 1
maxlag = 10
begin
    fig_ROR1mm = Figure(fontsize=17)
    wwww_1mm = 200
    hhhh_1mm = 150

    # Row 1: Distribution plots
    for m in eachindex(idx_seasons)
        xax = 0:(1/D):1.0
        xaxbin = vcat(xax, [1.01])

        row = 1
        col = m

        ax_dist = Axis(fig_ROR1mm[row, col],
            xlabel=L"\mathrm{ROR}_{1.0}",
            ylabel=col == 1 ? "Distribution" : "",
            title=seasonname[m],
            width=wwww_1mm,
            height=hhhh_1mm)

        errorlinehist!(ax_dist, [RORs1mm[i][idx_seasons[m]] for i in 1:Nb];
            label=m == 1 ? "HMM-SPA" : nothing,
            color=mycolors[4],
            secondarycolor=mycolors[4],
            normalization=:probability,
            bins=xaxbin,
            errortype=:percentile,
            percentiles=[0, 100],
            alpha=1,
            secondaryalpha=0.4,
            centertype=:median)

        scatter!(ax_dist, xax, [mean(RORo1mm[idx_seasons[m]] .== x) for x in xax],
            color=:blue, markersize=6, label=m == 1 ? "Observations" : nothing)

        ylims!(ax_dist, -0.01, 0.2)
        col > 1 && hideydecorations!(ax_dist, grid=false, minorgrid=false)
        if col == 1
            axislegend(ax_dist, position=:rt)
        end
    end

    # Row 2: ACF plots
    for m in eachindex(idx_seasons)
        rorsim = [RORs1mm[i][idx_seasons[m]] for i in 1:Nb]
        acf_sim = [autocor(rorsim[i], 0:maxlag) for i in 1:length(rorsim)]

        row = 2
        col = m

        ax_acf = Axis(fig_ROR1mm[row, col],
            xlabel="Lag",
            ylabel=col == 1 ? "ACF" : "",
            width=wwww_1mm,
            height=hhhh_1mm)

        errorline!(ax_acf, 0:maxlag, stack(acf_sim, dims=1)',
            color=mycolors[4],
            secondarycolor=mycolors[4],
            errortype=:percentile,
            percentiles=[0, 100],
            secondaryalpha=0.4,
            centertype=:median)

        acf_obs = autocor(RORo1mm, 0:maxlag)
        scatter!(ax_acf, 0:maxlag, acf_obs, color=:blue, markersize=7)
        col > 1 && hideydecorations!(ax_acf, grid=false, minorgrid=false)
    end

    resize_to_layout!(fig_ROR1mm)
    fig_ROR1mm
end

savefigcrop("./plots_paper/RER_1mm.pdf", fig_ROR1mm)

begin
    fig_spell_1mm = Figure(fontsize=17)
    wwwww = 220
    hhhhh = 150
    for m in eachindex(idx_seasons)
        row, col = (m - 1) ÷ 2 + 1, (m - 1) % 2 + 1
        ax = Axis(fig_spell_1mm[row, col], yscale=log10,
            xlabel=row == 2 ? "Nb of days" : "",
            ylabel=col == 1 ? "Probability" : "",
            title=seasonname[m],
            xticks=(0:3:15),
            width=wwwww,
            height=hhhhh)
        perc = 0.1
        len_ror_hist = pmf_spell(RORo1mm[idx_seasons[m]] .≤ perc, 1)
        len_ror_simu = [pmf_spell(RORs1mm[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]

        # Observations
        errorlinehist!(ax, [len_ror_hist], secondarycolor=:black, color=:blue, linewidth=2,
            normalization=:probability, bins=make_range(len_ror_hist),
            errortype=:percentile, label=(m == 1 ? "Obs" : nothing))

        # HMM-SPA K=my_K
        sim_range = make_range(reduce(vcat, len_ror_simu))
        errorlinehist!(ax, len_ror_simu,
            secondarycolor=mycolors[4],
            color=mycolors[4],
            normalization=:probability, bins=sim_range,
            errortype=:percentile, percentiles=QQ, secondaryalpha=0.2,
            centertype=:median, alpha=0.6, linewidth=1.5,
            label=(m == 1 ? "HMM-SPA - K = $my_K" : nothing))

        ylims!(ax, 9e-4, 1)
        xlims!(ax, 0, 13)
        if col > 1
            hideydecorations!(ax, grid=false, ticks=false,
                minorgrid=false, minorticks=false)
        end
    end

    Legend(fig_spell_1mm[:, 3],
        [
            [LineElement(color=mycolors[4]), PolyElement(color=mycolors[4], alpha=0.2)],
            MarkerElement(color=:blue, marker=:circle, markersize=8),
        ],
        ["HMM-SPA - K = $my_K", "Observations"],
    )

    resize_to_layout!(fig_spell_1mm)
    fig_spell_1mm
end

savefigcrop("./plots_paper" * "/dry_spells_ROR1mm" * string(perc) * "_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".pdf", fig_spell_1mm)
