cd(@__DIR__)
using LaTeXStrings
using CSV, DataFrames
using Random: Random
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

date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end

my_N = length(every_year)
n2t = dayofyear_Leap.(every_year)

Yobs = Matrix(CSV.read("../00data/transformedECAD_Yobs.csv", header = false, DataFrame))
data = JLD2.load("../23precip_intensity/res_real_data/periodicEGPD_K$(4)$(2)_Sim_ZYR.jld2")
Rs = data["Rs"]
Ys = data["Ys"]
Zs = data["Zs"]
Nb = size(Zs, 2)
D=size(Yobs, 1)
Robs = Matrix(CSV.read("../00data/transformedECAD_Robs.csv", header = false, DataFrame))


year_range = unique(year.(every_year));
idx_year = [findall(x -> year.(x) == m, every_year) for m in year_range];
select_month = 1:12
idx_months = [findall(x -> month.(x) == m, every_year) for m in 1:12]
idx_month_vcat = vcat(idx_months[select_month]...)
idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_months];

every_year = date_start:Day(1):date_end
seuil = 1
rainy = [(mean(r) .> seuil) for r in eachrow(Robs)]

plot(rainy)
rainysim = [collect(mean(r) > seuil for r in eachcol(rr)) for rr in eachslice(Rs, dims=3)]
rainysim[1]



function spell_info(mask::AbstractVector{Bool})
	values, lengths = rle(mask)

	ends   = cumsum(lengths)
	starts = ends .- lengths .+ 1

	keep = findall(values)   # runs where mask == true

	return (
		lengths = lengths[keep],
		starts  = starts[keep],
		ends    = ends[keep],
	)
end

ror_info = spell_info(rainy)

start_dates = every_year[ror_info.starts]
end_dates   = every_year[ror_info.ends]

for i in eachindex(ror_info.lengths)
	if ror_info.lengths[i]>20
		println("Wet episode $i:")
		println("  Start: ", start_dates[i])
		println("  End:   ", end_dates[i])
		println("  Length: ", ror_info.lengths[i], " days")
	end
end

# ror_info.length gives the length of rain episodes. I want a plot of the distrib of this,also highlighting the very long ones by giving the start date of them. then I also want to plot the distribution in simulation with 5-95 interquartile range of the distribution, stored in Rs.

obs_lengths = ror_info.lengths


D, N, Nsim = size(Rs)
size(Robs)

sim_distributions = [pmf_spell(rainysim[i], true) for i in 1:Nb]

begin
make_range(y, step=1) = range(extrema(y)..., step=step)

QQ=[0.0005,0.9555]
    fig_spell = Figure(fontsize=17)
    wwwww = 220
    hhhhh = 150
 
        ax = Axis(fig_spell[1,1],
            xlabel="Nb of days" ,yscale=log10,
            ylabel= "Probability" ,
            xticks=(0:5:55),
            width=wwwww,
            height=hhhhh)
        len_ror_hist = pmf_spell(rainy, true)

        # Observations 
        errorlinehist!(ax, [len_ror_hist], color=:blue, linewidth=2,
            normalization=:probability, bins=make_range(len_ror_hist),
            errortype=:percentile, label="Obs")

        # HMM-SPA K=my_K
        sim_range = make_range(reduce(vcat, sim_distributions))
        errorlinehist!(ax, sim_distributions,
            secondarycolor=:gray,
            color=:red,
            normalization=:probability, bins=sim_range,
            errortype=:percentile, percentiles=QQ, secondaryalpha=0.2,
            centertype=:median, alpha=0.6, linewidth=1.5,
            label=("sim"))

        # ylims!(ax, 9e-4, 1)
        # xlims!(ax, 0, 55)
    
    

    Legend(fig_spell[:, 2],
        [
            [LineElement(color=:red), PolyElement(color=:red, alpha=0.2)],
            MarkerElement(color=:blue, marker=:circle, markersize=8),
        ],
        ["Simulations", "Observations"],
    )

    resize_to_layout!(fig_spell)
    fig_spell
end

begin
    all_values = vcat(obs_lengths, reduce(vcat, sim_distributions))
bins = collect(1:maximum(all_values))

fig_spell = Figure(fontsize=17)
ax = Axis(fig_spell[1,1],
    xlabel="Nb of days",
    ylabel="Probability",
    yscale=log10,
    xticks=0:5:55)

# Observations
errorlinehist!(ax, [obs_lengths],
    color=:blue,
    linewidth=2,
    normalization=:probability,
    bins=bins,
    errortype=:percentile,
    label="Obs")

# Simulations
errorlinehist!(ax, sim_distributions,
    color=:red,
    alpha=0.6,
    secondarycolor=:gray,
    errortype=:percentile,
    percentiles=[0.05,0.95],
    centertype=:median,
    normalization=:probability,
    bins=bins,
    linewidth=1.5,
    label="Sim")


fig_spell
end


begin
    function wet_spell_lengths(mask::Vector{Bool})
        values, lengths = rle(mask)  # run-length encoding
        return lengths[values .== true]  # keep only wet spells
    end 
    obs_lengths = wet_spell_lengths(rainy)
    sim_lengths = [wet_spell_lengths(sim) for sim in rainysim]
    all_lengths = vcat(obs_lengths, reduce(vcat, sim_lengths))
bins = 1:maximum(all_lengths)
obs_hist = counts(obs_lengths, bins)
sim_hists = [counts(sim, bins) for sim in sim_lengths]
using Statistics

sim_matrix = hcat(sim_hists...)  # each column = one simulation
median_sim = mapslices(median, sim_matrix; dims=2)[:]
p5_sim = mapslices(x -> quantile(x, 0.0005), sim_matrix; dims=2)[:]
p95_sim = mapslices(x -> quantile(x, 0.9555), sim_matrix; dims=2)[:]
using CairoMakie

fig = Figure()
ax = Axis(fig[1,1], xlabel="Wet spell length (days)", ylabel="Count", yscale=log10)


# Simulations median + envelope
poly!(ax, vcat(bins, reverse(bins)), vcat(p5_sim, reverse(p95_sim)), color=:red, alpha=0.2)
lines!(ax, bins, median_sim, color=:red, linewidth=1.5, label="Sim median")
# Observation
lines!(ax, bins, obs_hist, color=:blue, linewidth=2, label="Obs")

Legend(fig[1,2], [LineElement(color=:red), LineElement(color=:blue)], ["Sim", "Obs"])

fig

end