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
rainysim = [collect(mean(r) > seuil for r in eachcol(rr)) for rr in eachslice(Rs, dims = 3)]
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


obs_lengths = ror_info.lengths


D, N, Nsim = size(Rs)
size(Robs)

sim_distributions = [pmf_spell(rainysim[i], true) for i in 1:Nb]

begin
	all_values = vcat(obs_lengths, reduce(vcat, sim_distributions))
	bins = collect(1:maximum(all_values)+5)

	fig_spell = Figure(fontsize = 17,size=(1000,300))
	ax = Axis(fig_spell[1, 1],
		xlabel = "Nb of days",
		ylabel = "Probability",
		yscale = log10,
		xticks = 0:5:60)

	# Observations
	errorscatterhist!(ax, [obs_lengths],
		color = :blue,
		normalization = :probability,
		bins = bins,
		errortype = :percentile,
		label = "Obs")

		
	# Simulations
	errorlinehist!(ax, sim_distributions;
	color=:red,
	secondarycolor=:grey,
	label=("Simu q_{0,100}" ),
	normalization=:probability,
	bins=bins,
	errortype=:percentile,
	percentiles=[0, 100],
	secondaryalpha=0.4,
	centertype=:median)

# Interquartile 5-95
errorlinehist!(ax, sim_distributions;
	color=:red,
	secondarycolor=:red,
	label=( "Simu q_{5,95}" ),
	normalization=:probability,
	bins=bins,
	errortype=:percentile,
	percentiles=[5, 95],
	secondaryalpha=0.5,
	centertype=:median)

	Legend(
        fig_spell[:, 2],
        [PolyElement(color=:grey, alpha=0.5), [PolyElement(color=:red, alpha=0.5),
        LineElement(color=:red)], LineElement(color=:blue)],
[L"Simu $q_{0,100}$", L"Simu $q_{5,95}$", "Obs"])
    resize_to_layout!(fig_spell)
	ylims!(ax, 8e-6, 1)
	fig_spell
end
