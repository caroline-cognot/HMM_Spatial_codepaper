# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
using Base.Threads

include("../23precip_intensity/EGPD_functions.jl")
skip_estim_marginals = true
skip_plots_marginals = true
########################## on real data #################
using Dates
date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end
my_N = length(every_year)
import StochasticWeatherGenerators.dayofyear_Leap
n2t = dayofyear_Leap.(every_year)
using JLD2
using CSV
using DataFrames
Robs = Matrix(CSV.read("./00data/transformedECAD_Robs.csv", DataFrame, header=false)[:, :])'
D = size(Robs, 1)
locsdata = CSV.read("./00data/transformedECAD_stations.csv", DataFrame, header=true)
locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
locsdata.LON = locations[:, 1]
locsdata.LAT = locations[:, 2]


y = Robs[1:D, :]

T = 366
left_bound = 0.1
middle_bound = 1.0
degree_of_P = 2
# y[(y.>left_bound).*(y.<=middle_bound)] .= rand(Uniform(left_bound,middle_bound), length(y[(y.>left_bound).*(y.<=middle_bound)]))

param_sigmainit = hcat(fill(log(5.0), D, 1), fill(0.0, D, 2 * degree_of_P))
param_xiinit = hcat(fill(-log(1/0.2-1), D, 1), fill(0.0, D, 2 * degree_of_P))
param_kappainit = hcat(fill(0.0, D, 1), fill(0.0, D, 2 * degree_of_P))

Threads.nthreads()
if !skip_estim_marginals
   @time  di, param_kappa, param_sigma, param_xi, param_proba_lowrain = fit_dists(y,param_kappainit, param_sigmainit, param_xiinit, Matrix{AbstractFloat}(undef, D, 2 * degree_of_P + 1), left_bound, middle_bound, T, n2t,maxiters=5000)
    jldsave("./21precip_intensity_marginal_noclass/res_real_data/periodicEGPD"*string(degree_of_P)*".jld2"; di=di)
end
@load "./21precip_intensity_marginal_noclass/res_real_data/periodicEGPD"*string(degree_of_P)*".jld2" di



using StatsBase
using LaTeXStrings
if !skip_plots_marginals
    p = [plot() for i in 1:D]
    for i in 1:D
        Ri = Robs[i, :]
        samples = Ri[Ri.>0]
        dlist1 = fit_mix(MixedUniformTail, samples; left=left_bound, middle=middle_bound)

        p1 = plot([di[i][t].p for t in 1:T], label=ifelse(i == 9, "fitted proba low rain t", :none))
        hline!(p1, [dlist1.p], color="red", label=ifelse(i == 9, "ExtendedExtremes constant estimate", :none))
        p2 = plot([di[i][t].tail_part.G.σ for t in 1:T], label=ifelse(i == 9, "fitted sigma(t)", :none))
        hline!(p2, [dlist1.tail_part.G.σ], color="red", label=ifelse(i == 9, "ExtendedExtremes constant estimate", :none))

        # horizontal line at y = 0.5
        p3 = plot([di[i][t].tail_part.G.ξ for t in 1:T], label=ifelse(i == 9, "fitted xi(t)", :none))
        hline!(p3, [dlist1.tail_part.G.ξ], color="red", label=ifelse(i == 9, "ExtendedExtremes constant estimate", :none))

        p4 = plot([di[i][t].tail_part.V.α for t in 1:T], label=ifelse(i == 9, "fitted kappa(t)", :none))
        hline!(p4, [dlist1.tail_part.V.α], color="red", label=ifelse(i == 9, "ExtendedExtremes constant estimate", :none))

        p[i] = plot(p1, p2, p3, p4, suptitle=locsdata.STANAME[i])
    end
    savefig(plot(p[[9, 16, 14, 21, 3, 6]]..., size=(2000, 2000)), "./21precip_intensity_marginal_noclass/res_real_data/trydeg" * string(degree_of_P) * "sim.png")

end
if !skip_plots_marginals

    ysim = my_rand(di, (vcat(n2t, n2t, n2t, n2t, n2t, n2t, n2t, n2t, n2t)))

    begin
        i = 12
        datet = 1:100
        bins = 0:0.5:100
        Ri = Robs[i, :]
        samples = Ri[Ri.>0]
        dlist1 = fit_mix(MixedUniformTail, samples; left=left_bound, middle=middle_bound)
        ysim1c = rand(dlist1, length(vcat(n2t, n2t, n2t, n2t, n2t, n2t, n2t, n2t, n2t, n2t, n2t, n2t)))


        # Extract data subsets
        data_sim1 = ysim[i, vcat([findall(n2t .== t) for t = 1:T][datet]...)]

        data_true1 = Robs[i, vcat([findall(n2t .== t) for t = 1:T][datet]...)]
        data_true1 = data_true1[data_true1.>0]

        p1 = histogram(data_sim1, bins=bins, label="sim fitted model", alpha=0.3,
            title="Histogram r date 1 to 100", normalize=true)
        histogram!(p1, data_true1, bins=bins, label="obs rain >0", alpha=0.3, normalize=true)
        data_sim1c = ysim1c[vcat([findall(n2t .== t) for t = 1:T][datet]...)]
        histogram!(p1, data_sim1c, bins=bins, label="sim extendedGP constant model >0", alpha=0.1, normalize=true)



        datet = 200:300
        data_sim2 = ysim[i, vcat([findall(n2t .== t) for t = 1:T][datet]...)]
        data_true2 = Robs[i, vcat([findall(n2t .== t) for t = 1:T][datet]...)]
        data_true2 = data_true2[data_true2.>0]

        p2 = histogram(data_sim2, bins=bins, label="sim fitted model", alpha=0.3,
            title="Histogram r date 200 to 300", normalize=true)
        histogram!(p2, data_true2, bins=bins, label="obs rain >0", alpha=0.3, normalize=true)
        data_sim1c = ysim1c[vcat([findall(n2t .== t) for t = 1:T][datet]...)]
        histogram!(p2, data_sim1c, bins=bins, label="sim extendedGP constant model >0", alpha=0.1, normalize=true)



        data_sim2 = ysim[i, :]
        data_true2 = Robs[i, :]
        data_true2 = data_true2[data_true2.>0]

        p3 = histogram(data_sim2, bins=bins, label="sim fitted model", alpha=0.3,
            title="Histogram r>0 all dates", normalize=true)
        histogram!(p3, data_true2, bins=bins, label="obs rain >0", alpha=0.3, normalize=true)
        data_sim1c = ysim1c[:]
        histogram!(p3, data_sim1c, bins=bins, label="sim extendedGP constant model >0", alpha=0.1, normalize=true)

        p = plot(p1, p2, p3, suptitle="Period T = $T", xlims=(0, 50), layout=(3, 1))
    end
    savefig(plot(p, size=(1000, 1000)), "./21precip_intensity_marginal_noclass/res_real_data/trydeg" * string(degree_of_P) * "histsim.png")

   
end
