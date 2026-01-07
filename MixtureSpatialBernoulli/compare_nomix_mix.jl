using CSV
using DataFrames
using JLD2
using Plots
using Dates
# Spatial Bernoulli model
include("../SpatialBernoulli/SpatialBernoulli.jl")

# validation plots
include("../SpatialBernoulli/plot_validation.jl")


md"""
Compare real data to simulated data : get the real data.
"""

station_50Q = CSV.read("data/transformedECAD_stations.csv",DataFrame)
Yobs=Matrix(CSV.read("data/transformedECAD_Yobs.csv",header=false,DataFrame))
my_distance =Matrix(CSV.read("data/transformedECAD_locsdistances.csv",header=false,DataFrame))

my_locations = hcat(station_50Q.LON_idx, station_50Q.LAT_idx)


nlocs = length(my_locations[:, 1])


select_month = function (m::Int64, dates, Y::AbstractMatrix)
    indicesm = findall(month.(dates) .== m)
    return Y[:, indicesm]
end



date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end

Ymonths = [select_month(m, every_year, Yobs) for m in 1:12]

md"""
get the parameters estimated using SpatialBernoulli

"""
vec_models = Vector{SpatialBernoulli}(undef, 12)
for imonth in 1:12
    vec_models[imonth] = load("./SpatialBernoulli/fitted_month_QMC100" * string(imonth) * ".jld2")["d"]
end
vec_models


md"""
get the parameters estimated using Mixture of SpatialBernoulli
change name of file if necessary

"""
vec_modelsZ = Vector{MixtureModel}(undef, 12)
for imonth in 1:12
    nt = length(Ymonths[imonth][1, :])
    vec_modelsZ[imonth] = load("./MixtureSpatialBernoulli/res_real_data/3classes_fitted_month" * string(imonth) * "_" * string(20) * "iterEM_" * string(20) * "iterM_" * string(nt) * "days.jld2")["d"]
end
vec_modelsZ

using Base.Threads
monthname = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
p = [plot() for i in 1:12]
@threads for imonth in 1:12
    Nb = 500
    ntot = length(Ymonths[imonth][1, :])
    Ys = zeros(Bool, nlocs, ntot, Nb)
    @time "Simulations  Y" for i in 1:Nb
        Ys[:, :, i] = rand(vec_models[imonth], ntot)

    end

    YsZ = zeros(Bool, nlocs, ntot, Nb)
    @time "Simulations  Y" for i in 1:Nb
        YsZ[:, :, i] = rand(vec_modelsZ[imonth], ntot)

    end

    p[imonth] = compare_ROR_histogram90(Ymonths[imonth], Ys, YsZ;show = imonth ==4)
    plot!(p[imonth], title=monthname[imonth])
end

for imonth in 1:12
    plot!(p[imonth],xlabel=ifelse(imonth in 9:12, "ROR", ""),ylabel=ifelse(imonth in [1,5,9], "frequency", ""); ylim=(0, 0.06))
end

default(fontfamily="Computer Modern")
pp=plot(p..., layout=(3, 4), size=(1000, 800);leg=:topright)

savefig(pp, "./MixtureSpatialBernoulli/res_real_data/3classes_RORresults.png"
)

plt = [plot() for i in 1:6] 

[plot!(plt[i], 1:5, 1:5, label = ifelse(i==2, "label", :none)) for i in 1:6] 

plot(plt...) 
md"""
Plot the parameters as function of time for each state.
"""
# plot the parameters
d = vec_models
indices = 1:4
lambda = hcat([vec_models[i].λ[indices] for i in 1:length(d)]...);
rho = [d[i].range for i in 1:length(d)];
p1 = scatter(monthname, rho,c=1, label="no Z",title="Estimated range parameter ρ");
plot!(p1,monthname, rho, c=1,label="",title="Estimated range parameter ρ");

# Initialize plot p2
p2 = plot(title="Estimated rain probabilities λₛ - no hidden state", ylim=(0, 1));

# Correctly assign labels to each station
for (idx, station) in enumerate(indices)
    plot!(p2, monthname, c=idx, lambda[idx, :], label="Station $station")
    scatter!(p2, monthname, c=idx, lambda[idx, :], label="")
end
p2




p3 = plot(title="Estimated rain probabilities λₛ - with hidden state", ylim=(0, 1));

shapes = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5]

for k in 1:3
    lambdak = hcat([vec_modelsZ[m].components[k].λ[indices] for m in 1:12]...)
    for (idx, station) in enumerate(indices)
        println(idx)
        # plot!(p3, monthname, c=k, lambdak[idx, :], label="")
        scatter!(p3, monthname, c=k, markershape=shapes[idx], lambdak[idx, :], label="Station $station - state $k")
    end
end


pp = plot(p2, p3, layout=(2, 1), legend=:outerright, size=(1000, 800))
savefig(pp, "./MixtureSpatialBernoulli/res_real_data/3classes_lambda_results.png"
)



for k in 1:3
    rhok = [vec_modelsZ[m].components[k].range for m in 1:12]
    scatter!(p1,monthname, rhok, c=k+1, label="State $k")
    # plot!(p1,monthname,rhok, c=k+1, label="")

end

p1
savefig(p1, "./MixtureSpatialBernoulli/res_real_data/3classes_range_results.png"
)

