# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
using CSV,DataFrames
include("../11SpatialBernoulli/SpatialBernoulli.jl")
station_50Q = CSV.read("./00data/transformedECAD_stations.csv",DataFrame)
Yobs=Matrix(CSV.read("./00data/transformedECAD_Yobs.csv",header=false,DataFrame))
my_distance =Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv",header=false,DataFrame))

my_locations = hcat(station_50Q.LON_idx, station_50Q.LAT_idx)
nlocs = length(my_locations[:, 1])

select_month = function (m::Int64, dates, Y::AbstractMatrix)
    indicesm = findall(month.(dates) .== m)
    return Y[:, indicesm]
end

using Dates
date_start = Date(1973)


date_end = Date(2024) - Day(1)

every_year = date_start:Day(1):date_end
dates = every_year


Ymonths = [select_month(m, every_year, Yobs) for m in 1:12]

##################################################################################
using JLD2
using Base.Threads
Threads.@threads for imonth in 1:12
    y = Ymonths[imonth][:, 1:100] #select 100 first days of mobth 1 for trial
    y = Ymonths[imonth][:, 1:300]
    y = Ymonths[imonth]
    n = length(y[1, :])


    init_range = 600
    init_order = 0.5
    init_lambda = fill(0.4, nlocs)
    init_d = SpatialBernoulli(init_range, 1.0, init_order, init_lambda, my_distance)
   
    #pairwise solution
    tdist = maximum(my_distance) / 3
    wp = 1.0 .* (my_distance .< tdist)
    # @time sol = fit_mle(init_d,y,wp; m = 100*2, return_sol = false)

    @time sol_fixednu = fit_mle(init_d, y, wp; order=1/2,maxiters = 10, m=100 * 2, return_sol=false)

    # Save to JLD2 file
    save("./11SpatialBernoulli/fitted_month_QMC100" * string(imonth) * ".jld2", Dict("d" => sol_fixednu))
    
end

Threads.@threads for imonth in 1:12
    y = Ymonths[imonth][:, 1:100] #select 100 first days of mobth 1 for trial
    y = Ymonths[imonth][:, 1:300]
    y = Ymonths[imonth]
    n = length(y[1, :])


    init_range = 600
    init_order = 0.5
    init_lambda = fill(0.4, nlocs)
    init_d = SpatialBernoulli(init_range, 1.0, init_order, init_lambda, my_distance)
   
    #pairwise solution
    tdist = maximum(my_distance) / 3
    wp = 1.0 .* (my_distance .< tdist)
    # @time sol = fit_mle(init_d,y,wp; m = 100*2, return_sol = false)

    @time sol_fixednu = fit_mle_vfast(init_d, y, wp; order=1/2,maxiters = 10, return_sol=false)

    # Save to JLD2 file
    save("./11SpatialBernoulli/vfastfitted_month_QMC100" * string(imonth) * ".jld2", Dict("d" => sol_fixednu))
    
end

vec_models = Vector{SpatialBernoulli}(undef, 12)
for imonth in 1:12
    vec_models[imonth] = load("./11SpatialBernoulli/fitted_month_QMC100" * string(imonth) * ".jld2")["d"]
end

vec_models_vfast = Vector{SpatialBernoulli}(undef, 12)
for imonth in 1:12
    vec_models_vfast[imonth] = load("./11SpatialBernoulli/vfastfitted_month_QMC100" * string(imonth) * ".jld2")["d"]
end
ntot = length(Yobs[1,:])

using Dates
date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end

# there is probably a better way to do this ?
Nb = 500
nlocs= length(my_locations[:,1])
begin
    Ys = zeros(Bool, nlocs, ntot, Nb)
    @time "Simulations  Y" for i in 1:Nb
        for t in 1:ntot
        m = month(every_year[t])
        Ys[:, t, i] = rand(vec_models[m]);
        end
    end
end

begin
    Ysvf = zeros(Bool, nlocs, ntot, Nb)
    @time "Simulations  Y" for i in 1:Nb
        for t in 1:ntot
        m = month(every_year[t])
        Ysvf[:, t, i] = rand(vec_models_vfast[m]);
        end
    end
end


Ys
Ysvf
Yobs

include("../11SpatialBernoulli/plot_validation.jl")
p=compare_ROR_density(Yobs,Ys)
savefig(p, "./11SpatialBernoulli/ROR_500sim.png")

p=compare_ROR_histogram(Yobs,Ys)
Plots.plot!(p, title= "Rain Occurrence Ratio for 37 stations, 500 simulations",size=(1000,500))
savefig(p, "./11SpatialBernoulli/ROR_500sim_histo.png")


p=compare_ROR_histogram(Yobs,Ysvf)
Plots.plot!(p, title= "Rain Occurrence Ratio for 37 stations, 500 simulations",size=(1000,500))
savefig(p, "./11SpatialBernoulli/vfastROR_500sim_histo.png")


# plot the parameters
function Plot(d::Vector{SpatialBernoulli})
    lambda= hcat([d[i].λ for i in 1:length(d)]...);
    rho = [d[i].range for i in 1:length(d)];
    p1 = Plots.scatter(1:length(d),rho,title = "Estimated range parameter ρ");
    p2 = Plots.scatter(1:length(d),lambda',label="",title="Estimated rain probabilities λₛ");
    p2 = Plots.plot!(p2, 1:length(d),lambda',label="");
return p1,p2


end

p1,p2=Plot( vec_models)
p1
pp=Plots.plot(p,p1,p2,layout= @layout [a;b c])
savefig(pp, "./11SpatialBernoulli/ROR_500sim_andparams.png")

p1,p2=Plot( vec_models_vfast)
p1
pp=Plots.plot(p,p1,p2,layout= @layout [a;b c])
savefig(pp, "./11SpatialBernoulli/vfastROR_500sim_andparams.png")

