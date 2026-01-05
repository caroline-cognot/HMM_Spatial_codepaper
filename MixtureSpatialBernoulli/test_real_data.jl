

# Spatial Bernoulli model
include("../SpatialBernoulli/SpatialBernoulli.jl")

# Mixture fitting 
include("../MixtureSpatialBernoulli/ExpectationMaximization_source.jl")
include("../MixtureSpatialBernoulli/estimation_functions.jl")
include("/home/caroline/Gitlab_SWG_Caro/hmmspa/utils/random_utilities.jl")
include("../SpatialBernoulli/plot_validation.jl")

########################################
# real data - one model for each month #
########################################
using CSV
using DataFrames
using JLD2
using Dates

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
get the parameters estimated using no mixture.jl

"""
vec_models = Vector(undef, 12)
for imonth in 1:12
    vec_models[imonth] = load("SpatialBernoulli/fitted_month_QMC100" * string(imonth) * ".jld2")["d"]
end
vec_models

# study month 1

maxiter = 4
maxiter_m = 10


for im in 1:12

    nt = length(Ymonths[im][1, :])
    Y = Ymonths[im][:, 1:nt]
    model_full = vec_models[im]


    #first try : 4 classes. 
    model_full.range
    model_full.λ
    D₁g = SpatialBernoulli(model_full.range + 200, 1.0, 1 / 2, min.(model_full.λ .+ 0.3, 1), my_distance)
    # rainy, very correlated state
    D₂g = SpatialBernoulli(model_full.range - 200, 1.0, 1 / 2, max.(model_full.λ .- 0.3, 0), my_distance)
    # dry, less correlated state

    mlat = median(my_locations[:, 2])
    ns = (-1) .^ ((my_locations[:, 2] .> mlat))


    # france cut in half ! wet north dry south and contrary.
    D₃g = SpatialBernoulli(model_full.range, 1.0, 1 / 2, in_zero_one.(model_full.λ .- 0.2 * ns), my_distance)
    D₄g = SpatialBernoulli(model_full.range, 1.0, 1 / 2, in_zero_one.(model_full.λ .+ 0.2 * ns), my_distance)

    mix = MixtureModel([D₁g, D₂g, D₃g, D₄g], [1 / 4, 1 / 4, 1 / 4, 1 / 4])

    Random.seed!(1)
    PlotSim(Y[:, 1:9], my_locations)

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

   
    model_full = vec_models[im]


    Nb = 100
    nlocs = length(my_locations[:, 1])
    begin
        Ys = zeros(Bool, nlocs, nt, Nb)
        @time "Simulations  Y" for i in 1:Nb
            Ys[:, :, i] = rand(model_full, nt)
        end

    end
    pbefore = compare_ROR_histogram(Y, Ys)
    plot!(pbefore, title="Model with no classes")

    @time dfit = fit_mle!(α, dists, Y, PairwiseClassicEM(); order="fixed", maxiter=maxiter, QMC_m=30, factor_dmax=1 / 3, maxiter_m=maxiter_m)

    # each month for QMC_m=30, maxiter = 10, maxiter_m = 5 for 51*(30 or 31) used to take 3h.
    #from this point, initial guess has to be taken as previous result.

    # each month for QMC_m=30, maxiter = 10, maxiter_m = 10 for 51*(30 or 31) takes 250s.


    dists
    plot(dfit["logtots"])
    α
    sol = MixtureModel(dists, α)
    save("MixtureSpatialBernoulli/res_real_data/fitted_month_QMC100" * string(im) * ".jld2", Dict("d" => sol))
    savefig(plot(dfit["logtots"]), "MixtureSpatialBernoulli/res_real_data/fitted_month" * string(im) * "_" * string(maxiter) * "iterEM_" * string(maxiter_m) * "iterM_" * string(nt) * "days_logtots.png")


    Nb = 100
    nlocs = length(my_locations[:, 1])
    begin
        Ys = zeros(Bool, nlocs, nt, Nb)
        @time "Simulations  Y" for i in 1:Nb
            Ys[:, :, i] = rand(MixtureModel(dists, α), nt)
        end

    end

    pafter = compare_ROR_histogram(Y, Ys)
    p = plot(pbefore, pafter)
    dd = MixtureModel(dists, α)
    p2 = PlotPi_beforeafter(mix, dd)
    p3 = PlotCovParam_beforeafter(mix, dd)
    p4 = PlotLambda_beforeafter(mix, dd)
    l = @layout [a; b c; d]
    p5 = plot(p, p2, p3, p4, layout=l, size=(1000, 1000))

    savefig(p5, "MixtureSpatialBernoulli/res_real_data/beforeafter_month" * string(im) * "_" * string(maxiter) * "iterEM_" * string(maxiter_m) * "iterM_" * string(nt) * "days.png")
end


maxiter = 10
maxiter_m = 10

K=3
for im in 1:12

    nt = length(Ymonths[im][1, :])
    Y = Ymonths[im][:, 1:nt]
    model_full = vec_models[im]


    #second try : 3 classes. 
    model_full.range
    model_full.λ
    D₁g = SpatialBernoulli(model_full.range + 200, 1.0, 1 / 2, min.(model_full.λ .+ 0.3, 1), my_distance)
    # rainy, very correlated state
    D₂g = SpatialBernoulli(model_full.range - 200, 1.0, 1 / 2, max.(model_full.λ .- 0.3, 0), my_distance)
    # dry, less correlated state

    mlat = median(my_locations[:, 2])
    ns = (-1) .^ ((my_locations[:, 2] .> mlat))


    # france cut in half ! wet north dry south and contrary.
    D₃g = SpatialBernoulli(model_full.range, 1.0, 1 / 2, in_zero_one.(model_full.λ .- 0.2 * ns), my_distance)

    mix = MixtureModel([D₁g, D₂g, D₃g], [1 / 4, 1 / 4, 1 / 2])

    Random.seed!(1)
    PlotSim(Y[:, 1:9], my_locations)

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

    
    model_full = vec_models[im]


    Nb = 100
    nlocs = length(my_locations[:, 1])
    begin
        Ys = zeros(Bool, nlocs, nt, Nb)
        @time "Simulations  Y" for i in 1:Nb
            Ys[:, :, i] = rand(model_full, nt)
        end

    end
    pbefore = compare_ROR_histogram(Y, Ys)
    plot!(pbefore, title="Model with no classes")

    @time dfit = fit_mle!(α, dists, Y, PairwiseClassicEM(); order="fixed", maxiter=maxiter, QMC_m=30, factor_dmax=1 / 3, maxiter_m=maxiter_m)

    # each month for QMC_m=30, maxiter = 10, maxiter_m = 5 for 51*(30 or 31) used to take 3h.
    #from this point, initial guess has to be taken as previous result.

    # each month for QMC_m=30, maxiter = 10, maxiter_m = 10 for 51*(30 or 31) takes 250s.


    dists
    plot(dfit["logtots"])
    α
    sol = MixtureModel(dists, α)
    save("MixtureSpatialBernoulli/res_real_data/3classes_fitted_month" * string(im) * "_" * string(maxiter) * "iterEM_" * string(maxiter_m) * "iterM_" * string(nt) * "days.jld2", Dict("d" => sol))
    savefig(plot(dfit["logtots"]), "MixtureSpatialBernoulli/res_real_data/3classes_fitted_month" * string(im) * "_" * string(maxiter) * "iterEM_" * string(maxiter_m) * "iterM_" * string(nt) * "days_logtots.png")


    Nb = 100
    nlocs = length(my_locations[:, 1])
    begin
        Ys = zeros(Bool, nlocs, nt, Nb)
        @time "Simulations  Y" for i in 1:Nb
            Ys[:, :, i] = rand(MixtureModel(dists, α), nt)
        end

    end

    pafter = compare_ROR_histogram(Y, Ys)
    p = plot(pbefore, pafter)
    dd = MixtureModel(dists, α)
    p2 = PlotPi_beforeafter(mix, dd)
    p3 = PlotCovParam_beforeafter(mix, dd)
    p4 = PlotLambda_beforeafter(mix, dd)
    l = @layout [a; b c; d]
    p5 = plot(p, p2, p3, p4, layout=l, size=(1000, 1000))

    savefig(p5, "MixtureSpatialBernoulli/res_real_data/3classes_beforeafter_month" * string(im) * "_" * string(maxiter) * "iterEM_" * string(maxiter_m) * "iterM_" * string(nt) * "days.png")
end