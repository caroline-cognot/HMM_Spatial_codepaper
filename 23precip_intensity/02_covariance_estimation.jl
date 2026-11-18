# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
using JLD2
using CSV
using DataFrames
using PeriodicHiddenMarkovModels
include("EGPD_functions.jl")
include("EGPD_class.jl")
include("covariance_functions.jl")
using Distributions

z_hat = CSV.read("./00data/transformedECAD_zhat.csv", DataFrame, header=false)[:, 1]

N = length(z_hat)
# z_hat=fill(1,N) #test if setting this to 1 gives same result. as previous no-class periodic estimation.
# z_hat[z_hat.==4] .=3

K = length(unique(z_hat))
Robs = Matrix(CSV.read("./00data/transformedECAD_Robs.csv", DataFrame, header=false)[:, :])'
D = size(Robs, 2)
locsdata = CSV.read("./00data/transformedECAD_stations.csv", DataFrame, header=true)
locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
locsdata.LON = locations[:, 1]
locsdata.LAT = locations[:, 2]
using Dates
date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end
my_N = length(every_year)
import StochasticWeatherGenerators.dayofyear_Leap
n2t = dayofyear_Leap.(every_year)

begin
    Mat_h = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))

    include("../13PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
    include("../13PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl") #for my_polynomial_trigo

    my_K = 4
    my_degree_of_P = 1
    maxiter = 100
    my_autoregressive_order = 1
    R0 = 500
    QMC_m = 30
    datafile = "./13PeriodicHMMSpatialBernoulli/res_real_data/" * "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"


    hmmspa = load(datafile)["hmm"]


end
T = 366

##########"" get pdry[s,t] for all s all t as observed in the data/
degree_of_P = 2
XR = Matrix(CSV.read("./23precip_intensity/res_real_data/EGPD_periodic_K_normalized" * string(K) * string(degree_of_P) * ".csv", DataFrame, header=false))'
D = size(XR, 1)
Yobs = Matrix(CSV.read("./00data/transformedECAD_Yobs.csv", DataFrame, header=false)[:, :])'
Yprevious = fill(1, 1, size(hmmspa, 2))
Ymoins1 = vcat(Yprevious, Yobs)[1:my_N, :]
Pdry = similar(XR)
for s in 1:D
    for n in 1:my_N
        t = n2t[n]
        k = z_hat[n]
        h = Ymoins1[n, s]
        Pdry[s, n] = 1 - hmmspa.B[k, t, s, h+1]
    end
end
phimPdry = quantile.(Normal(), Pdry)
data = XR
Mat_h
include("/home/caroline/Gitlab_SWG_Caro/hmmspa/utils/fast_bivariate_cdf.jl")
using Base.Threads



param0 = [250. 3.]
fitted = fit_mple(data, Mat_h, ExpExp, param0, phimPdry, maxiter=20)

# took a while but i also closed the computer at some point;
# 1066.181343 seconds (1.99 M allocations: 180.534 MiB, 0.00% gc time, 0.10% compilation time)
# ExpExp{Float64, Float64}(1.0, 280.2402620839839, 0.6647444627840139)

jldsave("23precip_intensity/res_real_data/ExpExpcov_withmarginalsdeg" * string(degree_of_P) * ".jld2"; fitted=fitted)






param0=[ 280.0, 2/3,  1.0, 0.5, 0.1, 0.5]
fitted=fit_mple(data,Mat_h,GneitingMatern,param0,phimPdry,lower=[200., 0.1 ,0. ,0. ,0.00, 0.1],upper= [1000. ,10., 1., 1., 10., 10.],maxiter=100)
# 100 iter : 6279.295689 seconds (35.85 M allocations: 3.355 GiB, 0.01% gc time, 0.26% compilation time: <1% of which was recompilation)

# 100 iter : 3780.496105 seconds (24.23 M allocations: 2.086 GiB, 0.01% gc time, 0.08% compilation time)
# GneitingMatern{Float64, Float64}(1.0, 305.2396076771512, 0.6097296932197163, 0.9994210930169677, 0.755855842863924, 0.5928448637557749, 0.4482260014469662)

jldsave("23precip_intensity/res_real_data/GMcov_withmarginalsdeg" * string(degree_of_P) * ".jld2"; fitted=fitted)
