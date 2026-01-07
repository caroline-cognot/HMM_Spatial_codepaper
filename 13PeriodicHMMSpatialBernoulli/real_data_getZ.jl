using Pkg
Pkg.activate("HMMSPAcodepaper")
Pkg.instantiate()

   # ## Utilities
   using ArgCheck
   using Base: OneTo
   using ShiftedArrays: lead, lag
   using Distributed
   # ## Optimization
   using JuMP, Ipopt

   using CSV,DataFrames,DataFramesMeta
   using JLD2
   using Dates
   import StochasticWeatherGenerators.dayofyear_Leap

include("../11SpatialBernoulli/SpatialBernoulli.jl")
station_50Q = CSV.read("./00data/transformedECAD_stations.csv",DataFrame)
Yobs=Matrix(CSV.read("./00data/transformedECAD_Yobs.csv",header=false,DataFrame))'
my_distance =Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv",header=false,DataFrame))

my_locations = hcat(station_50Q.LON_idx, station_50Q.LAT_idx)
heatmap(my_distance)
nlocs = length(my_locations[:, 1])
my_D = size(my_locations, 1)


include("../PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
include("../PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl")
# means we are using the  EM and not CLEM,  and estimating B then R.




STAID = station_50Q.STAID #[32, 33, 39, 203, 737, 755, 758, 793, 11244, 11249];

station_name = station_50Q.STANAME


date_start = Date(1973)


date_end = Date(2024) - Day(1)
using Printf
every_year = date_start:Day(1):date_end
n = length(every_year)
path = "./00data/ECA_blended_custom"
function collect_data_ECA(STAID::Integer, path::String, var::String="RR"; skipto=21, header=20, url=false)
    file = url ? Base.download(string(path, @sprintf("STAID%06.d.txt", STAID))) : joinpath(path, string("$(uppercase(var))_", @sprintf("STAID%06.d.txt", STAID)))
    if isfile(file)
        return CSV.read(file, DataFrame, skipto=skipto, header=header, comment="#", normalizenames=true, dateformat="yyyymmdd", types=Dict(:DATE => Date))
    else
        return @warn "STAID $STAID File does not exists $(file)"
    end
end
onefy(x::Number) = x == 0 ? 0 : 1
onefy(x::Missing) = missing
onefy(x::Number, eps) = abs(x) <= eps ? 0 : 1
onefy(x::Missing, eps) = missing
begin
    data_stations = collect_data_ECA.(STAID, path)
    for i = eachindex(data_stations)
        @transform!(data_stations[i], :RO = onefy.(:RR))
        @subset!(data_stations[i], date_start .≤ :DATE .≤ date_end)
    end
end
data_stations
my_N = length(every_year)

n2t = dayofyear_Leap.(every_year)


tdist = 0.3
maxiter = 100
doss_save = "./PeriodicHMMSpatialBernoulli/res_real_data/"

my_K = 4
my_degree_of_P = 1
maxiter = 100
my_autoregressive_order = 1
my_size_order = 2^my_autoregressive_order
R0 = 500
QMC_m = 30
my_T = 366
    datafile = doss_save * "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"


    hmmspa = load(datafile)["hmm"]

Y_past = BitMatrix(Yobs[1:my_autoregressive_order, :]) # rand(Bool, local_order, D)

function viterbi(hmm::PeriodicHMMSpaMemory, Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool}; robust=false, n2t=n_to_t(size(Y, 1), size(hmm.B, 2))::AbstractVector{<:Integer}, QMC_m=30)
    LL = loglikelihoods(hmm, Y, Y_past; n2t=n2t, robust=robust, QMC_m=QMC_m)
    return viterbi(hmm.a, hmm.A, LL; n2t=n2t)
end
z_hat = viterbi(hmmspa, Yobs, Y_past; n2t=n2t, QMC_m=30)
plot(z_hat[1:365])
include("../utils/seasons_and_other_dates.jl")

# get table of seasons/states
using FreqTables
using LatexPrint
dataframe = DataFrame(season=season(every_year), z_hat=z_hat)
freq_zhat = freqtable(dataframe, :season, :z_hat)
lap(freq_zhat)

ds = length(data_stations)
Robs = Matrix(reduce(hcat, [0.1 .* data_stations[j].RR for j = 1:ds]))
Yobs
z_hat
using CSV, DelimitedFiles
CSV.write("./00data/transformedECAD_Robs.csv", DataFrame(Robs, :auto), header=false)
CSV.write("./00data/transformedECAD_zhat.csv", DataFrame(z_hat=(z_hat)), header=false)
CSV.write("./00data/transformedECAD_season.csv", DataFrame(season=season(every_year)), header=false)
CSV.write("./00data/transformedECAD_stations.csv", station_50Q, header=true)
z_hat=CSV.read("./00data/transformedECAD_zhat.csv", DataFrame, header=false)
CSV.write("./00data/transformedECAD_zhatbis.csv", DataFrame(z_hat=(z_hat[:,1]),dates=every_year), header=false)
