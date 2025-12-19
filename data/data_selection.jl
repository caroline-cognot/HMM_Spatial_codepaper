
using CSV, JLD, DelimitedFiles # File Read/Load/Save
using Printf
using DataFrames # DataFrames
using DataFramesMeta
using LaTeXStrings  # ✅ Import this to use @L_str


using Dates
begin
    station_all = CSV.read("./data/ECA_blended_custom/stations.txt", DataFrame, header=18, normalizenames=true, ignoreemptyrows=true)
    station_all = @chain station_all begin
        @transform(:CN = rstrip.(:CN), :STANAME = rstrip.(:STANAME))
        # @subset(:STAID .∈ tuple([32, 33, 34, 36, 39, 203, 322, 323, 434, 736, 737, 738, 740, 742, 745, 749, 750, 755, 756, 757, 758, 786, 793, 2192, 2203, 2205, 2207, 2209, 11244, 11245, 11247, 11249]))
    end
end



STAID = station_all.STAID #[32, 33, 39, 203, 737, 755, 758, 793, 11244, 11249];

station_name = station_all.STANAME


date_start = Date(1973)


date_end = Date(2024) - Day(1)

every_year = date_start:Day(1):date_end
n = length(every_year)
path = "./data/ECA_blended_custom"
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
print([data_stations[i].DATE[1] for i in 1:60])
# seules la 5, 9, 46,53,n'ont pasde données sur la période souhaitée 
# on les enlève
leave_out = [5, 9, 46, 53]
station_50 = station_all[Not(leave_out), :]
data_stations_50 = data_stations[Not(leave_out)]
STAID_50 = STAID[Not(leave_out)]
station_name_50 = station_name[Not(leave_out)]
print([data_stations_50[i].DATE[1] for i in 1:56])
print([data_stations_50[i].DATE[end] for i in 1:56])
print([length(data_stations_50[i].DATE) for i in 1:56])
# on doit aussi enlever celles qui ont des données peu satisfaisantes : un taux de code 9 trop élevé !

Missing_amount = [sum(data_stations_50[j].Q_RR .== 9) / n for j in 1:length(STAID_50)]
scatter(Missing_amount)
leave_out = findall(Missing_amount .> 0)
data_stations_50Q = data_stations_50[Not(leave_out)]
STAID_50Q = STAID_50[Not(leave_out)]
station_name_50Q = station_name_50[Not(leave_out)]
station_50Q = station_50[Not(leave_out), :]
ds = length(data_stations_50Q)
Yobs = BitMatrix(reduce(hcat, [data_stations_50Q[j].RO for j = 1:ds]))'


select_month = function (m::Int64, dates, Y::AbstractMatrix)
    indicesm = findall(month.(dates) .== m)
    return Y[:, indicesm]
end

dates = every_year


Ymonths = [select_month(m, every_year, Yobs) for m in 1:12]

##################################################### coordinates and distance ###################"
# * Station Coordinates * #
using Geodesy
"""
    dms_to_dd(l)
Convert `l` in Degrees Minutes Seconds to Decimal Degrees. Inputs are strings of the form
* LAT    : Latitude in degrees:minutes:seconds (+: North, -: South)
* LON    : Longitude in degrees:minutes:seconds (+: East, -: West)
"""
function dms_to_dd(l)
    deg, minutes, seconds = parse.(Float64, split(l, ":"))
    (abs(deg) + minutes / 60 + seconds / (60 * 60)) * sign(deg)
end

#! Type piracy
Geodesy.LLA(x::String, y::String, z) = LLA(dms_to_dd(x), dms_to_dd(y), z)

"""
    distance_x_to_y(station_x, station_y)
Distance in km between two stations. Does not take into account altitude. `station` must have a field `LAT` and `LON` in Decimal Degree.
"""
function distance_x_to_y(station_x, station_y)
    coord_station_x = LLA(station_x.LAT, station_y.LON, 0.0)
    coord_station_y = LLA(station_y.LAT, station_y.LON, 0)
    return Geodesy.distance(coord_station_x, coord_station_y) / 1000 # distance in km
end

function distance_x_to_all_stations(central_row, station)
    coord_central = LLA(central_row.LAT, central_row.LON, 0.0)
    coord_stations = [LLA(station[i, :].LAT, station[i, :].LON, 0) for i in 1:nrow(station)]
    return [Geodesy.distance(coord_central, pos) for pos in coord_stations] / 1000 # distance in km
end




station_50Q.LAT_idx = dms_to_dd.(station_50Q.LAT)

station_50Q.LON_idx = dms_to_dd.(station_50Q.LON)

scatter(station_50Q.LON_idx, station_50Q.LAT_idx)
my_locations = hcat(station_50Q.LON_idx, station_50Q.LAT_idx)
using Distances
my_distance = [haversine(my_locations[i, :], my_locations[j, :]) / 1000 for i in axes(my_locations, 1), j in axes(my_locations, 1)]
heatmap(my_distance)

nlocs = length(my_locations[:, 1])

CSV.write("./data/transformedECAD_locs.csv", DataFrame(my_locations, :auto), header=false)
CSV.write("./data/transformedECAD_Yobs.csv", DataFrame(Yobs, :auto), header=false)
CSV.write("./data/transformedECAD_locsdistances.csv", DataFrame(my_distance, :auto), header=false)
CSV.write("./data/transformedECAD_stations.csv", station_50Q, header=true)

