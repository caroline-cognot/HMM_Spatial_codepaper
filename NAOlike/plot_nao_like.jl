import Pkg
cd(@__DIR__)
Pkg.activate(@__DIR__)
Pkg.instantiate()
using JLD2
using AxisArrays
using CSV, DataFrames, DataFramesMeta
using Dates
using StatsBase
using GeoMakie, CairoMakie, NaturalEarth

function savefigcrop(plt::CairoMakie.Figure, save_name)
    CairoMakie.save(string(save_name, ".pdf"), plt)
    run(`pdfcrop $(string(save_name,".pdf"))`) # Petit délire pour croper proprement la figure 
    mv(string(save_name, "-crop", ".pdf"), string(save_name, ".pdf"), force=true)
end
date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end
K = 4
Deg = 1
local_memory = 1

df_z = CSV.read("../data/transformedECAD_zhat.csv", DataFrame,header=false)
df_z.DATE = every_year
rename!(df_z,:Column1 => :z)
DJF = [11, 12, 1, 2, 3]
DJF = [12, 1, 2]

path_data(year) = "year_by_year_mean/msl_pp_big_$(year)_daymean.jld2"

map_data = load(path_data(1979))["map_data"]

function agg!(mean_season, mean_season_k, count_k, M, dates)
    df_date = DataFrame([:DATE, :idx] .=> [dates, 1:length(dates)])

    df = leftjoin(df_date, df_z, on=:DATE)

    df_season = @subset(df, month.(:DATE) .∈ tuple(DJF))

    df_season_k = groupby(df_season, :z)
    nrows = [nrow(df_season_k[k]) for k in 1:K]
    count_k .+= nrows

    mean_season .+= sum(M[:, :, s] for s in df_season.idx)
    for k in 1:K
        mean_season_k[k] .+= sum(M[:, :, s] for s in df_season_k[k].idx)
    end
    return nothing
end

ratio_K = round.([mean(@subset(df_z, month.(:DATE) .∈ tuple(DJF)).z .== k)*100 for k in 1:K], digits = 1)
ratio_K = round.([mean(@subset(df_z, month.(:DATE) .∈ tuple(DJF)).z .== k)*100 for k in 1:K], digits = 1)
sum(ratio_K)
@time begin
    M_leap_no = similar(map_data.data)
    M_leap = similar(map_data.data, size(map_data)[1:2]..., 366)
    mean_season = zeros(eltype(map_data.data), size(map_data)[1:2]...)
    mean_season_k = [zeros(eltype(map_data.data), size(map_data)[1:2]...) for k in 1:K]
    count_k = zeros(Int, K)
    for ye in setdiff(1979:2017, 1989)
        dates = Date(ye):Day(1):Date(ye, 12, 31)
        if isleapyear(ye)
            M_leap .= load(path_data(ye))["map_data"].data
            agg!(mean_season, mean_season_k, count_k, M_leap, dates)
        else
            M_leap_no .= load(path_data(ye))["map_data"].data
            agg!(mean_season, mean_season_k, count_k, M_leap_no, dates)
        end
    end
    Δmean_season_k = [mean_season_k[k] / count_k[k] - mean_season / sum(count_k) for k in 1:K]
end

max_diff = (Δmean_season_k / 100 .|> x -> abs.(x)) .|> maximum |> maximum |> x -> ceil(Int, x)

function positions(k)
    if k == 1
        return (1, 1)
    elseif k == 4
        return (1, 2)
    elseif k == 2
        return (2, 1)
    elseif k == 3
        return (2, 2)
    end
end

lons = map_data.axes[1].val
lats = map_data.axes[2].val
fontsize = 24
coastlines_ne = naturalearth("coastline", 110)
LON_min, LON_max = extrema(lons)
LAT_min, LAT_max = extrema(lats)

fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do

    fig = GeoMakie.Figure(size=(1000, 1000))
    ## Dict("projection" => ccrs.PlateCarree(central_longitude = mean([LON_min, LON_max]))) # in cartopy
    ax = [GeoMakie.GeoAxis(fig[positions(k)...],
        ## dest="+proj=natearth", 
        ## dest = "+proj=ortho +lon_0=$(mean(lons)) +lat_0=$(mean(lats))",
        dest="+proj=longlat +datum=WGS84",
        xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false, title=L"$\,\,\, Z = %$k$ (%$(ratio_K[k])%)") for k in 1:K]
    # Divide by 100 to get hPa
    sp = [GeoMakie.contourf!(ax[k], lons, lats, Δmean_season_k[k] / 100; colormap=:coolwarm, levels=range(-max_diff, max_diff, step=2)) for k in 1:K]

    for k in 1:K
        GeoMakie.xlims!(ax[k], LON_min, LON_max)
        GeoMakie.ylims!(ax[k], LAT_min, LAT_max)
        GeoMakie.lines!(ax[k], GeoMakie.to_multilinestring.(coastlines_ne.geometry); color=:black, linewidth=2)
        k ∈ [4, 3] && GeoMakie.hideydecorations!(ax[k], ticks=false)
        k ∈ [1, 4] && GeoMakie.hidexdecorations!(ax[k], ticks=false)
        ax[k].aspect = 1.65
    end
    colorrange = range(-max_diff, max_diff, step=2)
    cb_cell = fig[1:2, 3] = GeoMakie.GridLayout(height=555) # treat the colorbar-and-label system as a single grid cell
    cb = GeoMakie.Colorbar(cb_cell[2, 1], colormap=cgrad(:coolwarm, length(colorrange), categorical=true), colorrange=extrema(colorrange), ticks=-24:3:24)
    cb_label = GeoMakie.Label(cb_cell[1, 1], "hPa"; tellheight=true, tellwidth=false)
    # GeoMakie.Colorbar(fig[:, 3], sp[4], label="hPa", ticks=-24:3:24, height=GeoMakie.Relative(0.6))
    GeoMakie.colgap!(fig.layout, 16)
    GeoMakie.rowgap!(fig.layout, -390)

    fig
end

savefigcrop(fig, "FR_mean_DJF_pressure_diff_K_$(K)_d_$(Deg)_m_$(local_memory)_WGS84")
# fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do

#     fig = GeoMakie.Figure(size=(1000, 1000))
#     ## Dict("projection" => ccrs.PlateCarree(central_longitude = mean([LON_min, LON_max]))) # in cartopy
#     ax = [GeoMakie.GeoAxis(fig[positions(k)...],
#         ## dest="+proj=natearth", 
#         ## dest = "+proj=ortho +lon_0=$(mean(lons)) +lat_0=$(mean(lats))",
#         dest="+proj=longlat +datum=WGS84",
#         xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false, title=L"Z = %$k") for k in 1:K]
#     # Divide by 100 to get hPa
#     sp = [GeoMakie.contourf!(ax[k], lons, lats, Δmean_season_k[k] / 100; colormap=:coolwarm, levels=range(-max_diff, max_diff, step=2)) for k in 1:K]

#     for k in 1:K
#         GeoMakie.xlims!(ax[k], LON_min, LON_max)
#         GeoMakie.ylims!(ax[k], LAT_min, LAT_max)
#         GeoMakie.lines!(ax[k], GeoMakie.to_multilinestring.(coastlines_ne.geometry); color=:black, linewidth=2)
#         k ∈ [4, 3] && GeoMakie.hideydecorations!(ax[k], ticks=false)
#         k ∈ [1, 4] && GeoMakie.hidexdecorations!(ax[k], ticks=false)
#         ax[k].aspect = 1.65
#     end
#     GeoMakie.Colorbar(fig[:, 3], sp[4], label="hPa", ticks=-24:3:24, height=GeoMakie.Relative(0.6))
#     GeoMakie.colgap!(fig.layout, -6)
#     GeoMakie.rowgap!(fig.layout, -390)

#     fig
# end


SON = [9, 10, 11]

ratio_K = round.([mean(@subset(df_z, month.(:DATE) .∈ tuple(SON)).z .== k)*100 for k in 1:K], digits = 1)
sum(ratio_K)

function agg!(mean_season, mean_season_k, count_k, M, dates)
    df_date = DataFrame([:DATE, :idx] .=> [dates, 1:length(dates)])

    df = leftjoin(df_date, df_z, on=:DATE)

    df_season = @subset(df, month.(:DATE) .∈ tuple(SON))

    df_season_k = groupby(df_season, :z)
    nrows = [nrow(df_season_k[k]) for k in 1:K]
    count_k .+= nrows

    mean_season .+= sum(M[:, :, s] for s in df_season.idx)
    for k in 1:K
        mean_season_k[k] .+= sum(M[:, :, s] for s in df_season_k[k].idx)
    end
    return nothing
end
@time begin
    M_leap_no = similar(map_data.data)
    M_leap = similar(map_data.data, size(map_data)[1:2]..., 366)
    mean_season = zeros(eltype(map_data.data), size(map_data)[1:2]...)
    mean_season_k = [zeros(eltype(map_data.data), size(map_data)[1:2]...) for k in 1:K]
    count_k = zeros(Int, K)
    for ye in setdiff(1979:2017, 1989)
        dates = Date(ye):Day(1):Date(ye, 12, 31)
        if isleapyear(ye)
            M_leap .= load(path_data(ye))["map_data"].data
            agg!(mean_season, mean_season_k, count_k, M_leap, dates)
        else
            M_leap_no .= load(path_data(ye))["map_data"].data
            agg!(mean_season, mean_season_k, count_k, M_leap_no, dates)
        end
    end
    Δmean_season_k = [mean_season_k[k] / count_k[k] - mean_season / sum(count_k) for k in 1:K]
end

max_diff = (Δmean_season_k / 100 .|> x -> abs.(x)) .|> maximum |> maximum |> x -> ceil(Int, x)

fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do

    fig = GeoMakie.Figure(size=(1000, 1000))
    ## Dict("projection" => ccrs.PlateCarree(central_longitude = mean([LON_min, LON_max]))) # in cartopy
    ax = [GeoMakie.GeoAxis(fig[positions(k)...],
        ## dest="+proj=natearth", 
        ## dest = "+proj=ortho +lon_0=$(mean(lons)) +lat_0=$(mean(lats))",
        dest="+proj=longlat +datum=WGS84",
        xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false, title=L"$\,\,\, Z = %$k$ (%$(ratio_K[k])%)") for k in 1:K]
    # Divide by 100 to get hPa
    sp = [GeoMakie.contourf!(ax[k], lons, lats, Δmean_season_k[k] / 100; colormap=:coolwarm, levels=range(-max_diff, max_diff, step=2)) for k in 1:K]

    for k in 1:K
        GeoMakie.xlims!(ax[k], LON_min, LON_max)
        GeoMakie.ylims!(ax[k], LAT_min, LAT_max)
        GeoMakie.lines!(ax[k], GeoMakie.to_multilinestring.(coastlines_ne.geometry); color=:black, linewidth=2)
        k ∈ [4, 3] && GeoMakie.hideydecorations!(ax[k], ticks=false)
        k ∈ [1, 4] && GeoMakie.hidexdecorations!(ax[k], ticks=false)
        ax[k].aspect = 1.65
    end
    colorrange = range(-max_diff, max_diff, step=2)
    cb_cell = fig[1:2, 3] = GeoMakie.GridLayout(height=555) # treat the colorbar-and-label system as a single grid cell
    cb = GeoMakie.Colorbar(cb_cell[2, 1], colormap=cgrad(:coolwarm, length(colorrange), categorical=true), colorrange=extrema(colorrange), ticks=-24:3:24)
    cb_label = GeoMakie.Label(cb_cell[1, 1], "hPa"; tellheight=true, tellwidth=false)
    # GeoMakie.Colorbar(fig[:, 3], sp[4], label="hPa", ticks=-24:3:24, height=GeoMakie.Relative(0.6))
    GeoMakie.colgap!(fig.layout, 16)
    GeoMakie.rowgap!(fig.layout, -390)

    fig
end

savefigcrop(fig, "FR_mean_SON_pressure_diff_K_$(K)_d_$(Deg)_m_$(local_memory)_WGS84")


MAM = [3, 4, 5]
function agg!(mean_season, mean_season_k, count_k, M, dates)
    df_date = DataFrame([:DATE, :idx] .=> [dates, 1:length(dates)])

    df = leftjoin(df_date, df_z, on=:DATE)

    df_season = @subset(df, month.(:DATE) .∈ tuple(MAM))

    df_season_k = groupby(df_season, :z)
    nrows = [nrow(df_season_k[k]) for k in 1:K]
    count_k .+= nrows

    mean_season .+= sum(M[:, :, s] for s in df_season.idx)
    for k in 1:K
        mean_season_k[k] .+= sum(M[:, :, s] for s in df_season_k[k].idx)
    end
    return nothing
end
ratio_K = round.([mean(@subset(df_z, month.(:DATE) .∈ tuple(MAM)).z .== k)*100 for k in 1:K], digits = 1)
sum(ratio_K)
@time begin
    M_leap_no = similar(map_data.data)
    M_leap = similar(map_data.data, size(map_data)[1:2]..., 366)
    mean_season = zeros(eltype(map_data.data), size(map_data)[1:2]...)
    mean_season_k = [zeros(eltype(map_data.data), size(map_data)[1:2]...) for k in 1:K]
    count_k = zeros(Int, K)
    for ye in setdiff(1979:2017, 1989)
        dates = Date(ye):Day(1):Date(ye, 12, 31)
        if isleapyear(ye)
            M_leap .= load(path_data(ye))["map_data"].data
            agg!(mean_season, mean_season_k, count_k, M_leap, dates)
        else
            M_leap_no .= load(path_data(ye))["map_data"].data
            agg!(mean_season, mean_season_k, count_k, M_leap_no, dates)
        end
    end
    Δmean_season_k = [mean_season_k[k] / count_k[k] - mean_season / sum(count_k) for k in 1:K]
end

max_diff = (Δmean_season_k / 100 .|> x -> abs.(x)) .|> maximum |> maximum |> x -> ceil(Int, x)

fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do

    fig = GeoMakie.Figure(size=(1000, 1000))
    ## Dict("projection" => ccrs.PlateCarree(central_longitude = mean([LON_min, LON_max]))) # in cartopy
    ax = [GeoMakie.GeoAxis(fig[positions(k)...],
        ## dest="+proj=natearth", 
        ## dest = "+proj=ortho +lon_0=$(mean(lons)) +lat_0=$(mean(lats))",
        dest="+proj=longlat +datum=WGS84",
        xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false, title=L"$\,\,\, Z = %$k$ (%$(ratio_K[k])%)") for k in 1:K]
    # Divide by 100 to get hPa
    sp = [GeoMakie.contourf!(ax[k], lons, lats, Δmean_season_k[k] / 100; colormap=:coolwarm, levels=range(-max_diff, max_diff, step=2)) for k in 1:K]

    for k in 1:K
        GeoMakie.xlims!(ax[k], LON_min, LON_max)
        GeoMakie.ylims!(ax[k], LAT_min, LAT_max)
        GeoMakie.lines!(ax[k], GeoMakie.to_multilinestring.(coastlines_ne.geometry); color=:black, linewidth=2)
        k ∈ [4, 3] && GeoMakie.hideydecorations!(ax[k], ticks=false)
        k ∈ [1, 4] && GeoMakie.hidexdecorations!(ax[k], ticks=false)
        ax[k].aspect = 1.65
    end
    colorrange = range(-max_diff, max_diff, step=2)
    cb_cell = fig[1:2, 3] = GeoMakie.GridLayout(height=555) # treat the colorbar-and-label system as a single grid cell
    cb = GeoMakie.Colorbar(cb_cell[2, 1], colormap=cgrad(:coolwarm, length(colorrange), categorical=true), colorrange=extrema(colorrange), ticks=-24:3:24)
    cb_label = GeoMakie.Label(cb_cell[1, 1], "hPa"; tellheight=true, tellwidth=false)
    # GeoMakie.Colorbar(fig[:, 3], sp[4], label="hPa", ticks=-24:3:24, height=GeoMakie.Relative(0.6))
    GeoMakie.colgap!(fig.layout, 16)
    GeoMakie.rowgap!(fig.layout, -390)

    fig
end

savefigcrop(fig, "FR_mean_MAM_pressure_diff_K_$(K)_d_$(Deg)_m_$(local_memory)_WGS84")


JJA = [6, 7, 8]
function agg!(mean_season, mean_season_k, count_k, M, dates)
    df_date = DataFrame([:DATE, :idx] .=> [dates, 1:length(dates)])

    df = leftjoin(df_date, df_z, on=:DATE)

    df_season = @subset(df, month.(:DATE) .∈ tuple(JJA))

    df_season_k = groupby(df_season, :z)
    nrows = [nrow(df_season_k[k]) for k in 1:K]
    count_k .+= nrows

    mean_season .+= sum(M[:, :, s] for s in df_season.idx)
    for k in 1:K
        mean_season_k[k] .+= sum(M[:, :, s] for s in df_season_k[k].idx)
    end
    return nothing
end
ratio_K = round.([mean(@subset(df_z, month.(:DATE) .∈ tuple(JJA)).z .== k)*100 for k in 1:K], digits = 1)
sum(ratio_K)
@time begin
    M_leap_no = similar(map_data.data)
    M_leap = similar(map_data.data, size(map_data)[1:2]..., 366)
    mean_season = zeros(eltype(map_data.data), size(map_data)[1:2]...)
    mean_season_k = [zeros(eltype(map_data.data), size(map_data)[1:2]...) for k in 1:K]
    count_k = zeros(Int, K)
    for ye in setdiff(1979:2017, 1989)
        dates = Date(ye):Day(1):Date(ye, 12, 31)
        if isleapyear(ye)
            M_leap .= load(path_data(ye))["map_data"].data
            agg!(mean_season, mean_season_k, count_k, M_leap, dates)
        else
            M_leap_no .= load(path_data(ye))["map_data"].data
            agg!(mean_season, mean_season_k, count_k, M_leap_no, dates)
        end
    end
    Δmean_season_k = [mean_season_k[k] / count_k[k] - mean_season / sum(count_k) for k in 1:K]
end

max_diff = (Δmean_season_k / 100 .|> x -> abs.(x)) .|> maximum |> maximum |> x -> ceil(Int, x)

fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do

    fig = GeoMakie.Figure(size=(1000, 1000))
    ## Dict("projection" => ccrs.PlateCarree(central_longitude = mean([LON_min, LON_max]))) # in cartopy
    ax = [GeoMakie.GeoAxis(fig[positions(k)...],
        ## dest="+proj=natearth", 
        ## dest = "+proj=ortho +lon_0=$(mean(lons)) +lat_0=$(mean(lats))",
        dest="+proj=longlat +datum=WGS84",
        xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false, title=L"$\,\,\, Z = %$k$ (%$(ratio_K[k])%)") for k in 1:K]
    # Divide by 100 to get hPa
    sp = [GeoMakie.contourf!(ax[k], lons, lats, Δmean_season_k[k] / 100; colormap=:coolwarm, levels=range(-max_diff, max_diff, step=2)) for k in 1:K]

    for k in 1:K
        GeoMakie.xlims!(ax[k], LON_min, LON_max)
        GeoMakie.ylims!(ax[k], LAT_min, LAT_max)
        GeoMakie.lines!(ax[k], GeoMakie.to_multilinestring.(coastlines_ne.geometry); color=:black, linewidth=2)
        k ∈ [4, 3] && GeoMakie.hideydecorations!(ax[k], ticks=false)
        k ∈ [1, 4] && GeoMakie.hidexdecorations!(ax[k], ticks=false)
        ax[k].aspect = 1.65
    end
    colorrange = range(-max_diff, max_diff, step=2)
    cb_cell = fig[1:2, 3] = GeoMakie.GridLayout(height=555) # treat the colorbar-and-label system as a single grid cell
    cb = GeoMakie.Colorbar(cb_cell[2, 1], colormap=cgrad(:coolwarm, length(colorrange), categorical=true), colorrange=extrema(colorrange), ticks=-24:3:24)
    cb_label = GeoMakie.Label(cb_cell[1, 1], "hPa"; tellheight=true, tellwidth=false)
    # GeoMakie.Colorbar(fig[:, 3], sp[4], label="hPa", ticks=-24:3:24, height=GeoMakie.Relative(0.6))
    GeoMakie.colgap!(fig.layout, 16)
    GeoMakie.rowgap!(fig.layout, -390)

    fig
end

savefigcrop(fig, "FR_mean_JJA_pressure_diff_K_$(K)_d_$(Deg)_m_$(local_memory)_WGS84")

