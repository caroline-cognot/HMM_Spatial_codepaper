using JLD2
using CSV
using DataFrames
skip_sim=true
include("EGPD_functions.jl")
include("EGPD_class.jl")
include("covariance_functions.jl")
using Distributions
using Measures
z_hat = CSV.read("./00data/transformedECAD_zhat.csv", DataFrame, header=false)[:, 1]

N = length(z_hat)
# z_hat=fill(1,N) #test if setting this to 1 gives same result. as previous no-class periodic estimation.
# z_hat[z_hat.==4] .=3
K = length(unique(z_hat))
Robs = Matrix(CSV.read("./00data/transformedECAD_Robs.csv", DataFrame, header=false)[:, :])'
D = size(Robs, 1)
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
Nb = 500

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
Yobs = Matrix(CSV.read("./00data/transformedECAD_Yobs.csv", DataFrame, header=false)[:, :])'

degree_of_P = 2
if !skip_sim

XR = Matrix(CSV.read("23precip_intensity/res_real_data/EGPD_periodic_K_normalized" * string(K) * string(degree_of_P) * ".csv", DataFrame, header=false))'
D = size(XR, 1)
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
# fitted=fit_mple(data,Mat_h,model_type,param0,maxiter=20)
include("../utils/fast_bivariate_cdf.jl")

covmodel = load("./23precip_intensity/res_real_data/GMcov_withmarginalsdeg" * string(degree_of_P) * ".jld2")["fitted"]

################# plot the covariance model ###################################
using Measures
coordll = [locsdata.LON locsdata.LAT]
gf = GaussianField(coordll, covmodel)
u_lags=0:3
h_vals=0:50:900
begin
plt = plot(; xlabel="Distance h (km)", ylabel= "Covariance",
               legend=:topright, lw=2,size=(1000,600),left_margin=10mm,title="Spatio-temporal fitted covariance")

    for u in u_lags
        covs = cov_spatiotemporal.(Ref(covmodel), h_vals, u)
      
        plot!(plt, h_vals, covs, label="u = $u")
    end

annotate!(
    plt,
    (600, 0.7,
     text(L"C(h,u) = \frac{\sigma^2}{\left(\left(\frac{u}{a}\right)^{2\alpha} + 1\right)^{b+\delta}}\mathcal{M}\!\left(\frac{h}{\sqrt{\left(\left(\frac{u}{a}\right)^{2\alpha} + 1\right)^b}}; r; \nu\right)",
          12, :black, :relative))
)

annotate!(
    plt,
    (600, 0.55,
     text("\$ \\sigma = $(round(covmodel.σ², digits = 2)) , b = $(round(covmodel.β, digits = 2))   \$",
          12, :black, :relative))
)
annotate!(
    plt,
    (600, 0.5,
     text("\$ r = $(round(covmodel.c, digits = 2))  , \\nu = $(round(covmodel.ν, digits = 2))     \$",
          12, :black, :relative))
)

annotate!(
    plt,
    (600, 0.45,
     text("\$ a = $(round(covmodel.a, digits = 2))   , \\alpha =  $(round(covmodel.α, digits = 2))  , \\delta  = $(round(covmodel.δ, digits = 2))  \$",
          12, :black, :relative))
)
end

savefig(plt, "./23precip_intensity/res_real_data/covariance_fitted.pdf")
    display(plt)
###############################################################################

coordll = [locsdata.LON locsdata.LAT]
gf = GaussianField(coordll, covmodel)
times = 1:my_N

lagt = 4
Y = Yobs'
Pdry
using TruncatedMVN
function simulate_iterative(gf::GaussianField, times, lagt::Int, Y, Pdry)

    D = size(gf.Mat_h, 1)
    N = length(times)

    # Spatial distance matrix
    dist_s = gf.Mat_h

    phimPdry = quantile.(Normal(), Pdry)
    lb = ifelse.(Y .== 1, phimPdry, -Inf)
    ub = ifelse.(Y .== 0, phimPdry, Inf)


    # make covariance matrices of necessary lags
    Covariancesmatrices = [cov_spatiotemporal(gf.model, dist_s, l) for l in 0:(lagt+1)]
    C0 = Covariancesmatrices[0+1] # spatial cov at lag 0
    C_1tot = zeros(eltype(Covariancesmatrices[1]), lagt * D, lagt * D)

    for i in 1:lagt
        for j in 1:lagt
            C_1tot[(i-1)*D+1:i*D, (j-1)*D+1:j*D] .= Covariancesmatrices[abs(i - j)+1]
        end
    end
    C_ttotp1 = zeros(eltype(Covariancesmatrices[1]), D, lagt * D)
    for j in 1:lagt
        C_ttotp1[:, (j-1)*D+1:j*D] .= Covariancesmatrices[j+1]
    end





    matm = C_ttotp1 * inv(C_1tot)
    Sigmat = C0 - matm * transpose(C_ttotp1)

    # Storage
    Zsim = zeros(D, N)

    # # Initial block (lags)
    # lbinit=vec(lb[:,1:lagt])
    # ubinit=vec(ub[:,1:lagt])
    # Zinit = TruncatedMVN.sample(TruncatedMVNormal(zeros(lagt * D), Matrix(Symmetric(C_1tot)),lbinit,ubinit),1)
    # for l in 1:lagt
    #     Zsim[:, l] .= Zinit[(1+(l-1)*D):(l*D)]
    # end

    # unconditional (mean zero, full covariance for 1st day)
    Zsim[:, 1] .= TruncatedMVN.sample(TruncatedMVNormal(zeros(D), Matrix(Symmetric(C0)), lb[:, 1], ub[:, 1]), 1)

    for t in 2:lagt
        Zprec = stack_reverse_columns((Zsim[:, (1):(t-1)]))  # previous lagt day
        C_1tott = zeros(eltype(Covariancesmatrices[1]), (t - 1) * D, (t - 1) * D)

        for i in 1:(t-1)
            for j in 1:(t-1)
                C_1tott[(i-1)*D+1:i*D, (j-1)*D+1:j*D] .= Covariancesmatrices[abs(i - j)+1]
            end
        end
        C_ttotp1t = zeros(eltype(Covariancesmatrices[1]), D, (t - 1) * D)
        for j in 1:(t-1)
            C_ttotp1t[:, (j-1)*D+1:j*D] .= Covariancesmatrices[j+1]
        end





        matmt = C_ttotp1t * inv(C_1tott)
        Sigmatt = C0 - matmt * transpose(C_ttotp1t)
        Zsim[:, t] .= TruncatedMVN.sample(TruncatedMVNormal(matmt * Zprec, Matrix(Symmetric(Sigmatt)), lb[:, t], ub[:, t]), 1)

    end
    # Iterative simulation
    for t in (lagt+1):N
        Zprec = stack_reverse_columns((Zsim[:, (t-lagt):(t-1)]))  # previous lagt day
        Zsim[:, t] .= TruncatedMVN.sample(TruncatedMVNormal(matm * Zprec, Matrix(Symmetric(Sigmat)), lb[:, t], ub[:, t]), 1)
    end

    return Zsim
end

#test
# gf = GaussianField(coordll, GneitingMatern(1., 300.0, 1.0, 1.0, 1.0, 0.5, 0.5))
times = 1:my_N
Y = Yobs'[:, times]
@time data = simulate_iterative(gf, times, 10, Y, Pdry[:, times])
#1:1000 sims = done in 250s. for really random Pdry and Y
#re-run 1000 times in 3s, when using realistic Pdry and Y.
# complete 18627 sim : with observed Y and Pdry ; gives 36s for lagt=1. re-run a second time, gives 35s.
# for lagt=10, gives also 35s.


#make sims of the Z (hidden state) and Y (rain occurrence)
begin
    Ys = zeros(Bool, D, length(n2t), Nb)
    Zs = zeros(Int, length(n2t), Nb)
    @time "Simulations  Y" @threads for i in 1:Nb
        z, Y = my_rand(hmmspa, n2t; seq=true)
        Ys[:, :, i] = Y'
        Zs[:, i] = z
    end
end

#generate censorded Gaussian 

XR = zeros(D, length(n2t), Nb)
@time "Simulations  XR" @threads for i in 1:Nb
    Yprevious = fill(1, size(hmmspa, 2), 1)
    Ymoins1 = hcat(Yprevious, Ys[:, :, i])
    Pdry = similar(XR[:, :, i])
    for s in 1:D
        for n in 1:my_N
            t = n2t[n]
            k = Zs[n, i]
            h = Ymoins1[s, n]
            Pdry[s, n] = 1 - hmmspa.B[k, t, s, h+1]
        end
    end
    sim = simulate_iterative(gf, 1:length(n2t), 10, Ys[:, :, i], Pdry)
    XR[:, :, i] = sim
end
#2 sims took 55s.
# 150s for 10 sims
#2700 s for 500 sims

# get back to the rain intensity :
@load "./23precip_intensity/res_real_data/periodicEGPD_K" * string(K) * string(degree_of_P) * ".jld2" di



Rs = zeros(D, length(n2t), Nb)
@time "transformation to Rs" @threads for i in 1:Nb
    Yprevious = fill(1, size(hmmspa, 2), 1)
    Ymoins1 = hcat(Yprevious, Ys[:, :, i])
    Pdry = similar(XR[:, :, i])
    for s in 1:D
        for n in 1:my_N
            t = n2t[n]
            k = Zs[n, i]
            h = Ymoins1[s, n]
            Pdry[s, n] = 1 - hmmspa.B[k, t, s, h+1]
        end
    end
    sim = similar(Rs[:, :, i])
    for s in 1:D
        for n in 1:my_N
            k = Zs[n, i]
            h = Ymoins1[s, n]
            t = n2t[n]
            pdry = 1 - hmmspa.B[k, t, s, h+1]
            sim[s, n] = ifelse(XR[s, n, i] < quantile(Normal(), pdry), 0, quantile((di[s][Zs[n, i], t]), cdf(Truncated(Normal(), quantile(Normal(), pdry), Inf), XR[s, n, i])))
        end
    end
    Rs[:, :, i] = sim
end

idx = findall((Rs .== 0) .!= (Ys .== 0))

Rs
Ys
# Rs .= Rs .* Ys
println(findall(isinf, Rs))
# Rs[findall(isinf, Rs)].=0
JLD2.@save "./23precip_intensity/res_real_data/periodicEGPD_K" * string(my_K) * string(degree_of_P) * "_Sim_ZYR.jld2" Rs Ys Zs
end
data = JLD2.load("./23precip_intensity/res_real_data/periodicEGPD_K$(my_K)$(degree_of_P)_Sim_ZYR.jld2")
Rs = data["Rs"]
Ys = data["Ys"]
Zs = data["Zs"]


# ################# plots ##########################

# using SmoothPeriodicStatsModels # Name might change. Small collection of smooth periodic models e.g. AR, HMM
# using StochasticWeatherGenerators # interface to use with SmoothPeriodicStatsModels.jl

# station_50Q = CSV.read("./00data/transformedECAD_stations.csv", DataFrame)

# STAID = station_50Q.STAID #[32, 33, 39, 203, 737, 755, 758, 793, 11244, 11249];
# station_name = station_50Q.STANAME
# select_plot = [6,1, 16, 14, 21, 3]

# cor_bin_hist = cor(Yobs);

# cor_bin_mean_simu = mean(cor(Ys[:, :, i]') for i in 1:Nb);

# using Compose

# begin
#     plots_cor_bin = [plot(-0.1:0.1:0.8, -0.1:0.1:0.8, aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=13) for _ in 1:1]
#     scatter!(plots_cor_bin[1], vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu), label="", xlabel="Observations", ylabel="Simulations", c=1)
#     [xlims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
#     [ylims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
#     annotate!(0.2, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(cor_bin_hist) - vec_triu(cor_bin_mean_simu)), digits = 4))")
#     plot_cor_bin = plot(plots_cor_bin..., suptitle="Correlation in rain occurence Y", size=(500, 500), left_margin=15px, right_margin=8px, top_margin=2px, bottom_margin=40px)
# end

# println("Largest error between $(station_name[findmax(cor_bin_hist - cor_bin_mean_simu)[2][1]]) and $(station_name[findmax(cor_bin_hist - cor_bin_mean_simu)[2][2]])")

# ####### correlations between precip amounts 
# cor_hist = cor(Robs');

# corT_hist = corTail(Robs');

# cor_mean_simu = mean(cor(Rs[:, :, i]') for i in 1:Nb);

# corT_mean_simu = mean(corTail(Rs[:, :, i]') for i in 1:Nb);


# begin
#     plots_cor = [Plots.plot() for _ in 1:2]
#     scatter!(plots_cor[1], vec_triu(cor_hist), vec_triu(cor_mean_simu), label="Correlations", xlabel="Observations", ylabel="Simulations", c=2)
#     annotate!(plots_cor[1], 0.3, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(cor_hist) - vec_triu(cor_mean_simu)), digits = 4))")
#     Plots.abline!(plots_cor[1],1,0, line=:dash,aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=13)
#     scatter!(plots_cor[2], vec_triu(corT_hist), vec_triu(corT_mean_simu), label="Tail index", xlabel="Observations", ylabel="Simulations", c=3)
#     annotate!(plots_cor[2], 0.3, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(corT_hist) - vec_triu(corT_mean_simu)), digits = 4))")
#     Plots.abline!(plots_cor[2],1,0, line=:dash,aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=13)

#     [xlims!(plots_cor[i], -0.1, 1) for i in 1:2]
#     [ylims!(plots_cor[i], -0.1, 1) for i in 1:2]
#     plot_cor_all = plot(plots_cor..., size=(1000, 600), left_margin=15px, right_margin=8px, top_margin=2px, bottom_margin=40px, plot_title="Correlation in rain amounts R")
# end



# CRobs = continuity_ratio(Robs)
# CRsim = continuity_ratio(Rs)

# threshold=1.5
# dmax=800
# using LaTeXStrings

#     mask = .!(isnan.(CRobs) .| isnan.(CRsim) .| (Mat_h .> dmax))
#     x = CRobs[mask]
#     y = CRsim[mask]

#     p = scatter(x, y, xlabel="Observations", ylabel="Simulations",
#          label="CR",caption="Pairs with distance < $dmax km", aspect_ratio=true, xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=13);
#     Plots.abline!(p, 1, 0, line=:dash, label=:none);
#     ylims!(p, (0, threshold));
#     xlims!(p, (0, threshold));
# p
# plot_cor_all=plot(plots_cor...,p,layout=(1,3), size=(1000, 600), left_margin=15px, right_margin=8px, top_margin=-20px, bottom_margin=40px)

# savefig(plot_cor_all, "./23precip_intensity/res_real_data/corplot.pdf")

# acfrange = 0:9
# @views aa = [autocor(Rs[j, :, i], acfrange) for j in 1:D, i in 1:Nb]
# file_for_plot_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_plot.jl")
# include(file_for_plot_utilities)
# begin
#     p_spell_wet = [plot(xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=16) for j = 1:D]
#     for j = 1:D
#         errorline!(p_spell_wet[j], acfrange, stack(aa[j, :], dims=1)', groupcolor=:gray, label=islabel(j, 9, L"Simu $q_{0,100}$"), errortype=:percentile, percentiles=[0, 100], fillalpha=0.8, lw=2, centertype=:median)
#         plot!(p_spell_wet[j], acfrange, autocor(Robs[j, :], acfrange), label=islabel(j, 9, "Obs"), lw=2.0, c=1, markers=:circle, alpha=0.8)
#     end

#     [xlabel!(p_spell_wet[j], "Lag", xlabelfontsize=12) for j in select_plot[4:6]]
#     [ylabel!(p_spell_wet[j], "ACF", ylabelfontsize=12) for j in select_plot[[1,4]]]
#     [title!(p_spell_wet[j], locsdata.STANAME[j], titlefontsize=13) for j = 1:D]
#     pall_ACF = plot(p_spell_wet[select_plot]..., layout=(4, 3), size=(1190, 500), left_margin=19px)
# end
# savefig(pall_ACF, "./23precip_intensity/res_real_data/acfrr.pdf")

# select_month = 1:12
# my_autoregressive_order = 1
# idx_months = [findall(x -> month.(x) == m, every_year) for m in 1:12]

# idx_month_vcat = vcat(idx_months[select_month]...)
# year_range = unique(year.(every_year))

# idx_year = [findall(x -> year.(x) == m, every_year) for m in year_range]

# idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_months]

# month_rain_simu = [monthly_agg(Rs[j, :, i], idx_all) for j in 1:D, i in 1:Nb]


# month_rain_histo = [monthly_agg(Robs[j, :], idx_all) for j in 1:D]
# gr() # plotly() # for interactive plots
# default(fontfamily="Computer Modern")
# cur_colors = get_color_palette(:auto, 100)
# my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K);
# using StatsPlots.PlotMeasures # To play with margin in Plots

# qs = [0.9, 0.5, 0.1]
# @time "Plot monthly quantile" begin
#     p_month_RR = [scatter(xtickfontsize=10, ytickfontsize=11, ylabelfontsize=12, legendfontsize=12, foreground_color_legend=nothing) for j = 1:D]
#     for j = 1:D
#         for (α, per) in enumerate([[0, 100], [25, 75]])
#             for (cc, q) in enumerate(qs)
#                 errorline!(p_month_RR[j], [quantile(month_rain_simu[j, i][:, m], q) for m in 1:12, i in 1:Nb], label=(α == 1 ? islabel(j, 1, L"Simu  $q_{%$(Int(q*100))}$") : :none), fillalpha=0.18 * α^2, centertype=:median, errortype=:percentile, percentiles=per, groupcolor=my_palette(length(qs))[cc])
#             end
#         end
#         for q in qs
#             scatter!(p_month_RR[j], m -> quantile(month_rain_histo[j][:, m], q), 1:12, label=(q == qs[1] ? islabel(j, 1, "Obs") : :none), legend=:topleft, ms=2.5, c=:blue)
#             plot!(p_month_RR[j], m -> quantile(month_rain_histo[j][:, m], q), 1:12, label=:none, c=:blue, lw=1.75)
#         end
#         xticks!(p_month_RR[j], 1:12, string.(first.(monthabbr.(1:12))))
#         ylims!(p_month_RR[j], 0, Inf)
#     end
#     [ylabel!(p_month_RR[j], "Rain (mm)") for j in select_plot[[1,4]]]

#     [title!(p_month_RR[j], locsdata.STANAME[j], titlefontsize=12) for j = 1:D]
#     pall_month_RR = plot(p_month_RR[select_plot]..., layout=(2, 3), size=(1190, 500), left_margin=19px)
# end
# savefig(pall_month_RR, "./23precip_intensity/res_real_data/SpaTgen_PeriodicEGPDS_deg" * string(degree_of_P) * ".pdf")

# using RollingFunctions
# ndays=5
# Robs3 = hcat([vcat(fill(0,ndays), rolling(sum, Robs[s,:], ndays) ) for s in 1:D] ...)'
# Rs3 = cat(
#     [hcat([vcat(fill(0, ndays), rolling(sum, Rs[s, :, i], ndays)) for s in 1:D]...)'
#      for i in 1:Nb]...;
#     dims = 3
# )

# idx_months = [findall(x -> dayofyear.(x) == m, every_year) for m in 1:366]

# idx_month_vcat = vcat(idx_months[select_month]...)
# year_range = unique(year.(every_year))

# idx_year = [findall(x -> year.(x) == m, every_year) for m in year_range]

# idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_months]

# month_rain_simu = [monthly_agg(Rs3[j, :, i], idx_all) for j in 1:D, i in 1:Nb]


# month_rain_histo = [monthly_agg(Robs3[j, :], idx_all) for j in 1:D]
# qs = [0.9, 0.5, 0.1]
# @time "Plot monthly quantile" begin
#     p_month_RR = [scatter(xtickfontsize=10, ytickfontsize=11, ylabelfontsize=12, legendfontsize=12, foreground_color_legend=nothing) for j = 1:D]
#     for j = 1:D
#         for (α, per) in enumerate([[0, 100], [25, 75]])
#             for (cc, q) in enumerate(qs)
#                 errorline!(p_month_RR[j], [quantile(month_rain_simu[j, i][:, m], q) for m in 1:366, i in 1:Nb], label=(α == 1 ? islabel(j, 1, L"Simu  $q_{%$(Int(q*100))}$") : :none), fillalpha=0.18 * α^2, centertype=:median, errortype=:percentile, percentiles=per, groupcolor=my_palette(length(qs))[cc])
#             end
#         end
#         for q in qs
#             scatter!(p_month_RR[j], m -> quantile(month_rain_histo[j][:, m], q), 1:366, label=(q == qs[1] ? islabel(j, 1, "Obs") : :none), legend=:topleft, ms=2.5, c=:blue)
#             plot!(p_month_RR[j], m -> quantile(month_rain_histo[j][:, m], q), 1:366, label=:none, c=:blue, lw=1.75)
#         end
#         ylims!(p_month_RR[j], 0, Inf)
#     end
#     [ylabel!(p_month_RR[j], "Rain (mm)") for j in select_plot[[1,4]]]
#     [xlabel!(p_month_RR[j], "Day of year") for j in select_plot[4:6]]

#     [title!(p_month_RR[j], locsdata.STANAME[j], titlefontsize=12) for j = 1:D]
#     pall_month_RR = plot(p_month_RR[select_plot]..., layout=(2, 3), size=(1190, 500), left_margin=19px)
# end
# savefig(pall_month_RR, "./23precip_intensity/res_real_data/5daycumulSpaTgen_PeriodicEGPDS_deg" * string(degree_of_P) * ".pdf")


# begin
#     qmax = 0.75
#     RRmax = [quantile(r, qmax) for r in eachrow(Robs)]
#     RORo = mean(Robs .> RRmax, dims=1)
#     RORs = [mean(rr .> RRmax, dims=1) for rr in eachslice(Rs, dims=3)]

#     JJA = [6, 7, 8]
#     MAM = [3, 4, 5]
#     SON = [9, 10, 11]
#     DJF = [12, 1, 2]
#     SEASONS = [DJF, MAM, JJA, SON]
#     seasonname = ["Winter : DJF", "Spring : MAM", "Summer : JJA", "Autumn : SON"]
#     local_order = my_autoregressive_order
#     idx_seasons = [findall(month.(every_year) .∈ tuple(season)) for season in SEASONS]

#     ## ROR density
#     maxlag = 4
#     K = 4
#     m = 1

#     #RORo,RORs

#     pROR = [plot() for m in idx_seasons]
#     pRORautocor = [plot() for m in idx_seasons]
#     for m in eachindex(idx_seasons)
#         xax = 0:(1/D):1
#         hRORobs = [mean(RORo[idx_seasons[m]] .== xax[k]) for k in 1:(D+1)]

#         xax2, hRORsim = hist_vectors([RORs[i][idx_seasons[m]] for i in 1:Nb], xax)
#         mindy, maxdy, medy, y = enveloppe_minmax(hRORsim, xax)
#         label1 = "HMM-SPA - K = $K, m=1 + EGPD + GM cov"

#         plot!(pROR[m], y, medy; ribbon=(medy - mindy, -medy + maxdy), fillalpha=0.2, label=label1)

#         rorsim = [RORs[i][idx_seasons[m]] for i in 1:Nb]
#         acf_sim = [autocor(rorsim[i]) for i in 1:length(rorsim)]
#         miniacf = [minimum([acf_sim[i][ilag] for i in 1:length(rorsim)]) for ilag in 1:maxlag+1]
#         maxiacf = [maximum([acf_sim[i][ilag] for i in 1:length(rorsim)]) for ilag in 1:maxlag+1]
#         moyacf = [mean([acf_sim[i][ilag] for i in 1:length(rorsim)]) for ilag in 1:maxlag+1]

#         # Plot ACF with bars
#         plot!(pRORautocor[m], 0:maxlag, moyacf; ribbon=(moyacf - miniacf, -moyacf + maxiacf), fillalpha=0.2, label=label1)




#         scatter!(pROR[m], xax, hRORobs, label="Observations", xlabel="ROR", title=seasonname[m])
#         # ylims!(pROR[m], 0, 0.05)

#         acf_obs = autocor(RORo[1, :], 0:maxlag)# Compute ACF for lags 0 to maxlag
#         scatter!(pRORautocor[m], 0:maxlag, acf_obs, label="Observations", xlabel="Lag", ylabel="Autocorrelation of ROR", title=seasonname[m])
#     end
#     plot(pROR..., size=(1000, 600), top_margin=0.34cm, left_margin=0.3cm, bottom_margin=0.22cm, suptitle="Distribution of joint exceedance of quantile $qmax")
#     plot(pRORautocor..., size=(1000, 600), top_margin=0.34cm, left_margin=0.3cm, bottom_margin=0.22cm, suptitle="Autocorrelation of joint exceedance of quantile $qmax")
#     savefig(plot(pROR..., size=(1000, 600), top_margin=0.34cm, left_margin=0.3cm, bottom_margin=0.22cm, suptitle="Distribution of joint exceedance of quantile $qmax"), "./23precip_intensity/res_real_data/QER" * string(qmax) * "" * ".pdf")
#     savefig(plot(pRORautocor..., size=(1000, 600), top_margin=0.34cm, left_margin=0.3cm, bottom_margin=0.22cm, suptitle="Autocorrelation of joint exceedance of quantile $qmax"), "./23precip_intensity/res_real_data/QER" * string(qmax) * "autocor" * ".pdf")

# end

# ## ROR density
# begin
# qmax = 0.95
# RRmax = [quantile(r, qmax) for r in eachrow(Robs)]
# RORo = mean(Robs .> RRmax, dims=1)
# RORs = [mean(rr .> RRmax, dims=1) for rr in eachslice(Rs, dims=3)]

# JJA = [6, 7, 8]
# MAM = [3, 4, 5]
# SON = [9, 10, 11]
# DJF = [12, 1, 2]
# SEASONS = [DJF, MAM, JJA, SON]
# seasonname = ["Winter : DJF", "Spring : MAM", "Summer : JJA", "Autumn : SON"]
# local_order = my_autoregressive_order
# idx_seasons = [findall(month.(every_year) .∈ tuple(season)) for season in SEASONS]

# ## ROR density
# maxlag = 4
# K = 4
# m = 1
# maxlag = 10
# #RORo,RORs
# pROR = [plot() for m in idx_seasons]
# pRORautocor = [plot() for m in idx_seasons]
# for m in eachindex(idx_seasons)
#     xax = 0:(1/D):1
#     hRORobs = [mean(RORo[idx_seasons[m]] .== xax[k]) for k in 1:(D+1)]

#     xax2, hRORsim = hist_vectors([RORs[i][idx_seasons[m]] for i in 1:Nb], xax)
#     mindy, maxdy, medy, y = enveloppe_minmax(hRORsim, xax)
#     label1 = "HMM-SPA + EGPD + GM cov"

#     plot!(pROR[m], y, medy; ribbon=(medy - mindy, -medy + maxdy), fillalpha=0.2, label=:none)

#     rorsim = [RORs[i][idx_seasons[m]] for i in 1:Nb]
#     acf_sim = [autocor(rorsim[i]) for i in 1:length(rorsim)]
#     miniacf = [minimum([acf_sim[i][ilag] for i in 1:length(rorsim)]) for ilag in 1:maxlag+1]
#     maxiacf = [maximum([acf_sim[i][ilag] for i in 1:length(rorsim)]) for ilag in 1:maxlag+1]
#     moyacf = [mean([acf_sim[i][ilag] for i in 1:length(rorsim)]) for ilag in 1:maxlag+1]

#     # Plot ACF with bars
#     plot!(pRORautocor[m], 0:maxlag, moyacf; ribbon=(moyacf - miniacf, -moyacf + maxiacf), fillalpha=0.2, label=ifelse(m==1,label1,:none))




#     scatter!(pROR[m], xax, hRORobs, label=ifelse(m==1,:none,:none), xlabel="\$ QER_{$qmax} \$ ", title=seasonname[m],ylabel=ifelse(m==1,"Distribution",""))
#     # ylims!(pROR[m], 0, 0.05)

#     acf_obs = autocor(RORo[1, :], 0:maxlag)# Compute ACF for lags 0 to maxlag
#     scatter!(pRORautocor[m], 0:maxlag, acf_obs, label=ifelse(m==1,"Observations",:none), xlabel="Temporal lag", ylabel=ifelse(m==1,"Autocorrelation"," "), title=" ")

# end

# p=plot(pROR...,pRORautocor...,layout=(2,4), size=(1200, 600), top_margin=0.34cm, left_margin=0.45cm, bottom_margin=0.3cm)
# savefig(p, "./23precip_intensity/res_real_data/QER" * string(qmax)  * ".pdf")
# end

# JJA = [6, 7, 8]
# MAM = [3, 4, 5]
# SON = [9, 10, 11]
# DJF = [12, 1, 2]
# SEASONS = [DJF, MAM, JJA, SON]
# seasonname = ["Winter : DJF", "Spring : MAM", "Summer : JJA", "Autumn : SON"]
# idx_seasons = [findall(month.(every_year) .∈ tuple(season)) for season in SEASONS]

# using Statistics, Dates, Plots

# # --- Assumed inputs:
# # Yobs :: Array{Int,2}  (D×Nt)
# # Ysim :: Array{Int,3}  (D×Nt×Nb)
# # every_year :: Vector{Date} of length Nt
# # JJA, MAM, SON, DJF, SEASONS, seasonname defined as in your code

# # --- Define helper functions

# # Event definitions
# f00(a,b)     = (a .== 0) .& (b .== 0)
# f01(a,b)     = (a .== 0) .& (b .== 1)
# f00lag(a,b)  = (a[2:end] .== 0) .& (b[1:end-1] .== 0)
# f01lag(a,b)  = (a[2:end] .== 0) .& (b[1:end-1] .== 1)

# # Pairwise probability function
# function pairwise_prob(Y, f)
#     D = size(Y, 1)
#     P = zeros(D, D)
#     for i in 1:D, j in 1:D
#         P[i, j] = mean(f(Y[i, :], Y[j, :]))
#     end
#     return P
# end

# # Function to compute seasonal pairwise probability
# function seasonal_pairwise(Yobs, Ysim, idx_seasons, f)
#     D, Nt = size(Yobs)
#     Nb = size(Ysim, 3)
#     nseasons = length(idx_seasons)
#     Pobs = Vector{Matrix{Float64}}(undef, nseasons)
#     Psim = Vector{Matrix{Float64}}(undef, nseasons)

#     for s in 1:nseasons
#         idx = idx_seasons[s]
#         # slice time indices for the season
#         Yobs_s = Yobs[:, idx]
#         Ysim_s = Ysim[:, idx, :]

#         # observed
#         Pobs[s] = pairwise_prob(Yobs_s, f)
#         # simulated (average over Nb)
#         Psim[s] = mean([pairwise_prob(Ysim_s[:, :, b], f) for b in 1:Nb])
#     end
#     return Pobs, Psim
# end

# # --- Compute seasonal probabilities for each event
# Pobs_00,    Psim_00    = seasonal_pairwise(Yobs', Ys, idx_seasons, f00)
# Pobs_01,    Psim_01    = seasonal_pairwise(Yobs', Ys, idx_seasons, f01)
# Pobs_00lag, Psim_00lag = seasonal_pairwise(Yobs', Ys, idx_seasons, f00lag)
# Pobs_01lag, Psim_01lag = seasonal_pairwise(Yobs', Ys, idx_seasons, f01lag)

# # --- Plotting helpers
# function scatter_panel(Pobs, Psim; title_str="", xlabel="", ylabel="")
#     scatter(
#         vec(Pobs), vec(Psim),
#         xlabel=xlabel,
#         ylabel=ylabel,
#         label="", title=title_str,
#         legend=false, xlim=(0,1), ylim=(0,1),
#         tickfontsize=8, guidefontsize=10
#     )
#     plot!([0,1],[0,1], lw=1, lc=:black, ls=:dash)
# end

# function dummy_label_panel(text)
#     plot(legend=false, framestyle=:none, grid=false, axis=false,
#         title=text)
# end

# # --- Build 4×5 grid (first column season labels)
# titles = [
#     "\$ \\mathbb{P}(Y_i^{(n)}=0, Y_j^{(n)}=0)\$",
#     "\$ \\mathbb{P}(Y_i^{(n)}=0, Y_j^{(n)}=1)\$",
#     "\$ \\mathbb{P}(Y_i^{(n)}=0, Y_j^{(n-1)}=0)\$",
#     "\$ \\mathbb{P}(Y_i^{(n)}=0, Y_j^{(n-1)}=1)\$"
# ]

# plots = []

# for s in 1:4  # season
#     push!(plots, dummy_label_panel(seasonname[s]))  # first col: season
#     for (i,title) in enumerate(titles)
#         Pobs_s = [Pobs_00,Pobs_01,Pobs_00lag,Pobs_01lag][i][s]
#         Psim_s = [Psim_00,Psim_01,Psim_00lag,Psim_01lag][i][s]
#         xlabel_txt = (s==4 ? "Observed" : "")
#         ylabel_txt = (i==1 ? "Simulated" : "")
#         push!(plots, scatter_panel(Pobs_s, Psim_s, title_str=title,
#                                    xlabel=xlabel_txt, ylabel=ylabel_txt))
#     end
# end

# # --- Compose final grid: 4 rows × 5 columns
# plot(plots..., layout=(4,5), size=(1000,600),xtickfontsize=10, ytickfontsize=10, ylabelfontsize=10, legendfontsize=10,titlefontsize=10)
# savefig("./23precip_intensity/res_real_data/probs01.pdf")


# ########## with seasons ################
# function continuity_ratio(Y::AbstractMatrix)
#     D, N = size(Y)
#     CR = fill(NaN, D, D)
#     for k in 1:D
#         for l in 1:D
#             if k == l
#                 continue
#             end
#             yk, yl = Y[k, :], Y[l, :]
#             mask1 = (yk .> 0) .& (yl .== 0)
#             mask2 = (yk .> 0) .& (yl .> 0)

#             if any(mask1) && any(mask2)
#                 num = mean(yk[mask1])
#                 den = mean(yk[mask2])
#                 CR[k, l] = num / den
#             end
#         end
#     end
#     return CR
# end

# function continuity_ratio(Rsim::Array{<:Real,3})
#     D, N, Nsim = size(Rsim)
#     CRs = Array{Float64}(undef, D, D, Nsim)
#     for s in 1:Nsim
#         CRs[:, :, s] = continuity_ratio(Rsim[:, :, s])
#     end
#     return mean(CRs, dims=3)[:, :, 1]
# end

# function seasonal_continuity_ratio(Yobs, Ysim, idx_seasons)
#     nseasons = length(idx_seasons)
#     D = size(Yobs, 1)
#     Pobs = Vector{Matrix{Float64}}(undef, nseasons)
#     Psim = Vector{Matrix{Float64}}(undef, nseasons)

#     for s in 1:nseasons
#         idx = idx_seasons[s]
#         Yobs_s = Yobs[:, idx]
#         Ysim_s = Ysim[:, idx, :]
#         Pobs[s] = continuity_ratio(Yobs_s)
#         Psim[s] = continuity_ratio(Ysim_s)
#     end

#     return Pobs, Psim
# end

# CRobs, CRsim = seasonal_continuity_ratio(Robs, Rs, idx_seasons)

# function continuity_ratio_lag(Y::AbstractMatrix)
#     D, N = size(Y)
#     CR = fill(NaN, D, D)

#     # Shift in time: compare Yk(t) with Yl(t-1)
#     for k in 1:D
#         for l in 1:D
#             if k == l
#                 continue
#             end
#             # Align so Yk at t corresponds to Yl at t-1
#             yk = Y[k, 2:end]
#             yl = Y[l, 1:end-1]

#             # same logic as before
#             mask1 = (yk .> 0) .& (yl .== 0)
#             mask2 = (yk .> 0) .& (yl .> 0)

#             if any(mask1) && any(mask2)
#                 num = mean(yk[mask1])
#                 den = mean(yk[mask2])
#                 CR[k, l] = num / den
#             end
#         end
#     end
#     return CR
# end
# function continuity_ratio_lag(Rsim::Array{<:Real,3})
#     D, N, Nsim = size(Rsim)
#     CRs = Array{Float64}(undef, D, D, Nsim)
#     for s in 1:Nsim
#         CRs[:, :, s] = continuity_ratio_lag(Rsim[:, :, s])
#     end
#     return mean(CRs, dims=3)[:, :, 1]
# end

# function seasonal_continuity_ratio_lag(Yobs, Ysim, idx_seasons)
#     nseasons = length(idx_seasons)
#     D = size(Yobs, 1)
#     Pobs = Vector{Matrix{Float64}}(undef, nseasons)
#     Psim = Vector{Matrix{Float64}}(undef, nseasons)

#     for s in 1:nseasons
#         idx = idx_seasons[s]
#         # Avoid misalignment if the season has very few days
#         if length(idx) <= 1
#             Pobs[s] = fill(NaN, D, D)
#             Psim[s] = fill(NaN, D, D)
#             continue
#         end

#         Yobs_s = Yobs[:, idx]
#         Ysim_s = Ysim[:, idx, :]
#         Pobs[s] = continuity_ratio_lag(Yobs_s)
#         Psim[s] = continuity_ratio_lag(Ysim_s)
#     end

#     return Pobs, Psim
# end


# CRobs1, CRsim1 = seasonal_continuity_ratio_lag(Robs, Rs, idx_seasons)

# function scatter_panel_CR(CRobs, CRsim,lims; xlabel="", ylabel="",title="")
#     scatter(
#         vec(CRobs), vec(CRsim),
#         xlabel=xlabel, ylabel=ylabel,title=title,
#         label="", legend=false,
#         xlim=lims, ylim=lims,  # adjust if needed
#         tickfontsize=8, guidefontsize=10
#     )
#     plot!([-1.5,1.5],[-1.5,1.5], lw=1, lc=:black, ls=:dash)
# end

# cor_hists = [cor(Robs[:,idx_seasons[s]]') for s in 1:4];


# cor_mean_simu =[ mean(cor(Rs[:,idx_seasons[s], i]') for i in 1:Nb) for s in 1:4 ];

# corT_hist = [corTail(Robs[:,idx_seasons[s]]') for s in 1:4 ];

# corT_mean_simu = [mean(corTail(Rs[:, idx_seasons[s], i]') for i in 1:Nb) for s in 1:4 ];


# function cor_lag1(Y::AbstractMatrix)
#     if size(Y, 2) < 2
#         return fill(NaN, size(Y,1), size(Y,1))
#     end
#     return cor(Y[:,2:end]', Y[:,1:end-1]')
# end
# cor_hists1 = [cor_lag1(Robs[:,idx_seasons[s]]) for s in 1:4];


# cor_mean_simu1 =[ mean(cor_lag1(Rs[:,idx_seasons[s], i]) for i in 1:Nb) for s in 1:4 ];

# plots = []


# for s in 1:4
#     push!(plots, dummy_label_panel(seasonname[s]))  # left column: season name
#     push!(plots, scatter_panel_CR(CRobs[s], CRsim[s],(0,1.6),
#                                   xlabel=(s==4 ? "Observed" : ""),
#                                   ylabel="Simulated", title=L"CR_0 = \frac{(E[R_k^{(n)} \mid R_k^{(n)} > 0,\, R_\ell^{(n)} = 0]} {(E[R_k^{(n)} \mid R_k^{(n)} > 0,\, R_\ell^{(n)} > 0]}"))
#     # Repeat 3 dummy plots to fill 4 columns (if desired, or only one column for CR)
#     push!(plots, scatter_panel_CR(CRobs1[s], CRsim1[s],(0.25,1.2),
#     xlabel=(s==4 ? "Observed" : ""),
#     ylabel="", title=L"CR_1 = \frac{(E[R_k^{(n)}\mid R_k^{(n)} > 0,\, R_\ell^{(n-1)} = 0]} {(E[R_k^{(n)} \mid R_k^{(n)} > 0,\, R_\ell^{(n-1)} > 0]}"))
#     push!(plots, scatter_panel_CR(cor_hists[s], cor_mean_simu[s],(-0.20,1.02),
#     xlabel=(s==4 ? "Observed" : ""),
#     ylabel="", title=L"Cor(R_k,R_\ell)"))
#     push!(plots, scatter_panel_CR(cor_hists1[s], cor_mean_simu1[s],(-0.2,0.6),
#     xlabel=(s==4 ? "Observed" : ""),
#     ylabel="", title=L"Cor(R_k^{(n)},R_\ell^{(n-1)})"))
#     push!(plots, scatter_panel_CR(corT_hist[s], corT_mean_simu[s],(0,1),
#     xlabel=(s==4 ? "Observed" : ""),
#     ylabel="", title="Tail index"))
# end

# plot(plots..., layout=(4,6), size=(2000,1200),xtickfontsize=10, ytickfontsize=10, ylabelfontsize=10, legendfontsize=10,titlefontsize=9)
# savefig("./23precip_intensity/res_real_data/CR.pdf")

