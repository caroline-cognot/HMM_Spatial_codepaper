using SmoothPeriodicStatsModels
using Distributions
using Optimization, OptimizationOptimJL, LineSearches
using ForwardDiff
import Random
using StochasticWeatherGenerators
gr() # plotly() # for interactive plots
default(fontfamily="Computer Modern")
cur_colors = get_color_palette(:auto, 100);
my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K)

file_for_plot_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_plot.jl")
include(file_for_plot_utilities)


using CSV
using DataFrames
using JLD2


##### load parameters from parameter files estimated on the real data.
my_distance = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))

my_D = length(my_locations[:, 1])


using Dates
date_start = Date(1973)
# date_end = Date(2024) - Day(1)
date_end = Date(1994) - Day(1)

every_year = date_start:Day(1):date_end

my_N = length(every_year)
n2t = dayofyear_Leap.(every_year)

doss_save = "./13PeriodicHMMSpatialBernoulli/res_real_data/"

Mat_h = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))

# get the parameters for Z,Y.
hmmspadata = load("./13PeriodicHMMSpatialBernoulli/res_real_data/parameters/K4_resu_18627days_degree1_maxiter100_m1_R0500_QMCm30.jld2")



all_thetaR_iterations = hmmspadata["all_thetaR_iterations"]
all_thetaB_iterations = hmmspadata["all_thetaB_iterations"]
all_thetaA_iterations = hmmspadata["all_thetaA_iterations"]
logtots = hmmspadata["logtots"]
iter=argmax(logtots)
@show iter                        

thetaR = all_thetaR_iterations[iter+1]
thetaB = all_thetaB_iterations[iter+1]
thetaA = all_thetaA_iterations[iter+1]

a = hmmspadata["hmm"].a
modeltrue = Trig2ARPeriodicHMMSpatial(a,thetaA,thetaB,thetaR,366,my_distance)

# get the parameters for R 
include("./23precip_intensity/EGPD_functions.jl")
include("./23precip_intensity/EGPD_class.jl")
include("./23precip_intensity/covariance_functions.jl")
margins_true= load("./23precip_intensity/res_real_data/full_periodicEGPD_K42.jld2")
covmodel_true = load("./23precip_intensity/res_real_data/GMcov_withmarginalsdeg2.jld2")["fitted"]

##########################################################################################################################

## generate synthetic data from the estimated parameters of the real data. in Bootstrap-like procedure, they will be used to re-estimate the parameters.

Z,Yall = rand(modeltrue,my_N,seq=true)
Z
Yall

coordll = [locsdata.LON locsdata.LAT]
gf = GaussianField(coordll, covmodel)

Yprevious = fill(1, size(modeltrue, 2), 1)
Yprevious=convert(BitMatrix,Yprevious)
Ymoins1 = hcat(Yprevious,Yall')
Pdry =zeros(D, length(n2t))
for s in 1:my_D
    for n in 1:my_N
        t = n2t[n]
        k = Z[n]
        h = Ymoins1[s, n]
        Pdry[s, n] = 1 - modeltrue.B[k, t, s, h+1]
        end
    end
    XR = simulate_iterative(gf, 1:length(n2t), 10, Yall', Pdry)
R = zeros(D, length(n2t))
for s in 1:D
        for n in 1:my_N
            h = Ymoins1[s, n]
            t = n2t[n]
            pdry = Pdry[s,n]
            R[s, n] = ifelse(XR[s, n] < quantile(Normal(), pdry), 0, quantile((di[s][Z[n], t]), cdf(Truncated(Normal(), quantile(Normal(), pdry), Inf), XR[s, n])))
        end
end

# now we have one realisation of Z,Y,R, where Z is the weather type, Y is precip occurence and R is precip intensity with same 0/non0 pattern as Y.


# # Obsolete code to fit the model with not-too-far starting values - testing our hopes.
# thetaA_start = copy(thetaA)
# thetaA_start[2:end,:,:] .= 0.0
# thetaB_start = copy(thetaB)
# thetaB_start[:,:,:,2:end] .= 0.0
# thetaR_start = copy(thetaR)
# thetaR_start[:,2:end] .= 0.0
# model_cst = Trig2ARPeriodicHMMSpatial(a,thetaA_start,thetaB_start,thetaR_start,366,my_distance)

# solver = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking())

# Y_past = BitMatrix(rand(Bool, my_autoregressive_order, my_D))
# Y = Yall
# @time "FitMLE HMMSpa (EM)" history, all_θᴬ_iter, all_θᴮ_iter, all_θᴿ_iter = fit_mle!(model_cst, thetaA_start, thetaB_start, thetaR_start, Y, Y_past; solver=solver, n2t=n2t, maxiter=20, tol=1e-3, maxiters_R=100, display=:iter, tdist=tdist, QMC_m=30);

# begin
#     iter_best = argmax(history["logtots"])
#     p_logtots = plot(history["logtots"], xlabel="EM iteration", ylabel="Log-likelihood", title="Convergence of the spatial HMM EM", legend=:none, lw=2)
#     vline!(p_logtots, [iter_best], c=:red, s=:dash, label="best iter")
# end

# begin
#     pr = plot(title="Spatial range per hidden state", ylabel="Range (km)", tickfont=12, legendfontsize=14)
#     for k in 1:my_K
#         plot!(pr, 1:my_T, model_cst.R[k, :], c=my_color(k, my_K), label=L"k = %$k", lw=2)
#     end
#      for k in 1:my_K
#         plot!(pr, 1:my_T, modeltrue.R[k, :], c=my_color(k, my_K), label=L"k = %$k", lw=2, ls=:dash)
#     end
#     xticks!(pr, vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367))
# end

### inference with procedure similar to the one used on real data. Start by fitting the HMM indep.
ref_station = 1
hmm_random = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order; ξ=[1; zeros(my_K - 1)], ref_station=ref_station);

@time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1);
θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, 1);

@time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past, maxiter=100, robust=true; display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t);



R0=500
θᴿ = zeros(my_K, 3)
θᴿ[:, 1] .= log.(R0)

hmm_spa_init = Trig2ARPeriodicHMMSpatial(fill(1 / my_K, my_K), θq_fit, θy_fit, θᴿ, 366, my_distance)



solver = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking())

Y_past = BitMatrix(rand(Bool, my_autoregressive_order, my_D))
Y = Yall
@time "FitMLE HMMSpa (EM)" history, all_θᴬ_iter, all_θᴮ_iter, all_θᴿ_iter = fit_mle!(hmm_spa_init, θq_fit, θy_fit, θᴿ, Y, Y_past; solver=solver, n2t=n2t, maxiter=20, tol=1e-3, maxiters_R=100, display=:iter, tdist=tdist, QMC_m=30);

begin
    iter_best = argmax(history["logtots"])
    p_logtots = plot(history["logtots"], xlabel="EM iteration", ylabel="Log-likelihood", title="Convergence of the spatial HMM EM", legend=:none, lw=2)
    vline!(p_logtots, [iter_best], c=:red, s=:dash, label="best iter")
end


iter=argmax(history["logtots"])
@show iter                        
iter=20

# this is what we are going to save for comparison with true parameters.
thetaR = all_θᴿ_iter[iter+1]
thetaB = all_θᴮ_iter[iter+1]
thetaA = all_θᴬ_iter[iter+1]
hmm_spa_fitted = Trig2ARPeriodicHMMSpatial(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, 366, my_distance)
# or, should we compare the actual functions ? or both ? ie hmm_spa_fitted.R and modeltrue.R, hmm_spa_fitted.A and modeltrue.A, hmm_spa_fitted.B and modeltrue.B

# just to look at the fitted parameters - not to be used for comparison as we are going to do 100 inferences at least...
begin
    pr = plot(title="Spatial range per hidden state", ylabel="Range (km)", tickfont=12, legendfontsize=14)
    for k in 1:my_K
        plot!(pr, 1:my_T, hmm_spa_fitted.R[k, :], c=my_color(k, my_K), label=L"k = %$k", lw=2)
    end
     for k in 1:my_K
        plot!(pr, 1:my_T, modeltrue.R[k, :], c=my_color(k, my_K), label=L"k = %$k", lw=2, ls=:dash)
    end
    xticks!(pr, vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367))
end

begin
    pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=4, tickfont=12, legendfontsize=16) for k in 1:my_K]
    for k in 1:my_K
        [plot!(pA[k], hmm_spa_fitted.A[k, l, :], c=my_color(l, my_K), label=L"Q_{%$(k)\to %$(l)}", legend=:top, lw=1.75) for l in 1:my_K]
        [plot!(pA[k], modeltrue.A[k, l, :], c=my_color(l, my_K), label=L"Q_{%$(k)\to %$(l)}", legend=:top, lw=1.75,linestyle=:dash) for l in 1:my_K]
        hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
        xticks!(pA[k], vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367), ylims=(0, 1))
    end
    pallA = plot(pA..., size=(1000, 500))
end

#### retrieve z_hat from estimated parameters

# i need to compare z_hat with true z.
Zhat  = viterbi(hmm_spa_fitted, Yall, Yprevious'; n2t=n2t, QMC_m=30)
plot(Zhat[1:365])
plot!(Z[1:365],c=:red)

countmap_matrix = [sum((Z .== k) .& (Zhat .== l)) for k in 1:my_K, l in 1:my_K]
df = DataFrame(countmap_matrix, ["̂̂Z=$l" for l in 1:my_K])
insertcols!(df, 1, :Z => ["Z=$k" for k in 1:my_K])
#### fit marginal parameters on true Z and on zhat
#### compare with fitted on Z_hat
left_bound = 0.1
middle_bound = 1.0
param_sigmainit = hcat(fill(log(5.0), my_D, 1), fill(0.0, my_D, 2 * 2))
    param_xiinit = hcat(fill(-log(1/0.2-1), my_D, 1), fill(0.0, my_D, 2 * 2))
    param_kappainit = hcat(fill(0.0, my_D, 1), fill(0.0, my_D, 2 * 2))
    
    di1, param_kappa1, param_sigma1, param_xi1, param_proba_lowrain = fit_dists(R,param_kappainit, param_sigmainit, param_xiinit, Matrix{AbstractFloat}(undef, D, 2 * 2 + 1), left_bound, middle_bound, 366, n2t,maxiters=50)
    
    param_kappainit = repeat(reshape(param_kappa1, 1, D, 2 * degree_of_P+1), my_K, 1, 1)
    param_sigmainit = repeat(reshape(param_sigma1, 1, D, 2 * degree_of_P+1), my_K, 1, 1)
    param_xiinit = repeat(reshape(param_xi1, 1, D, 2 * degree_of_P+1), my_K, 1, 1)

    diz, param_kappaz, param_sigmaz, param_xiz, param_proba_lowrainz = fit_dists(R, param_kappainit,param_sigmainit, param_xiinit, Array{AbstractFloat}(undef,my_K, D, 2 * 2 + 1), left_bound, middle_bound, 366, Z,n2t,maxiters=10000)
    dizhat, param_kappazhat, param_sigmazhat, param_xizhat, param_proba_lowrainzhat = fit_dists(R, param_kappainit,param_sigmainit, param_xiinit, Array{AbstractFloat}(undef,my_K, D, 2 * 2 + 1), left_bound, middle_bound, 366, Zhat,n2t,maxiters=10000)

#### Fit covariance parameters on true Z and on zhat

Pdryhat =zeros(D, length(n2t))
for s in 1:my_D
    for n in 1:my_N
        t = n2t[n]
        k = Z[n]
        h = Ymoins1[s, n]
        Pdryhay[s, n] = 1 - hmm_spa_fitted.B[k, t, s, h+1]
        end
    end

param0=[ 280.0, 2/3,  1.0, 0.5, 0.1, 0.5]
fittedz=fit_mple(XR,Mat_h,GneitingMatern,param0,quantile.(Normal(),Pdry),lower=[200., 0.1 ,0. ,0. ,0.00, 0.1],upper= [1000. ,10., 1., 1., 10., 10.],maxiter=100)
fittedzhat=fit_mple(XR,Mat_h,GneitingMatern,param0,quantile.(Normal(),Pdryhat),lower=[200., 0.1 ,0. ,0. ,0.00, 0.1],upper= [1000. ,10., 1., 1., 10., 10.],maxiter=100)
#each take 1h on my computer. I defaulted Julia as starting with 8 cores.