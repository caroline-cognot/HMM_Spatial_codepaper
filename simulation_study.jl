##############################################################################
# Strategy (single replicate, to be run many times on cluster):
#   1. Load parameters estimated on real data ("true" parameters)
#   2. Simulate one replicate (Z, Y, R) from the true model
#   3. Fit the HMM on (Y) → get Ẑ via Viterbi
#   4. Fit intensity margins twice: once with Z, once with Ẑ
#   5. Fit covariance parameters twice: once with Z, once with Ẑ
#   6. Save all results for comparison
##############################################################################
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using SmoothPeriodicStatsModels
using Distributions
using Optimization, OptimizationOptimJL, LineSearches
using ForwardDiff
using Random
using StochasticWeatherGenerators
using CSV, DataFrames, Dates, JLD2
using Base.Threads
number=Threads.nthreads()

include("./23precip_intensity/EGPD_functions.jl")
include("./23precip_intensity/EGPD_class.jl")
include("./23precip_intensity/covariance_functions.jl")

##############################################################################
# 0. Global settings
##############################################################################

Random.seed!(2024)  # for reproducibility; change per replicate on cluster

const my_K  = 4
const my_T  = 366
const my_autoregressive_order = 1
const degree_of_P = 1
const tdist   = 0.3
const QMC_m   = 30
const left_bound   = 0.1
const middle_bound = 1.0

##############################################################################
# 1. Load spatial metadata
##############################################################################

my_distance = Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv", DataFrame, header=false))
my_locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
my_D = size(my_locations, 1)
locsdata = CSV.read("./00data/transformedECAD_stations.csv", DataFrame, header=true)
locsdata.LON = my_locations[:, 1]
locsdata.LAT = my_locations[:, 2]
coordll  = [locsdata.LON locsdata.LAT]


##############################################################################
# 2. Define time axis
##############################################################################

date_start = Date(1973)
date_end   = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end
my_N       = length(every_year)
n2t        = dayofyear_Leap.(every_year)

##############################################################################
# 3. Load true parameters (estimated on real data)
##############################################################################

hmmspadata = load("./13PeriodicHMMSpatialBernoulli/res_real_data/parameters/K4_resu_18627days_degree1_maxiter100_m1_R0500_QMCm30.jld2")

iter_true = argmax(hmmspadata["logtots"])
thetaA_true = hmmspadata["all_thetaA_iterations"][iter_true + 1]
thetaB_true = hmmspadata["all_thetaB_iterations"][iter_true + 1]
thetaR_true = hmmspadata["all_thetaR_iterations"][iter_true + 1]
a_true      = hmmspadata["hmm"].a

modeltrue = Trig2ARPeriodicHMMSpatial(a_true, thetaA_true, thetaB_true, thetaR_true, my_T, my_distance)

margins_true  = load("./23precip_intensity/res_real_data/full_periodicEGPD_K42.jld2")
covmodel_true = load("./23precip_intensity/res_real_data/GMcov_withmarginalsdeg2.jld2")["fitted"]

##############################################################################
# 4. Simulate one replicate (Z, Y, R) from the true model
##############################################################################
for replicate in 1:2
seed = 2026 + replicate
Random.seed!(seed)

Z, Yall = rand(modeltrue, my_N, seq=true)

# Y_past: random initial condition 
Y_past = BitMatrix(rand(Bool, my_D, my_autoregressive_order))
Y = Yall  # (my_N × my_D), aligned with n2t

# Lag matrix for R simulation (prepend one dry day as initial condition)
Yprevious = BitMatrix(fill(1, my_D, 1))
Ymoins1   = hcat(Yprevious, Yall')  # (my_D × (my_N + 1)), Ymoins1[:, n] is Y at n-1

# Dry probabilities under true model
Pdry_true = zeros(my_D, my_N)
for n in 1:my_N
    t = n2t[n]
    k = Z[n]
    for s in 1:my_D
        h = Ymoins1[s, n]
        Pdry_true[s, n] = 1 - modeltrue.B[k, t, s, h + 1]
    end
end

# Simulate latent Gaussian field and rainfall intensities
gf       = GaussianField(coordll, covmodel_true)
XR       = simulate_iterative(gf, 1:my_N, 10, Yall', Pdry_true)

R = zeros(my_D, my_N)
for n in 1:my_N
    t = n2t[n]
    k = Z[n]
    for s in 1:my_D
        h      = Ymoins1[s, n]
        pdry   = Pdry_true[s, n]
        thresh = quantile(Normal(), pdry)
        R[s, n] = ifelse(
            XR[s, n] < thresh,
            0.0,
            quantile(margins_true["di"][s][k, t], cdf(Truncated(Normal(), thresh, Inf), XR[s, n]))
        )
    end
end

##############################################################################
# 5. Fit HMM on simulated Y → get Ẑ
##############################################################################

# 5a. Independent HMM initialisation (slice + Baum-Welch)
hmm_random = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order;
                                ξ=[1; zeros(my_K - 1)], ref_station=1)

@time "Slice init" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past';
    n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1)

    
θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, 1)

@time "Baum-Welch" hmm_ind, θq_fit, θy_fit, hist_ind, _, _ = fit_mle(
    hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past';
    maxiter=100, robust=true, display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)

# 5b. Spatial HMM (EM with LBFGS for R update)
θᴿ_init        = zeros(my_K, 2 * degree_of_P + 1)
θᴿ_init[:, 1] .= log(500.0)

hmm_spa_init = Trig2ARPeriodicHMMSpatial(
    fill(1 / my_K, my_K), θq_fit, θy_fit, θᴿ_init, my_T, my_distance)

solver = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking())

@time "Spatial EM" history_spa, all_θᴬ, all_θᴮ, all_θᴿ = fit_mle!(
    hmm_spa_init, θq_fit, θy_fit, θᴿ_init, Y, Y_past';
    solver=solver, n2t=n2t, maxiter=20, tol=1e-3,
    maxiters_R=100, display=:iter, tdist=tdist, QMC_m=QMC_m)

iter_best  = argmax(history_spa["logtots"])
thetaA_fit = all_θᴬ[iter_best + 1]
thetaB_fit = all_θᴮ[iter_best + 1]
thetaR_fit = all_θᴿ[iter_best + 1]

hmm_spa_fit = Trig2ARPeriodicHMMSpatial(
    fill(1 / my_K, my_K), thetaA_fit, thetaB_fit, thetaR_fit, my_T, my_distance)

# 5c. Viterbi decoding
Zhat = viterbi(hmm_spa_fit, Y, Y_past'; n2t=n2t, QMC_m=QMC_m)

##############################################################################
# 6. Compare Z and Ẑ
##############################################################################

contingency = [sum((Z .== k) .& (Zhat .== l)) for k in 1:my_K, l in 1:my_K]
accuracy    = mean(Z .== Zhat)

##############################################################################
# 7. Fit intensity margins with Z and with Ẑ
##############################################################################

param_kappainit = hcat(fill(0.0,    my_D, 1), fill(0.0, my_D, 2 * 2))
param_sigmainit = hcat(fill(log(5.0), my_D, 1), fill(0.0, my_D, 2 * 2))
param_xiinit    = hcat(fill(-log(1 / 0.2 - 1), my_D, 1), fill(0.0, my_D, 2 * 2))
di1, param_kappa1, param_sigma1, param_xi1, param_proba_lowrain = fit_dists(R,param_kappainit, param_sigmainit, param_xiinit, Matrix{AbstractFloat}(undef, my_D, 2 * 2 + 1), left_bound, middle_bound, 366, n2t,maxiters=50)
    
    param_kappainit = repeat(reshape(param_kappa1, 1, my_D, 2 * 2+1), my_K, 1, 1)
    param_sigmainit = repeat(reshape(param_sigma1, 1, my_D, 2 * 2+1), my_K, 1, 1)
    param_xiinit = repeat(reshape(param_xi1, 1, my_D, 2 * 2+1), my_K, 1, 1)

# With true Z
@time "Margins (Z)" diz, param_kappaz, param_sigmaz, param_xiz, param_proba_lowrainz =
    fit_dists(R, param_kappainit,param_sigmainit, param_xiinit, Array{AbstractFloat}(undef,my_K, my_D, 2 * 2 + 1), left_bound, middle_bound, 366, Z,n2t,maxiters=10000)


# With Viterbi Ẑ
@time "Margins (Ẑ)" dizhat, param_kappazhat, param_sigmazhat, param_xizhat, param_proba_lowrainzhat =
        fit_dists(R, param_kappainit,param_sigmainit, param_xiinit, Array{AbstractFloat}(undef,my_K, my_D, 2 * 2 + 1), left_bound, middle_bound, 366, Zhat,n2t,maxiters=10000)


##############################################################################
# 8. Fit covariance parameters with Z and with Ẑ
##############################################################################

# Dry probabilities under fitted model, using Ẑ
Pdry_zhat = zeros(my_D, my_N)
for n in 1:my_N
    t = n2t[n]
    k = Zhat[n]
    for s in 1:my_D
        h = Ymoins1[s, n]
        Pdry_zhat[s, n] = 1 - hmm_spa_fit.B[k, t, s, h + 1]
    end
end

param0_cov = [280.0, 2/3, 1.0, 0.5, 0.1, 0.5]
lower_cov  = [200.0, 0.1, 0.0, 0.0, 0.00, 0.1]
upper_cov  = [1000.0, 10.0, 1.0, 1.0, 10.0, 10.0]

# With true Z (Pdry_true)
@time "Covariance (Z)" fitted_cov_z = fit_mple(XR, my_distance, GneitingMatern, param0_cov,
    quantile.(Normal(), Pdry_true); lower=lower_cov, upper=upper_cov, maxiter=100)

# With Viterbi Ẑ (Pdry_zhat)
@time "Covariance (Ẑ)" fitted_cov_zhat = fit_mple(XR, my_distance, GneitingMatern, param0_cov,
    quantile.(Normal(), Pdry_zhat); lower=lower_cov, upper=upper_cov, maxiter=100)

##############################################################################
# 9. Save results
##############################################################################

save("./simulation_study/replicate"*string(replicate)*".jld2",
    # True parameters
    "thetaA_true",   thetaA_true,
    "thetaB_true",   thetaB_true,
    "thetaR_true",   thetaR_true,
    # Fitted HMM parameters
    "thetaA_fit",    thetaA_fit,
    "thetaB_fit",    thetaB_fit,
    "thetaR_fit",    thetaR_fit,
    # State classification
    "Z",             Z,
    "Zhat",          Zhat,
    "contingency",   contingency,
    "accuracy",      accuracy,
    # Intensity margins with Z
    "param_kappaz",          param_kappaz,
    "param_sigmaz",          param_sigmaz,
    "param_xiz",             param_xiz,
    "param_proba_lowrainz",  param_proba_lowrainz,
    # Intensity margins with Ẑ
    "param_kappazhat",          param_kappazhat,
    "param_sigmazhat",          param_sigmazhat,
    "param_xizhat",             param_xizhat,
    "param_proba_lowrainzhat",  param_proba_lowrainzhat,
    # Covariance parameters
    "fitted_cov_z",    fitted_cov_z,
    "fitted_cov_zhat", fitted_cov_zhat,
)

println("Done. Accuracy Z vs Ẑ: ", round(accuracy, digits=3))
println("Contingency table (rows=Z, cols=Ẑ):")
display(contingency)
end

