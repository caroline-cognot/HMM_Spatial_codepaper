# # Parametric bootstrap of the rain generator, one replicate per run
#
#     julia --project=. --threads=auto simulation_study.jl 17    # writes results/replicate_17.jld2
#
# On a SLURM array, pass `$SLURM_ARRAY_TASK_ID`; the replicate number is the random seed.
#
# A replicate draws hidden states z, occurrences Y, latent field X and amounts R over 1973-2023
# from the model fitted to the ECA&D stations, then refits:
# - the spatial HMM on Y, and its Viterbi path ẑ;
# - the rain margins on R, given z and given ẑ;
# - the covariance of the latent field, on X with its dry sites censored, and on the field rebuilt
#   from R with the refitted HMM, ẑ and the margins given ẑ, as on the real data.
# A refit numbers its states as it found them: match them to the true ones through z and ẑ before
# comparing coefficients.
cd(@__DIR__)
using Pkg
Pkg.activate(".")

println("Number of threads: ", Threads.nthreads())
println("Hostname: ", gethostname())
println("CPU: ", Sys.cpu_info()[1].model)
println("Number of CPU cores: ", Sys.CPU_THREADS)

using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve   # load the fitting routines of SmoothPeriodicStatsModels
using OptimizationOptimJL, LineSearches
using StochasticWeatherGenerators: dayofyear_Leap
using Distributions, LogExpFunctions, Random, Dates
using DelimitedFiles, JLD2

include("rain_margins.jl")
include("gaussian_field.jl")

# ## Settings, those of the fit to the real data unless noted

const INPUT = joinpath(@__DIR__, "..")   # the model fitted to the ECA&D data
const OUTPUT = joinpath(@__DIR__, "results")
isdir(OUTPUT) || mkpath(OUTPUT)

const K = 4                              # hidden states
const T = 366                            # period, the days of a leap year
const DATES = Date(1973):Day(1):Date(2023, 12, 31)
# occurrences
const ORDER = 1                          # days of local memory
const DEGREE_HMM = 1                     # trigonometric degree of the HMM coefficients
const R0 = 500.0                         # starting spatial range, km
const TDIST = 0.3                        # the range M-step uses pairs closer than TDIST × the largest distance
const QMC_M = 30                         # quasi-Monte-Carlo points per site in the spatial likelihood
const MAXITER_SPATIAL = 20               # EM iterations of the spatial HMM (100 on the real data)
# amounts
const DEGREE_RAIN = 2                    # trigonometric degree of the margins
const WET = 0.1                          # smallest wet-day amount, mm
const SMALL = 1.0                        # amounts in [WET, SMALL] are uniform, those above SMALL + EGPD
# latent field
const LAG = 10                           # days each simulated day is conditioned on
const COV_START = [280.0, 2 / 3, 1.0, 0.5, 0.1, 0.5]   # c, a, α, β, δ, ν
const COV_LOWER = [200.0, 0.1, 0.0, 0.0, 0.0, 0.1]
const COV_UPPER = [1000.0, 10.0, 1.0, 1.0, 10.0, 10.0]

"""
    load_truth() -> NamedTuple

The model fitted to the ECA&D data: `distances`, spatial `hmm`, rain `margins` and latent `covariance`.
"""
function load_truth()
    distances = readdlm(joinpath(INPUT, "transformedECAD_locsdistances.csv"), ',')
    logtots, θᴬ, θᴮ, θᴿ = load(joinpath(INPUT, "K4_resu_18627days_degree1_maxiter100_m1_R0500_QMCm30.jld2"),
        "logtots", "all_thetaA_iterations", "all_thetaB_iterations", "all_thetaR_iterations")
    i = argmax(logtots) + 1   # the iterates start with the starting point
    hmm = SmoothARPeriodicHMMSpatial(fill(1 / K, K), θᴬ[i], θᴮ[i], θᴿ[i], T, distances; QMC_m=QMC_M)
    p, κ, σ, ξ = load(joinpath(INPUT, "full_periodicEGPD_K42.jld2"),
        "param_proba_lowrain", "param_kappa", "param_sigma", "param_xi")
    # that fit wrote p and ξ as 1 / (1 + exp(η)), the logistic of -η
    margins = (p=-Float64.(p), κ, σ, ξ=-ξ)
    return (; distances, hmm, margins, covariance=load(joinpath(INPUT, "GMcov_withmarginalsdeg2.jld2"), "fitted"))
end

"""
    dry_probability(hmm::ARPeriodicHMMSpatial, z, Y, Y_past, n2t) -> Matrix

Probability that site `s` is dry on day `n`, in state `z[n]` and given its previous day.
"""
function dry_probability(hmm::ARPeriodicHMMSpatial, z, Y, Y_past, n2t)
    @assert size(hmm.B, 4) == 2 "written for one day of memory"
    previous = [Y_past[:, end] Y[:, 1:end-1]]
    return [1 - hmm.B[z[n], n2t[n], s, previous[s, n]+1] for s in axes(Y, 1), n in axes(Y, 2)]
end

"""
    simulate(truth, n2t) -> NamedTuple

States `z`, occurrences `Y` after `Y_past`, latent field `X`, amounts `R` and dry probabilities `pdry` drawn from `truth`.
"""
function simulate(truth, n2t)
    Y_past = rand(Bool, size(truth.distances, 1), ORDER)
    z, Y = rand(truth.hmm, n2t; Y_past, seq=true)
    pdry = dry_probability(ARPeriodicHMMSpatial(truth.hmm), z, Y, Y_past, n2t)
    X = simulate_latent_field(truth.covariance, truth.distances, Y, pdry; lag=LAG)
    R = rain_amounts(X, Y, z, n2t, pdry, rain_distributions(truth.margins, T; a=WET, b=SMALL))
    return (; z, Y, Y_past, X, R, pdry)
end

"""
    fit_occurrences(Y, Y_past, n2t, distances) -> (hmm, history)

The spatial HMM fitted to the occurrences `Y`, started from the HMM with independent sites.
"""
function fit_occurrences(Y, Y_past, n2t, distances)
    solver = LBFGS(linesearch=LineSearches.BackTracking())
    slices = fit_mle_all_slices(randARPeriodicHMM(K, T, size(Y, 1), ORDER; ξ=[1; zeros(K - 1)], ref_station=1),
        Y, Y_past; n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13])
    θᴬ, θᴮ = fit_θ!(slices, DEGREE_HMM)
    @time "Ind fit" independent, _ = fit_mle(SmoothARPeriodicHMM(slices.a, θᴬ, θᴮ, T), Y, Y_past;
        n2t, solver, ensemble=EnsembleThreads(), maxiter=10_000, tol=1e-3, robust=true)
    θᴿ = [j == 1 ? log(R0) : 0.0 for _ in 1:K, j in 1:2DEGREE_HMM+1]
    @time "Spatial HMM" start = SmoothARPeriodicHMMSpatial(fill(1 / K, K), parameters(independent).A, parameters(independent).B,
        θᴿ, T, distances; QMC_m=QMC_M)
    return fit_mle(start, Y, Y_past; n2t, solver, ensemble=EnsembleThreads(),
        maxiter=MAXITER_SPATIAL, tol=1e-3, maxiters_R=100, tdist=TDIST)
end

hmm_coefficients(m) = (A=copy(parameters(m).A), B=copy(parameters(m).B), R=copy(parameters(m).R))
covariance_parameters(m::GneitingMatern) = (; m.c, m.a, m.α, m.β, m.δ, m.ν)

"""
    run_replicate(replicate) -> String

Simulate the data set of seed `replicate`, refit the chain, and return the file the estimates are saved to.
"""
function run_replicate(replicate::Integer)
    file = joinpath(OUTPUT, "replicate_$replicate.jld2")
    if !isfile(file)
    Random.seed!(replicate)
    truth = load_truth()
    n2t = dayofyear_Leap.(DATES)
    (; z, Y, Y_past, X, R, pdry) = @time "simulation" simulate(truth, n2t)

    hmm, history = @time "spatial HMM" fit_occurrences(Y, Y_past, n2t, truth.distances)
    ẑ = viterbi(hmm, Y, Y_past; n2t)

    # A fit with the states pooled starts the fits by state. It starts from κ = 1, σ = 5 and ξ = 0.8,
    # as on the real data.
    D, n = size(Y, 1), 2DEGREE_RAIN + 1
    flat(η) = [j == 1 ? η : 0.0 for _ in 1:1, _ in 1:D, j in 1:n]
    pooled = @time "margins, pooled" fit_rain_margins(R, ones(Int, length(n2t)), n2t,
        (κ=flat(0.0), σ=flat(log(5.0)), ξ=flat(logit(0.8))); T, a=WET, b=SMALL, maxiters=5000)
    θ0 = map(θ -> repeat(θ, K), pooled[(:κ, :σ, :ξ)])
    margins_z = @time "margins | z" fit_rain_margins(R, z, n2t, θ0; T, a=WET, b=SMALL, maxiters=10_000)
    margins_ẑ = @time "margins | ẑ" fit_rain_margins(R, ẑ, n2t, θ0; T, a=WET, b=SMALL, maxiters=10_000)

    @time "dry probability" pdry_ẑ = dry_probability(ARPeriodicHMMSpatial(hmm), ẑ, Y, Y_past, n2t)

    X̂ = latent_field(R, Y, ẑ, n2t, pdry_ẑ, rain_distributions(margins_ẑ, T; a=WET, b=SMALL))
    X̂z = latent_field(R, Y, z, n2t, pdry, rain_distributions(margins_z, T; a=WET, b=SMALL))

    fit_field(X, pdry) = fit_covariance(X, quantile.(Normal(), pdry), truth.distances, COV_START;
        lower=COV_LOWER, upper=COV_UPPER, maxiters=100)
    covariance_Xz = @time "covariance | X true z " fit_field(X̂z, pdry)
    covariance_ẑ = @time "covariance | ẑ" fit_field(X̂, pdry_ẑ)

    jldsave(file; replicate, z, z_hat=ẑ,
        hmm=hmm_coefficients(hmm), hmm_logtots=history.logtots, hmm_best_iteration=history.best_iteration,
        margins_z, margins_z_hat=margins_ẑ,
        covariance_Xz=covariance_parameters(covariance_Xz), covariance_ẑ=covariance_parameters(covariance_ẑ),
        truth=(hmm=hmm_coefficients(truth.hmm), truth.margins, covariance=covariance_parameters(truth.covariance)))
    return file
    else
        println("Replicate $replicate already exists, skipping.")
        return file
    end
end

if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    @time "total" run_replicate(parse(Int, ENV["SLURM_ARRAY_TASK_ID"]))
end
