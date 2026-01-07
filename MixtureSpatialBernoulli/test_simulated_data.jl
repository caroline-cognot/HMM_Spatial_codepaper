# Spatial Bernoulli model
include("../11SpatialBernoulli/SpatialBernoulli.jl")

# Mixture fitting 
include("../12MixtureSpatialBernoulli/ExpectationMaximization_source.jl")
include("../12MixtureSpatialBernoulli/estimation_functions.jl")
include("../11SpatialBernoulli/plot_validation.jl")


# try the code before starting
nlocs = 37
my_locations = rand(nlocs, 2)
my_distance = [sqrt(sum(abs2, my_locations[i, :] - my_locations[j, :])) for i in axes(my_locations, 1), j in axes(my_locations, 1)]
my_sill = 1.0
my_λ1 = fill(0.5, nlocs) #wet state
my_range1 = 0.5 #long range
my_order1 = 1 / 2

my_λ2 = fill(0.2, nlocs) #dry state
my_range2 = 0.2 # low range (almost uncorrelated)
my_order2 = 1 / 2

D₁ = SpatialBernoulli(my_range1, my_sill, my_order1, my_λ1, my_distance)
D₂ = SpatialBernoulli(my_range2, my_sill, my_order2, my_λ2, my_distance)
mix_real = MixtureModel([D₁, D₂], [1 / 4, 3 / 4])

Ymix = rand(mix_real, 30)
D₁guess = SpatialBernoulli(my_range1, my_sill, my_order1, my_λ1, my_distance)
D₂guess = SpatialBernoulli(my_range2, my_sill, my_order2, my_λ2, my_distance)
mix = MixtureModel([D₁guess, D₂guess], [1 / 4, 3 / 4])

y = Ymix


PlotSim(Ymix[:, 1:9], my_locations)


# Initial parameters
α = copy(probs(mix))
dists = copy(components(mix))


# Estimation
@time dfit = fit_mle!(α, dists, y, PairwiseClassicEM(); order="fixed", maxiter=20, QMC_m=30, factor_dmax=1 / 3, maxiter_m=10)
dists
plot(dfit["logtots"])
α


Nb = 100
nlocs = length(my_locations[:, 1])
begin
    Ys = zeros(Bool, nlocs, 30, Nb)
    @time "Simulations  Y" for i in 1:Nb
        Ys[:, :, i] = rand(MixtureModel(dists, α), 30)
    end

end

using Plots
p = compare_ROR_histogram(Ymix, Ys)

# try the code before, starting with 4 mix
n=100
nlocs = 37
my_locations = rand(nlocs, 2)
my_distance = [sqrt(sum(abs2, my_locations[i, :] - my_locations[j, :])) for i in axes(my_locations, 1), j in axes(my_locations, 1)]

D₁ = SpatialBernoulli(0.5, 1.0, 1 / 2, fill(0.5, nlocs), my_distance)
D₂ = SpatialBernoulli(0.2, 1.0, 1 / 2, fill(0.2, nlocs), my_distance)
D₃ = SpatialBernoulli(0.2, 1.0, 1 / 2, fill(0.5, nlocs), my_distance)
D₄ = SpatialBernoulli(0.8, 1.0, 1 / 2, fill(0.2, nlocs), my_distance)

mix_real = MixtureModel([D₁, D₂, D₃, D₄], [1 / 4, 1 / 4, 1 / 4, 1 / 4])

Ymix = rand(mix_real, n)

D₁g = SpatialBernoulli(0.5, 1.0, 1 / 2, fill(0.5, nlocs), my_distance)
D₂g = SpatialBernoulli(0.2, 1.0, 1 / 2, fill(0.2, nlocs), my_distance)
D₃g = SpatialBernoulli(0.2, 1.0, 1 / 2, fill(0.5, nlocs), my_distance)
D₄g = SpatialBernoulli(0.85, 1.0, 1 / 2, fill(0.2, nlocs), my_distance)

mix = MixtureModel([D₁g, D₂g, D₃g, D₄g], [1 / 4, 1 / 4, 1 / 4, 1 / 4])


y = Ymix
PlotSim(Ymix[:, 1:9], my_locations)

# Initial parameters
α = copy(probs(mix))
dists = copy(components(mix))
@time dfit = fit_mle!(α, dists, y, PairwiseClassicEM(); order="fixed", maxiter=15, QMC_m=30, factor_dmax=1 / 3, maxiter_m=10)
dists
plot(dfit["logtots"])
α


Nb = 100
nlocs = length(my_locations[:, 1])
begin
    Ys = zeros(Bool, nlocs, n, Nb)
    @time "Simulations  Y" for i in 1:Nb
        Ys[:, :, i] = rand(MixtureModel(dists, α), n)
    end

end

p = compare_ROR_histogram(Ymix, Ys)
