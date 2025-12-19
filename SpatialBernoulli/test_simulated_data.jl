include("../SpatialBernoulli/SpatialBernoulli.jl")



Random.seed!(0)


# define locations in the unit square
my_locations = vcat(([x y] for x in 0:0.2:1 for y in 0:0.2:1)...)
nlocs = length(my_locations[:, 1])


my_distance = [sqrt(sum(abs2, my_locations[i, :] - my_locations[j, :])) for i in axes(my_locations, 1), j in axes(my_locations, 1)]

# randomly generate a SpatialBernoulli
my_λ = rand(nlocs)# [0.5+ 0.1*i for i in 1:nlocs] 
my_range = 0.3
my_sill = 1.0
my_order = 1 / 2
d = SpatialBernoulli(my_range, my_sill, my_order, my_λ, my_distance)
heatmap(d.ΣU)

# generate data
n = 20
y=rand(d,n)

######## plot simulation ###############
using CairoMakie
yplot=y[:,2]
begin
fig = Figure()
ax = Axis(fig[1, 1])

scatter!(
    ax,
    my_locations[yplot .== 0, 1],
    my_locations[yplot .== 0, 2];
    color = :black,
    label = "y = 0"
)

scatter!(
    ax,
    my_locations[yplot .== 1, 1],
    my_locations[yplot .== 1, 2];
    color = :white,
    strokecolor = :black,
    label = "y = 1"
)

axislegend(ax)
fig
end


# fit a model using initial values
init_range = 0.5
init_order = 1.5
init_lambda = fill(0.4, nlocs)
init_d = SpatialBernoulli(init_range, 1.0, init_order, init_lambda, my_distance)
heatmap(init_d.ΣU)

# using full likelihood is a bad idea.
# @timed sol = fit_mle(init_d,y; m = 100*length(init_d), return_sol = false)
# #maximise without the order !
 @timed sol = fit_mle(init_d,y; m = 10*length(init_d), return_sol = false, order = my_order)

# pairwise maximisation is a lot better
tdist = maximum(my_distance) / 1
wp = 1.0 .* (my_distance .< tdist)
# @timed sol1 = fit_mle(init_d, y, wp; m=100 * 2, return_sol=false,maxiters=200)
@timed sol2 = fit_mle(init_d, y, wp; order=my_order, m=30 * 2, return_sol=false, maxiters = 2000)
# end


