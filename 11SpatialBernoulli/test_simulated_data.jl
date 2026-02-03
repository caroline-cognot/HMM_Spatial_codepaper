# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
include("../11SpatialBernoulli/SpatialBernoulli.jl")



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
# heatmap(d.ΣU)

# generate data
n = 2000
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
#  @timed sol = fit_mle(init_d,y; m = 10*length(init_d), return_sol = false, order = my_order)

# pairwise maximisation is a lot better
tdist = maximum(my_distance) / 1
wp = 1.0 .* (my_distance .< tdist)
@timed sol1 = fit_mle(init_d, y, wp; order=my_order, m=100 * 2, return_sol=true,maxiters=2000)
@timed sol2 =  fit_mle(init_d, y, wp; order=my_order, m=30 * 2, return_sol=true, maxiters = 2000)
@timed sol3 = fit_mle_vfast(init_d, y, wp; order=my_order, return_sol=true, maxiters = 2000)
# end


# results :
# (value = (SpatialBernoulli{Float64, Float64, Float64, Vector{Float64}, Matrix{Float64}, Matrix{Float64}}(
# range: 0.3044696545714648
# sill: 1.0
# order: 0.5
# λ: [0.459, 0.5555, 0.783, 0.927, 0.0295, 0.7485, 0.731, 0.9775, 0.349, 0.672  …  0.24, 0.5005, 0.615, 0.6505, 0.903, 0.568, 0.0845, 0.3535, 0.1945, 0.112]
# h: [0.0 0.2 … 1.2806248474865698 1.4142135623730951; 0.2 0.0 … 1.1661903789690602 1.2806248474865698; … ; 1.2806248474865698 1.1661903789690602 … 0.0 0.19999999999999996; 1.4142135623730951 1.2806248474865698 … 0.19999999999999996 0.0]
# ΣU: [1.0 0.5184664743498574 … 0.014904625355195301 0.009611044199657649; 0.5184664743498574 1.0 … 0.021704510791816813 0.014904625355195301; … ; 0.014904625355195301 0.021704510791816813 … 1.0 0.5184664743498575; 0.009611044199657649 0.014904625355195301 … 0.5184664743498575 1.0]
# )
# , retcode: Success
# u: [-1.1891838533779866]
# Final objective value:     2.5765759720319216e6
# ), time = 3.008514067, bytes = 2896970080, gctime = 0.169236212, gcstats = Base.GC_Diff(2896970080, 340, 0, 101768159, 0, 3663, 169236212, 15, 0), lock_conflicts = 0, compile_time = 0.0, recompile_time = 0.0)

# (value = (SpatialBernoulli{Float64, Float64, Float64, Vector{Float64}, Matrix{Float64}, Matrix{Float64}}(
# range: 0.3055046402943637
# sill: 1.0
# order: 0.5
# λ: [0.459, 0.5555, 0.783, 0.927, 0.0295, 0.7485, 0.731, 0.9775, 0.349, 0.672  …  0.24, 0.5005, 0.615, 0.6505, 0.903, 0.568, 0.0845, 0.3535, 0.1945, 0.112]
# h: [0.0 0.2 … 1.2806248474865698 1.4142135623730951; 0.2 0.0 … 1.1661903789690602 1.2806248474865698; … ; 1.2806248474865698 1.1661903789690602 … 0.0 0.19999999999999996; 1.4142135623730951 1.2806248474865698 … 0.19999999999999996 0.0]
# ΣU: [1.0 0.5196215396327505 … 0.015118526642551753 0.009763477397893903; 0.5196215396327505 1.0 … 0.021987984474643327 0.015118526642551753; … ; 0.015118526642551753 0.021987984474643327 … 1.0 0.5196215396327506; 0.009763477397893903 0.015118526642551753 … 0.5196215396327506 1.0]
# )
# , retcode: Success
# u: [-1.185790311306255]
# Final objective value:     2.5765806196812e6
# ), time = 1.520585057, bytes = 992816432, gctime = 0.044798032, gcstats = Base.GC_Diff(992816432, 314, 0, 34254723, 0, 765, 44798032, 5, 0), lock_conflicts = 0, compile_time = 0.0, recompile_time = 0.0)

# (value = (SpatialBernoulli{Float64, Float64, Float64, Vector{Float64}, Matrix{Float64}, Matrix{Float64}}(
# range: 0.30530135017021937
# sill: 1.0
# order: 0.5
# λ: [0.459, 0.5555, 0.783, 0.927, 0.0295, 0.7485, 0.731, 0.9775, 0.349, 0.672  …  0.24, 0.5005, 0.615, 0.6505, 0.903, 0.568, 0.0845, 0.3535, 0.1945, 0.112]
# h: [0.0 0.2 … 1.2806248474865698 1.4142135623730951; 0.2 0.0 … 1.1661903789690602 1.2806248474865698; … ; 1.2806248474865698 1.1661903789690602 … 0.0 0.19999999999999996; 1.4142135623730951 1.2806248474865698 … 0.19999999999999996 0.0]
# ΣU: [1.0 0.5193950792524831 … 0.015076386579870836 0.009733429083686173; 0.5193950792524831 1.0 … 0.021932166664302293 0.015076386579870836; … ; 0.015076386579870836 0.021932166664302293 … 1.0 0.5193950792524832; 0.009733429083686173 0.015076386579870836 … 0.5193950792524832 1.0]
# )
# , retcode: Success
# u: [-1.186455956813418]
# Final objective value:     2.5765997563123205e6
# ), time = 0.42062003, bytes = 13979352, gctime = 0.0, gcstats = Base.GC_Diff(13979352, 222, 0, 347455, 0, 0, 0, 0, 0), lock_conflicts = 0, compile_time = 0.0, recompile_time = 0.0)