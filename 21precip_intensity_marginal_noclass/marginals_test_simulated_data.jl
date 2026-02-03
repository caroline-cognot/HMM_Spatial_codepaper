# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
include("../23precip_intensity/EGPD_functions.jl")

# test on simulated data with no zeroes
using Random
Random.seed!(1234)  
D = 2
T = 365
degree_of_P = 2
size_P = 2 * degree_of_P + 1
param_sigma0 = (rand(D, size_P) .- 1 / 2)
param_xi0 =  (rand(D, size_P))
param_kappa0 =(rand(D, size_P) .- 1 / 2)
middle_bound = 1.0
left_bound = 0.1
param_proba_lowrain0 = (rand(D, size_P) .- 1 / 2)
param_proba_lowrain0[:, 1] .= 0.9
param_sigma0[:, 1] .+= log(5)
param_xi0[:, 1] .-= log(1/0.2-1)
param_kappa0[:, 1] .= 0.

dists = Trig2myEGPDPeriodicDistribution(param_sigma0, param_xi0, param_kappa0, param_proba_lowrain0, left_bound, middle_bound, T)


using PeriodicHiddenMarkovModels
N = 50 * T
n2t = n_to_t(N, T)

y = my_rand(dists, n2t)
histogram(y[1, :], xlim=(0, 50), bins=0:0.5:50);



param_sigmainit = hcat(fill(log(5.0), D, 1), fill(0.0, D, 2 * degree_of_P))
param_xiinit = hcat(fill(-log(1/0.2-1), D, 1), fill(0.0, D, 2 * degree_of_P))
param_kappainit = hcat(fill(0, D, 1), fill(0.0, D, 2 * degree_of_P))

@time di,param_kappa,param_sigma,param_xi,param_proba_lowrain = fit_dists(y,param_kappainit, param_sigmainit, param_xiinit, Matrix{AbstractFloat}(undef, D, size_P), left_bound, middle_bound, T, n2t,maxiters=5000);
# param_sigmainit = copy(param_sigma)
# param_xiinit   = copy(param_xi)
# param_kappainit = copy(param_kappa)

# di2,param_kappa,param_sigma,param_xi,param_proba_lowrain = fit_dists(y, param_kappainit,param_sigmainit, param_xiinit, Matrix{AbstractFloat}(undef, D, size_P), left_bound, middle_bound, T, n2t,maxiters=5000);
# di3,param_kappa,param_sigma,param_xi,param_proba_lowrain = fit_dists(y, param_sigma, param_xi, param_kappa, Matrix{AbstractFloat}(undef, D, size_P), left_bound, middle_bound, T, n2t);
# di4,param_kappa,param_sigma,param_xi,param_proba_lowrain = fit_dists(y, param_sigma, param_xi, param_kappa, Matrix{AbstractFloat}(undef, D, size_P), left_bound, middle_bound, T, n2t);
# di5,param_kappa,param_sigma,param_xi,param_proba_lowrain = fit_dists(y, param_sigma, param_xi, param_kappa, Matrix{AbstractFloat}(undef, D, size_P), left_bound, middle_bound, T, n2t);
using Plots
p = [plot() for i in 1:D]
for i in 1:D

    p1 = plot([di[i][t].p for t in 1:T], label="1st run fitted proba low rain t")
    #  plot!(p1,[di2[i][t].p for t in 1:T], label="2nd run")
    # plot!(p1,[di3[i][t].p for t in 1:T], label="3rd run")

    plot!(p1, [dists[i][t].p for t in 1:T], label="true low rain t")

    p2 = plot([di[i][t].tail_part.G.σ for t in 1:T], label="fitted sigma(t)")
    plot!(p2, [dists[i][t].tail_part.G.σ for t in 1:T], label="original sigma(t)")
    # plot!(p2,[di2[i][t].tail_part.G.σ  for t in 1:T], label="2nd run")
    # plot!(p2,[di3[i][t].tail_part.G.σ  for t in 1:T], label="3rd run")
    # plot!(p2,[di4[i][t].tail_part.G.σ  for t in 1:T], label="4th run")
    # plot!(p2,[di5[i][t].tail_part.G.σ  for t in 1:T], label="5th run")
    # horizontal line at y = 0.5
    p3 = plot([di[i][t].tail_part.G.ξ for t in 1:T], label="fitted xi(t)")
    plot!(p3, [dists[i][t].tail_part.G.ξ for t in 1:T], label="original xi(t)")
    #  plot!(p3,[di2[i][t].tail_part.G.ξ  for t in 1:T], label="2nd run")
    # plot!(p3,[di3[i][t].tail_part.G.ξ  for t in 1:T], label="3rd run")
    # plot!(p3,[di4[i][t].tail_part.G.ξ  for t in 1:T], label="4th run")
    # plot!(p3,[di5[i][t].tail_part.G.ξ  for t in 1:T], label="5th run")

    p4 = plot([di[i][t].tail_part.V.α for t in 1:T], label="fitted kappa(t)")
    plot!(p4, [dists[i][t].tail_part.V.α for t in 1:T], label="original kappa(t)")
    # plot!(p4,[di2[i][t].tail_part.V.α  for t in 1:T], label="2nd run")
    # plot!(p4,[di3[i][t].tail_part.V.α  for t in 1:T], label="3rd run")
    # plot!(p4,[di4[i][t].tail_part.V.α  for t in 1:T], label="4th run")
    # plot!(p4,[di5[i][t].tail_part.V.α  for t in 1:T], label="5th run")

    p[i] = plot(p1, p2, p3, p4, suptitle="location $i")
end
display(plot(p..., size=(1000, 1000)))
savefig(plot(p..., size=(1000, 1000)), "./21precip_intensity_marginal_noclass/res_sim_data/trydeg" * string(degree_of_P) * "sim.png")



using StatsPlots  # for density()

ytrue = my_rand(dists, vcat(n2t, n2t, n2t, n2t, n2t,n2t,n2t,n2t,n2t))
ysim = my_rand(di, (vcat(n2t, n2t, n2t, n2t, n2t,n2t,n2t,n2t,n2t)))

begin
    datet = 1:100
    bins = 0:0.5:50

    # Extract data subsets
    data_sim1 = ysim[1, vcat([findall(n2t .== t) for t = 1:T][datet]...)]
    data_true1 = ytrue[1, vcat([findall(n2t .== t) for t = 1:T][datet]...)]

    p1 = histogram(data_sim1, bins=bins, label="sim fitted model", alpha=0.5,
                   title="Histogram r date 1 to 100", normalize=true)
    histogram!(p1, data_true1, bins=bins, label="sim true model", alpha=0.5, normalize=true)



    datet = 200:300
    data_sim2 = ysim[1, vcat([findall(n2t .== t) for t = 1:T][datet]...)]
    data_true2 = ytrue[1, vcat([findall(n2t .== t) for t = 1:T][datet]...)]

    p2 = histogram(data_sim2, bins=bins, label="sim fitted model", alpha=0.5,
                   title="Histogram r date 200 to 300", normalize=true)
    histogram!(p2, data_true2, bins=bins, label="sim true model", alpha=0.5, normalize=true)

    
  

    p = plot(p1, p2,  suptitle="Period T = $T",xlims=(0,40),layout=(2,1))
end

savefig(plot(p, size=(1000, 1000)), "./21precip_intensity_marginal_noclass/res_sim_data/trydeg" * string(degree_of_P) * "histsim.png")

