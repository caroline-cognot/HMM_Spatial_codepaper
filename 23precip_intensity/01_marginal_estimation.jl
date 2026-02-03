# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
skip_estim_marginals = true
skip_plots_marginals = false
skip_normalization =true
skip_resu_marginals = true
using JLD2
using CSV
using DataFrames
include("../23precip_intensity/EGPD_functions.jl")
include("../23precip_intensity/EGPD_class.jl")
using Distributions

z_hat = CSV.read("./00data/transformedECAD_zhat.csv", DataFrame, header=false)[:,1]

N=length(z_hat)
# z_hat=fill(1,N) #test if setting this to 1 gives same result. as previous no-class periodic estimation.
# z_hat[z_hat.==4] .=3
K=length(unique(z_hat))
Robs = Matrix(CSV.read("./00data/transformedECAD_Robs.csv", DataFrame, header=false)[:, :])'
D= size(Robs,2)
locsdata = CSV.read("./00data/transformedECAD_stations.csv", DataFrame, header=true)
locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
locsdata.LON = locations[:,1]
locsdata.LAT = locations[:,2]
using Dates
date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end
my_N = length(every_year)
import StochasticWeatherGenerators.dayofyear_Leap
n2t = dayofyear_Leap.(every_year)

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
    datafile = "./13PeriodicHMMSpatialBernoulli/res_real_data/"* "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"


        hmmspa = load(datafile)["hmm"]

    
end
cur_colors = get_color_palette(:auto, 100)

my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K)

T = 366
left_bound = 0.1
middle_bound = 1.0
degree_of_P = 2


y=Robs[:,:]
D=size(y,1)

 if !skip_estim_marginals    
    # start with parameters with no class.

    param_sigmainit = hcat(fill(log(5.0), D, 1), fill(0.0, D, 2 * degree_of_P))
    param_xiinit = hcat(fill(-log(1/0.2-1), D, 1), fill(0.0, D, 2 * degree_of_P))
    param_kappainit = hcat(fill(0.0, D, 1), fill(0.0, D, 2 * degree_of_P))
    
    di1, param_kappa1, param_sigma1, param_xi1, param_proba_lowrain = fit_dists(y,param_kappainit, param_sigmainit, param_xiinit, Matrix{AbstractFloat}(undef, D, 2 * degree_of_P + 1), left_bound, middle_bound, T, n2t,maxiters=5000)
    
    param_kappainit = repeat(reshape(param_kappa1, 1, D, 2 * degree_of_P+1), K, 1, 1)
    param_sigmainit = repeat(reshape(param_sigma1, 1, D, 2 * degree_of_P+1), K, 1, 1)
    param_xiinit = repeat(reshape(param_xi1, 1, D, 2 * degree_of_P+1), K, 1, 1)

    di, param_kappa, param_sigma, param_xi, param_proba_lowrain = fit_dists(y, param_kappainit,param_sigmainit, param_xiinit, Array{AbstractFloat}(undef,K, D, 2 * degree_of_P + 1), left_bound, middle_bound, T, z_hat,n2t,maxiters=10000)
    jldsave("./23precip_intensity/res_real_data/periodicEGPD_K"*string(K)*string(degree_of_P)*".jld2"; di=di)
end

@load "./23precip_intensity/res_real_data/periodicEGPD_K"*string(K)*string(degree_of_P)*".jld2" di

if !skip_plots_marginals
    # plot the parameters
    dists = JLD2.load("./22precip_intensity_marginal_withclass/res_real_data/EGPD_constant_K.jld2", "modellist")
    
    modellist_noK= JLD2.load("./22precip_intensity_marginal_withclass/res_real_data/EGPD_constant_noK.jld2", "modellist")
    modellist= JLD2.load( "./21precip_intensity_marginal_noclass/res_real_data/periodicEGPD"*string(degree_of_P)*".jld2" ,"di")
    

    using Plots
p = [plot() for i in 1:D]
for i in 1:D
    dlistnoK= modellist_noK[i]

   
    p1 = plot(title="\$ p_{k,s}^{(t)}\$",ylim=(-0.01,1))
    plot!(p1,[modellist[i][t].p for t in 1:T], color="black", label=ifelse(i == 6, "no class", :none))

    # hline!(p1, [dlistnoK.p],  color="black", label=ifelse(i == 9, "constant + no K", :none),linestyle=:dash)
    p2 = plot(title="\$\\sigma_{k,s}^{(t)}\$")
    # hline!(p2, [dlistnoK.tail_part.G.σ], color="black", label=ifelse(i == 9, "constant + no K", :none),linestyle=:dash)
    plot!(p2,[modellist[i][t].tail_part.G.σ for t in 1:T], color="black", label=ifelse(i == 889, "no class", :none))

    p3 = plot(title="\$\\xi_{k,s}^{(t)}\$")
    # hline!(p3, [dlistnoK.tail_part.G.ξ],  color="black", label=ifelse(i == 9, "constant +no K", :none),linestyle=:dash)
    plot!(p3,[modellist[i][t].tail_part.G.ξ for t in 1:T], color="black", label=ifelse(i == 999, "no class", :none))

    p4 = plot(title="\$\\kappa_{k,s}^{(t)}\$")
    # hline!(p4, [dlistnoK.tail_part.V.α], color="black", label=ifelse(i == 9, "constant +no K", :none),linestyle=:dash)
    plot!(p4,[modellist[i][t].tail_part.V.α for t in 1:T], color="black", label=ifelse(i == 9999, "no class", :none))

    for k in 1:K
     
        plot!(p1,[di[i][k,t].p for t in 1:T], c=my_palette(K)[k],label=ifelse(i == 6, "\$k= $k \$", :none))
        # plot!(p1, [dists[i].dists[k].p for t in 1:T],c=k, linestyle=:dash,label=ifelse(i == 9, "Cst + class $k", :none))

        plot!(p2,[di[i][k,t].tail_part.G.σ for t in 1:T],c=my_palette(K)[k],label=ifelse(i == 999, "\$k= $k \$", :none))
        # plot!(p2, [dists[i].dists[k].tail_part.G.σ for t in 1:T],c=k, linestyle=:dash,label=ifelse(i == 9, "Cst + class $k", :none))


        plot!(p3,[di[i][k,t].tail_part.G.ξ for t in 1:T], c=my_palette(K)[k],label=ifelse(i == 9999, "\$k= $k \$", :none))
        # plot!(p3, [dists[i].dists[k].tail_part.G.ξ for t in 1:T],c=k, linestyle=:dash,label=ifelse(i == 9, "Cst + class $k", :none))


        plot!(p4,[di[i][k,t].tail_part.V.α for t in 1:T], c=my_palette(K)[k],label=ifelse(i == 9999, "\$k= $k \$", :none))
        # plot!(p4, [dists[i].dists[k].tail_part.V.α for t in 1:T],c=k, linestyle=:dash,label=ifelse(i == 9, "Cst + class $k", :none))

    end
     

 
    p[i] = plot(p1, p2, p3, p4,suptitle=locsdata.STANAME[i], layout=(2,2),legendfont=10,titlefont=20)
end
select_plot = [6,1, 16, 14, 21, 3]

savefig(plot(p[select_plot]..., layout= (2,3), size=(2000, 1200),plot_titlevspan = 0.01), "./23precip_intensity/res_real_data/"*string(K)*"trydeg" * string(degree_of_P) * "sim.pdf")

end

using StatsPlots
using StatsBase
if !skip_normalization

    function to_uniform_continuous(values::Vector{Float64}, middle_bound::Float64; rng=Random.GLOBAL_RNG)
    #used to make the small precip values into a uniform variable.
        step = 0.1
        jittered = [v - step/2 + step * rand(rng) for v in values]  # add uniform jitter in each bin
         ecdf_func = StatsBase.ecdf(jittered)
         uniformized = [0.1 + (middle_bound - 0.1) * ecdf_func(v) for v in jittered]
     
         return uniformized              # keep inside bounds
    end

    #need the rain probability 
Uobs = similar(Robs')
Vobs = similar(Robs')

# transform the observed R>0 as uniform values
for i in 1:D
    data = Robs[i, :]
    data[(data.>0.0).*(data.<=middle_bound)] .= to_uniform_continuous((data[(data.>0.0).*(data.<=middle_bound)]),middle_bound)
    n_in_t_inK = [findall((n2t .== t).&& (z_hat .== k)) for k = 1:K, t = 1:T]
    uniformized= similar(data)
        for k in 1:K
            for t in 1:T

            uniformized[n_in_t_inK[k,t]] = cdf.(di[i][k,t],data[n_in_t_inK[k,t]])
            end
        end
    histogram(uniformized)
    uniformized[7110]
    uniformized=clamp!(uniformized,0.000001,0.999999)
    Uobs[:, i] .= uniformized
end

# add censored information  : transform to truncated normal, according to rain probability at time t depending on previous day info
Yobs = Matrix(CSV.read("./00data/transformedECAD_Yobs.csv", DataFrame, header=false)[:, :])'
Yprevious = fill(1,1,size(hmmspa,2)) 
Ymoins1= vcat(Yprevious, Yobs)[1:my_N,:]
for i in 1:D
    uniformized = Uobs[:,i]
    for n in 1:my_N  
        t=n2t[n]
        k=z_hat[n]
        h=Ymoins1[n,i]
        pdry= 1-hmmspa.B[k,t,i,h+1]
        Vobs[n,i] = quantile(Truncated(Normal(),quantile(Normal(),pdry),Inf),uniformized[n])
    end
   
end

#just to check the completely normal distribution - fill censored values with "could-be" values 
Vobsfilled = copy(Vobs)
for i in 1:D
    uniformized = Uobs[:,i]
    for n in 1:my_N  
        t=n2t[n]
        k=z_hat[n]
        h=Ymoins1[n,i]
        pdry= 1-hmmspa.B[k,t,i,h+1]
        if isnan(uniformized[n])
        Vobsfilled[n,i] = rand(Truncated(Normal(),-Inf,quantile(Normal(),pdry)))
        end
    end
   
end

using DataFramesMeta
using StatsPlots
df = DataFrame(Vobs, :auto)              # Auto col names :x1, :x2, ...
df_long = stack(df)                      # Columns: :variable, :value

# Facet plot: one histogram per column
plt = @df df_long histogram(
    :value,
    group=:variable,
    layout=(6, 7),   # 37 facets in ~6 rows × 7 columns
    legend=false
)
savefig(plt, "./23precip_intensity/res_real_data/EGPD_periodic_K_normal"*string(K)*string(degree_of_P)*".pdf")

df = DataFrame(Uobs, :auto)              # Auto col names :x1, :x2, ...
df_long = stack(df)                      # Columns: :variable, :value

# Facet plot: one histogram per column
plt = @df df_long histogram(
    :value,
    group=:variable,
    layout=(6, 7),   # 37 facets in ~6 rows × 7 columns
    legend=false
)
savefig(plt, "./23precip_intensity/res_real_data/EGPD_periodic_K_uniform"*string(K)*string(degree_of_P)*".pdf")

CSV.write("./23precip_intensity/res_real_data/EGPD_periodic_K_normalized"*string(K)*string(degree_of_P)*".csv", DataFrame(Vobs, :auto), header=false)

df = DataFrame(Vobsfilled, :auto)              # Auto col names :x1, :x2, ...
df_long = stack(df)                      # Columns: :variable, :value

# Facet plot: one histogram per column
plt = @df df_long histogram(
    :value,
    group=:variable,
    layout=(6, 7),   # 37 facets in ~6 rows × 7 columns
    legend=false
)
savefig(plt, "./23precip_intensity/res_real_data/EGPD_periodic_K_normalfilled"*string(K)*string(degree_of_P)*".pdf")


end

mean(Vobsfilled)
std(Vobsfilled)

