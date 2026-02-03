# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
using JLD2
using CSV
using DataFrames
include("../23precip_intensity/EGPD_class.jl")

z_hat = CSV.read("./00data/transformedECAD_zhat.csv", DataFrame, header=false)[:,1]
N=length(z_hat)
Robs = Matrix(CSV.read("./00data/transformedECAD_Robs.csv", DataFrame, header=false)[:, :])
D= size(Robs,2)
locsdata = CSV.read("./00data/transformedECAD_stations.csv", DataFrame, header=true)
locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
locsdata.LON = locations[:,1]
locsdata.LAT = locations[:,2]

left=0.1
middle=1.0


########################### no class (for comparison) #########################################################################
#fit model for each station
dlist=Vector(undef,D)
for i in 1:D
    Ri = Robs[:, i]
    samples = Ri[Ri.>0]
    dlist[i] =  fit_mix(MixedUniformTail,samples,left=left,middle=middle)
end

jldsave("./22precip_intensity_marginal_withclass/res_real_data/EGPD_constant_noK.jld2"; modellist=dlist)


############################# with class ###########################################################
#fit model for each station
dlist=Vector(undef,D)
for i in 1:D
    Ri = Robs[:, i]
    samples = Ri[Ri.>0]
    zsamples = z_hat[Ri.>0]

    dlist[i] =  fit_mix_model(MixedUniformTail, samples, zsamples; left=left, middle=middle)
end

jldsave("./22precip_intensity_marginal_withclass/res_real_data/res_real_data/EGPD_constant_K.jld2"; modellist=dlist)


@load "./22precip_intensity_marginal_withclass/res_real_data/EGPD_constant_K.jld2" modellist
modellist_noK= JLD2.load("./22precip_intensity_marginal_withclass/res_real_data/EGPD_constant_noK.jld2", "modellist")

p = [plot() for i in 1:D]
for i in 1:D
    dlist1 = modellist[i]
    dlistnoK= modellist_noK[i]
    K=dlist1.K

    p1 = plot(title="Proba low rain",ylim=(-0.01,1))
    hline!(p1, [dlistnoK.p], c=K+1, label=ifelse(i == 9, "constant estimate", :none))
    p2 = plot(title="σ",ylim=(3,20))
    hline!(p2, [dlistnoK.tail_part.G.σ], c="black", label=ifelse(i == 9, "constant estimate", :none))

    p3 = plot(title="ξ",ylim=(0,1))
    hline!(p3, [dlistnoK.tail_part.G.ξ], c="black", label=ifelse(i == 9, "constant estimate", :none))

    p4 = plot(title="κ",ylim=(0,3))
    hline!(p4, [dlistnoK.tail_part.V.α], c="black", label=ifelse(i == 9, "constant estimate", :none))

    for k in 1:K
        hline!(p1, [dlist1.dists[k].p], c=k, label=ifelse(i == 9, "Class $k", :none))
        hline!(p2, [dlist1.dists[k].tail_part.G.σ], c=k, label=ifelse(i == 9, "Class $k", :none))
        hline!(p3, [dlist1.dists[k].tail_part.G.ξ], c=k, label=ifelse(i == 9, "Class $k", :none))
        hline!(p4, [dlist1.dists[k].tail_part.V.α], c=k, label=ifelse(i == 9, "Class $k", :none))

    end


    p[i] = plot(p1, p2, p3, p4, suptitle=locsdata.STANAME[i])
end
savefig(plot(p[[9, 16, 14, 21, 3, 6]]..., size=(2000, 2000)), "./22precip_intensity_marginal_withclass/res_real_data/EGPD_constant_noK_fitted.png")

