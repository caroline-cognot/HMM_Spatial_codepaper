# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
# study ROR
using Plots
function ROR(y::AbstractMatrix)
    [mean(y[:, t]) for t in 1:length(y[1, :])]
end

function ROR(y::Array{Bool,3})
    [[mean(y[:, t, i]) for t in 1:length(y[1, :, 1])] for i in 1:length(y[1, 1, :])]
end



using KernelDensity

function density_vectors(y::Vector{Vector{Float64}})
    my_range = 0:0.001:1
    dy = Vector(undef, length(y))
    # enveloppe of all densities
    for i in 1:length(y)
        dy[i] = kde(y[i], my_range).density
    end
    return dy, my_range
end



function enveloppe_minmax(dy, y)
    mindy = [min([dy[i][x] for i in 1:length(dy)]...) for x in 1:length(y)]
    maxdy = [max([dy[i][x] for i in 1:length(dy)]...) for x in 1:length(y)]
    medy = [quantile([dy[i][x] for i in 1:length(dy)], 0.5) for x in 1:length(y)]

    return mindy, maxdy, medy, y

end

function enveloppe_90(dy, y)
    mindy = [quantile([dy[i][x] for i in 1:length(dy)], 0.05) for x in 1:length(y)]
    maxdy = [quantile([dy[i][x] for i in 1:length(dy)], 0.95) for x in 1:length(y)]
    medy = [quantile([dy[i][x] for i in 1:length(dy)], 0.5) for x in 1:length(y)]

    return mindy, maxdy, medy, y

end


function compare_ROR_density(yobs, ysim)


    RORobs = ROR(yobs)
    RORsim = ROR(ysim)
    dy, y = density_vectors(RORsim)

    mindy, maxdy, medy, y = enveloppe_minmax(dy, y)
    begin
        p = Plots.plot(y, medy; ribbon=(medy - mindy, -medy + maxdy), label="sim")
        Plots.plot!(p, 0:0.001:1, kde(RORobs, 0:0.001:1).density, label="obs", title="Density estimation", xlabel="ROR = 1/dₛ∑ₛYₛ ∈ [0,1]")
    end

end







function hist_ROR(y)
    ror = ROR(y)
    d= length(y[:,1])
    xax = 0:(1/d):1
    vec= [mean(ror.==xax[k]) for k in 1:(d+1)]
    return xax,vec
end


function hist_vectors(y::Vector{Vector{Float64}},xax)
    dy = Vector(undef, length(y))
    # enveloppe of all densities
    for i in 1:length(y)
        dy[i] = [mean(y[i].==xax[k]) for k in 1:length(xax)]

    end
    return xax,dy
end


function compare_ROR_histogram(yobs, ysim)

    xax,hRORobs = hist_ROR(yobs)
    xax2,hRORsim = hist_vectors(ROR(ysim),xax)

    length(hRORsim)
    length(xax)
    mindy, maxdy, medy, y = enveloppe_minmax(hRORsim, xax)
    begin
        p =  Plots.scatter(y, medy,c=1; ribbon=(medy - mindy, -medy + maxdy), label="sim")
        Plots.plot!(p,y, medy,c=1; ribbon=(medy - mindy, -medy + maxdy),label="")
        Plots.scatter(p, xax, hRORobs, label="obs", xlabel="ROR = 1/dₛ∑ₛYₛ ∈ [0,1]")
    end

end



function compare_ROR_histogram(yobs, ysim,ysim2;label1 = "sim 1",label2="sim 2")

    xax,hRORobs = hist_ROR(yobs)
    xax2,hRORsim = hist_vectors(ROR(ysim),xax)
    xax22,hRORsim2 = hist_vectors(ROR(ysim2),xax)


    mindy, maxdy, medy, y = enveloppe_minmax(hRORsim, xax)
    mindy2, maxdy2, medy2, y2 = enveloppe_minmax(hRORsim2, xax)

    begin
        p =  Plots.scatter(y, medy,c=1; ribbon=(medy - mindy, -medy + maxdy), label=label1)
        Plots.plot!(p,y, medy,c=1; ribbon=(medy - mindy, -medy + maxdy),label="")
        Plots.scatter!(p,y2, medy2,c=2; ribbon=(medy2 - mindy2, -medy2 + maxdy2), label=label2)
        Plots.plot!(p,y2, medy2,c=2; ribbon=(medy2 - mindy2, -medy2 + maxdy2),label="")
        Plots.scatter(p, xax, hRORobs, label="Observations", xlabel="ROR")
    end

end



function compare_ROR_histogram90(yobs, ysim,ysim2;show=false)

    xax,hRORobs = hist_ROR(yobs)
    xax2,hRORsim = hist_vectors(ROR(ysim),xax)
    xax22,hRORsim2 = hist_vectors(ROR(ysim2),xax)


    mindy, maxdy, medy, y = enveloppe_90(hRORsim, xax)
    mindy2, maxdy2, medy2, y2 = enveloppe_90(hRORsim2, xax)

    begin
        p =  Plots.scatter(y, medy,c=1; ribbon=(medy - mindy, -medy + maxdy), label=ifelse(show,"sim - no hidden states",""))
        Plots.plot!(p,y, medy,c=1; ribbon=(medy - mindy, -medy + maxdy),label="")
        Plots.scatter!(p,y2, medy2,c=2; ribbon=(medy2 - mindy2, -medy2 + maxdy2), label=ifelse(show,"sim - with hidden states",""))
        Plots.plot!(p,y2, medy2,c=2; ribbon=(medy2 - mindy2, -medy2 + maxdy2),label="")
        Plots.scatter(p, xax, hRORobs, label=ifelse(show,"obs",""), xlabel="ROR",ylabel="frequency")
    end

end


# make ROR histogram (and plot it) in observation according to a given hidden state.

function hist_ROR(y, z; K=length(unique(z)))
    ror = fill(NaN, length(y[1, :]), K)
    roro = ROR(y)
    for t in 1:length(y[1, :])
        ror[t, z[t]] = roro[t]
    end
    d = length(y[:, 1])
    xax = 0:(1/d):1
    vec = [[mean(filter(!isnan, ror[:, k] .== xax[id])) for id in 1:(d+1)] for k in 1:K]
    return xax, vec
end

function Plot_hist_ROR(xax, vec::Vector{Vector{Float64}})
    p =  Plots.plot(title="ROR by class")
    K = length(vec)
    for k in 1:K
        Plots.scatter!(p, xax, vec[k], label=k)
        Plots.plot!(p, xax, vec[k], label="")
    end
    p
end


function PlotSim(y::AbstractMatrix, locations)
    n = length(y[1, :])
    plotss = [ Plots.plot(locations[:, 1][y[:, i].==0], locations[:, 2][y[:, i].==0], seriestype=:scatter, color=:black, label="y = 0") for i in 1:n]
    [ Plots.plot!(plotss[i], locations[:, 1][y[:, i].==1], locations[:, 2][y[:, i].==1], seriestype=:scatter, color=:white, markerstrokecolor=:black, label="y = 1") for i in 1:n]
    Plots.plot(plotss...)
end

using StatsBase
function compare_ROR_autocorr(yobs, ysim;maxlag=20)
    rorobs = ROR(yobs')
    rorsim = ROR(ysim)
    acf_obs = autocor(rorobs, 0:maxlag)  # Compute ACF for lags 0 to maxlag
    acf_sim =[autocor(rorsim[i]) for i in 1:length(rorsim)]
    miniacf = [minimum([acf_sim[i][ilag] for i in 1:length(rorsim) ]) for ilag in 1:maxlag+1]
    maxiacf = [maximum([acf_sim[i][ilag] for i in 1:length(rorsim) ]) for ilag in 1:maxlag+1]
    moyacf = [mean([acf_sim[i][ilag] for i in 1:length(rorsim) ]) for ilag in 1:maxlag+1]

    # Plot ACF with bars
    p =  Plots.scatter(0:maxlag, moyacf,c=1; ribbon=(moyacf - miniacf, -moyacf + maxiacf), label="sim", title="Autocorrelation Function (ACF) of the ROR")
    Plots.plot!(p,0:maxlag, moyacf,c=1; ribbon=(moyacf - miniacf, -moyacf + maxiacf),label="")
    Plots.scatter!(p, 0:maxlag, acf_obs, label="obs", xlabel="Lag",ylabel="Autocorrelation")
   return(p)

end

using Statistics, HypothesisTests



# Function to compute Kolmogorov-Smirnov distance
function ks_distance(observed::Vector{Float64}, simulated::Vector{Float64})
    test = ApproximateTwoSampleKSTest(observed, simulated)
    return test.δ  # KS statistic
end
