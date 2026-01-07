using Statistics
using StatsPlots

"""
    spatiotemporal_correlation(data, mat_dist, max_lag_t, max_dist; min_pairs=30)

Compute empirical spatio-temporal correlation for a matrix `data` of size (D × N)
where D = number of locations, N = number of time points.

- `mat_dist` : D × D matrix of distances
- `max_lag_t` : maximum temporal lag (in days)
- `max_dist` : maximum spatial distance (in km)
- `min_pairs` : minimum number of point pairs required to compute correlation

Returns:
    corr_matrix : matrix of correlations indexed by (time_lag, spatial_bin)
    time_lags   : vector of time lags
    dist_bins   : vector of spatial bin centers
"""
function spatiotemporal_correlation(data, mat_dist, max_lag_t, max_dist; min_pairs=30, n_dist_bins=15
)
    D, N = size(data)
    @show D,N

    # Compute pairwise distances in km (Haversine)

    # Define spatial bins (uniform)
    dist_edges = range(0, max_dist; length=n_dist_bins + 1)
    dist_bins = 0.5 .* (dist_edges[1:end-1] .+ dist_edges[2:end])

    time_lags = 0:max_lag_t
    corr_matrix = fill(NaN, length(time_lags), length(dist_bins))

    for (li, lag) in enumerate(time_lags)
        # For each spatial bin
        for (bi, (dmin, dmax)) in enumerate(zip(dist_edges[1:end-1], dist_edges[2:end]))
            corrs = Float64[]

            # Loop over all station pairs
            for s1 in 1:D
                for s2 in (s1):D
                    dist = mat_dist[s1, s2]
                    if dmin <= dist < dmax
                        # Build paired time series for this lag
                        tmax = N - lag
                        x = data[s1, 1:tmax]
                        y = data[s2, 1+lag:tmax+lag]

                        # Only if both have enough non-missing datan
                        valid = .!isnan.(x) .& .!isnan.(y) 

                        if count(valid) >= min_pairs
                            xv, yv = x[valid], y[valid]
                            # if std(xv) > 0 && std(yv) > 0   # avoid NaN correlations
                                push!(corrs, cor(xv, yv))
                            # end
                        end
                    end
                end
            end

            if !isempty(corrs)
                corr_matrix[li, bi] = mean(corrs)
            end
        end
    end

    return corr_matrix, time_lags, dist_bins
end

function check_nonseparability(corr_mat, dist_bins, lags)
    # 1. Spatial correlation: lag = 0
    spatial_corr = corr_mat[findfirst(==(0), lags), :]

    # 2. Temporal correlation: closest to distance = 0
    dist_idx0 = argmin(dist_bins)
    temporal_corr = corr_mat[:, dist_idx0]

    # 3. Predicted separable correlation
    pred_sep = fill(NaN, size(corr_mat))
    for li in 1:length(lags)
        for bi in 1:length(dist_bins)
            if !isnan(spatial_corr[bi]) && !isnan(temporal_corr[li])
                pred_sep[li, bi] = spatial_corr[bi] * temporal_corr[li]
            end
        end
    end

    # 4. Ratio — only where both are valid
    ratio = fill(NaN, size(corr_mat))
    for li in 1:length(lags)
        for bi in 1:length(dist_bins)
            if !isnan(corr_mat[li, bi]) && !isnan(pred_sep[li, bi])
                ratio[li, bi] = corr_mat[li, bi] / pred_sep[li, bi]
            end
        end
    end

    return spatial_corr, temporal_corr, pred_sep, ratio
end


#################### Fitting an exp-exp or a matern model #####################
using LinearAlgebra
using Distributions
using Random

# =====================================================
# Covariance model definition
# =====================================================
abstract type CovarianceStructure end

struct ExpExp{T1,T2<:Real} <: CovarianceStructure
    sill::T1      # variance
    range_s::T2  # spatial range
    range_t::T2   # temporal range
end

# =====================================================
# Covariance computation
# =====================================================
function cov_spatiotemporal(model::ExpExp, h, u)
    return model.sill .* exp.(-h ./ model.range_s) .* exp(-u / model.range_t)
end

using BesselK: _gamma
using BesselK
function matern0(h, c, ν)
    iszero(h) && return 1.0
    arg = h / c
    return ((2^(1 - ν)) / _gamma(ν)) * adbesselkxv(ν, arg)
end

struct GneitingMatern{T1,T<:Real} <: CovarianceStructure
    σ²::T1 #sill
    c::T #spatial scale
    a::T #temporal scale
    α::T # temporal power
    β::T #sep
    δ::T #temporal covariance 
    ν::T # regularity
end

function cov_spatiotemporal(p::GneitingMatern, h, u)

    arg = 1 + (u / p.a)^(2p.α)
    argpow = arg^(p.β / 2)
    rho = matern0.(h, p.c * argpow, p.ν) / arg^(p.δ + p.β)
    return p.σ² .* rho

end



# =====================================================
# Gaussian Field definition
# =====================================================
struct GaussianField
    coords::Matrix{Float64}         # D × 2
    model::CovarianceStructure
    Mat_h::Matrix{Float64}
end


function GaussianField(coords, model)
    D = size(coords, 1)
    Mat_h = zeros((D, D))
    for j in 1:D
        for i in (j+1):D

            Mat_h[i, j] = Mat_h[j, i] = haversine(coordll[i, :], coordll[j, :]) / 1000
        end
    end
    return GaussianField(coords, model, Mat_h)
end


using Distances
# =====================================================
# Iterative simulation 
# =====================================================
function simulate_iterative(gf::GaussianField, times, lagt::Int)

    D = size(gf.Mat_h, 1)
    N = length(times)

    # Spatial distance matrix
    dist_s = gf.Mat_h



    # make covariance matrices of necessary lags
    Covariancesmatrices = [cov_spatiotemporal(gf.model, dist_s, l) for l in 0:(lagt+1)]
    C0 = Covariancesmatrices[0+1] # spatial cov at lag 0
    C_1tot = zeros(eltype(Covariancesmatrices[1]), lagt * D, lagt * D)

    for i in 1:lagt
        for j in 1:lagt
            C_1tot[(i-1)*D+1:i*D, (j-1)*D+1:j*D] .= Covariancesmatrices[abs(i - j)+1]
        end
    end
    C_ttotp1 = zeros(eltype(Covariancesmatrices[1]), D, lagt * D)
    for j in 1:lagt
        C_ttotp1[:, (j-1)*D+1:j*D] .= Covariancesmatrices[j+1]
    end





    matm = C_ttotp1 * inv(C_1tot)
    Sigmat = C0 - matm * transpose(C_ttotp1)
    Lchol = cholesky(Symmetric(Sigmat)).L

    # Storage
    Zsim = zeros(D, N)

    # Initial block (lags)
    Zinit = rand(MvNormal(zeros(lagt * D), Symmetric(C_1tot)))
    for l in 1:lagt
        Zsim[:, l] .= Zinit[(1+(l-1)*D):(l*D)]
    end

    # Iterative simulation
    for t in (lagt+1):N
        Zprec = stack_reverse_columns((Zsim[:, (t-lagt):(t-1)]))  # previous lagt days
        innov = Lchol * randn(D)
        Zsim[:, t] .= matm * Zprec + innov
    end

    return Zsim
end

function stack_reverse_columns(mat)
    D, C = size(mat)
    v = Vector{eltype(mat)}(undef, D*C)
    for (i, col) in enumerate(reverse(eachcol(mat)))
        v[(i-1)*D+1 : i*D] .= col
    end
    return v
end


# =====================================================
# MPLE fitting
# =====================================================

using Base.Threads
#attention : this only estimates for sigma=1.



using Optimization
using OptimizationOptimJL
import OptimizationOptimJL: LBFGS
using ForwardDiff

function fit_mple(data, Mat_h, model_type::Type{<:CovarianceStructure}, param0;lower=nothing,upper=nothing,maxiter=100)
    
    function pairwise_loglik(data,model_type::Type{<:CovarianceStructure},param,Mat_h;maxdist=maximum(Mat_h)/3*2, maxtime=10)
        D, Nt = size(data)
    
        space_mask = Mat_h .<= maxdist
    
        # Create a matrix for the observed data values
    
        # Precompute covariance matrix
        covmodel=model_type(1.0,param...,)
        cov_matrixes = [cov_spatiotemporal(covmodel,Mat_h, timediff) for timediff in 0:maxtime] # D x D matrix of covariances (spatial)
    
        c11 = cov_matrixes[1][1, 1] # variance
        partial_sums = zeros(eltype(cov_matrixes[1]), nthreads())
    
        @threads for t1 in 1:Nt
            tid_sum = 0  # local accumulator
    
            for t2 in 1:Nt
                u = abs(t1 - t2)
                if u <= maxtime
                    for i in 1:D
                        for j in 1:D
                            if space_mask[i, j] && !(i == j && u == 0)
                                # Extract observed values
                                z1 = data[ i,t1]
                                z2 = data[j, t2]
      # Skip NaNs
      if isnan(z1) || isnan(z2)
        continue
    end

                                # Covariances
    
                                # Correlation
                                ρ = cov_matrixes[u+1][i, j] / c11
                                ρ = clamp(ρ, -0.9999, 0.9999)  # avoid numerical instability
    
                                tid_sum += -log(2π) - log(c11) - 0.5 * log(1 - ρ^2) - (1 / (2 * c11 * (1 - ρ^2))) * (z1^2 + z2^2 - 2 * ρ * z1 * z2)
                            end
                        end
                    end
                end
            end
            partial_sums[threadid()] += tid_sum
        end
        return sum(partial_sums)
    end
    
    
    p = (data,  Mat_h , model_type)

    function optimfunction0(u, p)
        y = p[1]
        Mat_h = p[2]
        model_type=p[3]
        param=u
        llh = -pairwise_loglik(y,model_type,param,Mat_h)
        return llh
    end
    
    @show optimfunction0(param0, p)
    @show ForwardDiff.gradient(u -> optimfunction0(u, p), param0)

    optf = OptimizationFunction(optimfunction0, Optimization.AutoForwardDiff())

    prob = OptimizationProblem(optf, param0, p; lb=lower,ub=upper)
    @time sol = solve(prob, Optim.LBFGS(),maxiters=maxiter)
    #exp exp model for N = 1000 samples of D=37 stations, takes 12s.
    usol = sol.u
    return model_type(1.0,usol...)
end



"""
    plot_covariance_vs_distance(model::CovarianceStructure, h_vals, u_lags; normalize=false)

Plot the spatio-temporal covariance function as a function of spatial distance
for different temporal lags.

# Arguments
- `model`    : a fitted covariance model (ExpExp, GneitingMatern, etc.)
- `h_vals`   : vector of spatial distances (e.g. 0:5:500 km)
- `u_lags`   : vector of temporal lags to plot (e.g. [0,1,2,5,10])
- `normalize`: if true, plots correlation (cov/sill) instead of covariance
"""
function plot_covariance_vs_distance(model::CovarianceStructure, h_vals, u_lags; normalize=false)
    plt = plot(; xlabel="Distance (km)", ylabel=normalize ? "Correlation" : "Covariance",
               legend=:topright, lw=2)

    for u in u_lags
        covs = cov_spatiotemporal.(Ref(model), h_vals, u)
        if normalize
            covs ./= model.sill  # convert to correlation
        end
        plot!(plt, h_vals, covs, label="lag = $u")
    end

    display(plt)
    return plt
end

using StatsPlots

"""
    plot_covariance_vs_distance(model::CovarianceStructure, dist_bins, time_lags,
                                corr_mat; normalize=false,title="Spatio-temporal covariance")

Plot spatio-temporal covariance (or correlation) as a function of spatial distance
for different temporal lags, overlaying empirical estimates.
"""

function plot_covariance_vs_distance(model::CovarianceStructure, dist_bins, time_lags, 
                                     corr_mat; normalize=false,title="Spatio-temporal covariance")

    plt = plot(; xlabel="Distance (km)", ylabel=normalize ? "Correlation" : "Covariance",
               legend=:topright, lw=2,title=title )

    for u in time_lags
        # Theoretical curve
        covs = cov_spatiotemporal.(Ref(model), dist_bins, u)
        if normalize
            covs ./= model.sill
        end
        plot!(plt, dist_bins, covs, label="Model, lag=$u", lw=2,c=u)

        # Empirical estimates (only if lag exists in corr_mat)
        li = findfirst(==(u), time_lags)
        if li !== nothing
            emp = corr_mat[li, :]
            scatter!(plt, dist_bins, emp; label="Empirical, lag=$u", marker=:circle, ms=4,c=u)
        end
    end

    display(plt)
    return plt
end


"""
    spatiotemporal_correlation_sim(data, mat_dist, max_lag_t, max_dist; min_pairs=30, n_dist_bins=15)

Compute empirical spatio-temporal correlation for simulated datasets.

# Arguments
- `data`      : D × N × Nsim array (D locations, N time points, Nsim simulations)
- `mat_dist`  : D × D matrix of spatial distances
- `max_lag_t` : maximum temporal lag
- `max_dist`  : maximum spatial distance
- `min_pairs` : minimum number of valid pairs to compute correlation
- `n_dist_bins`: number of spatial bins

# Returns
- `corr_matrix_mean` : time_lag × spatial_bin mean correlation over simulations
- `time_lags`        : vector of time lags
- `dist_bins`        : vector of spatial bin centers
"""
function spatiotemporal_correlation_sim(data, mat_dist, max_lag_t, max_dist; min_pairs=30, n_dist_bins=15)
    D, N, Nsim = size(data)

    # Define spatial bins
    dist_edges = range(0, max_dist; length=n_dist_bins + 1)
    dist_bins = 0.5 .* (dist_edges[1:end-1] .+ dist_edges[2:end])

    time_lags = 0:max_lag_t
    corr_matrix = fill(NaN, length(time_lags), length(dist_bins), Nsim)

    for sim in 1:Nsim
        sim_data = data[:, :, sim]
        for (li, lag) in enumerate(time_lags)
            for (bi, (dmin, dmax)) in enumerate(zip(dist_edges[1:end-1], dist_edges[2:end]))
                corrs = Float64[]
                for s1 in 1:D
                    for s2 in s1:D
                        dist = mat_dist[s1, s2]
                        if dmin <= dist < dmax
                            tmax = N - lag
                            x = sim_data[s1, 1:tmax]
                            y = sim_data[s2, 1+lag:tmax+lag]

                            valid = .!isnan.(x) .& .!isnan.(y)
                            if count(valid) >= min_pairs
                                push!(corrs, cor(x[valid], y[valid]))
                            end
                        end
                    end
                end
                if !isempty(corrs)
                    corr_matrix[li, bi, sim] = mean(corrs)
                end
            end
        end
    end

    # Mean correlation over simulations
    corr_matrix_mean = mean(corr_matrix, dims=3)[:,:,1]

    return corr_matrix_mean, time_lags, dist_bins
end


    
"""
continuity_ratio(Y::AbstractMatrix)

Compute continuity ratios for all pairs (k,l) in a data matrix `Y` of size D×N.
Returns a D×D matrix of ratios.
"""
function continuity_ratio(Y::AbstractMatrix)
D, N = size(Y)
CR = fill(NaN, D, D)
for k in 1:D
    for l in 1:D
        if k == l
            continue
        end
        yk, yl = Y[k, :], Y[l, :]
        mask1 = (yk .> 0) .& (yl .== 0)
        mask2 = (yk .> 0) .& (yl .> 0)

        if any(mask1) && any(mask2)
            num = mean(yk[mask1])
            den = mean(yk[mask2])
            CR[k, l] = num / den
        end
    end
end
return CR
end

function continuity_ratio(Rsim::Array{<:Real,3})
D, N, Nsim = size(Rsim)
CRs = Array{Float64}(undef, D, D, Nsim)
for s in 1:Nsim
CRs[:, :, s] = continuity_ratio(Rsim[:, :, s])
end
return mean(CRs, dims=3)[:, :, 1]
end


"""
plot_obs_vs_sim(Robs, Rsim)

Compute continuity ratios for observed and simulated data,
then scatter plot obs vs. sim.
"""
function plotCR_obs_vs_sim(Robs, Rsim)
CR_obs = continuity_ratio(Robs)
CR_sim = continuity_ratio(Rsim)

mask = .!(isnan.(CR_obs) .| isnan.(CR_sim))
x = CR_obs[mask]
y = CR_sim[mask]

scatter(x, y, xlabel="Observed continuity ratio", ylabel="Simulated continuity ratio",
        title="Observed vs Simulated Continuity Ratios",label=:none)
        Plots.abline!(1, 0, line=:dash,label=:none)

end

function plotCR_obs_vs_sim(Robs, Rsim, locsdata; threshold=1.5)
    CR_obs = continuity_ratio(Robs)
    CR_sim = continuity_ratio(Rsim)

    mask = .!(isnan.(CR_obs) .| isnan.(CR_sim))
    x = CR_obs[mask]
    y = CR_sim[mask]

    p=scatter(x, y, xlabel="Observed continuity ratio", ylabel="Simulated continuity ratio",
            title="Observed vs Simulated Continuity Ratios", label=:none)
    Plots.abline!(p,1, 0, line=:dash, label=:none)

    # Add labels for observed > threshold
    idx = findall(CR_obs .> threshold)
    for I in idx
        i, j = I[1], I[2]            
        if !isnan(CR_obs[i,j]) && !isnan(CR_sim[i,j])
            labeltxt = string(locsdata.STANAME[i], " & ", locsdata.STANAME[j])
            annotate!(p,CR_obs[i,j], CR_sim[i,j], Plots.text(labeltxt, 3, :black))
        end
    end
    idx = findall(CR_sim .> threshold)
    for I in idx
        i, j = I[1], I[2]            
        if !isnan(CR_obs[i,j]) && !isnan(CR_sim[i,j])
            labeltxt = string(locsdata.STANAME[i], " & ", locsdata.STANAME[j])
            annotate!(p,CR_obs[i,j], CR_sim[i,j], Plots.text(labeltxt, 3, :red))
        end
    end
    return p
end

function plotCR_obs_vs_sim(Robs, Rsim, locsdata; threshold=1.5,dmax=300)
    CR_obs = continuity_ratio(Robs)
    CR_sim = continuity_ratio(Rsim)

    mask = .!(isnan.(CR_obs) .| isnan.(CR_sim) .| (Mat_h.>dmax))
    x = CR_obs[mask]
    y = CR_sim[mask]

    p=scatter(x, y, xlabel="Observed continuity ratio", ylabel="Simulated continuity ratio",
            title="Observed vs Simulated Continuity Ratios", label=:none)
    Plots.abline!(p,1, 0, line=:dash, label=:none)

    # Add labels for observed > threshold
    idx = findall(CR_obs .> threshold)
    for I in idx
        i, j = I[1], I[2]            
        if !isnan(CR_obs[i,j]) && !isnan(CR_sim[i,j])&& Mat_h[i,j]<dmax
            labeltxt = string(locsdata.STANAME[i], " & ", locsdata.STANAME[j])
            annotate!(p,CR_obs[i,j], CR_sim[i,j], Plots.text(labeltxt, 3, :black))
        end
    end
    idx = findall(CR_sim .> threshold)
    for I in idx
        i, j = I[1], I[2]            
        if !isnan(CR_obs[i,j]) && !isnan(CR_sim[i,j]) && Mat_h[i,j]<dmax
            labeltxt = string(locsdata.STANAME[i], " & ", locsdata.STANAME[j])
            annotate!(p,CR_obs[i,j], CR_sim[i,j], Plots.text(labeltxt, 3, :red))
        end
    end
    return p
end
