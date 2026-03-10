# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
dayofyear_Leap(d) = @. dayofyear(d) + ((!isleapyear(d)) & (month(d) > 2))

vec_triu(A) = A[tril(trues(size(A)), -1)] # return striclty upper diagonal element of matrix A

"""
    shortname(name::String)
Experimental function that returns only the most relevant part of a station name.
```julia
long_name = "TOULOUSE-BLAGNAC"
shortname(long_name) # "TOULOUSE"
```
"""
function shortname(name::AbstractString)
    n = uppercase(name)
    if startswith(n, "LA ") || startswith(n, "LE ") || startswith(n, "LES ") || startswith(n, "ST ") || startswith(n, "BELLE ")
        m = join(split(name)[1:2], " ")
    elseif startswith(n, "PTE DE LA ")
        m = join(split(name)[3:4], " ")
    elseif startswith(n, "PTE DE ")
        m = split(name)[3]
    elseif startswith(n, "ST-") || startswith(n, "MONT-")
        return rstrip(n)
    else
        m = split(n)[1]
    end
    return split(m, "-")[1]
end

"""
    pmf_spell(y::AbstractVector, value)
Return the distribution of spells (consecutive sequence of with the same value) length of `value` in `y`
"""
function pmf_spell(y::AbstractVector, value)
    spells = rle(y)
    idx_spells = findall(spells[1] .== value)
    return spells[2][idx_spells]
end

"""
    monthly_agg(y::AbstractArray, idxs)
```julia
using DataFrames, Dates
time_range = Date(1956):Day(1):Date(2019,12,31)
year_range = unique(year.(time_range))
df = DataFrame(:DATE => time_range, :Temperature => 20 .+ 5*randn(length(time_range)))
monthly_agg(df, :Temperature) 
monthly_agg(df, :Temperature, mean) 
# or
idx_year = [findall(x-> year.(x) == m, df[:, :DATE]) for m in year_range]
idx_month = [findall(x-> month.(x) == m, df[:, :DATE]) for m in 1:12]
idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_month]
monthly_agg(df.Temperature, idx_all)
```
"""
monthly_agg(y::AbstractArray, idxs, aggfunc=sum) = [aggfunc(y[idx]) for idx in idxs]


"""
    corTail(x::AbstractMatrix, q = 0.95)
Compute the (symmetric averaged) tail index matrix `M` of a vector x, i.e. M[i, j] = (ℙ(x[:,j] > Fxⱼ(q) ∣ x[:,i] > Fxᵢ(q)) + ℙ(x[:,i] > Fxᵢ(q) ∣ x[:,j] > Fxⱼ(q)))/2 where Fx(q) is the CDF of x.
Note it uses the same convention as `cor` function i.e. observations in rows and features in column.
"""
function corTail(x, q=0.95)
    n, k = size(x)
    c = zeros(k, k)
    for i in 1:k
        ind_i = findall(x[:, i] .>= quantile(x[:, i], q))
        for j in 1:k
            c[i, j] = count(x[ind_i, j] .>= quantile(x[:, j], q)) / length(ind_i)
        end
    end

    return (c + c') / 2
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

function ROR(y::AbstractMatrix)
    [mean(y[:, t]) for t in 1:length(y[1, :])]
end

function ROR(y::Array{Bool,3})
    [[mean(y[:, t, i]) for t in 1:length(y[1, :, 1])] for i in 1:length(y[1, 1, :])]
end

function hist_ROR(y)
    ror = ROR(y)
    d= length(y[:,1])
    xax = 0:(1/d):1
    vec= [mean(ror.==xax[k]) for k in 1:(d+1)]
    return xax,vec
end

function enveloppe_minmax(dy, y)
    mindy = [min([dy[i][x] for i in 1:length(dy)]...) for x in 1:length(y)]
    maxdy = [max([dy[i][x] for i in 1:length(dy)]...) for x in 1:length(y)]
    medy = [quantile([dy[i][x] for i in 1:length(dy)], 0.5) for x in 1:length(y)]

    return mindy, maxdy, medy, y

end

function hist_vectors(y::AbstractVector{<:AbstractVector},xax)
    dy = Vector(undef, length(y))
    # enveloppe of all densities
    for i in 1:length(y)
        dy[i] = [mean(y[i].==xax[k]) for k in 1:length(xax)]

    end
    return xax,dy
end





#### Plots

function savefigcrop(plt::CairoMakie.Figure, save_name::AbstractString)
    if !endswith(save_name, ".pdf")
        save_name = string(save_name, ".pdf")
    end
    save_name_with_ext = replace(save_name, ".pdf" => "")
    CairoMakie.save(save_name, plt)
    run(`pdfcrop $(save_name)`) # Petit délire pour croper proprement la figure pas forcément necessaire ici?? Ca demande peut être d'installer pdfcrop
    mv(string(save_name_with_ext, "-crop", ".pdf"), save_name, force=true)
end

savefigcrop(save_name::AbstractString, plt::CairoMakie.Figure) = savefigcrop(plt, save_name)

function compute_error(
    y::AbstractMatrix,
    centertype::Symbol,
    errortype::Symbol,
    percentiles::AbstractVector,
)
    y_central = fill(NaN, size(y, 1))

    # NaNMath doesn't accept Ints so convert to AbstractFloat if necessary
    if eltype(y) <: Integer
        y = float(y)
    end
    # First compute the center
    y_central = if centertype === :mean
        mapslices(NaNMath.mean, y, dims=2)
    elseif centertype === :median
        mapslices(NaNMath.median, y, dims=2)
    else
        error("Invalid center type. Valid symbols include :mean or :median")
    end

    # Takes 2d matrix [x,y] and computes the desired error type for each row (value of x)
    if errortype === :std || errortype === :sem
        y_error = mapslices(NaNMath.std, y, dims=2)
        if errortype == :sem
            y_error = y_error ./ sqrt(size(y, 2))
        end

    elseif errortype === :percentile
        y_lower = fill(NaN, size(y, 1))
        y_upper = fill(NaN, size(y, 1))
        if any(isnan.(y)) # NaNMath does not have a percentile function so have to go via StatsBase
            for i in axes(y, 1)
                yi = y[i, .!isnan.(y[i, :])]
                y_lower[i] = percentile(yi, percentiles[1])
                y_upper[i] = percentile(yi, percentiles[2])
            end
        else
            y_lower = mapslices(Y -> percentile(Y, percentiles[1]), y, dims=2)
            y_upper = mapslices(Y -> percentile(Y, percentiles[2]), y, dims=2)
        end

        y_error = (y_central .- y_lower, y_upper .- y_central) # Difference from center value
    else
        error("Invalid error type. Valid symbols include :std, :sem, :percentile")
    end

    return y_central, y_error
end


# Makie errorline recipe
"""
    errorline!(ax, x, y; kwargs...)
    errorline!(ax, y; kwargs...)
    
Function for creating error band plots with Makie, similar to ribbons, error bars, or plume plots.
Allows for easy control of error type and NaN handling.

# Arguments
- `x`: (vector, unit range) - the values along the x-axis for each y-point
- `y`: (matrix [x, repeat]) - values along y-axis wrt x. The first dimension must be of equal 
   length to that of x. The second dimension is treated as repeated observations and error is computed 
   along this dimension.

# Keyword Arguments
- `errorstyle::Symbol = :ribbon` - `:ribbon`, `:stick`, or `:plume`
- `centertype::Symbol = :mean` - `:mean` or `:median`
- `errortype::Symbol = :std` - `:std`, `:sem`, or `:percentile`
- `percentiles::Vector = [25, 75]` - percentiles to use if errortype === :percentile
- `secondarycolor = CairoMakie.Makie.Inherit(:linecolor, nothing)` - color for secondary elements (sticks/plume lines/ribbons)
- `secondaryalpha::Float64 = 0.1` - alpha value of plume/ribbon/sticks
- `stickwidth::Float64 = 0.01` - width of error sticks (fraction of x-range)
- `secondarylines::Union{Int, AbstractVector} = 100` - number of plume lines to plot or vector of indices
- `linewidth::Real = 2` - width of main line (inherited from Lines)
- `secondarylinewidth::Real = 1` - width of secondary lines (plume/error bars)
- `whiskerwidth::Real = 2

# Example
```julia
xerror = 1:10
yerror = fill(NaN, 10, 100, 3)
for i = axes(yerror, 3)
    yerror[:, :, i] = collect(1:2:20) .+ rand(10, 100) .* 5 .* collect(1:2:20) .+ rand() * 100
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    errorline!(ax, 1:10, yerror[:, :, 1], errorstyle=:ribbon, percentiles=[0,100], errortype=:percentile, centertype=:median)
    errorline!(ax, 1:10, yerror[:, :, 2], errorstyle=:stick, errortype=:std)
    errorline!(ax, 1:10, yerror[:, :, 3], errorstyle=:plume, errortype=:sem, secondarylines=50)
    fig
end
```
"""
CairoMakie.Makie.@recipe ErrorLine (x, y) begin
    CairoMakie.Makie.documented_attributes(Lines)...
    "Error style - :ribbon, :stick, or :plume"
    errorstyle = :ribbon
    "Center type - :mean or :median"
    centertype = :mean
    "Error type - :std, :sem, or :percentile"
    errortype = :std
    "Percentiles to use if errortype === :percentile"
    percentiles = [25, 75]
    "Color for secondary elements (sticks/plume lines)"
    secondarycolor = @inherit linecolor
    "Alpha value of plume/ribbon/sticks"
    secondaryalpha = 0.1
    "Width of error sticks (fraction of x-range)"
    stickwidth = 0.01
    "Number of plume lines to plot"
    secondarylines = 100
    "Width of secondary lines"
    secondarylinewidth = 1
    "Width of whiskers"
    whiskerwidth = 2
end

function CairoMakie.Makie.plot!(plt::ErrorLine)
    # Get converted arguments
    x = plt.x[]
    y = plt.y[]

    # Extract attributes (use [] to get values from observables)
    errorstyle = plt.errorstyle[]
    centertype = plt.centertype[]
    errortype = plt.errortype[]
    percentiles = plt.percentiles[]

    secondarylines = plt.secondarylines[]

    valid_attributes = CairoMakie.Makie.shared_attributes(plt, Lines)

    # Check y orientation
    ndims(y) > 2 && error("ndims(y) > 2")

    if size(y, 1) !== length(x)
        error("size(y, 1) !== length(x)")
    elseif ndims(y) == 2 && size(y, 1) != length(x) && size(y, 2) == length(x) # Check if y needs to be transposed
        error("y appears to be transposed. Ensure that the first dimension of y matches length(x)")
    end

    # Compute center and distribution for each value of x
    y_central, y_error = compute_error(y, centertype, errortype, percentiles)
    y_central = dropdims(y_central; dims=2)
    if errortype !== :percentile
        y_error = dropdims(y_error; dims=2)
    end

    if errorstyle === :ribbon
        if errortype !== :percentile
            band!(plt, x, y_central .- y_error, y_central .+ y_error,
                color=plt.secondarycolor, alpha=plt.secondaryalpha
            )
        else
            band!(plt, x, y_central - y_error[1][:, 1], y_central + y_error[2][:, 1],
                color=plt.secondarycolor, alpha=plt.secondaryalpha
            )
        end
    elseif errorstyle === :stick
        # Compute error bar bounds
        if errortype === :percentile
            error_low = y_error[1]
            error_high = y_error[2]
        else
            error_low = y_error
            error_high = y_error
        end

        # Draw error bars using Makie's errorbars!
        errorbars!(plt, x, y_central, error_low, error_high,
            color=plt.secondarycolor, alpha=plt.secondaryalpha, linewidth=plt.secondarylinewidth,
            whiskerwidth=20)

    elseif errorstyle === :plume
        if secondarylines isa Integer
            sub_idx = 1:secondarylines
        elseif secondarylines isa AbstractVector
            sub_idx = secondarylines
        else
            error("secondarylines must be Integer or AbstractVector, got $(typeof(secondarylines))")
        end

        for j in sub_idx
            lines!(plt, x, y[:, j],
                color=plt.secondarycolor, alpha=plt.secondaryalpha, linewidth=plt.secondarylinewidth,
            )
        end
    else
        error("Invalid error style. Valid symbols include :ribbon, :stick, or :plume.")
    end

    # Base line
    lines!(plt, valid_attributes, x, y_central)


    return nothing
end


# xerror = 1:10
# yerror = fill(NaN, 10, 100, 3)
# for i = axes(yerror, 3)
#     yerror[:, :, i] = collect(1:2:20) .+ rand(10, 100) .* 5 .* collect(1:2:20) .+ rand() * 100
# end

# begin
#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     errorline!(ax, 1:10, yerror[:, :, 1], errorstyle=:ribbon, percentiles=[0,100], errortype=:percentile, centertype=:median)
#     errorline!(ax, 1:10, yerror[:, :, 2], errorstyle=:stick, errortype=:std)
#     errorline!(ax, 1:10, yerror[:, :, 3], errorstyle=:plume, errortype=:sem, secondarylines=50)
#     fig
# end

"""
    errorlinehist(y; kwargs)
    Function for parsing inputs to easily make a [`ribbons`] (https://ggplot2.tidyverse.org/reference/geom_ribbon.html),
    stick errorbar (https://www.mathworks.com/help/matlab/ref/errorbar.html), or plume
    (https://stackoverflow.com/questions/65510619/how-to-prepare-my-data-for-plume-plots) with several histograms plot.

Inputs: default values are indicated with *s

y is a Vector of vector

    bins (*:auto*, AbstractVector)

    norm (`Symbol` - *:false*, `:pdf`, `:probability`)

    error_style (`Symbol` - *:ribbon*, :stick, :plume) - determines whether to use a ribbon style or stick style error
     representation.

    centertype (symbol - *:mean* or :median) - which approach to use to represent the central value of y at each x-value.

    errortype (symbol - *:std*, :sem, :percentile) - which error metric to use to show the distribution of y at each x-value.

    percentiles (Vector{Int64} *[25, 75]*) - if using errortype === :percentile then which percentiles to use as bounds.

    secondarycolor (`Symbol`, `RGB`, `:matched` - *:Gray60*) - When using stick mode this will allow for the setting of the stick color.
        If `:matched` is given then the color of the sticks with match that of the main line.

    secondarylinealpha (float *.1*) - alpha value of plume lines.

    numsecondarylines (int *100*) - number of plume lines to plot behind central line.

    stickwidth (Float64 *.01*) - How much of the x-axis the horizontal aspect of the error stick should take up.

Example

```julia
using Distributions
dist = Normal()
N = 20000 # number of sample used in each histogram
N_hist = 50 # number of histogram
yc = [rand(dist, N) for _ in 1:N_hist]
edges = -5:0.05:5
length(edges)

begin
    fig = Figure()
    ax = Axis(fig[1, 1])
    errorlinehist!(ax, yc, normalization=:pdf, bins=edges, errorstyle=:ribbon, errortype=:percentile, percentiles=[0, 100]
    )
    fig
end
```
"""
CairoMakie.Makie.@recipe ErrorLineHist (x, y) begin
    CairoMakie.Makie.documented_attributes(ErrorLine)...

    """
    Sets the number of bins if set to an integer or the edges of bins if set to
    an sorted collection of real numbers.
    """
    bins = 15 # Int or iterable of edges
    """
    Sets the normalization applied to the histogram. Possible values are:

    * `:pdf`: Normalize by sum of weights and bin sizes. Resulting histogram
      has norm 1 and represents a probability density function.
    * `:density`: Normalize by bin sizes only. Resulting histogram represents
      count density of input and does not have norm 1. Will not modify the
      histogram if it already represents a density (`h.isdensity == 1`).
    * `:probability`: Normalize by sum of weights only. Resulting histogram
      represents the fraction of probability mass for each bin and does not have
      norm 1.
    * `:none`: Do not normalize.
    """
    normalization = :pdf
    "Sets optional statistical weights."
    weights = nothing
    # "Scales the histogram by a common factor such that the largest bin reaches the given value."
    # scale_to = nothing
end


_filternans(vs::NTuple{1,AbstractVector}) = filter!.(isfinite, vs)
function _filternans(vs::NTuple{N,AbstractVector}) where {N}
    _invertedindex(v, not) = [j for (i, j) in enumerate(v) if !(i ∈ not)]
    nots = union(Set.(findall.(!isfinite, vs))...)
    return _invertedindex.(vs, Ref(nots))
end

function my_make_hist(
    vs::NTuple{N,AbstractVector},
    binning;
    normed=false,
    weights=nothing,
) where {N}
    localvs = _filternans(vs)
    edges = CairoMakie.Makie.pick_hist_edges(localvs, binning)
    h = float(
        isnothing(weights) ?
        StatsBase.fit(StatsBase.Histogram, localvs, (edges,), closed=:left) :
        StatsBase.fit(
            StatsBase.Histogram,
            localvs,
            StatsBase.Weights(weights),
            edges,
            closed=:left,
        ),
    )
    return LinearAlgebra.normalize!(h, mode=_hist_norm_mode(normed))
end

_hist_norm_mode(mode::Symbol) = mode
_hist_norm_mode(mode::Bool) = mode ? :pdf : :none

function CairoMakie.Makie.plot!(plt::ErrorLineHist)
    v = plt.args[][1]
    normed = plt.normalization[]
    weights = plt.weights[]
    plotattributes = CairoMakie.Makie.shared_attributes(plt, ErrorLine)

    vs = filter.(isfinite, (reduce(vcat, v),))
    # edges = Plots._hist_edges(vs, bins)
    # x = edges[1][1:end-1]
    # map!(CairoMakie.Makie.pick_hist_edges, plt, [vs, plt.bins[]], :edges)
    edges = CairoMakie.Makie.pick_hist_edges(vs, plt.bins[])
    nbins = length(edges) .- 1

    ngroups = length(v)

    ## compute weights (frequencies) by group using those edges
    y = zeros(nbins, ngroups)
    for i in 1:ngroups
        v_i = filter(isfinite, v[i])
        w_i = weights
        h_i = my_make_hist((v_i,), edges; normed=normed, weights=w_i)
        y[:, i] += h_i.weights .+ eps() # for numerical stability when in log-scale
    end
    errorline!(plt, plotattributes, edges[1:end-1], y)
end

# using Distributions
# dist = Normal()
# N = 20000 # number of sample used in each histogram
# N_hist = 50 # number of histogram
# yc = [rand(dist, N) for _ in 1:N_hist]
# edges = -5:0.05:5
# length(edges)

# begin
#     fig = Figure()
#     ax = Axis(fig[1, 1])
#     errorlinehist!(ax, yc, normalization=:pdf, bins=edges, errorstyle=:ribbon, errortype=:percentile, percentiles=[0, 100]
#     )
#     fig
# end
