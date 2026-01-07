include("../11SpatialBernoulli/SpatialBernoulli.jl")
include("../11SpatialBernoulli/plot_validation.jl")
import Base.size
using ArgCheck
using LaTeXStrings
using SmoothPeriodicStatsModels


#-----------------equivalent to periodichmm.jl  ---------------------------#

"""
    ARPeriodicHMMSpa([a, ]A, B) -> ARPeriodicHMMSpa

Build an Auto Regressive Periodic Hidden Markov Chain with Spatial Bernoulli emission `ARPeriodicHMMSpa` with transition matrix `A(t)` and observation distributions `B(t)`.  
If the initial state distribution `a` is not specified, it does not work. Please give initial state distribution.

Observations distributions can only be SpatialBernoulli.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractArray{T,3}`: transition matrix.
- `B`: rain probabilities
- `R` : range parameter
-  `h` distance matrix.
"""
struct PeriodicHMMSpaMemory{T}
    a::Vector{T}
    A::Array{T,3}
    R::Array{T,2}
    B::Array{T,4} 
    h::AbstractMatrix
end



size(hmm::PeriodicHMMSpaMemory, dim=:) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]
# K                # D             # T          # number of memory


#-----------------equivalent to trig_conversion.jl  ---------------------------#

function Trig2PeriodicHMMspaMemory(a::AbstractVector, my_trans_θ::AbstractArray{<:AbstractFloat,3}, Bernoulli_θ::AbstractArray{<:AbstractFloat,4}, Range_θ::AbstractArray{<:AbstractFloat,2}, my_T::Integer, my_h::AbstractMatrix)
    my_K, my_D, my_size_order = size(Bernoulli_θ)
    @assert my_K == size(my_trans_θ, 1)

    # make transition matrices as function of time
    if my_K == 1
        my_A = ones(my_K, my_K, my_T)
    else
        my_A = zeros(my_K, my_K, my_T)
        for k = 1:my_K, l = 1:my_K-1, t = 1:my_T
            my_A[k, l, t] = exp(polynomial_trigo(t, my_trans_θ[k, l, :], my_T))
        end
        for k = 1:my_K, t = 1:my_T
            my_A[k, my_K, t] = 1  # last colum is 1/normalization (one could do otherwise)
        end
        normalization_polynomial = [1 + sum(my_A[k, l, t] for l = 1:my_K-1) for k = 1:my_K, t = 1:my_T]
        for k = 1:my_K, l = 1:my_K, t = 1:my_T
            my_A[k, l, t] /= normalization_polynomial[k, t]
        end
    end
    my_A
    # A is a K*K* T matrix of transition.

    #make emission parameters
    my_p = [1 / (1 + exp(polynomial_trigo(t, Bernoulli_θ[k, s, h, :], my_T))) for k = 1:my_K, t = 1:my_T, s = 1:my_D, h = 1:my_size_order]
    # p is a K (states)* T(period) *  D (stations) * m+1 (memory) vector.
    my_range = [exp(polynomial_trigo(t, Range_θ[k, :], my_T)) for k = 1:my_K, t = 1:my_T]
    # range is a K (states)* T(period)  * m+1 (memory) vector.
    # return (my_A, p, range)


    model = PeriodicHMMSpaMemory(a, my_A, my_range, my_p, my_h)
    return model
end



# simulate with given z sequence.
function my_rand(hmm::PeriodicHMMSpaMemory,
    z::AbstractVector{<:Integer},
    n2t::AbstractVector{<:Integer}, y_ini
)
    N = length(n2t)
    y = Matrix{eltype(eltype(hmm.B))}(undef, size(hmm, 2), length(z))
    if size(hmm, 4) == 2
        y[:, 1] = y_ini
        for n in 2:N
            t = n2t[n] # periodic t
            y_previous = y[:, n-1]
            lambdas = (1 .- y_previous) .* hmm.B[z[n], t, :, 1] .+ y_previous .* hmm.B[z[n], t, :, 2]
            y[:, n] = rand(SpatialBernoulli(hmm.R[z[n], t], 1.0, 1 / 2, lambdas, hmm.h))
        end
        return y'
    elseif size(hmm,4) == 1
        for n in 1:N
            t = n2t[n] # periodic t
            lambdas = hmm.B[z[n], t, :, 1] 
            y[:, n] = rand(SpatialBernoulli(hmm.R[z[n], t], 1.0, 1 / 2, lambdas, hmm.h))
        end
        return y'
    end
end


function my_rand(hmm::PeriodicHMMSpaMemory,
    n2t::AbstractVector{<:Integer};
    z_ini=rand(Categorical(hmm.a))::Integer, y_ini=fill(0, size(hmm, 2)),
    seq=false
)
    N = length(n2t)
    z = zeros(Int, N)
    (N >= 1) && (z[1] = z_ini)
    for n = 2:N
        tₙ₋₁ = n2t[n-1] # periodic t-1
        z[n] = rand(Categorical(hmm.A[z[n-1], :, tₙ₋₁]))
    end
    y = my_rand(hmm, z, n2t, y_ini)
    return seq ? (z, y) : y
end

# ----------- plot parameters ---------------------------#

function PlotModel(hmm::PeriodicHMMSpaMemory; indices_sta=1:size(hmm, 2))
    K = size(hmm, 1)
    d = size(hmm, 2)
    T = size(hmm, 3)
    nsta = length(indices_sta)
    # plot the ranges
    ranges = zeros(K, T)
    p1 = Plots.plot(title="spatial range")
    for k in 1:K

        Plots.plot!(p1, 1:T, hmm.R[k, :], label=k)

    end

    # plot the transotion parameters
    begin
        pA = [Plots.plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing) for k in 1:K]
        for k in 1:K
            [Plots.plot!(pA[k], hmm.A[k, l, :], c=l, label=L"Q_{%$(k)\to %$(l)}", legend=:topleft) for l in 1:K]

            hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
            Plots.ylims!(0, 1)
        end
        pallA = Plots.plot(pA..., size=(1000, 500))
    end

    # plot the proba of rain for all stations

    begin
        pB = [Plots.plot() for j in 1:nsta]
        for j in 1:nsta
            [Plots.plot!(pB[j], [hmm.B[k, t, j, 1] for t in 1:T], c=k, label=:none) for k in 1:K]
            if size(hmm,4)==2
                [Plots.plot!(pB[j], [hmm.B[k, t, j, 2] for t in 1:T], c=k, label=:none, linestyle=:dash) for k in 1:K]
    
            end

            hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
            Plots.ylims!(pB[j], (0, 1))
            title!(pB[j], "P(rain) at Station " * string(indices_sta[j]))
            # Add dummy plots just for the legend
        Plots.plot!(pB[j], [NaN], [NaN], label="No rain previous day", linestyle=:solid, c=:black)
        Plots.plot!(pB[j], [NaN], [NaN], label="Rain previous day", linestyle=:dash, c=:black)
    
        end

       
        pallB = Plots.plot(pB...)
    end


    return p1, pallA, pallB


end


function PlotModel(hmm::PeriodicHMMSpaMemory,hmm_fit::PeriodicHMMSpaMemory; indices_sta=1:size(hmm, 2))
    K = size(hmm, 1)
    d = size(hmm, 2)
    T = size(hmm, 3)
    nsta = length(indices_sta)
    # plot the ranges
    p1 = plot(title="spatial range")
    for k in 1:K

        plot!(p1, 1:T, hmm.R[k, :],c=k, label=k)
        plot!(p1, 1:T, hmm_fit.R[k, :],c=k, label=:none,linestyle=:dash)

    end

    # plot the transition parameters
    begin
        pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing) for k in 1:K]
        for k in 1:K
            [plot!(pA[k], hmm.A[k, l, :], c=l, label=L"Q_{%$(k)\to %$(l)}", legend=:topleft) for l in 1:K]
            [plot!(pA[k], hmm_fit.A[k, l, :], c=l, label=:none, legend=:topleft,linestyle=:dash) for l in 1:K]

            hline!(pA[k], [0.5], c=:black, label=:none, linestyle=:dash)
            ylims!(0, 1)
        end
        pallA = plot(pA..., size=(1000, 500))
        title!(pallA,"transition probabilities")
    end

    # plot the proba of rain for all stations

    begin
        pB = [plot() for j in 1:nsta]
        for j in 1:nsta
            [plot!(pB[j], [hmm.B[k, t, indices_sta[j], 1] for t in 1:T], c=k, label=:none) for k in 1:K]
            [plot!(pB[j],[hmm_fit.B[k, t, indices_sta[j], 1] for t in 1:T], c=k, label=:none, linestyle=:dash) for k in 1:K]


            hline!(pB[j], [0.5], c=:black, label=:none, linestyle=:dash)
            ylims!(pB[j], (0, 1))
            title!(pB[j], "P(rain) at Station " * string(indices_sta[j]))
        end
        pallB = plot(pB...)
    end


    return p1, pallA, pallB


end

function PlotFitAndReal(hmm::PeriodicHMMSpaMemory,hmm2::PeriodicHMMSpaMemory; indices_sta=1:size(hmm, 2))
    K = size(hmm, 1)
    d = size(hmm, 2)
    T = size(hmm, 3)
    nsta = length(indices_sta)
    # plot the ranges
    ranges = zeros(K, T)
    p1 = plot(title="spatial range")
    for k in 1:K
        plot!(p1, 1:T, hmm.R[k, :], label=:none, c=k)
        plot!(p1, 1:T, hmm2.R[k, :], label=:none,c=:black)

    end

    # plot the transotion parameters
    begin
        pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing) for k in 1:K]
        for k in 1:K
            [plot!(pA[k], hmm.A[k, l, :], c=l, label=L"Q_{%$(k)\to %$(l)}", legend=:topleft) for l in 1:K]
            [plot!(pA[k], hmm2.A[k, l, :], c=:black, label="") for l in 1:K]

            hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
            ylims!(0, 1)
        end
        pallA = plot(pA..., size=(1000, 500))
    end

    # plot the proba of rain for all stations

    begin
        pB = [plot() for j in 1:nsta]
        for j in 1:nsta
            [plot!(pB[j], [hmm.B[k, t, j, 1] for t in 1:T], c=k, label=:none) for k in 1:K]
            [plot!(pB[j], [hmm2.B[k, t, j, 1] for t in 1:T], c=:black, label=:none) for k in 1:K]

            if size(hmm,4)==2
                [plot!(pB[j], [hmm.B[k, t, j, 2] for t in 1:T], c=k, label=:none, linestyle=:dash) for k in 1:K]
                [plot!(pB[j], [hmm2.B[k, t, j, 2] for t in 1:T], c=:black, label=:none, linestyle=:dash) for k in 1:K]

            end

            hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
            ylims!(pB[j], (0, 1))
            title!(pB[j], "P(rain) at Station " * string(indices_sta[j]))
            # Add dummy plots just for the legend
        plot!(pB[j], [NaN], [NaN], label="No rain previous day", linestyle=:solid, c=:black)
        plot!(pB[j], [NaN], [NaN], label="Rain previous day", linestyle=:dash, c=:black)
    
        end

       
        pallB = plot(pB...)
    end


    return p1, pallA, pallB


end