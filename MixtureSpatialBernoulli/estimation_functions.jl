using Base.Threads
####################################################################################################################################"
########### my additions -  ########################################################################
######################################################################################################################################


struct PairwiseClassicEM <: AbstractEM end

function M_step!(α, dists, y::AbstractVecOrMat, γ, method::ClassicEM; order=nothing, QMC_m=100, maxiters=100)
    α[:] = mean(γ, dims=1)
    if isnothing(order)
        @threads for k in eachindex(eachcol(γ))
            γₖ = eachcol(γ)[k]
            dists[k] = fit_mle(dists[k], y, γₖ, m=QMC_m * length(dists[k]), maxiters=maxiters)
        end
    else
        messages = Vector{String}(undef, length(eachcol(γ)))

        @threads for k in eachindex(eachcol(γ))
            @show Threads.threadid()
            messages[k] = "Thread ID: $(Threads.threadid()) handling index: $k"
            γₖ = eachcol(γ)[k]
            dists[k] = fit_mle(dists[k], y, γₖ; order=dists[k].order, m=QMC_m * length(dists[k]), maxiters=maxiters)
        end
    end
    println.(messages)

end

function E_step!(
    LL::AbstractMatrix,
    c::AbstractVector,
    γ::AbstractMatrix,
    dists::AbstractVector{F} where {F<:Distribution},
    α::AbstractVector,
    y::AbstractMatrix;
    robust=false,
    QMC_m=100,
)
    # evaluate likelihood for each type k
    @views for k in eachindex(dists)
        LL[:, k] .= log(α[k])
        @threads for n in axes(y, 2)
            LL[n, k] += logpdf(dists[k], y[:, n]; m=QMC_m * length(dists[1]))
        end
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    logsumexp!(c, LL) # c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)
end

# rewriting the EM procedure with fixed order possibility and choice of QMC
function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    method::ClassicEM;
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
    order=nothing, #added this : if nothing is given then the estimation will be done without fixing. If anything is given it will keep initial value.
    QMC_m=100,
    maxiter_m=1000)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size(y)[2], length(dists)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # E-step
    E_step!(LL, c, γ, dists, α, y; robust=robust, QMC_m=QMC_m)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter
        # M-step
        M_step!(α, dists, y, γ, method; order=order, QMC_m=QMC_m, maxiters=maxiter_m)

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust=robust, QMC_m=QMC_m)

        # Loglikelihood
        logtotp = sum(c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < atol || (rtol !== nothing && abs(logtotp - logtot) < rtol * (abs(logtot) + abs(logtotp)) / 2)
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            history["converged"] = true
            break
        end
        # Break if log-likelihood decreases
        if logtotp < logtot
            println("Warning: Log-likelihood decreased at iteration $it. Stopping optimization.")
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println(
                "EM has not converged after $(history["iterations"]) iterations, final loglikelihood = $logtot",
            )
        end
    end

    return history
end



function M_step!(α, dists, y::AbstractVecOrMat, wp::AbstractMatrix{<:Real}, γ, method::PairwiseClassicEM; order=nothing, QMC_m=100, maxiters=1000)
    α[:] = mean(γ, dims=1)


    if isnothing(order)
        @threads for k in eachindex(eachcol(γ))
            γₖ = eachcol(γ)[k]
            dists[k] = fit_mle(dists[k], y, wp, γₖ, m=QMC_m * 2, maxiters=maxiters)
        end
    else
        messages = Vector{String}(undef, length(eachcol(γ)))

        @threads for k in eachindex(eachcol(γ))
            messages[k] = "Thread ID: $(Threads.threadid()) handling index: $k"
            γₖ = eachcol(γ)[k]
            dists[k] = fit_mle(dists[k], y, wp, γₖ; order=dists[k].order, m=QMC_m * 2, maxiters=maxiters)
        end
    end
    println.(messages)


end


function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    method::PairwiseClassicEM;
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
    order=nothing, #added this : if nothing is given then the estimation will be done without fixing. If anything is given it will keep initial value.
    QMC_m=100,
    factor_dmax=1, #default all pairs,
    maxiter_m=1000)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size(y)[2], length(dists)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)


    tdist = maximum(dists[1].h) * factor_dmax
    wp = 1.0 .* (dists[1].h .< tdist)

    # E-step
    E_step!(LL, c, γ, dists, α, y; robust=robust, QMC_m=QMC_m)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter
        println("EM iteration number " * string(it))
        @show α
        @show [dists[k].range for k in 1:length(dists)]
        # M-step
        M_step!(α, dists, y, wp, γ, method; order=order, QMC_m=QMC_m, maxiters=maxiter_m)

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust=robust, QMC_m=QMC_m)

        # Loglikelihood
        logtotp = sum(c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < atol || (rtol !== nothing && abs(logtotp - logtot) < rtol * (abs(logtot) + abs(logtotp)) / 2)
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            history["converged"] = true
            break
        end
        # Break if log-likelihood decreases
        if logtotp < logtot && abs((logtotp-logtot)/logtot) > atol
            println("Warning: Log-likelihood decreased at iteration $it. Stopping optimization.")
            break
        end
        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println(
                "EM has not converged after $(history["iterations"]) iterations, final loglikelihood = $logtot",
            )
        end
    end

    return history
end

############## plots ##########



# plot results for an estimated Mixture Spatial Bernoulli
PlotPi_beforeafter = function (d1, d2)
    #d1 = initial, d2 = result
    probas1 = probs(d1)
    probas2 = probs(d2)

    K = length(probas1)
    p1 = scatter(1:K, probas1, label="initial", title="prior probabilities",ylim=(0,1))
    scatter!(p1, 1:K, probas2, label="estimated")

    return p1

end

PlotCovParam_beforeafter = function (d1, d2, order=false)
    K = length(probs(d1))
    rho1 = [components(d1)[k].range for k in 1:K]
    rho2 = [components(d2)[k].range for k in 1:K]


    p1 = scatter(1:K, rho1, label="initial", title="spatial range")
    scatter!(p1, 1:K, rho2, label="estimated")

    if !order
        return (p1)

    else

        nu1 = [components(d1)[k].order for k in 1:K]
        nu2 = [components(d2)[k].order for k in 1:K]


        p2 = scatter(1:K, nu1, label="initial", title="spatial regularity")
        scatter!(p2, 1:K, nu2, label="estimated")



        return plot(p1, p2)
    end

end


PlotLambda_beforeafter = function (d1, d2, order=false)
    K = length(probs(d1))
    nlocs = length(components(d1)[1])
    lambda1 = [components(d1)[k].λ for k in 1:K]
    lambda2 = [components(d2)[k].λ for k in 1:K]


    p = [scatter("λ" .* string.(1:nlocs),    lambda1[k],label="initial",title="class "*string(k),ylim=(0,1)) for k in 1:K]
     [scatter!(p[k],"λ" .* string.(1:nlocs),    lambda2[k],label="estimated",title="class "*string(k)) for k in 1:K]
    return(plot(p...))
   
end

