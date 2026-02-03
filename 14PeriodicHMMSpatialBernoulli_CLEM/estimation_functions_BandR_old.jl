# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
########### functions in progress #########
#### todo : update_TB_memory1!

function fit_mle_one_RB_memory1!(hmm, theta_R, theta_B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, solver=Optimization.LBFGS(), return_sol=false, solkwargs...)
    T = size(hmm, 3)
    size_order = size(hmm, 4)
    D = size(hmm, 2)
    # println("size(B,1) = T ? =",T)
    # println("inside updateR! - inside fit - before estim: ", theta_R)
    pairwise_indices2 = Tuple.(findall(wp .> 0))

    # Branch: Optimize only `range`, fixing `order`
    function optimfunction2(u, p)
        degP = Int((size(u, 1) / (D * size_order + 1) - 1) / 2)
        uR = u[1:2*degP+1]
        uB = reshape(u[2*degP+1+1:end], D, size_order, 2 * degP + 1)  # adjust reshape according to your actual shape


        Rt = ones(eltype(u), T)
        for t in 1:T
            Rt[t] = exp(mypolynomial_trigo(t, uR, T)) #u[1]= param for R
        end
        # println("u inside optimfun",u)
        B = [1 / (1 + exp(polynomial_trigo(t, uB[s, h, :], T))) for t = 1:T, s = 1:D, h = 1:size_order]

        return -my_loglikelihood_memory1(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2)

    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), AutoForwardDiff())
    u0 = collect(vcat(theta_R, vec(theta_B)))
    optimfunction2(u0, [Y, n_pair, h, wp])

    # println("inside updateR! - inside fit - u0: ", u0)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    # Solve the problem
    sol = solve(prob, solver; solkwargs...)

    # Check solution status
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end
    # println("inside updateR! - inside fit - after estim sol.u: ", sol.u)

    # Return the result
    degP = Int((size(sol.u, 1) / (D * size_order + 1) - 1) / 2)

    theta_R[:] .= sol.u[1:2*degP+1]
    theta_B[:, :, :] .= reshape(sol.u[2*degP+1+1:end], D, size_order, 2 * degP + 1)  # adjust reshape according to your actual shape

    # @show theta_R
end




function update_RB_memory1!(hmm::PeriodicHMMSpaMemory,
    Range_θ::AbstractArray{N,2} where {N}, theta_B::AbstractArray{N,4} where {N},
    γ::AbstractMatrix, wp, Y, Situations::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, maxiters=10)
    @argcheck size(γ, 1) == size(Y, 1)
    N = size(γ, 1)
    R = hmm.R
    D = size(hmm, 2)
    size_order = size(hmm, 4)
    K = size(R, 1)
    T = size(R, 2)
    # println("inside updateR! - before fit: ", Range_θ)
    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    # Parallelized loop
    @threads for k in 1:K
        B = hmm.B[k, :, :, :]  # B[k,t,h]
        h = hmm.h
        w = γ[:, k, :, :]
        n_pair = zeros(eltype(R), 16, D, D, T)

        @inbounds for tk in 1:N
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                w_k = w[tk, :, :]
                @views begin
                    for s in 1:16
                        n_pair[s, i, j, t] += w_k[i, j] * Situations[s, tk, i, j]

                    end

                end
            end
        end

        # Fix: Use `view` to pass mutable references
        # @show (Range_θ[ k, :])
        fit_mle_one_RB_memory1!(hmm, view(Range_θ, k, :), view(theta_B, k, :, :, :), h, Y, wp, n_pair; n2t=n2t, maxiters=maxiters)

        # @show (Range_θ[ k, :])
    end
    # println("inside updateR! - after fit: ", Range_θ)


    # Ensure in-place modification of R
    for k in 1:K
        R[k, :] .= [exp(mypolynomial_trigo(t, view(Range_θ, k, :), T)) for t = 1:T]
    end


    p = [1 / (1 + exp(polynomial_trigo(t, theta_B[k, s, h, :], T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    hmm.B[:, :, :, :] .= p
end



############################### done ##########################

function update_RB_CLEM!(hmm::PeriodicHMMSpaMemory,
    Range_θ::AbstractArray{N,2} where {N}, theta_B::AbstractArray{N,4} where {N},
    γ::AbstractArray, wp, Y, Situations::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, maxiters=10)
    @argcheck size(γ, 1) == size(Y, 1)
    N = size(γ, 1)
    R = hmm.R
    size_order = size(hmm, 4)
    K = size(R, 1)
    T = size(R, 2)
    D = size(hmm, 2)
    @show K, T
    # println("inside updateR! - before fit: ", Range_θ)
    pairwise_indices = findall(wp .> 0)
    pairwise_indices2 = [(pairwise_indices[i][1], pairwise_indices[i][2]) for i in 1:length(pairwise_indices)]

    # Parallelized loop
    @threads for k in 1:K

        @show k
        h = hmm.h
        w = γ[:, k, :, :]
        # println("B,h,w ok")
        n_pair = zeros(eltype(R), 4, D, D, T)

        @inbounds for tk in 1:N
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                w_k = w[tk, :, :]
                @views begin
                    n_pair[1, i, j, t] += w_k[i, j] * Situations[1, tk, i, j]
                    n_pair[2, i, j, t] += w_k[i, j] * Situations[2, tk, i, j]
                    n_pair[3, i, j, t] += w_k[i, j] * Situations[3, tk, i, j]
                    n_pair[4, i, j, t] += w_k[i, j] * Situations[4, tk, i, j]
                end
            end
        end
        # println("weight pairs ok")
        # Fix: Use `view` to pass mutable references
        # @show (Range_θ[ k, :])
        fit_mle_one_RB!(hmm, view(Range_θ, k, :), view(theta_B, k, :, :, :), h, Y, wp, n_pair; n2t=n2t, maxiters=maxiters)
        # @show (Range_θ[ k, :])
    end
    # println("inside updateR! - after fit: ", Range_θ)


    # Ensure in-place modification of R
    for k in 1:K
        R[k, :] .= [exp(mypolynomial_trigo(t, view(Range_θ, k, :), T)) for t = 1:T]
    end
    p = [1 / (1 + exp(polynomial_trigo(t, theta_B[k, s, h, :], T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_order]
    hmm.B[:, :, :, :] .= p
end

function my_loglikelihood(R, B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, eps=1e-10, pairwise_indices2=Tuple.(findall(wp .> 0))
)
    N, D = size(Y)
    T = size(R, 1)
    # println("T = size(R,1)",T)
    # @show R

    Iij = ones(eltype(R), 4, D, D, T)
    @inbounds for t in 1:T



        for (i, j) in pairwise_indices2
            # @show (i,j)
            B_ij = B[t, [i, j]]
            h_ij = @view h[[i, j], [i, j]]
            if i == j
                Iij[1, i, j, t] = B_ij[1]
                Iij[4, i, j, t] = 1 - B_ij[1]
            else
                Iij[1, i, j, t] = ifelse(Iij[1, j, i, t] != 1.0, Iij[1, j, i, t], norm_cdf_2d_vfast(quantile(Normal(), B_ij[1]), quantile(Normal(), B_ij[2]), exp(-h_ij[1, 2] / R[t])))
                Iij[2, i, j, t] = ifelse(Iij[3, j, i, t] != 1.0, Iij[3, j, i, t], B_ij[1] - Iij[1, i, j, t])
                Iij[3, i, j, t] = ifelse(Iij[2, j, i, t] != 1.0, Iij[2, j, i, t], B_ij[2] - Iij[1, i, j, t])
                Iij[4, i, j, t] = ifelse(i == j, 1.0 - Iij[1, i, j, t], 1.0 - Iij[1, i, j, t] - Iij[2, i, j, t] - Iij[3, i, j, t])
            end
        end
    end

    Iij .= max.(Iij, eps)  # Replace elements < eps with eps
    pairwise_sum = 0.0
    @inbounds for (i, j) in pairwise_indices2
        for t in 1:T
            if i != j
                pairwise_sum += wp[i, j] * n_pair[1, i, j, t] * log(Iij[1, i, j, t]) +
                                wp[i, j] * n_pair[2, i, j, t] * log(Iij[2, i, j, t]) +
                                wp[i, j] * n_pair[3, i, j, t] * log(Iij[3, i, j, t]) +
                                wp[i, j] * n_pair[4, i, j, t] * log(Iij[4, i, j, t])
            else
                pairwise_sum += wp[i, j] * n_pair[1, i, j, t] * log(Iij[1, i, j, t]) +
                                wp[i, j] * n_pair[4, i, j, t] * log(Iij[4, i, j, t])
            end
        end
    end
    return (pairwise_sum)
end





function fit_mle_one_RB!(hmm, theta_R, theta_B, h, Y::AbstractArray{<:Real}, wp::AbstractMatrix{<:Real}, n_pair::AbstractArray{<:Real}; n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer}, solver=Optimization.LBFGS(), return_sol=false, solkwargs...)
    T = size(hmm, 3)
    size_order = size(hmm, 4)
    D = size(hmm, 2)
    pairwise_indices2 = Tuple.(findall(wp .> 0))

    function optimfunction2(u, p)
        degP = Int((size(u, 1) / (D * size_order + 1) - 1) / 2)
        uR = u[1:2*degP+1]
        uB = reshape(u[2*degP+1+1:end], D, 1, 2 * degP + 1)


        Rt = ones(eltype(u), T)
        for t in 1:T
            Rt[t] = exp(mypolynomial_trigo(t, uR, T)) #u[1]= param for R
        end
        # println("u inside optimfun",u)
        B = [1 / (1 + exp(polynomial_trigo(t, uB[s, 1, :], T))) for t = 1:T, s = 1:D]

        return -my_loglikelihood(Rt, B, p[3], p[1], p[4], p[2]; n2t=n2t, pairwise_indices2=pairwise_indices2
        )
    end
    optf2 = OptimizationFunction((u, p) -> optimfunction2(u, p), AutoForwardDiff())
    u0 = vcat(theta_R, vec(theta_B))
    optimfunction2(u0, [Y, n_pair, h, wp])

    # println("inside updateR! - inside fit - u0: ", u0)

    prob = OptimizationProblem(optf2, u0, [Y, n_pair, h, wp])

    # Solve the problem
    sol = solve(prob, solver; solkwargs...)

    # Check solution status
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "sol.retcode = $(sol.retcode)"
    end
    # println("inside updateR! - inside fit - after estim sol.u: ", sol.u)

    # Return the result
    degP = Int((size(sol.u, 1) / (D * size_order + 1) - 1) / 2)
    @show degP


    theta_R[:] .= sol.u[1:2*degP+1]
    theta_B[:, :, :] .= reshape(sol.u[2*degP+1+1:end], D, 1, 2 * degP + 1)  # adjust reshape according to your actual shape

    # @show theta_R
end

########### functions done ###############


function update_a_CLEM!(a::AbstractVector, α::AbstractArray, β::AbstractArray, wp, pairwise_indices2=Tuple.(findall(wp .> 0)))
    @argcheck size(α, 1) == size(β, 1)
    @argcheck size(α, 2) == size(β, 2) == size(a, 1)

    K = length(a)
    normalizing = 0.0
    resu = zeros(K)

    for i in OneTo(K)
        for (ii, jj) in pairwise_indices2

            resu[i] += α[1, i, ii, jj] * β[1, i, ii, jj] * wp[ii, jj]
        end
        normalizing += resu[i]
    end
    for i in OneTo(K)
        a[i] = resu[i] / normalizing
    end
end

function update_a_CLEMbis!(a::AbstractVector, γ::AbstractArray, wp, pairwise_indices2=Tuple.(findall(wp .> 0)))

    K = length(a)
    normalizing = 0.0
    resu = zeros(K)

    for i in OneTo(K)
        for (ii, jj) in pairwise_indices2

            resu[i] += γ[1, i, ii, jj] * wp[ii, jj]
        end
        normalizing += resu[i]
    end
    for i in OneTo(K)
        a[i] = resu[i] / normalizing
    end
end



# function update_A_CLEMbis!(
#     A::AbstractArray{<:AbstractFloat,3},
#     θᴬ::AbstractArray{<:AbstractFloat,3},
#     ξ::AbstractArray,
#     s_ξ::AbstractArray,
#     α::AbstractArray,
#     β::AbstractArray,
#     LL::AbstractArray,
#     n2t::AbstractArray{Int},
#     n_in_t,
#     model_A::Model, wp, pairwise_indices2=Tuple.(findall(wp .> 0));
#     warm_start=true
# )
#     @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
#     @argcheck size(α, 2) ==
#               size(β, 2) ==
#               size(LL, 2) ==
#               size(A, 1) ==
#               size(A, 2) ==
#               size(ξ, 2) ==
#               size(ξ, 3)

#     N, K, D, D = size(LL)
#     T = size(A, 3)
#     println("begin getting weights")
#     temp = zeros(K, K)

#     ξij = zeros(N, K, K, D, D)
#     for (ii, jj) in pairwise_indices2

#         @threads for n in OneTo(N - 1)
#             t = n2t[n] # periodic t
#             m = maximum(view(LL[LL.<0], n + 1, :, :, :))
#             c = 0

#             for i in OneTo(K), j in OneTo(K)
#                 ξij[n, i, j, ii, jj] = α[n, i, ii, jj] * A[i, j, t] * exp(LL[n+1, j, ii, jj] - m) * β[n+1, j, ii, jj]
#                 c += ξij[n, i, j, ii, jj]
#             end

#             for i in OneTo(K), j in OneTo(K)
#                 ξij[n, i, j, ii, jj] /= c
#             end
#         end

#     end
#     @threads for n in 1:N
#         for i in OneTo(K), j in OneTo(K)
#             s = 0.0
#             for (ii, jj) in pairwise_indices2
#                 s += ξij[n, i, j, ii, jj] * wp[ii, jj]
#             end
#             ξ[n, i, j] = s
#         end
#     end

#     println("end getting weights")
#     ## 
#     # ξ are the filtering probablies

#     s_ξ!(s_ξ, ξ, n_in_t)


#     θ_res = pmap(k -> fit_mle_one_A(θᴬ[k, :, :], model_A, s_ξ[:, k, :]; warm_start=warm_start), 1:K)

#     for k = 1:K
#         θᴬ[k, :, :] = θ_res[k][:, :]
#     end

#     for k = 1:K, l = 1:K-1, t = 1:T
#         A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
#     end
#     for k = 1:K, t = 1:T
#         A[k, K, t] = 1  # last colum is 1/normalization (one could do otherwise)
#     end
#     normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
#     for k = 1:K, l = 1:K, t = 1:T
#         A[k, l, t] /= normalization_polynomial[k, t]
#     end
# end



# function update_A_CLEMbis!(
#     A::AbstractArray{<:AbstractFloat,3},
#     θᴬ::AbstractArray{<:AbstractFloat,3},
#     ξ::AbstractArray,
#     s_ξ::AbstractArray,
#     α::AbstractArray,
#     β::AbstractArray,
#     LL::AbstractArray,
#     n2t::AbstractArray{Int},
#     n_in_t,
#     model_A::Model, wp, pairwise_indices2=Tuple.(findall(wp .> 0));
#     warm_start=true
# )
#     @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
#     @argcheck size(α, 2) ==
#               size(β, 2) ==
#               size(LL, 2) ==
#               size(A, 1) ==
#               size(A, 2) ==
#               size(ξ, 2) ==
#               size(ξ, 3)

#     N, K, D, D = size(LL)
#     T = size(A, 3)
#     println("begin getting weights (optimized)")

#     ξ .= 0.0  # reset before accumulation

#     @threads for n in 1:(N-1)
#         @inbounds begin
#             t = n2t[n]  # periodic t
#             # the original code used `maximum(view(LL[LL.<0], ...))` which seems incorrect;
#             # it's safer to just take maximum over the relevant slice
#             m = maximum(@view LL[n+1, :, :, :])

#             for (ii, jj) in pairwise_indices2
#                 c = 0.0
#                 # first pass: compute normalization constant c
#                 for i in 1:K, j in 1:K
#                     c += α[n, i, ii, jj] * A[i, j, t] * exp(LL[n+1, j, ii, jj] - m) * β[n+1, j, ii, jj]
#                 end
#                 if c == 0.0
#                     continue  # avoid divide-by-zero
#                 end

#                 # second pass: accumulate contribution into ξ
#                 w = wp[ii, jj] / c
#                 for i in 1:K
#                     ai = α[n, i, ii, jj]
#                     for j in 1:K
#                         ξ[n, i, j] += w * ai * A[i, j, t] * exp(LL[n+1, j, ii, jj] - m) * β[n+1, j, ii, jj]
#                     end
#                 end
#             end
#         end
#     end

#     println("end getting weights")

#     ## 
#     # ξ are the filtering probablies

#     s_ξ!(s_ξ, ξ, n_in_t)


#     θ_res = pmap(k -> fit_mle_one_A(θᴬ[k, :, :], model_A, s_ξ[:, k, :]; warm_start=warm_start), 1:K)

#     for k = 1:K
#         θᴬ[k, :, :] = θ_res[k][:, :]
#     end

#     for k = 1:K, l = 1:K-1, t = 1:T
#         A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
#     end
#     for k = 1:K, t = 1:T
#         A[k, K, t] = 1  # last colum is 1/normalization (one could do otherwise)
#     end
#     normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
#     for k = 1:K, l = 1:K, t = 1:T
#         A[k, l, t] /= normalization_polynomial[k, t]
#     end
# end

# function update_A_CLEMbis!(
#     A::AbstractArray{<:AbstractFloat,3},
#     θᴬ::AbstractArray{<:AbstractFloat,3},
#     ξ::AbstractArray,
#     s_ξ::AbstractArray,
#     α::AbstractArray,
#     β::AbstractArray,
#     LL::AbstractArray,
#     n2t::AbstractArray{Int},
#     n_in_t,
#     model_A::Model,
#     wp,
#     pairwise_indices2=Tuple.(findall(wp .> 0));
#     warm_start=true
# )
#     @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
#     @argcheck size(α, 2) ==
#               size(β, 2) ==
#               size(LL, 2) ==
#               size(A, 1) ==
#               size(A, 2) ==
#               size(ξ, 2) ==
#               size(ξ, 3)

#     N, K, D1, D2 = size(LL)
#     @argcheck D1 == D2
#     D = D1
#     T = size(A, 3)

#     println("begin getting weights (fast version)")

#     # Pre-views to avoid repeated allocations
#     At_view = similar(A, K, K)      # temporary view for A[:,:,t] (will be reassigned)

#     # buffers reused per-thread
#     local_temp = zeros(K, K)
#     s_vec = zeros(K)
#     outer = zeros(K, K)

#     # Thread over n (work per time step)
#     Threads.@threads for n in 1:(N-1)
#         # each thread must have its own buffers
#         local_temp_th = copy(local_temp)
#         s_vec_th = copy(s_vec)
#         outer_th = copy(outer)

#         t = n2t[n]                       # periodic index
#         # compute stabilizer m (same as your code)
#         m = maximum(view(LL[LL.<0], n + 1, :, :, :))  # keep your original approach

#         # get A[:,:,t] view once
#         @views At = A[:, :, t]

#         # accumulate contributions for all pairs
#         @inbounds for (ii, jj) in pairwise_indices2
#             # s_j vector: length-K, note index order: LL[n+1, j, ii, jj], β[n+1, j, ii, jj]
#             @views begin
#                 # s_j = exp(LL[n+1, j, ii, jj] - m) .* β[n+1, j, ii, jj] .* wp[ii, jj]
#                 for j in 1:K
#                     s_vec_th[j] = exp(LL[n+1, j, ii, jj] - m) * β[n+1, j, ii, jj] * wp[ii, jj]
#                 end

#                 # αslice is α[n, i, ii, jj] for i=1:K
#                 # outer_th = αslice * s_vec_th'
#                 for i in 1:K
#                     let α_i = α[n, i, ii, jj]
#                         @inbounds for j in 1:K
#                             outer_th[i, j] = α_i * s_vec_th[j]
#                         end
#                     end
#                 end

#                 # multiply elementwise with At and accumulate
#                 @inbounds for i in 1:K, j in 1:K
#                     local_temp_th[i, j] += outer_th[i, j] * At[i, j]
#                 end
#             end
#         end

#         # normalize local_temp_th and write to ξ[n, :, :]
#         normalizing = sum(local_temp_th)
#         if normalizing == 0.0
#             # fallback: avoid division by 0, keep uniform small values
#             @warn "normalizing is zero at n=$n; filling uniform tiny probs"
#             normalizing = K * K * eps(Float64)
#             local_temp_th .+= eps(Float64)
#         end

#         @inbounds for i in 1:K, j in 1:K
#             ξ[n, i, j] = local_temp_th[i, j] / normalizing
#         end
#     end

#     println("end getting weights")
#     # for n in 1:10
#     #     println(sum(ξ[n, :, :]))
#     # end
#     # s_ξ aggregate across times
#     s_ξ!(s_ξ, ξ, n_in_t)

#     # Fit A using the aggregated s_ξ (keep pmap if needed)
#     θ_res = pmap(k -> fit_mle_one_A(θᴬ[k, :, :], model_A, s_ξ[:, k, :]; warm_start=warm_start), 1:K)

#     for k = 1:K
#         θᴬ[k, :, :] = θ_res[k][:, :]
#     end

#     # rebuild A from θᴬ
#     for k = 1:K, l = 1:K-1, t = 1:T
#         A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
#     end
#     for k = 1:K, t = 1:T
#         A[k, K, t] = 1
#     end
#     normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
#     for k = 1:K, l = 1:K, t = 1:T
#         A[k, l, t] /= normalization_polynomial[k, t]
#     end
# end



# function update_A_CLEMbis!(
#     A::AbstractArray{<:AbstractFloat,3},
#     θᴬ::AbstractArray{<:AbstractFloat,3},
#     ξ::AbstractArray,
#     s_ξ::AbstractArray,
#     α::AbstractArray,
#     β::AbstractArray,
#     LL::AbstractArray,
#     n2t::AbstractArray{Int},
#     n_in_t,
#     model_A::Model,
#     wp,
#     pairwise_indices2=Tuple.(findall(wp .> 0));
#     warm_start=true
# )
#     @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
#     @argcheck size(α, 2) ==
#               size(β, 2) ==
#               size(LL, 2) ==
#               size(A, 1) ==
#               size(A, 2) ==
#               size(ξ, 2) ==
#               size(ξ, 3)

#     N, K, D1, D2 = size(LL)
#     @argcheck D1 == D2
#     D = D1
#     T = size(A, 3)

#     println("begin getting weights (fast version)")

#     # Pre-views to avoid repeated allocations
#     At_view = similar(A, K, K)      # temporary view for A[:,:,t] (will be reassigned)

#     # buffers reused per-thread
#     local_temp = zeros(K, K)
#     s_vec = zeros(K)
#     outer = zeros(K, K)

#     # Thread over n (work per time step)
#     Threads.@threads for n in 1:(N-1)
#         # each thread must have its own buffers
#         local_temp_th = copy(local_temp)
#         s_vec_th = copy(s_vec)
#         outer_th = copy(outer)

#         t = n2t[n]                       # periodic index
       
#         # get A[:,:,t] view once
#         @views At = A[:, :, t]

#         # accumulate contributions for all pairs
#         @inbounds for (ii, jj) in pairwise_indices2
#              # compute stabilizer m (same as your code)
#             m = maximum(view(LL, n + 1, :, ii, jj))  # keep your original approach

#             # s_j vector: length-K, note index order: LL[n+1, j, ii, jj], β[n+1, j, ii, jj]
#             @views begin
#                 # s_j = exp(LL[n+1, j, ii, jj] - m) .* β[n+1, j, ii, jj] .* wp[ii, jj]
#                 for j in 1:K
#                     s_vec_th[j] = exp(LL[n+1, j, ii, jj] - m) * β[n+1, j, ii, jj] * wp[ii, jj]
#                 end

#                 # αslice is α[n, i, ii, jj] for i=1:K
#                 # outer_th = αslice * s_vec_th'
#                 for i in 1:K
#                     let α_i = α[n, i, ii, jj]
#                         @inbounds for j in 1:K
#                             outer_th[i, j] = α_i * s_vec_th[j]
#                         end
#                     end
#                 end

#                 # multiply elementwise with At and accumulate
#                 @inbounds for i in 1:K, j in 1:K
#                     local_temp_th[i, j] += outer_th[i, j] * At[i, j]
#                 end
#             end
#         end

#         # normalize local_temp_th and write to ξ[n, :, :]
#         normalizing = sum(local_temp_th)
#         if normalizing == 0.0
#             # fallback: avoid division by 0, keep uniform small values
#             @warn "normalizing is zero at n=$n; filling uniform tiny probs"
#             normalizing = K * K * eps(Float64)
#             local_temp_th .+= eps(Float64)
#         end

#         @inbounds for i in 1:K, j in 1:K
#             ξ[n, i, j] = local_temp_th[i, j] / normalizing
#         end
#     end

#     println("end getting weights")
#     # for n in 1:10
#     #     println(sum(ξ[n, :, :]))
#     # end
#     # s_ξ aggregate across times
#     s_ξ!(s_ξ, ξ, n_in_t)

#     # Fit A using the aggregated s_ξ (keep pmap if needed)
#     θ_res = pmap(k -> fit_mle_one_A(θᴬ[k, :, :], model_A, s_ξ[:, k, :]; warm_start=warm_start), 1:K)

#     for k = 1:K
#         θᴬ[k, :, :] = θ_res[k][:, :]
#     end

#     # rebuild A from θᴬ
#     for k = 1:K, l = 1:K-1, t = 1:T
#         A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
#     end
#     for k = 1:K, t = 1:T
#         A[k, K, t] = 1
#     end
#     normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
#     for k = 1:K, l = 1:K, t = 1:T
#         A[k, l, t] /= normalization_polynomial[k, t]
#     end
# end



# In-place forward pass, where α and c are allocated beforehand.
function forwardlog!(
    α::AbstractArray, # N*K -> N*K*D*D
    c::AbstractArray, # ? 
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractArray, pairwise_indices2;  # N*K -> N*K*D*D
    n2t=n_to_t(size(LL, 1), size(A, 3))::AbstractVector{<:Integer}
)
    @argcheck size(α, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(α, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    N, K, D, D = size(LL)

    fill!(α, 0)
    fill!(c, 0)

    for (ii, jj) in pairwise_indices2
        m = vec_maximum(view(LL, 1, :, ii, jj))

        for j in OneTo(K)
            α[1, j, ii, jj] = a[j] * exp(LL[1, j, ii, jj] - m)
            c[1, ii, jj] += α[1, j, ii, jj]
        end

        for j in OneTo(K)
            α[1, j, ii, jj] /= c[1, ii, jj]
        end

        c[1, ii, jj] = log(c[1, ii, jj]) + m

        @inbounds for n = 2:N
            tₙ₋₁ = n2t[n-1] # periodic t-1
            m = vec_maximum(view(LL, n, :, ii, jj))

            for j in OneTo(K)
                for i in OneTo(K)
                    α[n, j, ii, jj] += α[n-1, i, ii, jj] * A[i, j, tₙ₋₁]
                end
                α[n, j, ii, jj] *= exp(LL[n, j, ii, jj] - m)
                c[n, ii, jj] += α[n, j, ii, jj]
            end

            for j in OneTo(K)
                α[n, j, ii, jj] /= c[n, ii, jj]
            end

            c[n, ii, jj] = log(c[n, ii, jj]) + m
        end
    end
end


update_A_CLEMbis!(
    A::AbstractArray{<:AbstractFloat,3},
    θᴬ::AbstractArray{<:AbstractFloat,3},
    ξ::AbstractArray,
    s_ξ::AbstractArray,
    α::AbstractArray,
    β::AbstractArray,
    LL::AbstractArray,
    n2t::AbstractArray{Int},
    n_in_t,
    model_A::Nothing,
    wp,
    pairwise_indices2=Tuple.(findall(wp .> 0));
    warm_start=true
) = nothing

# In-place backward pass, where β and c are allocated beforehand.
function backwardlog!(
    β::AbstractArray,
    c::AbstractArray,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractArray, pairwise_indices2;
    n2t=n_to_t(size(LL, 1), size(A, 3))::AbstractVector{<:Integer}
)
    @argcheck size(β, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(β, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    N, K, D, D = size(LL)
    T = size(A, 3)
    L = zeros(K)
    (T == 0) && return

    fill!(β, 0)
    fill!(c, 0)
    for (ii, jj) in pairwise_indices2

        for j in OneTo(K)
            β[end, j, ii, jj] = 1
        end

        @inbounds for n = N-1:-1:1
            t = n2t[n] # periodic t
            m = vec_maximum(view(LL, n + 1, :, ii, jj))

            for i in OneTo(K)
                L[i] = exp(LL[n+1, i, ii, jj] - m)
            end

            for j in OneTo(K)
                for i in OneTo(K)
                    β[n, j, ii, jj] += β[n+1, i, ii, jj] * A[j, i, t] * L[i]
                end
                c[n+1, ii, jj] += β[n, j, ii, jj]
            end

            for j in OneTo(K)
                β[n, j, ii, jj] /= c[n+1, ii, jj]
            end

            c[n+1, ii, jj] = log(c[n+1, ii, jj]) + m
        end

        m = vec_maximum(view(LL, 1, :, ii, jj))

        for j in OneTo(K)
            c[1, ii, jj] += a[j] * exp(LL[1, j, ii, jj] - m) * β[1, j, ii, jj]
        end

        c[1, ii, jj] = log(c[1, ii, jj]) + m
    end
end
function posteriors!(γ::AbstractArray, α::AbstractArray, β::AbstractArray, pairwise_indices2)
    @argcheck size(γ) == size(α) == size(β)
    N, K, D, D = size(α)
    for (ii, jj) in pairwise_indices2

        for t in OneTo(N)
            c = 0.0
            for i in OneTo(K)
                γ[t, i, ii, jj] = α[t, i, ii, jj] * β[t, i, ii, jj]
                c += γ[t, i, ii, jj]
            end

            for i in OneTo(K)
                γ[t, i, ii, jj] /= c
            end
        end
    end
end

# function γₛ_CLEM!(γₛ, γ, n_all, wp, pairwise_indices2)
#     K, D, size_order, T, rain_cat = size(γₛ)
#     for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)
#         for k = 1:K
#             i = tup[1]
#             γₛ[k, tup...] = sum(mean([wp[i, j] * γ[n, k, i, j] for (j) in 1:D]) for n in n_all[tup...]; init=0)
#         end
#     end
# end


function mypolynomial_trigo(t, β, T)
    d = (length(β) - 1) ÷ 2
    # println("in poly trigo : d = ",d)
    # println("in poly trigo : T = ",T)

    if d == 0
        return β[1]
    else
        f = 2π / T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] + sum(β[2*l] * cos(f * l * t) + β[2*l+1] * sin(f * l * t) for l = 1:d)
    end
end


function s_ξ!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ)
    for t = 1:T
        for (k, l) in Iterators.product(1:K, 1:K)
            s_ξ[t, k, l] = sum(ξ[n, k, l] for n in n_in_t[t])
        end
    end
    # * We add ξ[N, k, l] but it should be zeros
end

function fit_mle_one_A(θᴬ, model, s_ξ; warm_start=true)
    T, K = size(s_ξ)
    pklj_jump = model[:pklj_jump]
    s_πk = model[:s_πk]
    s_πkl = model[:s_πkl]
    ## Update the smoothing parameters in the JuMP model
    for t = 1:T
        set_value(s_πk[t], sum(s_ξ[t, l] for l = 1:K))
        for l = 1:K-1
            set_value(s_πkl[t, l], s_ξ[t, l])
        end
    end
    warm_start && set_start_value.(pklj_jump, θᴬ[:, :])
    # Optimize the updated model
    optimize!(model)
    # Obtained the new parameters
    return value.(pklj_jump)
end
function update_A_CLEMbis!(
    A::AbstractArray{<:AbstractFloat,3},
    θᴬ::AbstractArray{<:AbstractFloat,3},
    ξ::AbstractArray,          # N × K × K
    s_ξ::AbstractArray,
    α::AbstractArray,
    β::AbstractArray,
    LL::AbstractArray,
    n2t::AbstractVector{Int},
    n_in_t,
    model_A::Model,
    wp::AbstractMatrix,
    pairwise_indices2=Tuple.(findall(wp .> 0));
    warm_start=true
) #chatgpt write this in desperation
    @argcheck size(ξ, 1) == size(LL, 1)
    @argcheck size(ξ, 2) == size(ξ, 3) == size(A, 1)

    N, K, _, _ = size(LL)
    T = size(A, 3)

    ξ .= 0.0  # reset accumulator

    println("begin getting weights (pairwise-correct)")

    Threads.@threads for n in 1:(N-1)
        t = n2t[n]
        At = @view A[:, :, t]

        for (ii, jj) in pairwise_indices2
            w = wp[ii, jj]

            # ---- compute unnormalized ξ_pair (K×K) ----
            ξ_pair = zeros(K, K)

            # stabilizer
            m = maximum(@view LL[n+1, :, ii, jj])

            for i in 1:K
                αi = α[n, i, ii, jj]
                for j in 1:K
                    ξ_pair[i, j] =
                        αi *
                        At[i, j] *
                        exp(LL[n+1, j, ii, jj] - m) *
                        β[n+1, j, ii, jj]
                end
            end

            Z = sum(ξ_pair)
            if Z == 0.0 || !isfinite(Z)
                continue
            end

            # ---- normalize per pair, then accumulate ----
            invZ = w / Z
            for i in 1:K, j in 1:K
                ξ[n, i, j] += ξ_pair[i, j] * invZ
            end
        end
    end

    # println("end getting weights")
    # for n in 1:10
    #     println(sum(ξ[n, :, :]))
    # end
    # ---- aggregate ξ over time ----
    s_ξ!(s_ξ, ξ, n_in_t)

    # ---- M-step for A ----
    θ_res = pmap(
        k -> fit_mle_one_A(
            θᴬ[k, :, :],
            model_A,
            s_ξ[:, k, :];
            warm_start=warm_start
        ),
        1:K
    )

    for k in 1:K
        θᴬ[k, :, :] .= θ_res[k]
    end

    # ---- rebuild A ----
    for k in 1:K, l in 1:K-1, t in 1:T
        A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
    end
    for k in 1:K, t in 1:T
        A[k, K, t] = 1
    end

    # normalize rows
    for k in 1:K, t in 1:T
        s = sum(A[k, :, t])
        A[k, :, t] ./= s
    end
end


function model_for_A(s_ξ::AbstractArray, d::Int; silence=true)
    T, K = size(s_ξ)
    @assert K > 1 "To define a transition matrix K ≥ 2, here K = $K"
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 200)
    silence && set_silent(model)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]

    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    @variable(model, pklj_jump[l=1:(K-1), j=1:(2d+1)], start = 0.01)
    # Polynomial P_kl
    @NLexpression(model, Pkl[t=1:T, l=1:K-1], sum(trig[t][j] * pklj_jump[l, j] for j = 1:length(trig[t])))

    @NLparameter(model, s_πkl[t=1:T, l=1:K-1] == s_ξ[t, l])
    #TODO? is it useful to define the extra parameter for the sum?
    @NLparameter(model, s_πk[t=1:T] == sum(s_ξ[t, l] for l = 1:K))

    @NLobjective(
        model,
        Max,
        sum(sum(s_πkl[t, l] * Pkl[t, l] for l = 1:K-1) - s_πk[t] * log1p(sum(exp(Pkl[t, l]) for l = 1:K-1)) for t = 1:T)
    )
    # To add NL parameters to the model for later use https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:s_πkl] = s_πkl
    model[:s_πk] = s_πk
    return model
end

# function loglikelihoods!(LL::AbstractArray, Iij::AbstractArray, hmm::PeriodicHMMSpaMemory, Situations::AbstractArray, pairwise_indices2; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer}, epsilon=1e-10)
#     K = size(LL)[2]
#     N = size(LL)[1]
#     T = size(hmm)[3]
#     @show Iij[1, 1, 1, 3, T]
#     @threads for t in 1:T
#         for (i, j) in pairwise_indices2
#             for k in 1:K
#                 # @show (i,j)
#                 B_ij = @view hmm.B[k, t, [i, j], 1]
#                 h_ij = @view hmm.h[[i, j], [i, j]]
#                 if i == j
#                     Iij[1, k, i, j, t] = B_ij[1]
#                     Iij[4, k, i, j, t] = 1 - B_ij[1]
#                 else
#                     compute_value = norm_cdf_2d_vfast(quantile(Normal(), B_ij[1]), quantile(Normal(), B_ij[2]), exp(-h_ij[1, 2] / hmm.R[k, t]))
#                     Iij[1, k, i, j, t] = compute_value
#                     Iij[2, k, i, j, t] = B_ij[1] - compute_value
#                     Iij[3, k, i, j, t] = B_ij[2] - compute_value
#                     Iij[4, k, i, j, t] = 1 + compute_value - B_ij[1] - B_ij[2]


#                 end
#             end
#         end
#     end
#     Iij .= max.(Iij, epsilon)  # Replace elements < eps with eps
#     @show Iij[1, 1, 1, 3, T]

#     @threads for tk in 1:N
#         for k in 1:K
#             t = n2t[tk]
#             for (i, j) in pairwise_indices2

#                 LL[tk, k, i, j] = log(Situations[1, tk, i, j] * Iij[1, k, i, j, t] + Situations[2, tk, i, j] * Iij[2, k, i, j, t] + Situations[3, tk, i, j] * Iij[3, k, i, j, t] + Situations[4, tk, i, j] * Iij[4, k, i, j, t])

#             end
#         end
#     end
# end

# #try using more precise pairwise likelihood
# function loglikelihoods!(LL::AbstractArray, Iij::AbstractArray, hmm::PeriodicHMMSpaMemory, Situations::AbstractArray, pairwise_indices2, QMC_m=100; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer}, epsilon=1e-10)
#     K = size(LL)[2]
#     N = size(LL)[1]
#     T = size(hmm)[3]
#     @show Iij[1, 1, 1, 3, T]
#     @threads for t in 1:T
#         for (i, j) in pairwise_indices2
#             for k in 1:K
#                 # @show (i,j)
#                 B_ij = @view hmm.B[k, t, [i, j], 1]
#                 h_ij = @view hmm.h[[i, j], [i, j]]
#                 if i == j
#                     Iij[1, k, i, j, t] = B_ij[1]
#                     Iij[4, k, i, j, t] = 1 - B_ij[1]
#                 else
#                     rho = exp(-h_ij[1, 2] / hmm.R[k, t])
#                     # compute_value = norm_cdf_2d_vfast(quantile(Normal(), B_ij[1]), quantile(Normal(), B_ij[2]), exp(-h_ij[1, 2] / hmm.R[k, t]))
#                     compute_value = mvnormcdf([0., 0.], [1 rho; rho 1], [-Inf, -Inf], [quantile(Normal(), B_ij[1]), quantile(Normal(), B_ij[2])]; m=2 * QMC_m)[1]
#                     Iij[1, k, i, j, t] = compute_value
#                     Iij[2, k, i, j, t] = B_ij[1] - compute_value
#                     Iij[3, k, i, j, t] = B_ij[2] - compute_value
#                     Iij[4, k, i, j, t] = 1 + compute_value - B_ij[1] - B_ij[2]


#                 end
#             end
#         end
#     end
#     Iij .= max.(Iij, epsilon)  # Replace elements < eps with eps
#     @show Iij[1, 1, 1, 3, T]

#     @threads for tk in 1:N
#         for k in 1:K
#             t = n2t[tk]
#             for (i, j) in pairwise_indices2

#                 LL[tk, k, i, j] = log(Situations[1, tk, i, j] * Iij[1, k, i, j, t] + Situations[2, tk, i, j] * Iij[2, k, i, j, t] + Situations[3, tk, i, j] * Iij[3, k, i, j, t] + Situations[4, tk, i, j] * Iij[4, k, i, j, t])

#             end
#         end
#     end
# end

#try avoiding some issues with rho not working and some proba being zero
function loglikelihoods!(LL::AbstractArray, Iij::AbstractArray,
    hmm::PeriodicHMMSpaMemory, Situations::AbstractArray,
    pairwise_indices2; n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer},
    epsilon=1e-10)

    K = size(LL, 2)
    N = size(LL, 1)
    T = size(hmm, 3)

    @threads for t in 1:T
        for (i, j) in pairwise_indices2
            for k in 1:K
                B_ij = @view hmm.B[k, t, [i, j], 1]
                h_ij = @view hmm.h[[i, j], [i, j]]

                if i == j
                    # Diagonal: trivial safe probabilities
                    p11 = clamp(B_ij[1], epsilon, 1.0 - epsilon)
                    p10 = epsilon
                    p01 = epsilon
                    p00 = clamp(1.0 - p11, epsilon, 1.0 - epsilon)
                else
                    # Off-diagonal: safe computation
                    # Clamp marginals to avoid ±Inf in quantile
                    p1 = clamp(B_ij[1], epsilon, 1 - epsilon)
                    p2 = clamp(B_ij[2], epsilon, 1 - epsilon)

                    # Safe correlation: ρ ∈ [-0.999999, 0.999999]
                    ρ = clamp(exp(-h_ij[1, 2] / hmm.R[k, t]), -0.999999, 0.999999)

                    # Joint probability P(Y_i=1,Y_j=1)
                    p11 = norm_cdf_2d_vfast(quantile(Normal(), p1),
                        quantile(Normal(), p2), ρ)

                    # Remaining probabilities
                    p10 = max(epsilon, p1 - p11)
                    p01 = max(epsilon, p2 - p11)
                    p00 = max(epsilon, 1.0 - p11 - p10 - p01)

                    # Renormalize to sum = 1
                    s = p11 + p10 + p01 + p00
                    p11 /= s
                    p10 /= s
                    p01 /= s
                    p00 /= s
                end

                # Assign to Iij
                Iij[1, k, i, j, t] = p11
                Iij[2, k, i, j, t] = p10
                Iij[3, k, i, j, t] = p01
                Iij[4, k, i, j, t] = p00
            end
        end
    end



    # Compute weighted log-likelihoods
    @threads for tk in 1:N
        for k in 1:K
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                # Sum over situations safely
                total_prob = Situations[1, tk, i, j] * Iij[1, k, i, j, t] +
                             Situations[2, tk, i, j] * Iij[2, k, i, j, t] +
                             Situations[3, tk, i, j] * Iij[3, k, i, j, t] +
                             Situations[4, tk, i, j] * Iij[4, k, i, j, t]

                # Avoid log(0)
                LL[tk, k, i, j] = log(max(total_prob, epsilon))
            end
        end
    end
end

function loglikelihoods!(LL::AbstractArray, Iij::AbstractArray,
    hmm::PeriodicHMMSpaMemory, Situations::AbstractArray,
    pairwise_indices2;
    n2t=n_to_t(size(LL, 1), size(hmm, 3))::AbstractVector{<:Integer},
    epsilon=1e-10,
    QMC_m=100)

    K = size(LL, 2)
    N = size(LL, 1)
    T = size(hmm, 3)

    # Step 1: Compute pairwise probabilities safely
    @threads for t in 1:T
        for (i, j) in pairwise_indices2
            for k in 1:K
                B_ij = @view hmm.B[k, t, [i, j], 1]
                h_ij = @view hmm.h[[i, j], [i, j]]

                if i == j
                    # Diagonal: trivial probabilities
                    p11 = clamp(B_ij[1], epsilon, 1 - epsilon)
                    p10 = epsilon
                    p01 = epsilon
                    p00 = clamp(1.0 - p11, epsilon, 1 - epsilon)
                else
                    # Off-diagonal: safe marginals
                    p1 = clamp(B_ij[1], epsilon, 1 - epsilon)
                    p2 = clamp(B_ij[2], epsilon, 1 - epsilon)

                    # Safe correlation
                    ρ = clamp(exp(-h_ij[1, 2] / hmm.R[k, t]), -0.999999, 0.999999)

                    # Bivariate probability using mvnormcdf
                    p11 = mvnormcdf(
                        [0.0, 0.0],
                        [1.0 ρ; ρ 1.0],
                        [-Inf, -Inf],
                        [quantile(Normal(), p1), quantile(Normal(), p2)];
                        m=2 * QMC_m
                    )[1]

                    # Remaining probabilities
                    p10 = max(epsilon, p1 - p11)
                    p01 = max(epsilon, p2 - p11)
                    p00 = max(epsilon, 1.0 - p11 - p10 - p01)

                    # Renormalize
                    s = p11 + p10 + p01 + p00
                    p11 /= s
                    p10 /= s
                    p01 /= s
                    p00 /= s
                end

                # Assign to Iij
                Iij[1, k, i, j, t] = p11
                Iij[2, k, i, j, t] = p10
                Iij[3, k, i, j, t] = p01
                Iij[4, k, i, j, t] = p00
            end
        end
    end

    # Step 2: Compute log-likelihoods safely
    @threads for tk in 1:N
        for k in 1:K
            t = n2t[tk]
            for (i, j) in pairwise_indices2
                total_prob = Situations[1, tk, i, j] * Iij[1, k, i, j, t] +
                             Situations[2, tk, i, j] * Iij[2, k, i, j, t] +
                             Situations[3, tk, i, j] * Iij[3, k, i, j, t] +
                             Situations[4, tk, i, j] * Iij[4, k, i, j, t]

                LL[tk, k, i, j] = log(max(total_prob, epsilon))
            end
        end
    end
end

function fit_mle_CLEM!(
    hmm::PeriodicHMMSpaMemory,
    thetaA::AbstractArray{<:AbstractFloat,3},
    thetaB::AbstractArray{<:AbstractFloat,4},
    thetaR::AbstractArray{<:AbstractFloat,2}, Y::AbstractArray{<:Bool},
    Y_past::AbstractArray{<:Bool};
    n2t=n_to_t(size(Y, 1), size(hmm, 3))::AbstractVector{<:Integer},
    display=:none,
    maxiter=100,
    tol=1e-3,
    robust=false,
    silence=true,
    warm_start=true,
    tdist=1,
    maxiters_R=10, wp=1.0 .* (hmm.h .< maximum(hmm.h) * tdist), QMC_m=100
)
    # println("tdist = ",tdist)
    N, K, T, size_order, D = size(Y, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4), size(hmm, 2)
    @show N, K, T, size_order, D



    deg_A = (size(thetaA, 3) - 1) ÷ 2
    deg_B = (size(thetaB, 4) - 1) ÷ 2
    # println("wp= ",wp)
    rain_cat = 2 # dry or wet
    @argcheck T == size(hmm.B, 2)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => Float64[])

    all_thetaA_iterations = [copy(thetaA)]
    all_thetaB_iterations = [copy(thetaB)]



    all_thetaR_iterations = [copy(thetaR)]

    # Allocate order for in-place updates
    c = zeros(N, D, D)
    α = zeros(N, K, D, D) #forward ? but ij
    β = zeros(N, K, D, D) #backward ?
    γ = zeros(N, K, D, D) # regular smoothing proba

    ξ = zeros(N, K, K) # sum_ wij pi_kl(t)^ij ?
    s_ξ = zeros(T, K, K) #? somme pi_kl(t) pour t de même périodicité

    LL = zeros(N, K, D, D) # value of pariwise likelihood for each time step
    Iij = zeros(4, K, D, D, T) #possible values of pairwise for all times in periodic
    # store the integral values for the pairwise loglikelihood for each state
    # assign category for observation depending in the Y_past Y


    order = Int(log2(size_order))
    lag_cat = conditional_to(Y, Y_past)

    n_in_t = [findall(n2t .== t) for t = 1:T]
    n_occurence_history = [findall(.&(Y[:, j] .== y, lag_cat[:, j] .== h)) for j = 1:D, h = 1:size_order, y = 0:1] # dry or wet
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_order, 1:T, 1:rain_cat)]

    pairwise_indices2 = Tuple.(findall(wp .> 0))

    model_A = K ≥ 2 ? model_for_A(s_ξ[:, 1, :], deg_A, silence=silence) : nothing # JuMP Model for transition matrix





    # generate situations
    if size_order == 1
        Situations = zeros(Int, 4, N, D, D)

        for k in 1:N
            for i in 1:D
                for j in 1:D
                    Situations[1, k, i, j] = (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[2, k, i, j] = (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[3, k, i, j] = (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[4, k, i, j] = (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0
                end
            end
        end
    elseif size_order == 2
        # generate situations
        Situations = zeros(Int, 16, N, D, D)

        for k in 2:N
            for i in 1:D
                for j in 1:D
                    Situations[1, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[2, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[3, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[4, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[5, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[6, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[7, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[8, k, i, j] = (Y[k-1, i] == 1 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[9, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[10, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[11, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[12, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 1) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0

                    Situations[13, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 1) ? 1 : 0
                    Situations[14, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 1 && Y[k, j] == 0) ? 1 : 0
                    Situations[15, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 1) ? 1 : 0
                    Situations[16, k, i, j] = (Y[k-1, i] == 0 && Y[k-1, j] == 0) && (Y[k, i] == 0 && Y[k, j] == 0) ? 1 : 0
                end
            end
        end
        for i in 1:D
            for j in 1:D
                Situations[1, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[2, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[3, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[4, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[5, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[6, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[7, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[8, 1, i, j] = (Y_past[i] == 1 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[9, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[10, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[11, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[12, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 1) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0

                Situations[13, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 1) ? 1 : 0
                Situations[14, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 1 && Y[1, j] == 0) ? 1 : 0
                Situations[15, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 1) ? 1 : 0
                Situations[16, 1, i, j] = (Y_past[i] == 0 && Y_past[j] == 0) && (Y[1, i] == 0 && Y[1, j] == 0) ? 1 : 0
            end
        end
    elseif size_order > 2
        println("memory of more than 2 not yet implemented for the mle estimation")
        return

    end
    println("Situations generated")



    loglikelihoods!(LL, Iij, hmm, Situations, pairwise_indices2, QMC_m=QMC_m; n2t)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))




    forwardlog!(α, c, hmm.a, hmm.A, LL, pairwise_indices2; n2t=n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL, pairwise_indices2; n2t=n2t)

    posteriors!(γ, α, β, pairwise_indices2)
    τ = 1.
    γ .= γ .^ τ
    γ ./= sum(γ, dims=2)

    @show c[T, 1, 3]
    @show norm(hmm.B[1, :, :] .- hmm.B[2, :, :])

    logtot = sum(sum(c[i, :, :] .* wp) for i in 1:N)


    (display == :iter) && println("Iteration 0: composite logtot = $logtot")
    println("Iteration 0: composite logtot = $logtot")

    for it = 1:maxiter
        println(it)
        @show mean_gamma_per_state(γ, wp)

        @assert all(isfinite, hmm.R)
        @assert all(0 .< hmm.B .< 1)
        @assert all(isfinite, LL)
        @assert all(isfinite, γ[:,:,wp.>0])
        @assert all(isfinite, ξ)
        #  update_a_CLEM!(hmm.a, α, β, wp)
        update_a_CLEMbis!(hmm.a, γ, wp)


        # DONE :need to check update_A
        update_A_CLEMbis!(hmm.A, thetaA, ξ, s_ξ, α, β, LL, n2t, n_in_t, model_A, wp, pairwise_indices2; warm_start=warm_start)
        println("done updating A")
        push!(all_thetaA_iterations, copy(thetaA))
        @show sum(γ[T, :, 1, 3])
        @show sum(ξ[T, :, :])

        #######

        if size_order == 1
            update_RB_CLEM!(hmm, thetaR, thetaB, γ, wp, Y, Situations; n2t=n2t, maxiters=maxiters_R)

        elseif size_order == 2
            update_RB_memory1_CLEM!(hmm, thetaR, thetaB, γ, wp, Y, Situations; n2t=n2t, maxiters=maxiters_R)

            # update_R_memory1!(hmm, thetaR, γ, wp, Y, Situations; n2t=n2t, maxiters=maxiters_R)
        end
        push!(all_thetaR_iterations, copy(thetaR))
        push!(all_thetaB_iterations, copy(thetaB))
        @show norm(hmm.B[1, :, :] .- hmm.B[2, :, :])


        println("done updating R")

        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check all(t -> istransmat(hmm.A[:, :, t]), 1:T)



        # loglikelihoods!(LL, hmm, Y, n2t)
        loglikelihoods!(LL, Iij, hmm, Situations, pairwise_indices2, QMC_m=QMC_m; n2t) #only written for memory 0, no 1.
        @show Iij[1, 1, 1, 3, T]
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

        forwardlog!(α, c, hmm.a, hmm.A, LL, pairwise_indices2; n2t=n2t)
        backwardlog!(β, c, hmm.a, hmm.A, LL, pairwise_indices2; n2t=n2t)
        @show c[T, 1, 3]
        posteriors!(γ, α, β, pairwise_indices2)
        γ .= γ .^ τ
        γ ./= sum(γ, dims=2)

        println("A rows:")
        @show hmm.A[1, :, 1]
        @show hmm.A[2, :, 1]

        logtotp = sum(sum(c[i, :, :] .* wp) for i in 1:N)


        if display == :iter
            ΔmaxA = round(maximum(abs, (all_thetaA_iterations[it+1] - all_thetaA_iterations[it]) ./ all_thetaA_iterations[it]), digits=5)
            ΔmaxB = round(maximum(abs, (all_thetaB_iterations[it+1] - all_thetaB_iterations[it]) ./ all_thetaB_iterations[it]), digits=5)
            ΔmaxR = round(maximum(abs, (all_thetaR_iterations[it+1] - all_thetaR_iterations[it]) ./ all_thetaR_iterations[it]), digits=5)
            println("Iteration $it: logtot = $(round(logtotp, digits = 6)), max(|θᴬᵢ-θᴬᵢ₋₁|/|θᴬᵢ₋₁|) = ", ΔmaxA, " & max(|θᴮᵢ-θᴮᵢ₋₁|/|θᴮᵢ₋₁|) = ", ΔmaxB, " & max(|θRᵢ-θRᵢ₋₁|/|θRᵢ₋₁|) = ", ΔmaxR)
            # flush(stdout)
        end

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if (ΔmaxA < tol && ΔmaxB < tol && ΔmaxR < tol)
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end
        if abs((logtotp - logtot) / logtotp) > tol && logtotp < logtot
            (display in [:iter, :final]) &&
                println("stop the loglikelihood has deacreased dramatically")
            history["converged"] = false
            break
        end
        logtot = logtotp

    end
    # if abs((logtotp - logtot) / logtotp) > tol && logtotp < logtot
    #     (display in [:iter, :final]) &&
    #         println("stop the loglikelihood has deacreased dramatically")
    #     history["converged"] = false
    #     break
    # end

    if !history["converged"]
        if display in [:iter, :final]
            println("EM has not converged after $(history["iterations"]) iterations, logtot = $logtot")
        end
    end
    history, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations

end

function mean_gamma_per_state(γ, wp)
    N, K, D, _ = size(γ)
    g = zeros(K)
    W = sum(wp)

    for n in 1:N, k in 1:K, i in 1:D, j in 1:D
        if wp[i,j] > 0
            g[k] += wp[i,j] * γ[n, k, i, j]
        end
    end

    g ./= (N * W)
    return g
end



include("/home/caroline/Gitlab_SWG_Caro/hmmspa/utils/fast_bivariate_cdf.jl")

begin
    interleave2(args...) = collect(Iterators.flatten(zip(args...)))

    remaining(N::Int) = N > 0 ? range(1, length=N) : Int64[]

    argmaxrow(A::AbstractMatrix{<:Real}) = [argmax(A[i, :]) for i = axes(A, 1)]


    n_per_category(s, h, t, y, n_in_t, n_occurence_history) = (n_in_t[t] ∩ n_occurence_history[s, h, y])

    bin2digit(x) = sum(x[length(x)-i+1] * 2^(i - 1) for i = 1:length(x)) + 1
    bin2digit(x::Tuple) = bin2digit([x...])

    function dayx(lag_obs::AbstractArray)
        order = length(lag_obs)
        t = tuple.([lag_obs[m] for m = 1:order]...)
        bin2digit.(t)
    end

    function conditional_to(Y::AbstractArray{<:Bool}, Y_past::AbstractArray{<:Bool})
        order = size(Y_past, 1)
        if order == 0
            return ones(Int, size(Y))
        else
            lag_obs = [copy(lag(Y, m)) for m = 1:order]  # transform dry=0 into 1 and wet=1 into 2 for array indexing
            for m = 1:order
                lag_obs[m][1:m, :] .= reverse(Y_past[1:m, :], dims=1) # avoid the missing first row
            end
            return dayx(lag_obs)
        end
    end

   
end



# function update_A_CLEM!(
#     A::AbstractArray{<:AbstractFloat,3},
#     θᴬ::AbstractArray{<:AbstractFloat,3},
#     ξ::AbstractArray,
#     s_ξ::AbstractArray,
#     α::AbstractArray,
#     β::AbstractArray,
#     LL::AbstractArray,
#     n2t::AbstractArray{Int},
#     n_in_t,
#     model_A::Model, wp, pairwise_indices2=Tuple.(findall(wp .> 0));
#     warm_start=true
# )
#     @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
#     @argcheck size(α, 2) ==
#               size(β, 2) ==
#               size(LL, 2) ==
#               size(A, 1) ==
#               size(A, 2) ==
#               size(ξ, 2) ==
#               size(ξ, 3)

#     N, K, D, D = size(LL)
#     T = size(A, 3)
#     println("begin getting weights")
#     temp = zeros(K, K)

#     @threads for n in OneTo(N - 1)
#         t = n2t[n] # periodic t
#         m = maximum(view(LL[LL.<0], n + 1, :, :, :))

#         normalizing = 0.0
#         # resu = zeros(N, K, K)
#         local_temp = similar(temp)
#         fill!(local_temp, 0.0)        # clear the K×K buffer

#         for i in OneTo(K), j in OneTo(K)
#             for (ii, jj) in pairwise_indices2
#                 local_temp[i, j] += α[n, i, ii, jj] * A[i, j, t] * exp(LL[n+1, j, ii, jj] - m) * β[n+1, j, ii, jj] * wp[ii, jj]
#             end
#             normalizing += local_temp[i, j]
#         end

#         for i in OneTo(K), j in OneTo(K)
#             ξ[n, i, j] = local_temp[i, j] / normalizing
#         end



#     end
#     println("end getting weights")
#     ## 
#     # ξ are the filtering probablies

#     s_ξ!(s_ξ, ξ, n_in_t)


#     θ_res = pmap(k -> fit_mle_one_A(θᴬ[k, :, :], model_A, s_ξ[:, k, :]; warm_start=warm_start), 1:K)

#     for k = 1:K
#         θᴬ[k, :, :] = θ_res[k][:, :]
#     end

#     for k = 1:K, l = 1:K-1, t = 1:T
#         A[k, l, t] = exp(polynomial_trigo(t, θᴬ[k, l, :], T))
#     end
#     for k = 1:K, t = 1:T
#         A[k, K, t] = 1  # last colum is 1/normalization (one could do otherwise)
#     end
#     normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
#     for k = 1:K, l = 1:K, t = 1:T
#         A[k, l, t] /= normalization_polynomial[k, t]
#     end
# end