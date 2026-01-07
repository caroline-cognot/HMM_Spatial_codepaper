using Pkg
Pkg.activate("HMMSPAcodepaper")
Pkg.instantiate()

begin
    include("../13PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")
    Random.seed!(0)
    # ## Utilities
    using ArgCheck
    using Base: OneTo
    using ShiftedArrays: lead, lag
    using Distributed
    # ## Optimization
    using JuMP, Ipopt
    using Optimization, OptimizationMOI
    using LsqFit
    using Dates
    using LaTeXStrings
    using Profile
    using BenchmarkTools
    using CSV
    using DataFrames
    include("../13PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl")
    # means we are using the EM (and not CLEM this is not in this git),  and estimating B then R. Change the file name if you want to maximise for B and R simultaneously.
    using JLD2
end
   import StochasticWeatherGenerators.dayofyear_Leap

include("../SpatialBernoulli/SpatialBernoulli.jl")
station_50Q = CSV.read("./00data/transformedECAD_stations.csv",DataFrame)
Yobs=Matrix(CSV.read("./00data/transformedECAD_Yobs.csv",header=false,DataFrame))
my_distance =Matrix(CSV.read("./00data/transformedECAD_locsdistances.csv",header=false,DataFrame))

my_locations = hcat(station_50Q.LON_idx, station_50Q.LAT_idx)
heatmap(my_distance)
nlocs = length(my_locations[:, 1])
my_D = size(my_locations, 1)


include("../13PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl")



            
                 

                 

                 


date_start = Date(1973)
date_end = Date(2024) - Day(1)
every_year = date_start:Day(1):date_end
my_N = length(every_year)
n2t = dayofyear_Leap.(every_year)

tdist = 0.3
maxiter = 100
my_T = 366 # Period


Threads.nthreads()
doss_save = "./13PeriodicHMMSpatialBernoulli/res_real_data/"

for my_autoregressive_order in 0:1 #do not do 1 again yet.

for QMC_m in [30]

        for my_K in 1:5
            for my_degree_of_P in 0:2
                for R0 in [500]







                    my_size_order = 2^my_autoregressive_order
                    my_size_degree_of_P = 2 * my_degree_of_P + 1

                    datafile = doss_save * "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"

                    if !isfile(datafile)


                        # copy fit_mle with different names - from the code in update_A_B_jump.jl
                        Y = convert(Array{Bool}, Yobs)'

                        D = size(my_locations, 1)
                        Y_past = rand(Bool, my_autoregressive_order, D)

                        println("K = $my_K, ", "local_order = $my_autoregressive_order, ", "degree = $my_degree_of_P")


                        Yall = convert(Array{Bool}, Yobs)'
                        Y_past = rand(Bool, my_autoregressive_order, my_D)
                        ξ = [1; zeros(my_K - 1)]  # 1 jan 1956 was most likely a type Z = 1 wet day all over France
                        # Y = Yall[1+my_autoregressive_order:end, :]
                        Y = Yall
                        ref_station = 1

                        begin
                            hmm_random = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order; ξ=ξ, ref_station=ref_station)

                            @time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13])

                            θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)

                            @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
                                maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)

                            thetaA = θq_fit
                            thetaB = θy_fit


                            # now on to the next part : spatial model ----------------------------------------------------
                            thetaR = zeros(my_K, my_size_degree_of_P)

                            thetaR[:, 1] .= log(R0)
                            hmm = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)


                            p1, p2, p3 = PlotModel(hmm; indices_sta=[1, 3, 5, 7])
                            plot(p1, p2, p3, layout=@layout [a b; c]; size=(1000, 1000))


                            @time history, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations = fit_mle!(hmm, thetaA, thetaB, thetaR, Y, Y_past; n2t=n2t, maxiter=maxiter, tol=1e-3, maxiters_R=100, display=:iter, tdist=tdist, QMC_m=QMC_m)
                            pp1 = plot(history["logtots"])
                            savefig(pp1, doss_save * "logtots/K" * string(my_K) * "_t" * string(tdist) * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * "logtots.png")
                            save(doss_save * "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2", Dict("hmm" => hmm, "logtots" => history["logtots"], "all_thetaA_iterations" => all_thetaA_iterations, "all_thetaB_iterations" => all_thetaB_iterations, "all_thetaR_iterations" => all_thetaR_iterations))

                        end  
                    end  
                        all_thetaR_iterations = load(datafile)["all_thetaR_iterations"]
                        all_thetaB_iterations = load(datafile)["all_thetaB_iterations"]
                         all_thetaA_iterations = load(datafile)["all_thetaA_iterations"]
                         logtots = load(datafile)["logtots"]
                        @show length(all_thetaA_iterations)
                        @show length(logtots)
                        iter=argmax(logtots)
                        @show iter                        

                            R_all = [exp(mypolynomial_trigo(t, all_thetaR_iterations[iter+1][k, :]', my_T)) for k in 1:my_K, t = 1:my_T]
                            B_all = [1 / (1 + exp(polynomial_trigo(t, all_thetaB_iterations[iter+1][k, s, h, :], my_T))) for k = 1:my_K, t = 1:my_T, s = 1:my_D, h = 1:my_size_order] 
                            
                            if my_K == 1
                                all_A = ones(my_K, my_K, my_T)
                            else
                                all_A = zeros(my_K, my_K, my_T)
                                    for k = 1:my_K, l = 1:my_K-1, t = 1:my_T
                                        all_A[ k, l, t] = exp(polynomial_trigo(t, all_thetaA_iterations[iter+1][k, l, :], my_T))
                                    end
                                    for k = 1:my_K, t = 1:my_T
                                        all_A[ k, my_K, t] = 1  # last colum is 1/normalization (one could do otherwise)
                                    end
                                    normalization_polynomial = [1 + sum(all_A[k, l, t] for l = 1:my_K-1) for k = 1:my_K, t = 1:my_T]
                                    for k = 1:my_K, l = 1:my_K, t = 1:my_T
                                        all_A[ k, l, t] /= normalization_polynomial[k, t]
                                    end
                            end
                            hmm= PeriodicHMMSpaMemory(fill(1/my_K,my_K), all_A[ :, :, :], R_all, B_all, my_distance)
                            save(doss_save * "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2", Dict("hmm" => hmm, "logtots" => logtots, "all_thetaA_iterations" => all_thetaA_iterations, "all_thetaB_iterations" => all_thetaB_iterations, "all_thetaR_iterations" => all_thetaR_iterations))



                            begin
                                pr = plot(title=" range ", ylim=(200, 800))
                                for k in 1:my_K

                                    plot!(pr, 1:my_T, hmm.R[k, :], c=k, label=k)

                                end



                                p1 = plot(title=" proba rain Lille", ylim=(0, 1))
                                p2 = plot(title=" proba rain Nice", ylim=(0, 1))

                                for k in 1:my_K

                                    plot!(p1, 1:my_T, hmm.B[k, :, 6, 1], c=k, label=:none)
                                    plot!(p2, 1:my_T, hmm.B[k, :, 16, 1], c=k, label=:none)

                                    if my_autoregressive_order > 0
                                        plot!(p1, 1:my_T, hmm.B[k, :, 6, 2], c=k, label=:none, linestyle=:dash)

                                        plot!(p2, 1:my_T, hmm.B[k, :, 16, 2], c=k, label=:none, linestyle=:dash)
                                    end

                                end
                                plot!(p2, [NaN], [NaN], label="No rain previous day", linestyle=:solid, c=:black)

                                plot!(p1, [NaN], [NaN], label="No rain previous day", linestyle=:solid, c=:black)
                                if my_autoregressive_order > 0
                                    plot!(p1, [NaN], [NaN], label="Rain previous day", linestyle=:dash, c=:black)

                                    plot!(p2, [NaN], [NaN], label="Rain previous day", linestyle=:dash, c=:black)
                                end


                                begin
                                    pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing) for k in 1:my_K]
                                    for k in 1:my_K
                                        [plot!(pA[k], hmm.A[k, l, :], c=l, label=L"Q_{%$(k)\to %$(l)}", legend=:topleft) for l in 1:my_K]

                                        hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
                                        ylims!(0, 1)
                                    end
                                    pallA = plot(pA..., size=(1000, 500))
                                end

                                p = plot(p1, p2, size=(1000, 500))
                            end
                            savefig(plot(pr, pallA, p, layout=@layout([a b; c])), doss_save * "/parameterplot/K" * string(my_K) * "_t" * string(tdist) * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * "plotparams.png")
                        end
                    

                    # commented code : making with triangulated pairs (useless - results are sensibly the same)

                    # datafile = doss_save * "/parameters/tripair_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"

                    # if !isfile(datafile)

                    #     begin
                    #         wp = zeros(size(my_locations)[1], size(my_locations)[1])
                    #         using DelaunayTriangulation

                    #         points = [
                    #             (my_locations[:, 1][j], my_locations[:, 2][j]) for j in 1:size(my_locations)[1]
                    #         ]
                    #         tri = triangulate(points)



                    #         for i in 1:size(my_locations)[1]
                    #             set = get_neighbours(tri, i)
                    #             for j in set
                    #                 if j > 0
                    #                     wp[i, j] = 1.0 / length(set)
                    #                 end
                    #             end
                    #         end

                    #         hmm_random = randARPeriodicHMM(my_K, my_T, my_D, my_autoregressive_order; ξ=ξ, ref_station=ref_station)

                    #         @time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13])

                    #         θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, my_degree_of_P)

                    #         @time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
                    #             maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t)

                    #         thetaA = θq_fit
                    #         thetaB = θy_fit


                    #         # now on to the next part : spatial model ----------------------------------------------------
                    #         thetaR = zeros(my_K, my_size_degree_of_P)

                    #         thetaR[:, 1] .= log(R0)
                    #         hmm = Trig2PeriodicHMMspaMemory(fill(1 / my_K, my_K), thetaA, thetaB, thetaR, my_T, my_distance)


                    #         p1, p2, p3 = PlotModel(hmm; indices_sta=[1, 3, 5, 7])
                    #         plot(p1, p2, p3, layout=@layout [a b; c]; size=(1000, 1000))


                    #         @time history, all_thetaA_iterations, all_thetaB_iterations, all_thetaR_iterations = fit_mle!(hmm, thetaA, thetaB, thetaR, Y, Y_past; n2t=n2t, maxiter=maxiter, tol=1e-4, maxiters_R=100, display=:iter, tdist=tdist, QMC_m=QMC_m, wp=wp)
                    #         save(doss_save * "/parameters/tripair_K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2", Dict("hmm" => hmm, "logtots" => history["logtots"], "all_thetaA_iterations" => all_thetaA_iterations, "all_thetaB_iterations" => all_thetaB_iterations, "all_thetaR_iterations" => all_thetaR_iterations))
                    #         pp1 = plot(history["logtots"])
                    #         savefig(pp1, doss_save * "logtots/tripair_K" * string(my_K) * "_t" * string(tdist) * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * "logtots.png")



                    #         begin
                    #             pr = plot(title=" range ", ylim=(200, 800))
                    #             for k in 1:my_K

                    #                 plot!(pr, 1:my_T, hmm.R[k, :], c=k, label=k)

                    #             end



                    #             p1 = plot(title=" proba rain Lille", ylim=(0, 1))
                    #             p2 = plot(title=" proba rain Nice", ylim=(0, 1))

                    #             for k in 1:my_K

                    #                 plot!(p1, 1:my_T, hmm.B[k, :, 6, 1], c=k, label=:none)
                    #                 plot!(p2, 1:my_T, hmm.B[k, :, 16, 1], c=k, label=:none)

                    #                 if my_autoregressive_order > 0
                    #                     plot!(p1, 1:my_T, hmm.B[k, :, 6, 2], c=k, label=:none, linestyle=:dash)

                    #                     plot!(p2, 1:my_T, hmm.B[k, :, 16, 2], c=k, label=:none, linestyle=:dash)
                    #                 end

                    #             end
                    #             plot!(p2, [NaN], [NaN], label="No rain previous day", linestyle=:solid, c=:black)

                    #             plot!(p1, [NaN], [NaN], label="No rain previous day", linestyle=:solid, c=:black)
                    #             if my_autoregressive_order > 1
                    #                 plot!(p1, [NaN], [NaN], label="Rain previous day", linestyle=:dash, c=:black)

                    #                 plot!(p2, [NaN], [NaN], label="Rain previous day", linestyle=:dash, c=:black)
                    #             end


                    #             begin
                    #                 pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing) for k in 1:my_K]
                    #                 for k in 1:my_K
                    #                     [plot!(pA[k], hmm.A[k, l, :], c=l, label=L"Q_{%$(k)\to %$(l)}", legend=:topleft) for l in 1:my_K]

                    #                     hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
                    #                     ylims!(0, 1)
                    #                 end
                    #                 pallA = plot(pA..., size=(1000, 500))
                    #             end

                    #             p = plot(p1, p2, size=(1000, 500))
                    #         end
                    #         savefig(plot(pr, pallA, p, layout=@layout([a b; c])), doss_save * "/parameterplot/tripair_K" * string(my_K) * "_t" * string(tdist) * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * "plotparams.png")
                    #     end
                    # end

                
            end
        end
    end
end



#############################################################

# Make the ICL for each model   - give : K=4,deg=1
# ICL 
# Complete likelihood (smaller QMC_m)


# make many viterbi with "small" number of m 
dficl = DataFrame(
    K=Int[],
    deg=Int[],
    memory=Int[],
    R0=Int[],
    QMC_m=Int[],
    n_param=Int[],
    LC=Float64[],
    ICL=Float64[], rep=Int[]
)

for my_K in 1:5
    for my_autoregressive_order in 0:1
        for my_degree_of_P in 0:2
            QMC_m = 30
            
            R0 = 500
        
            my_T = 366 # Period

            my_size_order = 2^my_autoregressive_order
            my_size_degree_of_P = 2 * my_degree_of_P + 1
            Y_past = rand(Bool, my_autoregressive_order, D)
            Y = convert(Array{Bool}, Yobs)
            @show(my_autoregressive_order, my_K, my_degree_of_P, QMC_m, R0)

            datafile = doss_save * "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"
            Y_past = rand(Bool, my_autoregressive_order, D)
            Y = convert(Array{Bool}, Yobs)
          
                    hmmspa = load(datafile)["hmm"]

                
                for rep in 1
                    ẑ = viterbi(hmmspa, Y, Y_past; n2t=n2t, QMC_m=30)
                    ẑ_per_cat = [findall(ẑ .== k) for k in 1:my_K]
                    CLKdegp = complete_loglikelihood(hmmspa, Yobs, Y_past, ẑ; n2t=n2t, QMC_m=30)
                    n_param_val = nb_param_HMMSpa(my_K, my_autoregressive_order, my_degree_of_P, D)
                    icl_val = CLKdegp - log(length(ẑ)) / 2 * n_param_val

                    push!(dficl, (K=my_K, deg=my_degree_of_P, memory=my_autoregressive_order, LC=CLKdegp, n_param=n_param_val, ICL=icl_val, R0=R0, QMC_m=QMC_m, rep=rep))

                end
            
        end
    end
end






CSV.write(doss_save * "/ICLmemory01_m30_endll.csv", dficl)
