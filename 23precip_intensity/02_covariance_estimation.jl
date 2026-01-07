using JLD2
using CSV
using DataFrames
include("EGPD_functions.jl")
include("EGPD_class.jl")
include("covariance_functions.jl")
using Distributions

z_hat = CSV.read("./00data/transformedECAD_zhat.csv", DataFrame, header=false)[:, 1]

N = length(z_hat)
# z_hat=fill(1,N) #test if setting this to 1 gives same result. as previous no-class periodic estimation.
# z_hat[z_hat.==4] .=3
K = length(unique(z_hat))
Robs = Matrix(CSV.read("./00data/transformedECAD_Robs.csv", DataFrame, header=false)[:, :])'
D = size(Robs, 2)
locsdata = CSV.read("./00data/transformedECAD_stations.csv", DataFrame, header=true)
locations = Matrix(CSV.read("./00data/transformedECAD_locs.csv", DataFrame, header=false))
locsdata.LON = locations[:, 1]
locsdata.LAT = locations[:, 2]
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
    datafile = "./13PeriodicHMMSpatialBernoulli/res_real_data/" * "/parameters/K" * string(my_K) * "_resu_" * string(length(n2t)) * "days" * "_degree" * string(my_degree_of_P) * "_maxiter" * string(maxiter) * "_m" * string(my_autoregressive_order) * "_R0" * string(R0) * "_QMCm" * string(QMC_m) * ".jld2"


    hmmspa = load(datafile)["hmm"]


end
T = 366

##########"" get pdry[s,t] for all s all t as observed in the data/
degree_of_P = 2
XR = Matrix(CSV.read("./23precip_intensity/res_real_data/EGPD_periodic_K_normalized" * string(K) * string(degree_of_P) * ".csv", DataFrame, header=false))'
D = size(XR, 1)
Yobs = Matrix(CSV.read("./00data/transformedECAD_Yobs.csv", DataFrame, header=false)[:, :])'
Yprevious = fill(1, 1, size(hmmspa, 2))
Ymoins1 = vcat(Yprevious, Yobs)[1:my_N, :]
Pdry = similar(XR)
for s in 1:D
    for n in 1:my_N
        t = n2t[n]
        k = z_hat[n]
        h = Ymoins1[n, s]
        Pdry[s, n] = 1 - hmmspa.B[k, t, s, h+1]
    end
end
phimPdry = quantile.(Normal(), Pdry)
data = XR
Mat_h
# fitted=fit_mple(data,Mat_h,model_type,param0,maxiter=20)
include("/home/caroline/Gitlab_SWG_Caro/hmmspa/utils/fast_bivariate_cdf.jl")
# using StatsFuns
#chatgpt ay - never worked...
# function fit_mple(data, Mat_h, model_type::Type{<:CovarianceStructure}, param0, phimPdry; lower=nothing, upper=nothing, maxiter=100, maxdist=maximum(Mat_h) / 3 * 2, maxtime=10)

#     function pairwise_loglik(data, model_type::Type{<:CovarianceStructure}, param, phimPdry, Mat_h; maxdist=maximum(Mat_h) / 3 * 2, maxtime=10)
#         D, Nt = size(data)

#         space_mask = Mat_h .<= maxdist

#         # Create a matrix for the observed data values

#         # Precompute covariance matrix
#         covmodel = model_type(1.0, param...,)
#         cov_matrixes = [cov_spatiotemporal(covmodel, Mat_h, timediff) for timediff in 0:maxtime] # D x D matrix of covariances (spatial)

#         c11 = cov_matrixes[1][1, 1] # variance
#         partial_sums = zeros(eltype(cov_matrixes[1]), nthreads())

#         #  for t1 in 1:Nt
#         #     tid_sum = 0  # local accumulator

#         # for t2 in 1:Nt
#         #     u = abs(t1 - t2)
#         #     if u <= maxtime
#         @threads for u in 0:maxtime
#             tid_sum = 0  # local accumulator
#             for t1 in 1:(Nt-u)
#                 t2 = t1 + u
#                 for i in 1:D
#                     for j in 1:D
#                         if space_mask[i, j] && !(i == j && u == 0)
#                             # Extract observed values
#                             z1 = data[i, t1]
#                             z2 = data[j, t2]
#                             ρ = cov_matrixes[u+1][i, j] / c11
#                             ρ = clamp(ρ, -0.9999, 0.9999)  # avoid numerical instability

#                             if !isnan(z1) && !isnan(z2)

#                                 tid_sum += -log(2π) - log(c11) - 0.5 * log(1 - ρ^2) - (1 / (2 * c11 * (1 - ρ^2))) * (z1^2 + z2^2 - 2 * ρ * z1 * z2)

#                             elseif isnan(z1) && !isnan(z2)  #mean we have z1 a 0 and z2 non-zero
#                                 tid_sum += log(StatsFuns.normcdf((phimPdry[i, t1] - ρ * z2) / (sqrt(1 - ρ^2))))

#                             elseif !isnan(z1) && isnan(z2)  #mean we have z1 a 0 and z2 non-zero
#                                 tid_sum += log(StatsFuns.normcdf((phimPdry[j, t2] - ρ * z1) / (sqrt(1 - ρ^2))))

#                             else
#                                 tid_sum += log(norm_cdf_2d_vfast(phimPdry[i, t1], phimPdry[j, t2], ρ))
#                             end
#                         end

#                     end
#                 end
#             end
#             # end
#             partial_sums[threadid()] += tid_sum
#         end
#         return sum(partial_sums)
#     end

#     p = (data, Mat_h, model_type, phimPdry)

#     function optimfunction0(u, p)
#         y = p[1]
#         Mat_h = p[2]
#         model_type = p[3]
#         phimPdry = p[4]
#         param = u
#         llh = -pairwise_loglik(y, model_type, param, phimPdry, Mat_h, maxdist=maxdist, maxtime=maxtime)
#         return llh
#     end

#     @show optimfunction0(param0, p)
#     # @show ForwardDiff.gradient(u -> optimfunction0(u, p), param0)

#     optf = OptimizationFunction(optimfunction0, Optimization.AutoForwardDiff())

#     prob = OptimizationProblem(optf, param0, p; lb=lower, ub=upper)
#     @time sol = solve(prob, Optim.LBFGS(), maxiters=maxiter)
#     usol = sol.u
#     return model_type(1.0, usol...)
# end

function fit_mple(data, Mat_h, model_type::Type{<:CovarianceStructure}, param0, phimPdry; lower=nothing, upper=nothing, maxiter=100)
    function pairwise_loglik(data, model_type::Type{<:CovarianceStructure}, param, phimPdry, Mat_h; maxdist=maximum(Mat_h) / 3 * 2, maxtime=10)
        D, Nt = size(data)
        space_mask = Mat_h .<= maxdist
        # Precompute covariance matrix 
        covmodel = model_type(1.0, param...,)
        cov_matrixes = [cov_spatiotemporal(covmodel, Mat_h, timediff) for timediff in 0:maxtime]
        c11 = cov_matrixes[1][1, 1] # variance 
        partial_sums = zeros(eltype(cov_matrixes[1]), nthreads())
        @threads for t1 in 1:Nt
            tid_sum = 0 # local accumulator
            for t2 in 1:Nt
                u = abs(t1 - t2)
                if u <= maxtime
                    for i in 1:D
                        for j in 1:D
                            if space_mask[i, j] && !(i == j && u == 0) # Extract observed values 
                                z1 = data[i, t1]
                                z2 = data[j, t2]
                                ρ = cov_matrixes[u+1][i, j] / c11
                                ρ = clamp(ρ, -0.9999, 0.9999) # avoid numerical instability 
                                if !isnan(z1) && !isnan(z2)
                                    tid_sum += -log(2π) - log(c11) - 0.5 * log(1 - ρ^2) - (1 / (2 * c11 * (1 - ρ^2))) * (z1^2 + z2^2 - 2 * ρ * z1 * z2)
                                elseif isnan(z1) && !isnan(z2) #mean we have z1 a 0 and z2 non-zero 
                                    tid_sum += log(cdf(Normal(), (phimPdry[i, t1] - ρ * z2) / (sqrt(1 - ρ^2))))
                                elseif !isnan(z1) && isnan(z2) #mean we have z1 a 0 and z2 non-zero 
                                    tid_sum += log(cdf(Normal(), (phimPdry[j, t2] - ρ * z1) / (sqrt(1 - ρ^2))))
                                else
                                    tid_sum += log(norm_cdf_2d_vfast(phimPdry[i, t1], phimPdry[j, t2], ρ))
                                end
                            end
                        end
                    end
                end
            end
            partial_sums[threadid()] += tid_sum
        end
        return sum(partial_sums)
    end
    p = (data, Mat_h, model_type, phimPdry)
    function optimfunction0(u, p)
        y = p[1]
        Mat_h = p[2]
        model_type = p[3]
        phimPdry = p[4]
        param = u
        llh = -pairwise_loglik(y, model_type, param, phimPdry, Mat_h)
        return llh
    end
    @show optimfunction0(param0, p)
    @show ForwardDiff.gradient(u -> optimfunction0(u, p), param0)
    optf = OptimizationFunction(optimfunction0, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, param0, p; lb=lower, ub=upper)
    @time sol = solve(prob, Optim.LBFGS(), maxiters=maxiter) #exp exp model for N = 1000 samples of D=37 stations, takes 12s. 
    usol = sol.u
    return model_type(1.0, usol...)
end

# param0 = [1000. 3.]
# fitted = fit_mple(data, Mat_h, ExpExp, param0, phimPdry, maxiter=20)
# # start from restarted computer - on sector;
# # 2 iter : 135.908599 seconds (2.50 M allocations: 144.182 MiB, 0.03% gc time, 3.51% compilation time)
# # ExpExp{Float64, Float64}(1.0, 999.9704311523624, 0.5956409205963817)
# #20 iter :  388.885219 seconds (22.28 k allocations: 43.094 MiB, 0.01% gc time)
#     #    ExpExp{Float64, Float64}(1.0, 267.14150731860417, 0.6014886137593465)

# param0 = [250. 0.5]
# fitted = fit_mple(data, Mat_h, ExpExp, param0, phimPdry, maxiter=20)
# # 185.621960 seconds (9.95 k allocations: 20.786 MiB, 0.01% gc time)
# # ExpExp{Float64, Float64}(1.0, 267.14150731860417, 0.6014886137593465)
# jldsave("23precip_intensity/res_real_data/ExpExpcov_withmarginalsdeg" * string(degree_of_P) * ".jld2"; fitted=fitted)






param0=[ 300.0, 1/3,  1.0, 0.5, 0.1, 0.5]
fitted=fit_mple(data,Mat_h,GneitingMatern,param0,phimPdry,lower=[200., 0.1 ,0. ,0. ,0.00, 0.1],upper= [1000. ,10., 1., 1., 10., 10.],maxiter=100)
# 100 iter : 6279.295689 seconds (35.85 M allocations: 3.355 GiB, 0.01% gc time, 0.26% compilation time: <1% of which was recompilation)

# GneitingMatern{Float64, Float64}(1.0, 274.5487918568977, 0.46817484026433887, 0.9999999999999125, 0.7438575419048332, 0.42334214393444314, 0.4759128763382056)

jldsave("23precip_intensity/res_real_data/GMcov_withmarginalsdeg" * string(degree_of_P) * ".jld2"; fitted=fitted)
