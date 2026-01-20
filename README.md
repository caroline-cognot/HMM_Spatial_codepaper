# HMMSPAcodepaper
Code for the paper "Spatio-temporal generation of precipitation using a Hidden Markov Model, extended extreme distributions and conditional Gaussian fields" 

![Overview of model parameters](hmmrain.schema.pdf)

**Info** : some scripts for tests use the `Plots.jl`package, others use the `Makie.jl`package. They are not compatible so it is recommended to kill the julia terminal between launches. Scripts relying on the latter are highlighted in this document. 

**Info** : the results of time-consuming inference are stored on this git, however simulations have not been pushed as they are too voluminous. For now, everything is stored in binary files, which is not efficient. 

The scripts are organised as follows :

## 0 - Data storing and processing
**00data/** contains the French station data used in the application section of the paper and a script to format the rain series into common dataframe format.
- **00data/data_selection.jl** : selects the station data with minimal missing values; store the contents of all individual station data in common csv files for rain occurrence, distances and station information.

## 1 - Fitting a Periodic Hidden Markov Model with spatially correlated emission
**11SpatialBernoulli/** contains the code for the simple spatial emission only. 
- **11SpatialBernoulli/SpatialBernoulli.jl** : 
    - *Dependencies* : `Distributions`, `MvNormalCDF`, `BesselK`, `Optimization`, `ForwardDiff`
    - Model definition : SpatialBernoulli(range, sill, order, λ, h) (possibility to use a non-exponential Matérn kernel). `SpatialBernoulli` is a distribution subtype of `DiscreteMultivariateDistribution` of the Julia `Distributions` package.
    - Random sampling using `rand`
    - `pdf`, `logpdf` evaluation, for full, weighted, pairwise and weighted pairwise log-likelihood (see docstrings inside the code)
    - MLE inference for both full and pairwise case (full not recommended for speed). Possible weighted MLE inference, appropriate for EM algorithm.
- **11SpatialBernoulli/test_simulated_data.jl** proposes some tests for the simulation, inference with or without fixing the Matérn regularity parameter.
- **11SpatialBernoulli/test_real_data.jl** proposes some tests for the simulation, inference with or without fixing the Matérn regularity parameter. Fits the model for all 12 months of the year, plots results according to the `Plots` library.

**12MixtureSpatialBernoulli/** contains the code for a mixture of the previously introduced distribution. By default, it DOES NOT USE the fast approximation of Genz. It is not planned to be corrected.
- **12MixtureSpatialBernoulli/ExpectationMaximization_source.jl** contains the source code from David Métivier's ExpectationMaximization.jl package.
- **12MixtureSpatialBernoulli/estimation_functions.jl** adds a method called (wrongly) *PairwiseClassicEM*. As described in the paper, the objective function maximised in the M step is the pariwise likelihood.
- **12MixtureSpatialBernoulli/test_simulated_data.jl** proposes some tests for the simulation and inference. Plots results according to the `Plots` library.
- **12MixtureSpatialBernoulli/test_real_data.jl** proposes some tests for the simulation and inference. Plots results according to the `Plots` library. Only a small number of EM iterations performed. Plots the before/after parameters and ROR distribution to check the optimisation's relevance. Only K=3 and K=4 are tested.
- **12MixtureSpatialBernoulli/compare_nomix_mix.jl** compares the results in ROR between basic SpatialBernoulli and MixtureSpatialBernoulli on real data. As it is, only the version with 3 classes is computed, but the file names can be changed inside the script to accomodate other choices

**13PeriodicHMMSpatialBernoulli/** (final version, model used in the paper) contains the code for estimating, simulating and evaluating the full HMM with spatial emissions as used in the stochastic weather generator. The period is always noted *T* or alternatively, *my_T*, while the number of hidden states is noted *K* or *my_K*. The memory *m* denotes the absence (*m=0*) or presence (*m=1*) of memory in the local rain probability. *D* is the number of locations in the model (in the paper, *D=37*).
- **13PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl**
    - *Dependencies* : `SmoothPeriodicStatsModels` for the periodic parameterisation
    - Model definition : PeriodicHMMSpaMemory(a= initial probabilities, A=transition probabilities, R= spatial scales, B=rain probabilities,h=distance matrix). A is of size *K \times K	\times T*, R of size *K \times T*, B is of size *K \times T \times D \times m* .
    - Transformation from periodic coefficients to model : Trig2PeriodicHMMspaMemory(a , theta_A,theta_B,theta_R, T,h )
    - random generator : my_rand(hmm:PeriodicHMMSpaMemory, z, n2t= periodic indices, initial value of y ) for generating with given markov sequence, remove entry z for full HMM sampling.
    - some plotting functions to display model parameters, relying on `Plots.jl`.
- **13PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl** 
    - *Dependencies* :  `LogExpFunctions`, `PeriodicHiddenMarkovModels`, `Optimization`, `JuMP`.
    Implements the inference procedure for the HMM with spatial emission. In the M-step, the parameters are estimated sequentially : A, B, then R. 
- **13PeriodicHMMSpatialBernoulli/estimation_functions_BandR.jl** 
    - *Dependencies* :  `LogExpFunctions`, `PeriodicHiddenMarkovModels`, `Optimization`, `JuMP`.
    Implements the inference procedure for the HMM with spatial emission. In the M-step, the parameters are estimated directly : A first (maximising independant sums), then B and R at the same time. Is not used by default in the tests.
- **13PeriodicHMMSpatialBernoulli/test_simulated_data.jl** 
    Relies on `CairoMakie.jl`for prettier plots. Makes a synthetic example with 2 states, using real data locations, to show the estimation procedure works for both *m=0* and *m=1*. Outputs graphs showing the comparison between true model, starting model in the inference, and estimated parameters.
- **13PeriodicHMMSpatialBernoulli/test_simulated_data_BandR.jl** 
    Same as previous file but using **estimation_functions_BandR.jl** instead, and does not use `Makie`.
- **13PeriodicHMMSpatialBernoulli/test_real_data.jl** 
   Implements the inference for real station data. Also get the ICL criterion for choice at the end.
- **13PeriodicHMMSpatialBernoulli/test_K1_ch.jl** 
Tries the inference method for *K=1* to compare between full and pairwise likelihood. In practice, this code should belong to the **SpatialBernoulli** folder, except the HMM was so much better implemented that considering *K=1, d=0, m=0* as a special case was better than just trying the SpatialBernoulli code.    
- **13PeriodicHMMSpatialBernoulli/real_data_getZ.jl** : uses the Viterbi algorithm to get most likely sequence of hidden states for use in the rain model. Also formats the rain intensity data to a dataframe.

## 2 - Fitting the intensity model
**21precip_intensity_marginal_noclass**
Contains the code for periodic inference and test on simulated data for the EGPD distribution, as well as the fit on real data. 

**22precip_intensity_marginal_withclass**
Contains the code for constant inference, by class.

**23precip_intensity** (final version, model used in the paper)
-  **23precip_intensity/EGPD_functions.jl** extends the `ExtendedExtremes.jl`package to periodic parameters, notably adding a low-rain marginal and extending the penalised likelihood to periodic parameters.
-  **23precip_intensity/EGPD_class.jl** extends the previous file to models with classes.
- **23precip_intensity/covariance_functions.jl** defines Spatial Gaussian processes and possible necessary functions. 
- **23precip_intensity/01_marginal_estimation.jl** is the script to be used in order to infer the marginal parameters for the rain intensity, then transform to a truncated normal distribution according to the occurrence and intensity model parameters. 
- **23precip_intensity/02_covariance_estimation.jl** is the script to be used in order to infer the covariance parameters for the rain intensity. Be careful that it can take a while as the data is quite large.
- **23precip_intensity/03_simulation.jl** is the script to be used in order to simulate the intensity model. Relies on dependency `TruncatedMVN`. The plotting parts are commented, as all plots are now in a common folder, however as they are written in the more basic `Plots.jl` one may want to use them instead.


## 3- Comparing with the conditionnaly independent model
**31HMMIndep/** contains the code and result for the conditionnaly independent model, infered on several subsets of stations to highlight the need of a more complex (spatially correlated) model. There are two scripts : **31HMMIndep/test_real_data.jl**, aiming at investigating different model sizes, and the well-named **31HMMIndep/prettyplots_indep.jl** which produces a beautiful plot comparing models inferred with different *K* and *D*.


## 4- Plotting 
**41Plots_folder** contains the code for all plots used in the paper and supplementary materials, as well as others to better understand the model's initial idea, such as results of the conditionnaly independent model in high dimension. They rely on the `Makie` environment, which may make it harder to handle at first but produces *really pretty* graphs. 

**42NAOlike** contains the code to see how the HMM relates to atmospheric circulation, in particular regarding the North Atlantic Oscillation (NOA). The data is not on the git however, as it is too voluminous. 




## Utilities
**utils/** contains code for several useful functions:

- **utils/fast_bivariate_cdf.jl** uses the approximation of https://github.com/david-cortes/approxcdf to evaluate bivariate normal integrals 
- **utils/maps.jl** has useful functions for plotting maps of France with some station information.
- **utils/seasons_and_other_dates.jl** has two functions to extract months and seasons in dates.

