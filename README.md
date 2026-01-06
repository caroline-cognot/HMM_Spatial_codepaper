# HMMSPAcodepaper
Code for the paper "Spatio-temporal generation of precipitation using a Hidden Markov Model, extended extreme distributions and conditional Gaussian fields" 

**Warning** : some scripts for tests use the `Plots.jl`package, others use the `Makie.jl`package. They are not compatible so it is recommended to kill the julia terminal between launches. Scripts relying on the latter are highlighted in this document. 


The scripts are organised as follows :

## Data storing and processing
**data/** contains the French station data used in the application section of the paper and a script to format the rain series into common dataframe format.
- **data/data_selection.jl** : selects the station data with minimal missing values; store the contents of all individual station data in common csv files for rain occurrence, distances and station information.

## Fitting a Periodic Hidden Markov Model with spatially correlated emission
**SpatialBernoulli/** contains the code for the simple spatial emission only. 
- **SpatialBernoulli/SpatialBernoulli.jl** : 
    - *Dependencies* : `Distributions`, `MvNormalCDF`, `BesselK`, `Optimization`, `ForwardDiff`
    - Model definition : SpatialBernoulli(range, sill, order, λ, h) (possibility to use a non-exponential Matérn kernel). `SpatialBernoulli` is a distribution subtype of `DiscreteMultivariateDistribution` of the Julia `Distributions` package.
    - Random sampling using `rand`
    - `pdf`, `logpdf` evaluation, for full, weighted, pairwise and weighted pairwise log-likelihood (see docstrings inside the code)
    - MLE inference for both full and pairwise case (full not recommended for speed). Possible weighted MLE inference, appropriate for EM algorithm.
- **SpatialBernoulli/test_simulated_data.jl** proposes some tests for the simulation, inference with or without fixing the Matérn regularity parameter.
- **SpatialBernoulli/test_real_data.jl** proposes some tests for the simulation, inference with or without fixing the Matérn regularity parameter. Fits the model for all 12 months of the year, plots results according to the `Plots` library.

**MixtureSpatialBernoulli/** contains the code for a mixture of the previously introduced distribution. By default, it DOES NOT USE the fast approximation of Genz. It is not planned to be corrected.
- **MixtureSpatialBernoulli/ExpectationMaximization_source.jl** contains the source code from David Métivier's ExpectationMaximization.jl package.
- **MixtureSpatialBernoulli/estimation_functions.jl** adds a method called (wrongly) *PairwiseClassicEM*. As described in the paper, the objective function maximised in the M step is the pariwise likelihood.
- **MixtureSpatialBernoulli/test_simulated_data.jl** proposes some tests for the simulation and inference. Plots results according to the `Plots` library.
- **MixtureSpatialBernoulli/test_real_data.jl** proposes some tests for the simulation and inference. Plots results according to the `Plots` library. Only a small number of EM iterations performed. Plots the before/after parameters and ROR distribution to check the optimisation's relevance. Only K=3 and K=4 are tested.
- **MixtureSpatialBernoulli/compare_nomix_mix.jl** compares the results in ROR between basic SpatialBernoulli and MixtureSpatialBernoulli on real data. As it is, only the version with 3 classes is computed, but the file names can be changed inside the script to accomodate other choices

**PeriodicHMMSpatialBernoulli/** contains the code for estimating, simulating and evaluating the full HMM with spatial emissions as used in the stochastic weather generator. The period is always noted *T* or alternatively, *my_T*, while the number of hidden states is noted *K* or *my_K*. The memory *m* denotes the absence (*m=0*) or presence (*m=1*) of memory in the local rain probability. *D* is the number of locations in the model (in the paper, *D=37*).
- **PeriodicHMMSpatialBernoulli/PeriodicHMMSpa.jl**
    - *Dependencies* : `SmoothPeriodicStatsModels` for the periodic parameterisation
    - Model definition : PeriodicHMMSpaMemory(a= initial probabilities, A=transition probabilities, R= spatial scales, B=rain probabilities,h=distance matrix). A is of size *K \times K	\times T*, R of size *K \times T*, B is of size *K \times T \times D \times m* .
    - Transformation from periodic coefficients to model : Trig2PeriodicHMMspaMemory(a , theta_A,theta_B,theta_R, T,h )
    - random generator : my_rand(hmm:PeriodicHMMSpaMemory, z, n2t= periodic indices, initial value of y ) for generating with given markov sequence, remove entry z for full HMM sampling.
    - some plotting functions to display model parameters, relying on `Plots.jl`.
- **PeriodicHMMSpatialBernoulli/estimation_functions_BthenR.jl** 
    - *Dependencies* :  `LogExpFunctions`, `PeriodicHiddenMarkovModels`, `Optimization`, `JuMP`.
    Implements the inference procedure for the HMM with spatial emission. In the M-step, the parameters are estimated sequentially : A, B, then R. 
- **PeriodicHMMSpatialBernoulli/estimation_functions_BandR.jl** 
    - *Dependencies* :  `LogExpFunctions`, `PeriodicHiddenMarkovModels`, `Optimization`, `JuMP`.
    Implements the inference procedure for the HMM with spatial emission. In the M-step, the parameters are estimated directly : A first (maximising independant sums), then B and R at the same time. Is not used by default in the tests.
- **PeriodicHMMSpatialBernoulli/test_simulated_data.jl** 
    Relies on `CairoMakie.jl`for prettier plots. Makes a synthetic example with 2 states, using real data locations, to show the estimation procedure works for both *m=0* and *m=1*. Outputs graphs showing the comparison between true model, starting model in the inference, and estimated parameters.
- **PeriodicHMMSpatialBernoulli/test_simulated_data.jl** 
    Relies on `CairoMakie.jl`for prettier plots. Makes a synthetic example with 2 states, using real data locations, to show the estimation procedure works for both *m=0* and *m=1*. Outputs graphs showing the comparison between true model, starting model in the inference, and estimated parameters.
    


## Utilities
**utils/** contains code for several useful functions:
    - **utils/fast_bivariate_cdf.jl** uses the approximation of https://github.com/david-cortes/approxcdf to evaluate bivariate normal integrals 
    - **utils/maps.jl** has useful functions for plotting maps of France with some station information.
    - **utils/seasons_and_other_dates.jl** has two functions to extract months and seasons in dates.

