# HMMSPAcodepaper
Code for the paper "Spatio-temporal generation of precipitation using a Hidden Markov Model, extended extreme distributions and conditional Gaussian fields" 

"data/" contains the French station data used in the application section of the paper and a script to format the rain series into common dataframe format.

The scripts are organised as follows :

## Data storing and processing
**data/** contains the code and data used in the paper. 
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

**MixtureSpatialBernoulli/** contains the code for a mixture of the previously introduced distribution.
- **MixtureSpatialBernoulli/ExpectationMaximization_source.jl** contains the source code from David Métivier's ExpectationMaximization.jl package.
- **MixtureSpatialBernoulli/estimation_functions.jl** adds a method called (wrongly) *PairwiseClassicEM*. As described in the paper, the objective function maximised in the M step is the pariwise likelihood.
- **MixtureSpatialBernoulli/test_simulated_data.jl** proposes some tests for the simulation and inference. Plots results according to the `Plots` library.

## Utilities
**utils/** contains code for several useful functions:
    - **utils/fast_bivariate_cdf.jl** uses the approximation of https://github.com/david-cortes/approxcdf to evaluate bivariate normal integrals
    
