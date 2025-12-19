# HMMSPAcodepaper
Code for the paper "Spatio-temporal generation of precipitation using a Hidden Markov Model, extended extreme distributions and conditional Gaussian fields" 

"data/" contains the French station data used in the application section of the paper

The scripts are organised as follows :


## Fitting a Periodic Hidden Markov Model with spatially correlated emission
* "SpatialBernoulli/" contains the code for the simple spatial emission only. 
    - SpatialBernoulli.jl : depends on `Distributions`, `MvNormalCDF`, `BesselK`,- `Optimization`, `ForwardDiff`
        - Model definition : SpatialBernoulli(range, sill, order, λ, h) (possibility to use a non-exponential Matérn kernel). `SpatialBernoulli` is a distribution subtype of `DiscreteMultivariateDistribution` of the Julia `Distributions` package.
        - Random sampling using `rand`
        - `pdf`, `logpdf` evaluation, for full, weighted, pairwise and weighted pairwise log-likelihood (see docstrings inside the code)
        - MLE inference for both full and pairwise case (full not recommended for speed). Possible weighted MLE inference, appropriate for EM algorithm.

* "utils/" contains code for several useful functions:
    - 

