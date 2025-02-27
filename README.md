### Introduction

This repository contains code accompanying the paper ["Tukey Depth Mechanisms for Practical Private Mean Estimation,"](https://arxiv.org/abs/2502.18698) 
by Gavin Brown and Lydia Zakynthinou.
It provides algorithms for differentially private mean estimation that use notions of Tukey depth.
In addition to providing privacy, these algorithms are accurate at small sample sizes, robust to misspecification or adversarial corruption, and invariant under affine rescalings of the data.
Their computational requirements scale poorly with the dimension, so they are best applied to datasets of modest dimension (e.g., <=5).

Please note two important caveats.
1. These implementations do not address the possibility of privacy leakage through floating-point attacks. They are for research purposes only and should not be used in privacy-critical applications.
1. This code includes functionality to approximate polytope volume via Markov Chain Monte Carlo methods. Given a sufficient number of steps, these approximations suffice for privacy. **However**, there are no theoretical mixing time guarantees that yield practical constants. Thus, within practical computational constraints, the MCMC-based functionality we provide is **not currently known to satisfy differential privacy**.

### Dependencies

Some of our code uses libraries only available in R. 
You will need an [R installation](https://www.r-project.org/) and two R packages: [TukeyRegion](https://rdrr.io/cran/TukeyRegion/) and [Volesti](https://journal.r-project.org/archive/2021/RJ-2021-077/index.html).
After installing R, from the command line type `R` to open up the R console and install the packages with `install.packages("TukeyRegion")` and `install.packages("volesti")`.

We will require a few Python packages:
- `numpy`
- `scipy`
- `polytope` ([documentation](https://tulip-control.github.io/polytope/))
- `rpy2` ([documentation](https://rpy2.github.io/doc/latest/html/index.html)

In d>2 dimensions, we use the [VINCI package](https://www.multiprecision.org/vinci/) for exact polytope volume computation.
See also [the documentation](https://www.multiprecision.org/downloads/vinci.pdf) and [code](https://github.com/xhub/vinci).

### Running the algorithms

The script `first_experiment.py` will create a 500-example dataset in 2 dimensions and run REM on it under 
random Tukey depth.

The script `full_experiment.py` runs five tests that demonstrate the different mechanisms, depth notions, and sampling routines.
Some of the algorithms the code uses are not currently available in Python, so we will have to do some extra work.

### Further Guide to Code

In addition to the scripts `first_experiment.py` and `full_experiment.py`, we provide four other files:
- `mechanisms.py` contains functions for the Restricted Exponential Mechanism (REM) and the standard exponential mechanism (BoxEM) over Tukey depth.
- `volume_computation.py` contains the core code for computing volumes of polytopes and Tukey depth upper-level sets.
- `sampling.py` provides functions for the sampling needed to run the exponential mechanism.
- `utils.py` collects miscellaneous functions.

