---
title: 'SBArchOpt: Surrogate-Based Architecture Optimization'
tags:
  - Python
  - optimization
  - engineering
  - system architecture optimization
  - MBSE
  - surrogate-based optimization
  - Bayesian optimization
  - multi-objective optimization
authors:
  - name: Jasper H. Bussemaker
    orcid: 0000-0002-5421-6419
    affiliation: 1
affiliations:
 - name: Institute of System Architectures in Aeronautics, German Aerospace Center (DLR), Hamburg, Germany
   index: 1
date: 25 May 2023
bibliography: paper.bib
---

# Summary

In engineered systems, the architecture of a system describes how the components of a system work together to fulfill
the system functions and meet stakeholder expectations [@Crawley2015].
As the architecture is developed in early conceptual design stages, decisions involving architecture often have a large
influence on final system performance, for example, in terms of how well the functions are fulfilled, or at what
cost and in what timeframe.
However, architecture design spaces (i.e., the set of all possible architecture alternatives) can be very large due to
the combinatorial nature of architecture decisions, making it infeasible to compare all alternatives to each other.
Additionally, for new systems there might not be any prior experience to start from, requiring the use of
(typically) computationally-expensive physics-based simulation to estimate system performance.

The field of *system architecture optimization* aims to enable the use of physics-based simulation for
exploring the large combinatorial architecture design space, by formulating the system architecting process as a
numerical optimization problem [@Bussemaker2022c]. In optimization, the goal is to minimize (or maximize) one or more
objective functions by modifying design variables, while ensuring that design constraints are satisfied.
For example, for an aircraft propulsion system, the objectives could be to minimize energy consumption and operating
costs, by changing number of propellers, mechanical power generation source, fuel type, and operating strategy
(i.e., the design variable), while ensuring that thermodynamic stress and reliability constraints are satisfied.
By applying architecture optimization, more architecture alternatives can be considered in the early design phase with
the expected result of better understanding of the design space and more informed decision-making.

Architecture optimization problems feature several behavioral characteristics that make them a particularly
challenging class of optimization problem [@Bussemaker2021]:

- Evaluation functions are non-linear *black-box* functions that are *expensive* to evaluate: it might for example
  take several hours to evaluate the performance of only one architecture alternative.
- There might be *multiple conflicting objectives* (i.e., design goals) to optimize for, meaning that rather than one
  optimal design, there might be a Pareto-set of optimal designs.
- Simulations used in performance evaluation might fail to converge, yielding Not-a-Number as evaluation result; this
  phenomenon is called *hidden constraints*, because they can be seen as design constraints that are "hidden" when
  defining the problem.
- The design space might contain both continuous and discrete variables, making the optimization problem a
  *mixed-discrete* problem.
- Decisions can be conditionally active based on other decisions: there is a *hierarchy* between decisions.

Such optimization problems can be readily solved by Multi-Objective Evolutionary Algorithms (MOEAs). However, these
need many function evaluations to converge [@Chugh2019], which is a problem for expensive evaluation functions.
Surrogate-Based Optimization (SBO) algorithms and in particular Bayesian Optimization (BO) algorithms [@Garnett2023]
instead build a surrogate model (also known as response surface or regression function) of the design space, use
that model to suggest new design points to evaluate, and repeat the process after updating the surrogate model.
This approach is powerful, although existing SBO algorithms need to be extended to support all architecture optimization
challenges described above.

# Statement of need

Several open-source Surrogate-Based Optimization (SBO) libraries already exist, in particular
BoTorch [@Balandat2020],
Trieste [@Picheny2023],
SMAC3 [@Lindauer2022],
and
HEBO [@CowenRivers2022].
These libraries support multi-objective and mixed-discrete optimization; however, they do not all support hidden constraints
and decision hierarchy. The latter requires the automatic correction and imputation of design vectors to ensure
no duplicate design vectors are generated [@Bussemaker2021].

The purpose of SBArchOpt is to provide a one-stop solution for solving architecture optimization problems, by:

1. providing a common interface for implementing architecture optimization problems, ensuring that all information
   needed by optimization algorithms is available; and
2. providing several options for optimization algorithms that work out-of-the-box for most architecture optimization
   problems.

*SBArchOpt* implements experience with solving architecture optimization problems
(e.g., [@Bussemaker2021; @Bussemaker2023]) in an open-source Python library.
Target users are systems engineers and researchers in the field of (architecture) optimization.

*SBArchOpt* will be used as the go-to library for solving architecture optimization problems modeled
using ADORE [@Bussemaker2022], software developed by the German Aerospace Center (DLR) and applied in several
German and European research projects.
ADORE already implements the problem definition API of *SBArchOpt*.

# Library features

The problem definition API `ArchOptProblemBase` extends the `Problem` class of pymoo [@Blank2020],
an evolutionary optimization framework, with several additional features:

1. a unified way to define continuous, integer, and categorical design variables;
2. an interface for accepting modified design vectors from the evaluation function;
3. a function for correcting design vectors without running an evaluation;
4. a function for returning which design variables are conditionally active;
5. an interface for storing and loading problem-specific intermediate results; and
6. several functions for compiling statistics about the design space, such as the number of valid architectures, the
   average number of active design variables, and more.

Optionally, the hierarchical design space structure can also be specified using the `ExplicitDesignSpace` class,
which then relieves the user from implementing correction, conditional-activeness and statistics-related functions.
The explicit design space definition uses [ConfigSpace](https://github.com/automl/ConfigSpace) [@Lindauer2022] to model
conditional activation and value-pair constraints.

Then, *SBArchOpt* implements several features that may be used by any optimizer, using pymoo's API:

1. a sampling algorithm for hierarchical design spaces;
2. a repair operator that calls the correction function of the problem class; and
3. intermediate results storage and restart capabilities.

To solve optimization problems, *SBArchOpt* implements the following (interfaces to) optimization libraries/algorithms:

1. pymoo: *SBArchOpt* provides a pre-configured version of the NSGA2 evolutionary optimization algorithm;
2. ArchSBO: a custom implementation of a mixed-discrete, multi-objective Surrogate-Based Optimization algorithm, with
   support for design variable correction, hidden constraints, and restart,
   using state-of-the-art mixed-discrete, hierarchical Gaussian Process models [@Saves2023];
3. three open-source Bayesian Optimization libraries:
   BoTorch (Ax) [@Balandat2020], Trieste [@Picheny2023], and HEBO [@CowenRivers2022];
4. two proprietary Bayesian Optimization libraries: SEGOMOE [@Bartoli2019] and SMARTy [@Bekemeyer2022]; and
5. a Tree Parzen Estimator (TPE) algorithm with support for hidden constraints.

Finally, to support development of optimization algorithms, *SBArchOpt* also provides a database of test functions:

1. many analytical test problems with various combinations of characteristics: continuous vs mixed-discrete,
   single- or multi-objective, with or without constraints, hierarchy, and/or hidden constraints;
2. a Guidance, Navigation and Control (GNC) optimization problem from [@Apaza2021] trading-off system mass against
   reliability, and a little over 79 million possible architectures; and
3. an aircraft jet engine architecture optimization problem from [@Bussemaker2021c] that uses a realistic engine
   simulation framework for performance evaluation, features hidden constraints, and trades fuel consumption against
   engine weight and emissions.

# Acknowledgements

I would like to thank my colleagues at the DLR in Hamburg (DE) and at ONERA in Toulouse (FR)
for supporting my research and for their helpful discussions and feedback. In particular,
I would like to thank Nathalie Bartoli, Thierry Lefebvre, Paul Saves, and Rémi Lafage for their
warm welcome in Toulouse, and Luca Boggero and Björn Nagel for supporting my secondment at ONERA.

# References
