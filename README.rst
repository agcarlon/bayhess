Bayesian Hessian Approximation for Stochastic Optimization
----------------------------------------------------------

The BayHess package uses noisy curvature pairs (noisy gradient differences computed at different points) to compute Hessian approximations. These Hessian approximations can be used to accelerate the convergence in stochastic optimization in a quasi-Newton fashion. To find a Hessian approximation, a posterior distribution of the Hessian is built. The prior distribution is based on the Frobenius norm with determinant constraints to impose extreme eigenvalues constraints and the likelihood distribution is built from the secant equations given the observed curvature pairs. To find the maximizer of the log posterior, the BayHess package uses the Newton-CG method with a homotopy approach to deal with the logarithmic barrier determinant constraints.

For a detailed description of the method, convergence analysis and numerical results, check our `manuscript`_ named "Approximating Hessian matrices using Bayesian inference: a new approach for quasi-Newton methods in stochastic optimization". This package can be used with the `MICE`_ estimator.

A BayHess object is created by giving the dimensionality of the problem, and lower and upper bounds of the eigenvalues of the Hessian; i.e., strong convexity and smoothness parameters.

>>> bay = BayHess(n_dim=10, strong_conv=1e-3, smooth=1e4)

Curvature information is passed using the 'update_curv_pairs' method

>>> bay.update_curv_pairs(sk, yk)

And, finally, computing the Hessian approximation is done by

>>> hess = bay.find_hess()

Install BayHess from PyPI as

>>> pip install bayhess

A repository with numerical examples can be found at

https://github.com/agcarlon/bayhess_numerics

The documentation of the BayHess package is available at

https://bayhess.readthedocs.io/




.. _manuscript: https://arxiv.org/abs/2208.00441
.. _MICE: https://pypi.org/project/mice/
