# TKM3D

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArrogantGao.github.io/TKM3D.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArrogantGao.github.io/TKM3D.jl/dev/)
[![Build Status](https://github.com/ArrogantGao/TKM3D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/TKM3D.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ArrogantGao/TKM3D.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/TKM3D.jl)

Implementation of the truncated kernel method for 3D systems with continuous and discrete sources, accelerated by `FINUFFT`. 

Currently, only the Laplace kernel is supported.
For continuous sources $\rho(y)$, `TKM3D` evaluates
$$
\phi(x) = \int_{\Omega} \frac{1}{4 \pi |x - y|} \rho(y) dy,
$$
where $\Omega$ is the support of $\rho$.
For discrete sources $\{(x_i, q_i)\}$, `TKM3D` evaluates
$$
\phi(x) = \int_{\Omega'} \sum_{i} \frac{W(x - x_i)}{4 \pi |y - x|} q_i,
$$
where $W$ is a smooth window function, for example, Guassian function in Ewald summation, and $\Omega'$ is the support of the discrete sources after convolution with the window function.