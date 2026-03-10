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
where $W$ is a smooth window function, for example, a Gaussian in Ewald summation, and $\Omega'$ is the support of the discrete sources after convolution with the window function.

The current discrete entry point is
`ltkm3dd(eps, sources; charges, targets=nothing, pg=0, pgt=0, windowhat, lw, kmax)`,
with these constraints:

- `sources` and `targets` use FMM3D-style `3 x N` and `3 x M` layouts
- `charges` is required, and dipole inputs are not supported
- `windowhat(k)` is the radial Fourier transform of the window
- `pg` and `pgt` support `0`, `1`, and `2`
- gradients are returned as `3 x N` and `3 x M` matrices through `grad` and `gradtarg`
- source outputs omit the diagonal self interaction
