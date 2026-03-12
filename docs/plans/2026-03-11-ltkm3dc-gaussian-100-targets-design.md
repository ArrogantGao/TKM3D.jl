# ltkm3dc Gaussian 100-Target Test Design

## Goal

Add a regression test that samples a truncated 3D Gaussian on a uniform mesh and compares `ltkm3dc` target potentials against the analytic Gaussian potential at 100 random targets in a slightly larger box.

## Scope

- keep the existing continuous API unchanged
- add a target-only analytic regression in `test/continuous.jl`
- use fixed random seeds for reproducibility

Out of scope:

- production code changes
- new public API
- benchmark or performance assertions

## Test Construction

Use the continuous Gaussian density

```math
\rho(x) = \frac{1}{(2\pi\sigma^2)^{3/2}} e^{-|x-c|^2/(2\sigma^2)}
```

truncated to the cube `[-R, R]^3`, where

```math
R = \sigma\sqrt{2\log(10^{12})}.
```

Sample this Gaussian on a uniform tensor-product mesh and form preweighted quadrature masses with `h^3`.

Draw 100 target points from a uniform random distribution in the larger box `[-1.2R, 1.2R]^3` with a fixed RNG seed.

Evaluate

```julia
ltkm3dc(1e-12, sources; charges, targets, pgt = 1, kmax = sqrt(2log(1e12)) / sigma)
```

and compare the target potential against the analytic Gaussian free-space potential.

## Accuracy Strategy

The regression should measure `ltkm3dc`, not source quadrature error. If the first uniform mesh is too coarse, increase the mesh resolution until the analytic comparison is stable and the test tolerance is comfortably satisfied.
