# TKM3D

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArrogantGao.github.io/TKM3D.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArrogantGao.github.io/TKM3D.jl/dev/)
[![Build Status](https://github.com/ArrogantGao/TKM3D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/TKM3D.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ArrogantGao/TKM3D.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/TKM3D.jl)

Implementation of the truncated kernel method for 3D systems with `FINUFFT`.

## Current Status

The current code implements:

- the **discrete long-range Laplace solver** `ltkm3dd`
- the **continuous free-space Laplace solver** `ltkm3dc`
- the **continuous spectral cutoff estimator** `estimate_kcut3dc`

The discrete solver evaluates the windowed long-range interaction
```math
\phi(y) = \int_{\Omega'} \sum_i \frac{W(x - x_i)}{4 \pi |y - x|} q_i \, dx,
```
where `W` is a smooth window, for example the Gaussian window used in Ewald
summation.

The continuous solver evaluates
```math
\phi(y) = \int_{\Omega} \frac{\rho(x)}{4 \pi |y - x|} \, dx,
```
using arbitrary quadrature points with preweighted masses supplied through
`charges`.

## Public API

The public entry points are

`ltkm3dd(eps, sources; charges, targets=nothing, pg=0, pgt=0, windowhat, lw, kmax)`

and

`ltkm3dc(eps, sources; charges, targets=nothing, pg=0, pgt=0, kmax=nothing)`

and

`estimate_kcut3dc(sources; charges, tol, eps=1e-12)`

with FMM3D-like source and target output flags.

### Arguments

- `eps`: requested NUFFT tolerance
- `sources`: source coordinates in `3 x ns` layout
- `charges`: source data, length `ns`
- `pg`: source output selector
- `pgt`: target output selector
- `targets`: optional target coordinates in `3 x nt` layout

Discrete-only arguments:

- `windowhat(k)`: radial Fourier transform of the window
- `lw`: effective real-space support of the window
- `kmax`: Fourier truncation radius

Continuous-only argument:

- `kmax`: Fourier truncation radius, or `nothing` to estimate it from average source spacing

Continuous spectral cutoff arguments:

- `tol`: relative pointwise tail tolerance
- `eps`: requested NUFFT tolerance

### Output Flags

- `pg = 0`: no source output
- `pg = 1`: source potentials only
- `pg = 2`: source gradients only
- `pgt = 0`: no target output
- `pgt = 1`: target potentials only
- `pgt = 2`: target gradients only

### Return Value

Both solvers return a `TKMVals` object with fields:

- `pot`: source potential, length `ns`, or `nothing`
- `grad`: source gradient, size `3 x ns`, or `nothing`
- `pottarg`: target potential, length `nt`, or `nothing`
- `gradtarg`: target gradient, size `3 x nt`, or `nothing`
- `ier`: currently `0` on success

`estimate_kcut3dc` returns a `KCut3DCResult` with fields:

- `kcut`: smallest cutoff satisfying the requested tail tolerance
- `kmax_nyquist`: largest inferred axis-wise Nyquist limit
- `axis_nyquist`: per-axis Nyquist limits
- `max_coeff`: largest coefficient magnitude on the sampled mode box
- `tail_ratio`: achieved relative pointwise tail ratio at `kcut`
- `nmodes`: sampled mode-box dimensions
- `Δk`: per-axis mode spacings

## Example

```julia
using TKM3D

sigma = 0.12
windowhat(k) = exp(-0.25 * sigma^2 * k^2)

sources = rand(3, 8)
charges = randn(8)
targets = rand(3, 5)

out = ltkm3dd(
    1e-12,
    sources;
    charges,
    targets,
    pg = 1,
    pgt = 2,
    windowhat = windowhat,
    lw = 5sigma,
    kmax = 2sqrt(log(1e12)) / sigma,
)

pot_src = out.pot
grad_src = out.grad
pot_targ = out.pottarg
grad_targ = out.gradtarg
```

In the example above:

- `pot_src` is populated because `pg = 1`
- `grad_targ` is populated because `pgt = 2`
- `grad_src` and `pot_targ` are `nothing`

Continuous example:

```julia
using TKM3D

sources = rand(3, 1000)
charges = rand(1000) .* 1e-3  # preweighted quadrature masses
targets = rand(3, 20)

out = ltkm3dc(
    1e-12,
    sources;
    charges,
    targets,
    pgt = 2,
    kmax = 20.0,
)
```

Cutoff estimation example:

```julia
using TKM3D

n = 6
h = 0.25
sources = Matrix{Float64}(undef, 3, n^3)
idx = 1
for x in 0:(n - 1), y in 0:(n - 1), z in 0:(n - 1)
    sources[:, idx] .= (x * h, y * h, z * h)
    idx += 1
end
charges = ones(size(sources, 2))

cut = estimate_kcut3dc(sources; charges, tol = 1e-10)

cut.kcut
cut.axis_nyquist
cut.nmodes
```

## Numerical Method

The discrete and continuous solvers both use an anisotropic Fourier-space
trapezoidal rule on a centered mode box up to `kmax`.

For a given source-target box, the implementation:

- computes anisotropic mode spacings `Δk_x`, `Δk_y`, `Δk_z`
- builds the centered Fourier mode grid up to `kmax`
- applies one type-1 NUFFT to accumulate source data on that grid
- multiplies the Fourier coefficients by the truncated Laplace kernel, and by
  `windowhat(k)` for `ltkm3dd`
- evaluates requested outputs with `FINUFFT` type-2 interpolation

Gradient outputs are formed spectrally by multiplying the scaled coefficients by
`im * k_x`, `im * k_y`, and `im * k_z`. The current implementation batches
those three derivative fields into one many-vector type-2 plan, so the target
points are set once and the three gradient interpolations run together.

When `pg = 2` or `pgt = 2`, the solvers skip the potential interpolation
entirely and return only the gradient field.

The source-target box determines:

- the truncated-kernel radius `L`
- anisotropic spacings `Δk_x`, `Δk_y`, `Δk_z`
- the centered Fourier mode grid up to `kmax`

## Current Assumptions and Limitations

- `sources` and `targets` use FMM3D-style `3 x N` and `3 x M` layouts
- only charge sources are supported; dipole inputs are not implemented
- `pg` and `pgt` support `0`, `1`, and `2`
- `pg = 2` and `pgt = 2` are gradient-only modes
- gradients are returned as `3 x N` and `3 x M` matrices
- `ltkm3dd` expects `windowhat(k)` to be the radial Fourier transform of the window
- `ltkm3dd` source potentials omit the diagonal self interaction
- source self gradients are not corrected separately; for the symmetric Gaussian
  long-range kernel used in tests, the self gradient is zero
- `ltkm3dc` expects `charges` to be preweighted quadrature masses
- `ltkm3dc(...; kmax=nothing)` estimates Nyquist from average positive source-coordinate gaps, so explicit `kmax` is more reliable for strongly irregular point clouds
- `estimate_kcut3dc` uses the same average positive source-coordinate gap heuristic to infer Nyquist

## Validation

The current tests validate the discrete solver against the analytic Gaussian
long-range potential and gradient for:

- target potential and gradient
- source potential with the diagonal term removed
- source gradient
- combined source and target outputs

With the current NUFFT-based backend, the Gaussian long-range tests pass at
roughly `1e-10` relative error for potential and `5e-11` relative error for
gradients on the tested random systems.

The continuous solver is validated against the analytic Gaussian free-space
potential and gradient using preweighted quadrature masses, and also exercises
the `kmax = nothing` spacing heuristic on a uniform source cloud.
