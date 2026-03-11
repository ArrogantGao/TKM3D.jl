# TKM3D

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArrogantGao.github.io/TKM3D.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArrogantGao.github.io/TKM3D.jl/dev/)
[![Build Status](https://github.com/ArrogantGao/TKM3D.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/TKM3D.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ArrogantGao/TKM3D.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/TKM3D.jl)

Implementation of the truncated kernel method for 3D systems with `FINUFFT`.

## Current Status

The current code implements only the **discrete long-range Laplace solver** `ltkm3dd`.

It evaluates the windowed long-range interaction
```math
\phi(y) = \int_{\Omega'} \sum_i \frac{W(x - x_i)}{4 \pi |y - x|} q_i \, dx,
```
where `W` is a smooth window, for example the Gaussian window used in Ewald
summation.

The continuous interface `ltkm3dc` is not implemented yet.

## Public API

The current entry point is

`ltkm3dd(eps, sources; charges, targets=nothing, pg=0, pgt=0, windowhat, lw, kmax)`

with FMM3D-like source and target output flags.

### Arguments

- `eps`: requested spread/interpolation tolerance
- `sources`: source coordinates in `3 x ns` layout
- `charges`: source strengths, length `ns`
- `targets`: optional target coordinates in `3 x nt` layout
- `pg`: source output selector
- `pgt`: target output selector
- `windowhat(k)`: radial Fourier transform of the window
- `lw`: effective real-space support of the window
- `kmax`: Fourier truncation radius

### Output Flags

- `pg = 0`: no source output
- `pg = 1`: source potentials only
- `pg = 2`: source potentials and gradients
- `pgt = 0`: no target output
- `pgt = 1`: target potentials only
- `pgt = 2`: target potentials and gradients

### Return Value

`ltkm3dd` returns a `TKMVals` object with fields:

- `pot`: source potential, length `ns`, or `nothing`
- `grad`: source gradient, size `3 x ns`, or `nothing`
- `pottarg`: target potential, length `nt`, or `nothing`
- `gradtarg`: target gradient, size `3 x nt`, or `nothing`
- `ier`: currently `0` on success

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
    pg = 2,
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

## Numerical Method

The discrete solver uses an anisotropic Fourier-space trapezoidal rule with the
truncated Laplace kernel, but the current backend no longer calls the full
type-1 and type-2 NUFFTs directly.

Instead it uses the `FINUFFT` guru interface in spread/interpolate-only mode:

- spread sources to a regular fine grid
- apply an explicit FFT on that fine grid
- deconvolve/shuffle from the fine grid to the requested centered mode box
- apply the truncated-kernel and window multipliers in Fourier space
- deconvolve/shuffle back to the fine grid
- apply an explicit inverse FFT
- interpolate from the fine grid to sources or targets

The fine-grid sizes are chosen with the same `next235even(max(ceil(σ m), 2nspread))`
logic used by `FINUFFT`, with the internal spread-only upsampling factor set to
`1.00001`. The current local `FINUFFT` build rejects exact `upsampfac = 1.0` in
spread-only mode, so the implementation uses a fixed accepted value above `1`
instead of probing `1.0` at runtime.

The source-target box determines:

- the truncated-kernel radius `L`
- anisotropic spacings `Δk_x`, `Δk_y`, `Δk_z`
- the centered Fourier mode grid up to `kmax`

Gradients are computed in Fourier space by multiplying the scaled coefficients by
`im * k_x`, `im * k_y`, and `im * k_z` before the inverse fine-grid FFT and
interpolation step.

## Current Assumptions and Limitations

- `sources` and `targets` use FMM3D-style `3 x N` and `3 x M` layouts
- only charge sources are supported; dipole inputs are not implemented
- `windowhat(k)` is the radial Fourier transform of the window
- `pg` and `pgt` support `0`, `1`, and `2`
- gradients are returned as `3 x N` and `3 x M` matrices
- source potentials omit the diagonal self interaction
- source self gradients are not corrected separately; for the symmetric Gaussian
  long-range kernel used in tests, the self gradient is zero
- the current spread-only backend uses an ES spreader kernel internally
- only the discrete Laplace long-range path is implemented

## Validation

The current tests validate the discrete solver against the analytic Gaussian
long-range potential and gradient for:

- target potential and gradient
- source potential with the diagonal term removed
- source gradient
- combined source and target outputs

With the current spread-only backend, the Gaussian long-range tests pass at
roughly `1e-10` relative error for potential and `5e-11` relative error for
gradients on the tested random systems.
