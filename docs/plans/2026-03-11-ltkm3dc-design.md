# ltkm3dc Continuous Backend Design

## Goal

Implement `ltkm3dc` for continuous free-space Laplace convolution using preweighted quadrature masses at arbitrary source points.

## Scope

- add a public `ltkm3dc(eps, sources; charges, targets=nothing, pg=0, pgt=0, kmax=nothing)` interface
- duplicate the continuous-source spectral flow instead of reusing the discrete windowed backend
- support potential-only and potential-plus-gradient outputs at sources and targets
- estimate `kmax` from approximate source spacing when it is not provided
- validate the implementation against an analytic Gaussian continuous-source reference

Out of scope:

- dipole support
- non-Laplace kernels
- exact bandwidth detection for irregular point clouds
- vectorized multiple densities

## Target Public API

```julia
ltkm3dc(eps, sources; charges, targets=nothing, pg=0, pgt=0, kmax=nothing)
```

Supported output selectors:

- `pg = 0, 1, 2`
- `pgt = 0, 1, 2`

Meaning:

- `0`: no output
- `1`: potential only
- `2`: potential and gradient

The return value is:

```julia
TKMVals(pot, grad, pottarg, gradtarg, ier)
```

with:

- `pot::Union{Nothing, Vector}`
- `grad::Union{Nothing, Matrix}` as `3 x ns`
- `pottarg::Union{Nothing, Vector}`
- `gradtarg::Union{Nothing, Matrix}` as `3 x nt`
- `ier == 0` on success

## Numerical Method

This path follows the continuous-source formulation in the note.

For the combined source-target box with edge lengths `(l_x, l_y, l_z)`, choose the truncated-kernel radius

```math
L = \sqrt{l_x^2 + l_y^2 + l_z^2}.
```

Then choose anisotropic Fourier spacings

```math
\Delta k_x \leq \frac{2\pi}{l_x + L}, \quad
\Delta k_y \leq \frac{2\pi}{l_y + L}, \quad
\Delta k_z \leq \frac{2\pi}{l_z + L},
```

implemented with `prevfloat(...)` to stay strictly below the aliasing limit.

The Fourier coefficient tensor is

```math
\widehat{u}(k) = \widehat{G}_L(k)\sum_j q_j e^{-i k \cdot x_j},
```

where `q_j` are already preweighted quadrature masses. There is no window factor and no self-term correction.

Potential evaluation uses the type-2 NUFFT with the usual trapezoidal prefactor

```math
\Delta k_x \Delta k_y \Delta k_z / (2\pi)^3.
```

Gradients are obtained by multiplying the scaled coefficients by `im*k_x`, `im*k_y`, and `im*k_z` before the type-2 NUFFT.

## kmax Selection

If `kmax` is provided, it must be positive.

If `kmax` is omitted, estimate average spacings along each coordinate axis:

- sort each source coordinate axis independently
- collect positive consecutive gaps
- use the mean positive gap on each axis as `dx`, `dy`, `dz`
- set `kmax = min(π/dx, π/dy, π/dz)`

This is a heuristic for nearly uniform point sets. If any axis lacks a usable positive-gap estimate, the call should throw and require explicit `kmax`.

## Internal Structure

Add a continuous-only backend in `src/continuous.jl`:

```julia
_ltkm3dc_estimate_kmax(sources)
_ltkm3dc_eval(sources, charges, targets, kmax, eps; need_grad=false)
ltkm3dc(eps, sources; charges, targets=nothing, pg=0, pgt=0, kmax=nothing)
```

This keeps the continuous source path independent from `ltkm3dd`, which depends on window parameters and source self-term handling that do not apply here.

## Validation Strategy

Use a Gaussian density sampled by quadrature:

```math
\rho(x) = \frac{1}{(2\pi\sigma^2)^{3/2}} e^{-|x-c|^2 / (2\sigma^2)}.
```

The exact potential and gradient are

```math
\phi(r) = \frac{\operatorname{erf}(r / (\sqrt{2}\sigma))}{4\pi r}
```

and

```math
\nabla \phi(r) = \phi'(r)\hat{r}.
```

Tests should cover:

- argument validation
- target potential/gradient accuracy for explicit `kmax`
- source and target output shape behavior
- `kmax = nothing` using the spacing heuristic on a near-uniform cloud
