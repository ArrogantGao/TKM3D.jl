# ltkm3dd Gradient Extension Design

## Goal

Extend the FMM3D-style `ltkm3dd` interface so `pg = 2` and `pgt = 2` return long-range gradients for the discrete Laplace TKM evaluator.

## Scope

- add gradient evaluation for source and target outputs
- keep the FMM3D-style public interface
- remove the `nd` parameter from the public API
- validate gradients against the analytic Gaussian long-range gradient

Out of scope:

- dipole support
- vectorized multiple densities
- non-Laplace kernels

## Target Public API

```julia
ltkm3dd(eps, sources; charges, targets=nothing, pg=0, pgt=0, windowhat, lw, kmax)
```

Supported output selectors:

- `pg = 0, 1, 2`
- `pgt = 0, 1, 2`

Meaning:

- `0`: no output
- `1`: potential only
- `2`: potential and gradient

The return value remains:

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

The existing Fourier-space long-range potential evaluator already forms the scaled spectral tensor

```math
\widehat{u}(k) = \widehat{G}_L(k)\widehat{W}(k)\sum_j q_j e^{-i k \cdot x_j}.
```

The gradient is obtained componentwise from

```math
\nabla u(y) = \frac{1}{(2\pi)^3}\int i k \widehat{u}(k)e^{i k \cdot y}\,dk.
```

Discretely, each component is evaluated by multiplying the spectral tensor by `im*kx`, `im*ky`, or `im*kz` before the type-2 NUFFT.

This matches the pattern already used in `FBCPoisson.jl`.

## Internal Structure

Replace the current private potential-only helper with a combined evaluator:

```julia
_ltkm3dd_eval(sources, charges, targets, windowhat, lw, kmax, eps;
              need_grad=false, return_selfconst=false)
```

Behavior:

- always returns the potential
- returns `grad::Matrix{T}` when `need_grad = true`
- returns the scalar spectral self constant when `return_selfconst = true`

This avoids duplicating the type-1 NUFFT and spectral scaling between the potential and gradient paths.

## Source Self Term

For source outputs:

- the potential diagonal is omitted, matching FMM3D semantics
- the diagonal correction is applied by subtracting `q_i * c_self` from the source potential
- no diagonal correction is applied to the gradient

For the Gaussian long-range kernel used in tests, the self gradient is zero by symmetry.

## Testing Strategy

Add analytic Gaussian references for both potential and gradient.

Use:

```math
\phi(r) = \frac{\operatorname{erf}(r/\sigma)}{4\pi r}
```

and

```math
\nabla \phi(r) = \phi'(r)\hat{r},
```

with the finite zero-gradient limit at `r = 0`.

Test coverage:

- validation that `nd` is no longer accepted
- target `pgt = 2` potential/gradient accuracy
- source `pg = 2` potential/gradient accuracy, with the source diagonal removed from the potential reference only
- combined `pg = 2, pgt = 2` output shape checks

## Documentation

Update the README to reflect:

- `pg/pgt = 2` support
- gradients returned as `3 x N` matrices
- `nd` removed
- no dipole support
