# ltkm3dd Design

## Goal

Implement the discrete truncated-kernel long-range evaluator `ltkm3dd` for windowed point sources, and validate it against the long-range Gaussian term used in Ewald summation.

## Scope

- Implement `ltkm3dd` for potential-only evaluation (`pgt = 1`)
- Accept source and target coordinates only in `N x 3` form
- Support anisotropic Fourier spacings `Δk_x`, `Δk_y`, `Δk_z`
- Add tests that compare against a direct pairwise double sum for the Gaussian-windowed long-range potential

Out of scope for this pass:

- Gradient evaluation (`pgt = 2`)
- Continuous-source solver work
- Precomputation APIs

## API

The package keeps the existing exported entry point:

```julia
ltkm3dd(source, charge, target, W_hat, l_W, k_max, pgt, tol)
```

Expected inputs:

- `source::AbstractMatrix` with shape `N x 3`
- `charge::AbstractVector` with length `N`
- `target::AbstractMatrix` with shape `M x 3`
- `W_hat::Function`, radial Fourier transform of the source window
- `l_W`, effective real-space support of the window
- `k_max`, truncation wavenumber
- `pgt = 1` only
- `tol`, FINUFFT tolerance

The function returns a length-`M` vector of potentials.

## Numerical Method

Let the source and target bounding box edge lengths be `l_x`, `l_y`, `l_z`, and let

```math
L = \sqrt{l_x^2 + l_y^2 + l_z^2} + l_W.
```

The long-range potential is approximated by the anisotropic trapezoidal rule

```math
\phi(y) \approx \frac{\Delta k_x \Delta k_y \Delta k_z}{(2\pi)^3}
\sum_{k_x,k_y,k_z}
\widehat{G}_L(k) \widehat{W}(k)
\left(\sum_j q_j e^{-i k \cdot x_j}\right)
e^{i k \cdot y},
```

with

```math
\widehat{G}_L(k) = 2\left(\frac{\sin(L |k| / 2)}{|k|}\right)^2,
\qquad
\widehat{G}_L(0) = \frac{L^2}{2}.
```

The Fourier grid is centered and tensor-product:

- `Δk_d = prevfloat(2π / (l_d + L + l_W))` for `d in {x, y, z}`
- `mmax_d = ceil(Int, k_max / Δk_d)`
- `k_d = Δk_d * (-mmax_d:mmax_d)`

This follows the note's pairwise aliasing condition independently in each direction.

## Implementation Structure

### Shared utilities

Add small helpers in `src/common.jl`:

- `as_nx3(points)` validation/conversion for `N x 3` inputs only
- `truncated_laplace3d_hat(k, L)` with the `k = 0` limit
- helpers for box lengths, centers, and centered mode axes

### Discrete evaluator

`src/discrete.jl` will:

1. validate `source`, `target`, `charge`, and `pgt`
2. compute box lengths from the union of sources and targets
3. build anisotropic mode axes from `k_max` and `l_W`
4. evaluate the type-1 NUFFT structure factor
5. multiply by `W_hat(|k|)` and `truncated_laplace3d_hat(|k|, L)`
6. evaluate targets with type-2 NUFFT
7. apply the quadrature prefactor

The code will use the physical coordinates directly with anisotropic mode arrays, instead of forcing a normalized cube. This keeps the three `Δk` values explicit and avoids hiding direction-dependent scaling.

## Validation Strategy

Tests use the Gaussian Ewald long-range window:

```julia
W_hat(k) = exp(-0.25 * sigma^2 * k^2)
```

The reference is a direct pairwise double sum:

```math
\phi_{\mathrm{ref}}(y_i)
= \sum_j q_j \frac{\operatorname{erf}(|y_i - x_j| / \sigma)}{4\pi |y_i - x_j|},
```

with the analytic self-limit used when `y_i = x_j`.

Test coverage:

- input shape and argument validation
- translation of `N x 3` arrays into correct potentials
- agreement with the pairwise Gaussian reference on random source/target sets
- a target-equals-source case that exercises the finite self term

## Accuracy Expectations

The tests should demonstrate that `ltkm3dd` reproduces the Gaussian-windowed long-range reference to a relative error commensurate with the chosen Fourier truncation and NUFFT tolerance. The dominant error target for the regression test is the TKM discretization, not direct-sum noise.
