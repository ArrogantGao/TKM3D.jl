# estimate_kcut3dc Design

## Goal

Add a new public TKM3D function that estimates a radial spectral cutoff from a
sampled continuous source cloud. The function should:

- accept the existing `3 x N` source API plus `charges`
- infer a Nyquist-limited Fourier box from the same average-gap heuristic used by
  `ltkm3dc(...; kmax = nothing)`
- run one 3D type-1 NUFFT on that full Nyquist box
- return the smallest `kcut` such that
  `max_{|q| > kcut} |F(q)| / max_q |F(q)| < tol`

## Chosen API

Add one new exported function:

```julia
estimate_kcut3dc(sources; charges, tol, eps = 1e-12)
```

Return a compact result object rather than only a scalar. The result should
include:

- `kcut`
- `kmax_nyquist`
- `axis_nyquist`
- `max_coeff`
- `tail_ratio`
- `nmodes`
- `Δk`

The full coefficient tensor should not be returned by default because it can be
large enough to turn a lightweight analysis helper into a memory hazard.

## Numerical Definition

For source coordinates `sources` in `3 x N` layout:

1. Reuse `_ltkm3dc_mean_positive_gap` to estimate per-axis average spacings
   `hx`, `hy`, `hz`.
2. Define axis Nyquist limits as `π / hx`, `π / hy`, `π / hz`.
3. Define the sampled box lengths as `extent + h` per axis, not only `max - min`,
   so regularly sampled source clouds behave like true grid samplings.
4. Define mode spacings by `Δk_x = 2π / L_x`, `Δk_y = 2π / L_y`, `Δk_z = 2π / L_z`.
5. Build centered mode boxes that extend to the per-axis Nyquist limits.
6. Shift and scale source coordinates into the same angular coordinates used by
   `nufft3d1`.
7. Run one type-1 NUFFT to obtain the coefficient box.
8. Convert centered integer modes to anisotropic physical wavevectors and compute
   their radii.
9. Find the smallest `kcut` whose relative pointwise tail is below `tol`.

`kmax_nyquist` should be reported as `maximum(axis_nyquist)`, while the actual
radial search is performed over the full anisotropic box and therefore may see
corner radii larger than that scalar.

## Placement

This belongs in the continuous side of TKM3D, not inside `ltkm3dc` itself.

Reason:
- it analyzes the source spectrum rather than evaluating the Laplace operator
- it naturally reuses the continuous-source spacing heuristic
- it keeps the `ltkm3dc` evaluator free of unrelated analysis behavior

The common cutoff helper and result type should live in `src/common.jl`, while
the source-only Nyquist box builder and public API should live in
`src/continuous.jl`.

## Verification

Use TDD with:

1. a focused helper test for the cutoff scan from synthetic radii/magnitudes
2. an integration test on a uniform Cartesian grid with constant charges, where
   only the zero mode should dominate and `kcut` should collapse to zero for a
   moderate tolerance
3. a README update with a short `estimate_kcut3dc` example
