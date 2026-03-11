# ltkm3dd FINUFFT Spread-Only Backend Design

## Goal

Replace the current high-level `nufft3d1` / `nufft3d2` backend inside `ltkm3dd`
with a FINUFFT guru-interface backend that uses `spreadinterponly=1` plus
explicit FFTs, so the internal upsampling factor can be driven to `1.0` when
possible and to `1.0001` otherwise.

## Scope

- keep the public `ltkm3dd(eps, sources; charges, targets, pg, pgt, windowhat, lw, kmax)` interface
- preserve current source/target potential and gradient semantics
- preserve the current source self-potential subtraction
- validate the new backend against the analytic Gaussian long-range reference
- add a deterministic fallback from `upsampfac=1.0` to `upsampfac=1.0001`

Out of scope:

- changing the public API
- adding dipole support
- adding batched densities
- fixing the pre-existing Julia shutdown segfault seen after tests complete

## Current Constraints

Local probing against the installed FINUFFT 3.5.0 Julia wrapper shows:

- the guru interface exposes `finufft_makeplan`, `finufft_setpts!`, and `finufft_exec`
- `spreadinterponly=1` is supported for type 1 and type 2 plans
- exact `upsampfac=1.0` is rejected with `ERR_UPSAMPFAC_TOO_SMALL`
- `upsampfac=1.0001` is accepted in spread-only mode

That means the backend must not assume that exact `1.0` is usable in the local
FINUFFT build.

## Numerical Architecture

The current code evaluates the discrete long-range TKM operator as

```math
u(y) = \frac{1}{(2\pi)^3} \int \widehat{G}_L(k)\widehat{W}(k)
\left(\sum_j q_j e^{-i k \cdot x_j}\right)e^{i k \cdot y}\,dk.
```

The spread-only backend keeps the same discrete Fourier grid and replaces the
exact type-1 and type-2 NUFFTs with the following pipeline:

1. type-1 spread-only plan: spread source charges to a uniform grid
2. explicit forward FFT on that grid
3. multiply in Fourier space by `truncated_laplace3d_hat(k, L) * windowhat(k)`
4. explicit inverse FFT back to a uniform grid
5. type-2 interpolate-only plan: interpolate grid values to targets

The key point is that the same FINUFFT spreading kernel is used at both ends.
The Fourier transform of the spread grid already carries one factor of the
FINUFFT kernel transform, and the interpolate-only stage expects exactly that
same factor on the inverse-FFT grid. Because of that pairing, no explicit NUFFT
deconvolution factor is needed in the first implementation. The only spectral
physics multiplier remains

```math
\widehat{G}_L(k)\widehat{W}(k).
```

This keeps `windowhat` as the user-specified source window from the TKM/Ewald
derivation. It is not replaced by the FINUFFT kernel.

## Grid and Ordering Choices

- keep the current anisotropic mode-count logic based on `Δk_x`, `Δk_y`, `Δk_z`
- use the same `(length(kx), length(ky), length(kz))` grid as the current code
- create spread-only guru plans with `modeord=1` so the spread grid stays in
  FFT ordering and does not need extra `fftshift` / `ifftshift` work
- prefer `upsampfac=1.0`, but on `ERR_UPSAMPFAC_TOO_SMALL` recreate the plan
  with `upsampfac=1.0001`

The first implementation should not silently enlarge the grid if FINUFFT says
the fine grid is too small for the requested spread width. That is a separate
policy decision. For now, plan creation errors other than the explicit
`upsampfac` fallback should surface.

## Gradients

Gradients stay in spectral form. After the forward FFT and base spectral
scaling, compute the three gradient spectra by multiplying by `im*kx`,
`im*ky`, and `im*kz`, then inverse FFT each one and interpolate them with the
same type-2 interpolate-only plan.

This preserves the current `pg = 2` / `pgt = 2` behavior and keeps gradient
logic aligned with the existing exact-backend implementation.

## Self Term

Keep the current source self-potential correction:

```math
c_{\mathrm{self}} =
\frac{\Delta k_x \Delta k_y \Delta k_z}{(2\pi)^3}
\sum_k \widehat{G}_L(k)\widehat{W}(k).
```

The public source potential path should continue to subtract
`charges .* selfconst`, while source gradients remain uncorrected. This matches
the current semantics and the Gaussian long-range validation setup.

## Migration Strategy

Implement the spread-only path alongside the current exact backend first.

- keep a private exact reference evaluator during the migration
- add a new private spread-only evaluator
- compare both on small random systems in tests
- switch `ltkm3dd` to the spread-only evaluator only after parity is verified

This reduces debugging ambiguity when the guru-plan path is first introduced.

## Testing Strategy

Keep the existing analytic Gaussian long-range potential/gradient tests and add
backend-specific coverage:

- unit coverage for the `upsampfac=1.0 -> 1.0001` fallback helper
- small-system parity tests between the current exact backend and the new
  spread-only backend
- existing target and source Gaussian long-range potential/gradient accuracy
  tests, with the source diagonal removed from the potential reference only

The worktree baseline already shows `24/24` passing before the existing Julia
shutdown segfault on process exit. That segfault should be treated as unrelated
background noise unless guru-plan lifecycle changes make it materially worse.

## Documentation Impact

Update the README after implementation to document:

- that `ltkm3dd` now uses a FINUFFT spread-only backend internally
- the `upsampfac` fallback policy
- that public semantics are unchanged
