# ltkm3dd FMM3D-Style Interface Design

## Goal

Revise `ltkm3dd` so its public API mirrors the FMM3D Julia interface style, while preserving the current truncated-kernel long-range evaluator internally.

## Target Public API

```julia
ltkm3dd(eps, sources; charges, targets=nothing, pg=0, pgt=0, nd=1, windowhat, lw, kmax)
```

with:

- `eps`: numerical tolerance, mapped to the FINUFFT tolerance
- `sources`: `3 x ns` source coordinates
- `charges`: required source strengths
- `targets`: optional `3 x nt` target coordinates
- `pg`: source-output selector
- `pgt`: target-output selector
- `nd`: number of density vectors
- `windowhat`: radial Fourier transform of the window function
- `lw`: effective window support size
- `kmax`: Fourier truncation radius

## FMM3D-Like Semantics

The interface should follow the FMM3D calling pattern:

- `pg = 0`: no source output
- `pg = 1`: source potentials only
- `pg = 2`: source potentials and gradients
- `pgt = 0`: no target output
- `pgt = 1`: target potentials only
- `pgt = 2`: target potentials and gradients

For this pass, only potential output is implemented, so:

- `pg ∈ {0, 1}` is supported
- `pgt ∈ {0, 1}` is supported
- `pg = 2` or `pgt = 2` throws `ArgumentError`

Source evaluation omits the self term, matching FMM3D behavior. Tests will remove the diagonal term manually from the pairwise Gaussian reference.

## Return Type

Return a small result struct shaped like the FMM3D output object:

```julia
struct TKMVals{T, VP, VG}
    pot::Union{Nothing, VP}
    grad::Union{Nothing, VG}
    pottarg::Union{Nothing, VP}
    gradtarg::Union{Nothing, VG}
    ier::Int
end
```

For this pass:

- `pot` is populated only when `pg == 1`
- `pottarg` is populated only when `pgt == 1`
- `grad` and `gradtarg` are always `nothing`
- `ier == 0` on success

## Internal Structure

The public interface should be a wrapper over a single internal evaluator that computes long-range potentials at arbitrary targets.

### Internal target-only evaluator

Keep a private helper with explicit TKM parameters:

```julia
_ltkm3dd_potential_only(sources, charges, targets, windowhat, lw, kmax, eps)
```

This helper:

- accepts coordinates in `3 x N` and `3 x M` form
- computes the combined source-target bounding box
- forms anisotropic `Δk_x`, `Δk_y`, `Δk_z`
- runs the type-1 NUFFT
- applies the truncated Laplace spectral factor and `windowhat`
- runs the type-2 NUFFT
- returns the target potential vector

### Public wrapper behavior

`ltkm3dd` should:

1. validate interface arguments
2. validate `sources` and `targets` are `3 x N` shaped
3. validate `charges` against `sources`
4. dispatch `_ltkm3dd_potential_only` on:
   - `sources` if `pg == 1`
   - `targets` if `pgt == 1`
5. wrap the outputs in `TKMVals`

## Data Layout

The implementation should switch from the current `N x 3` convention to FMM3D-compatible `3 x N` arrays.

Internally:

- source coordinates are read from rows `1:3`
- target coordinates are read from rows `1:3`
- returned `pot` has length `ns`
- returned `pottarg` has length `nt`

If `nd == 1`, `charges` is a vector of length `ns`.

For this pass, `nd != 1` is rejected. That keeps the interface aligned with FMM3D without prematurely implementing batched densities.

## Testing Strategy

Replace or revise the current tests to exercise the new interface:

- validation of `3 x N` source/target shape
- rejection of unsupported `pg`, `pgt`, and `nd`
- target-only evaluation through `pgt = 1`
- source-only evaluation through `pg = 1`
- combined source and target evaluation in one call
- Gaussian pairwise long-range accuracy against a direct double sum
- source-output accuracy with the pairwise reference diagonal removed manually

## Documentation

Update the README so the published API reflects:

- the FMM3D-style keyword-driven call pattern
- `3 x N` source/target coordinates
- required `charges`
- current support limited to potential-only evaluation
- current support limited to `nd = 1`

## Non-Goals

Not part of this redesign:

- gradient implementation
- `nd > 1` batched densities
- self-term recovery APIs
- a generic non-Laplace kernel backend
