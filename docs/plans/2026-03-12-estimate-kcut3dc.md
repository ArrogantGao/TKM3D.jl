# estimate_kcut3dc Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a public TKM3D helper that estimates a Nyquist-based radial spectral cutoff from sampled continuous source data.

**Architecture:** Reuse the continuous-source average-gap heuristic, construct a source-only Nyquist Fourier box, run one type-1 NUFFT, and scan the resulting anisotropic mode box for the smallest cutoff whose relative pointwise tail is below tolerance. Keep the cutoff scan and result container in shared helpers, and expose one small public API from the continuous module.

**Tech Stack:** Julia, FINUFFT.jl, existing TKM3D continuous helpers, TDD, README docs

---

### Task 1: Add failing tests for the new API

**Files:**
- Modify: `/mnt/home/xgao1/codes/TKM3D.jl/.worktrees/tkm3d-estimate-kcut/test/continuous.jl`

**Step 1: Write the failing tests**

Add:
- a focused test for a cutoff helper on synthetic radii/magnitudes
- an integration test for `estimate_kcut3dc` on a uniform Cartesian grid with
  constant charges

The integration test should verify:
- returned fields exist
- `kcut ≈ 0`
- `tail_ratio < tol`
- `kmax_nyquist > 0`

**Step 2: Run test to verify it fails**

Run:
`julia --project=. --color=no test/continuous.jl`

Expected:
- FAIL because the helper and public API do not exist yet.

### Task 2: Implement shared cutoff helpers and result type

**Files:**
- Modify: `/mnt/home/xgao1/codes/TKM3D.jl/.worktrees/tkm3d-estimate-kcut/src/common.jl`
- Modify: `/mnt/home/xgao1/codes/TKM3D.jl/.worktrees/tkm3d-estimate-kcut/src/TKM3D.jl`

**Step 1: Add the result container**

Add a small result type for:
- `kcut`
- `kmax_nyquist`
- `axis_nyquist`
- `max_coeff`
- `tail_ratio`
- `nmodes`
- `Δk`

**Step 2: Add helper routines**

Add shared helpers for:
- centered integer mode indices
- smallest cutoff from synthetic radii/magnitudes
- relative pointwise tail ratio

**Step 3: Re-run the focused helper test**

Run:
`julia --project=. --color=no test/continuous.jl`

Expected:
- helper portion passes
- integration test still fails until the public API is implemented.

### Task 3: Implement the Nyquist-box spectral estimator

**Files:**
- Modify: `/mnt/home/xgao1/codes/TKM3D.jl/.worktrees/tkm3d-estimate-kcut/src/continuous.jl`

**Step 1: Add source-only box helpers**

Implement helpers to:
- estimate per-axis average gaps
- derive per-axis Nyquist limits
- derive sampled box lengths as `extent + h`
- derive `Δk`
- convert centered integer modes to anisotropic physical radii

**Step 2: Implement `estimate_kcut3dc`**

Use one type-1 NUFFT over the Nyquist mode box and build the result object.

**Step 3: Re-run continuous tests**

Run:
`julia --project=. --color=no test/continuous.jl`

Expected:
- all continuous tests pass.

### Task 4: Update package docs

**Files:**
- Modify: `/mnt/home/xgao1/codes/TKM3D.jl/.worktrees/tkm3d-estimate-kcut/README.md`

**Step 1: Add the new API to the public README**

Document:
- function signature
- meaning of returned fields
- short example

**Step 2: Re-run tests**

Run:
`julia --project=. --color=no test/runtests.jl`

Expected:
- full suite passes.

### Task 5: Representative example run

**Files:**
- No new files required

**Step 1: Run a small direct example**

Run a Julia one-liner or short script in the worktree that calls
`estimate_kcut3dc` on a simple structured source grid and prints the result.

**Step 2: Confirm output sanity**

Verify:
- `kcut` is finite and nonnegative
- `tail_ratio < tol`
- `nmodes` and `Δk` are sensible
