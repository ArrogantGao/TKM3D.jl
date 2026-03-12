# ltkm3dc Gaussian 100-Target Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible Gaussian regression that samples a truncated 3D Gaussian on a uniform mesh and checks `ltkm3dc` target potentials at 100 random targets against the analytic reference.

**Architecture:** Extend `test/continuous.jl` with one helper for random `3 x N` target generation and one new testset. Keep the solver unchanged unless the new regression reveals a real discrepancy.

**Tech Stack:** Julia, Test, Random, LinearAlgebra, SpecialFunctions, TKM3D

---

### Task 1: Add the failing 100-target Gaussian regression

**Files:**
- Modify: `test/continuous.jl`
- Test: `test/continuous.jl`

**Step 1: Write the failing test**

```julia
@testset "ltkm3dc matches analytic Gaussian potential at 100 random targets" begin
    sigma = 0.2
    center = (0.1, -0.2, 0.3)
    region = sigma * sqrt(2.0 * log(1.0e12))
    sources, charges = make_uniform_gaussian_quadrature(26, region, center, sigma)
    targets = make_random_targets_3xn(100, 1.2 * region; seed = 20260311)

    out = ltkm3dc(1e-12, sources; charges, targets, pgt = 1, kmax = sqrt(2.0 * log(1.0e12)) / sigma)
    ref_pot = [...]

    @test norm(out.pottarg .- ref_pot) / norm(ref_pot) < 1e-8
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=. --startup-file=no -e 'using TKM3D, Test; include("test/continuous.jl")'`

Expected: FAIL on the new 100-target Gaussian accuracy assertion.

**Step 3: Write minimal implementation**

Add only the helper needed to generate random `3 x N` targets and keep the new regression self-contained in `test/continuous.jl`.

**Step 4: Run test to verify it passes**

Run: `julia --project=. --startup-file=no -e 'using TKM3D, Test; include("test/continuous.jl")'`

Expected: the file still fails on the accuracy threshold, not from missing test helpers.

**Step 5: Commit**

```bash
git add test/continuous.jl
git commit -m "test: add ltkm3dc 100-target Gaussian regression"
```

### Task 2: Refine the quadrature until the regression is stable

**Files:**
- Modify: `test/continuous.jl`
- Test: `test/continuous.jl`

**Step 1: Write the failing test**

Use the red test from Task 1.

**Step 2: Run test to verify it fails**

Run: `julia --project=. --startup-file=no -e 'using TKM3D, Test; include("test/continuous.jl")'`

Expected: FAIL because the initial uniform source mesh is too coarse for the requested analytic tolerance.

**Step 3: Write minimal implementation**

Increase only the source mesh resolution and relax the tolerance to a value justified by fresh measurements, while keeping:

- uniform Gaussian source sampling
- 100 random targets
- target region `[-1.2R, 1.2R]^3`
- analytic potential reference

**Step 4: Run test to verify it passes**

Run: `julia --project=. --startup-file=no -e 'using TKM3D, Test; include("test/continuous.jl")'`

Expected: all continuous tests pass.

**Step 5: Commit**

```bash
git add test/continuous.jl
git commit -m "test: stabilize ltkm3dc 100-target Gaussian regression"
```

### Task 3: Re-run the full suite

**Files:**
- Verify: `test/runtests.jl`
- Verify: `test/continuous.jl`

**Step 1: Write the failing test**

No new test. Use the completed regression from Tasks 1-2.

**Step 2: Run test to verify it fails**

Run: `julia --project=. --startup-file=no test/runtests.jl`

Expected: if any regressions exist, this command exposes them.

**Step 3: Write minimal implementation**

No additional code unless the full suite surfaces a genuine regression.

**Step 4: Run test to verify it passes**

Run: `julia --project=. --startup-file=no test/runtests.jl`

Expected: full suite passes.

**Step 5: Commit**

```bash
git add test/continuous.jl
git commit -m "test: verify Gaussian target regression"
```
