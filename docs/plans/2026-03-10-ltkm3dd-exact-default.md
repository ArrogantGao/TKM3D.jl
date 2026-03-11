# ltkm3dd Exact-Default Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the spread-only backend into its own source file and make public `ltkm3dd(...)` use `_ltkm3dd_eval_exact` by default.

**Architecture:** Keep the exact evaluator and public API in `src/discrete.jl`, extract spread-only helpers and `_ltkm3dd_eval_spreadonly` into `src/discrete_spreadonly.jl`, and update module includes so the internal spread-only API remains available. Drive the behavior change with a failing regression test that compares public `ltkm3dd(...)` against the exact backend.

**Tech Stack:** Julia, FFTW, FINUFFT, stdlib `Test`

---

### Task 1: Add the regression test for the public default backend

**Files:**
- Modify: `test/discrete.jl`

**Step 1: Write the failing test**

Rename the public-backend comparison test so it states that `ltkm3dd` matches
the exact backend at sources and targets, and compute the reference values with
`TKM3D._ltkm3dd_eval_exact(...)` instead of the spread-only evaluator.

**Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. --startup-file=no -e 'push!(LOAD_PATH, "@stdlib"); using TKM3D, Test, Random, LinearAlgebra, SpecialFunctions, Statistics; include("test/discrete.jl")'
```

Expected: the renamed public-backend test fails because `ltkm3dd(...)`
currently uses `_ltkm3dd_eval_spreadonly`.

**Step 3: Commit**

```bash
git add test/discrete.jl
git commit -m "test: assert ltkm3dd uses exact backend"
```

### Task 2: Extract the spread-only implementation into a separate file

**Files:**
- Create: `src/discrete_spreadonly.jl`
- Modify: `src/discrete.jl`
- Modify: `src/TKM3D.jl`

**Step 1: Write minimal implementation**

Move `_ltkm3dd_make_spreadonly_plan`, all `_ltkm3dd_spreadonly_*` helpers, and
`_ltkm3dd_eval_spreadonly` into `src/discrete_spreadonly.jl`. Remove those
definitions from `src/discrete.jl`. Update `src/TKM3D.jl` to include the new
file.

**Step 2: Switch the public default**

Update the two public `ltkm3dd(...)` call sites in `src/discrete.jl` so they
use `_ltkm3dd_eval_exact(...)` instead of `_ltkm3dd_eval_spreadonly(...)`.

**Step 3: Run tests to verify they pass**

Run:

```bash
julia --project=. --startup-file=no -e 'push!(LOAD_PATH, "@stdlib"); using TKM3D, Test, Random, LinearAlgebra, SpecialFunctions, Statistics; include("test/discrete.jl")'
```

Expected: the public-backend regression test passes, and the spread-only
parity tests still pass.

**Step 4: Commit**

```bash
git add src/TKM3D.jl src/discrete.jl src/discrete_spreadonly.jl test/discrete.jl
git commit -m "refactor: restore exact default backend for ltkm3dd"
```

### Task 3: Verify benchmark and internal references still point at the intended backends

**Files:**
- Check: `benchmark/compare_ltkm3dd_eval.jl`
- Check: `test/discrete.jl`

**Step 1: Verify references**

Confirm the benchmark still compares exact (`_ltkm3dd_eval`) against
spread-only (`_ltkm3dd_eval_spreadonly`) and that internal spread-only tests
still call the spread-only evaluator explicitly.

**Step 2: Run focused verification**

Run:

```bash
rg -n "_ltkm3dd_eval_spreadonly|_ltkm3dd_eval_exact|_ltkm3dd_eval\\b" src test benchmark
```

Expected: public `ltkm3dd(...)` uses `_ltkm3dd_eval_exact`, while tests and
benchmark still reference `_ltkm3dd_eval_spreadonly` intentionally.

**Step 3: Commit**

```bash
git add docs/plans/2026-03-10-ltkm3dd-exact-default-design.md docs/plans/2026-03-10-ltkm3dd-exact-default.md
git commit -m "docs: add exact-default backend design and plan"
```
