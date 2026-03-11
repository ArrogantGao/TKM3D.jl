# ltkm3dd Exact-Default Backend Design

## Goal

Move the spread-only evaluator into its own source file and make the public
`ltkm3dd` API use the exact evaluator by default again.

## Scope

- split the spread-only implementation out of `src/discrete.jl`
- keep `_ltkm3dd_eval_spreadonly` available for internal tests and benchmarks
- keep `_ltkm3dd_eval` and `_ltkm3dd_eval_exact` in `src/discrete.jl`
- switch public `ltkm3dd(...)` source and target evaluation paths to
  `_ltkm3dd_eval_exact`
- update tests so the public API is checked against the exact backend

Out of scope:

- changing the public `ltkm3dd` signature
- removing the spread-only backend
- changing benchmark coverage or benchmark data
- fixing the pre-existing Julia 1.12 test-environment dependency issue

## Current State

Today `src/discrete.jl` contains both the exact evaluator and all
spread-only helpers. The public `ltkm3dd(...)` entry point directly calls
`_ltkm3dd_eval_spreadonly` for both source and target work, while
`_ltkm3dd_eval_exact` is only a thin wrapper around `_ltkm3dd_eval`.

That means the default public behavior is coupled to the spread-only code path
even though an exact backend still exists.

## Design

Create a new `src/discrete_spreadonly.jl` file containing:

- `_ltkm3dd_make_spreadonly_plan`
- all `_ltkm3dd_spreadonly_*` helpers
- `_ltkm3dd_eval_spreadonly`

Leave the following in `src/discrete.jl`:

- `_ltkm3dd_eval`
- `_ltkm3dd_eval_exact`
- `ltkm3dd(...)`

Update `src/TKM3D.jl` to include `discrete_spreadonly.jl` alongside the
existing `discrete.jl` include so the internal spread-only API stays available
to tests and benchmark scripts.

## Public Behavior

The public `ltkm3dd(...)` implementation should call `_ltkm3dd_eval_exact` in
both places where it currently calls `_ltkm3dd_eval_spreadonly`:

- source evaluation with self-term subtraction
- target evaluation

This restores the exact backend as the default policy without changing the
function signature or result structure.

## Testing

Keep the spread-only parity tests, but change the public-backend regression
test so `ltkm3dd(...)` is compared against `_ltkm3dd_eval_exact` instead of the
spread-only evaluator. The tolerance should be tight enough to fail under the
current spread-only default and pass once the exact backend is wired back in.

## Risks

- include-order mistakes could make spread-only helpers unavailable
- moving code may accidentally leave references behind in `discrete.jl`
- public exact-default behavior may expose existing differences that were
  masked by the prior spread-only test expectations
