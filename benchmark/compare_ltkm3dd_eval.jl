using BenchmarkTools
using LinearAlgebra
using Printf
using Random
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
if REPO_ROOT ∉ LOAD_PATH
    pushfirst!(LOAD_PATH, REPO_ROOT)
end

using TKM3D

const DEFAULT_DIGITS = (3, 6, 9, 12)
const DEFAULT_SIGMAS = (0.1, 0.5, 1.0)
const DEFAULT_CASES = (
    (ns = 100, nt = 100, need_grad = false),
    (ns = 100, nt = 100, need_grad = true),
    (ns = 400, nt = 400, need_grad = false),
    (ns = 400, nt = 400, need_grad = true),
)
const DEFAULT_OUTPUT = joinpath(@__DIR__, "ltkm3dd_eval_comparison.csv")

digits_to_eps(digits) = [10.0^(-d) for d in digits]

gaussian_windowhat(sigma::Float64) = k -> exp(-0.25 * sigma^2 * k^2)

gaussian_kmax_from_eps(sigma::Float64, eps::Float64) = 2.0 * sqrt(log(1.0 / eps)) / sigma

function relerr(reference::AbstractArray, value::AbstractArray)
    denom = max(norm(reference), eps(Float64))
    return norm(reference .- value) / denom
end

achieved_digits(err::Real) = err > 0 ? -log10(float(err)) : Inf

function measure(f; samples::Int = 10)
    trial = @benchmark $f() samples = samples evals = 1
    med = median(trial)
    lo = minimum(trial)
    return (
        median_s = med.time / 1e9,
        min_s = lo.time / 1e9,
        median_bytes = med.memory,
        min_bytes = lo.memory,
        median_gctime_s = med.gctime / 1e9,
        min_gctime_s = lo.gctime / 1e9,
        median_allocs = med.allocs,
        min_allocs = lo.allocs,
    )
end

function benchmark_case(ns::Int, nt::Int, sigma::Float64, eps::Float64; need_grad::Bool, seed::Int, samples::Int)
    rng = MersenneTwister(seed)
    sources = rand(rng, 3, ns)
    targets = rand(rng, 3, nt)
    charges = randn(rng, ns)
    charges .-= mean(charges)

    lw = 4.0 * sigma
    kmax = gaussian_kmax_from_eps(sigma, eps)
    windowhat = gaussian_windowhat(sigma)

    f_exact() = TKM3D._ltkm3dd_eval(sources, charges, targets, windowhat, lw, kmax, eps; need_grad = need_grad)
    f_spread() = TKM3D._ltkm3dd_eval_spreadonly(sources, charges, targets, windowhat, lw, kmax, eps; need_grad = need_grad)

    pot_exact, grad_exact, _ = f_exact()
    pot_spread, grad_spread, _ = f_spread()

    exact_stats = measure(f_exact; samples = samples)
    spread_stats = measure(f_spread; samples = samples)

    pot_relerr = relerr(pot_exact, pot_spread)
    grad_relerr = need_grad ? relerr(grad_exact, grad_spread) : NaN

    return (
        ns = ns,
        nt = nt,
        sigma = sigma,
        lw = lw,
        kmax = kmax,
        need_grad = need_grad,
        eps = eps,
        exact_median_ms = 1e3 * exact_stats.median_s,
        spread_median_ms = 1e3 * spread_stats.median_s,
        exact_min_ms = 1e3 * exact_stats.min_s,
        spread_min_ms = 1e3 * spread_stats.min_s,
        exact_median_bytes = exact_stats.median_bytes,
        spread_median_bytes = spread_stats.median_bytes,
        exact_min_bytes = exact_stats.min_bytes,
        spread_min_bytes = spread_stats.min_bytes,
        exact_median_gc_ms = 1e3 * exact_stats.median_gctime_s,
        spread_median_gc_ms = 1e3 * spread_stats.median_gctime_s,
        exact_min_gc_ms = 1e3 * exact_stats.min_gctime_s,
        spread_min_gc_ms = 1e3 * spread_stats.min_gctime_s,
        exact_median_allocs = exact_stats.median_allocs,
        spread_median_allocs = spread_stats.median_allocs,
        exact_min_allocs = exact_stats.min_allocs,
        spread_min_allocs = spread_stats.min_allocs,
        exact_over_spread = exact_stats.median_s / spread_stats.median_s,
        pot_relerr = pot_relerr,
        grad_relerr = grad_relerr,
        pot_digits_achieved = achieved_digits(pot_relerr),
        grad_digits_achieved = need_grad ? achieved_digits(grad_relerr) : NaN,
    )
end

function build_benchmark_rows(;
    digits = DEFAULT_DIGITS,
    sigmas = DEFAULT_SIGMAS,
    cases = DEFAULT_CASES,
    samples::Int = 10,
    base_seed::Int = 1234,
)
    rows = NamedTuple[]
    eps_values = digits_to_eps(digits)

    for (digit, eps) in zip(digits, eps_values)
        for (sigma_idx, sigma) in enumerate(sigmas)
            for case in cases
                seed = base_seed + 100_000 * sigma_idx + 10_000 * digit + 100 * case.ns + 10 * case.nt + (case.need_grad ? 1 : 0)
                row = benchmark_case(case.ns, case.nt, sigma, eps; need_grad = case.need_grad, seed = seed, samples = samples)
                push!(rows, merge((
                    requested_digits = digit,
                    pot_meets_requested_digits = row.pot_digits_achieved >= digit,
                    grad_meets_requested_digits = case.need_grad ? (row.grad_digits_achieved >= digit) : false,
                ), row))
            end
        end
    end

    return rows
end

function csv_header()
    return (
        "requested_digits",
        "eps",
        "sigma",
        "lw",
        "kmax",
        "ns",
        "nt",
        "need_grad",
        "exact_median_ms",
        "spread_median_ms",
        "exact_min_ms",
        "spread_min_ms",
        "exact_median_bytes",
        "spread_median_bytes",
        "exact_min_bytes",
        "spread_min_bytes",
        "exact_median_gc_ms",
        "spread_median_gc_ms",
        "exact_min_gc_ms",
        "spread_min_gc_ms",
        "exact_median_allocs",
        "spread_median_allocs",
        "exact_min_allocs",
        "spread_min_allocs",
        "exact_over_spread",
        "pot_relerr",
        "grad_relerr",
        "pot_digits_achieved",
        "grad_digits_achieved",
        "pot_meets_requested_digits",
        "grad_meets_requested_digits",
    )
end

function row_to_csv_values(row)
    return (
        string(row.requested_digits),
        @sprintf("%.0e", row.eps),
        @sprintf("%.6f", row.sigma),
        @sprintf("%.6f", row.lw),
        @sprintf("%.6f", row.kmax),
        string(row.ns),
        string(row.nt),
        string(row.need_grad),
        @sprintf("%.6f", row.exact_median_ms),
        @sprintf("%.6f", row.spread_median_ms),
        @sprintf("%.6f", row.exact_min_ms),
        @sprintf("%.6f", row.spread_min_ms),
        string(row.exact_median_bytes),
        string(row.spread_median_bytes),
        string(row.exact_min_bytes),
        string(row.spread_min_bytes),
        @sprintf("%.6f", row.exact_median_gc_ms),
        @sprintf("%.6f", row.spread_median_gc_ms),
        @sprintf("%.6f", row.exact_min_gc_ms),
        @sprintf("%.6f", row.spread_min_gc_ms),
        string(row.exact_median_allocs),
        string(row.spread_median_allocs),
        string(row.exact_min_allocs),
        string(row.spread_min_allocs),
        @sprintf("%.6f", row.exact_over_spread),
        @sprintf("%.6e", row.pot_relerr),
        isnan(row.grad_relerr) ? "NaN" : @sprintf("%.6e", row.grad_relerr),
        @sprintf("%.6f", row.pot_digits_achieved),
        isnan(row.grad_digits_achieved) ? "NaN" : @sprintf("%.6f", row.grad_digits_achieved),
        string(row.pot_meets_requested_digits),
        string(row.grad_meets_requested_digits),
    )
end

function write_results_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(csv_header(), ","))
        for row in rows
            println(io, join(row_to_csv_values(row), ","))
        end
    end
    return path
end

function print_summary(rows, io::IO = stdout)
    println(io, "requested_digits,sigma,ns,nt,need_grad,exact_median_ms,spread_median_ms,exact_over_spread,pot_relerr,grad_relerr")
    for row in rows
        grad_relerr = isnan(row.grad_relerr) ? "NaN" : @sprintf("%.3e", row.grad_relerr)
        @printf(io, "%d,%.3f,%d,%d,%s,%.3f,%.3f,%.3f,%.3e,%s\n",
            row.requested_digits,
            row.sigma,
            row.ns,
            row.nt,
            string(row.need_grad),
            row.exact_median_ms,
            row.spread_median_ms,
            row.exact_over_spread,
            row.pot_relerr,
            grad_relerr,
        )
    end
end

function main(; output_path::AbstractString = DEFAULT_OUTPUT, samples::Int = 10)
    rows = build_benchmark_rows(; samples = samples)
    path = write_results_csv(output_path, rows)
    print_summary(rows)
    println("wrote benchmark results to $(path)")
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
