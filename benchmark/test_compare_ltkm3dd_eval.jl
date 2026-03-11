using Test

include(joinpath(@__DIR__, "compare_ltkm3dd_eval.jl"))

@testset "benchmark defaults" begin
    @test DEFAULT_DIGITS == (3, 6, 9, 12)
    @test DEFAULT_SIGMAS == (0.1, 0.5, 1.0)
    @test digits_to_eps(DEFAULT_DIGITS) == [1e-3, 1e-6, 1e-9, 1e-12]
end

@testset "gaussian kmax selection" begin
    sigma = 0.5
    eps = 1e-9
    kmax = gaussian_kmax_from_eps(sigma, eps)
    @test isapprox(exp(-0.25 * sigma^2 * kmax^2), eps; rtol = 1e-12, atol = 0.0)
end

@testset "benchmark row generation count" begin
    cases = ((ns = 8, nt = 5, need_grad = false), (ns = 8, nt = 5, need_grad = true))
    rows = build_benchmark_rows(; digits = (3, 6), sigmas = (0.1, 0.5), cases = cases, samples = 1)
    @test length(rows) == 8
    @test Set(getfield.(rows, :requested_digits)) == Set([3, 6])
    @test Set(getfield.(rows, :sigma)) == Set([0.1, 0.5])
    @test Set(getfield.(rows, :need_grad)) == Set([false, true])
end
