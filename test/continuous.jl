using LinearAlgebra
using Random
using SpecialFunctions: erf

@inline function gaussian_laplace3d_pot(center::NTuple{3, Float64}, target::NTuple{3, Float64}, sigma::Float64)
    r = norm(target .- center)
    if r < 10.0 * eps(Float64)
        return 1.0 / (2.0 * sqrt(2.0) * pi^(3.0 / 2.0) * sigma)
    end
    return erf(r / (sqrt(2.0) * sigma)) / (4.0 * pi * r)
end

@inline function gaussian_laplace3d_grad(center::NTuple{3, Float64}, target::NTuple{3, Float64}, sigma::Float64)
    rvec = collect(target .- center)
    r = norm(rvec)
    if r < 10.0 * eps(Float64)
        return zeros(Float64, 3)
    end

    a = 1.0 / (sqrt(2.0) * sigma)
    ddr = ((2a / sqrt(pi)) * exp(-(a * r)^2) * r - erf(a * r)) / (4.0 * pi * r^2)
    return (ddr / r) .* rvec
end

function make_uniform_gaussian_quadrature(
    n::Int,
    region::Float64,
    center::NTuple{3, Float64},
    sigma::Float64;
    jitter_scale::Float64 = 0.0,
    seed::Int = 0,
)
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    xs = collect(range(-region, region; length = n))
    h = xs[2] - xs[1]
    rng = MersenneTwister(seed)

    sources = Matrix{Float64}(undef, 3, n^3)
    charges = Vector{Float64}(undef, n^3)

    idx = 1
    jitter_bound = 0.5 * jitter_scale * h
    for x in xs, y in xs, z in xs
        px = clamp(x + (2rand(rng) - 1) * jitter_bound, -region, region)
        py = clamp(y + (2rand(rng) - 1) * jitter_bound, -region, region)
        pz = clamp(z + (2rand(rng) - 1) * jitter_bound, -region, region)
        sources[:, idx] .= (px, py, pz)

        r2 = (px - center[1])^2 + (py - center[2])^2 + (pz - center[3])^2
        rho = exp(-r2 / (2.0 * sigma^2)) / ((2.0 * pi)^(3.0 / 2.0) * sigma^3)
        charges[idx] = rho * h^3
        idx += 1
    end

    return sources, charges
end

function make_targets_3xn(points::Vector{NTuple{3, Float64}})
    targets = Matrix{Float64}(undef, 3, length(points))
    for (i, p) in pairs(points)
        targets[:, i] .= p
    end
    return targets
end

function make_random_targets_3xn(n::Int, halfwidth::Float64; seed::Int)
    rng = MersenneTwister(seed)
    return (2.0 * halfwidth) .* rand(rng, 3, n) .- halfwidth
end

function make_uniform_cartesian_sources(n::Int, h::Float64)
    sources = Matrix{Float64}(undef, 3, n^3)
    idx = 1
    for x in 0:(n - 1), y in 0:(n - 1), z in 0:(n - 1)
        sources[:, idx] .= (x * h, y * h, z * h)
        idx += 1
    end
    return sources
end

@testset "type-2 FINUFFT plan helper matches simple interface" begin
    rng = MersenneTwister(20260312)
    ms, mt, mu = 7, 5, 6
    targets = (2π) .* rand(rng, 3, 11) .- π
    fk = randn(rng, ComplexF64, ms, mt, mu)
    fk_many = cat(fk, 2 .* fk, 3 .* fk; dims = 4)

    ref_single = vec(TKM3D.nufft3d2(targets[1, :], targets[2, :], targets[3, :], 1, 1e-9, fk))
    ref_many = TKM3D.nufft3d2(targets[1, :], targets[2, :], targets[3, :], 1, 1e-9, fk_many)

    out_single = TKM3D._finufft_type2_eval_3d(targets[1, :], targets[2, :], targets[3, :], 1, 1e-9, fk)
    out_many = TKM3D._finufft_type2_eval_3d(targets[1, :], targets[2, :], targets[3, :], 1, 1e-9, fk_many)

    @test isapprox(out_single, ref_single; rtol = 1e-12, atol = 1e-12)
    @test isapprox(out_many, ref_many; rtol = 1e-12, atol = 1e-12)
end

@testset "spectral cutoff helper returns smallest admissible radius" begin
    radii = [0.0, 1.0, 1.0, 2.0, 3.0, 3.0]
    magnitudes = [10.0, 5.0, 4.0, 0.8, 0.09, 0.08]

    cutoff = TKM3D._smallest_kcut_from_tail(radii, magnitudes, 0.1)
    tail_ratio = TKM3D._pointwise_tail_ratio(radii, magnitudes, cutoff)

    @test cutoff == 2.0
    @test tail_ratio < 0.1
end

@testset "estimate_kcut3dc collapses constant Cartesian grid to zero mode" begin
    n = 6
    h = 0.25
    sources = make_uniform_cartesian_sources(n, h)
    charges = ones(Float64, size(sources, 2))

    out = estimate_kcut3dc(sources; charges, tol = 1e-10, eps = 1e-12)

    @test out isa TKM3D.KCut3DCResult
    @test out.kcut ≈ 0.0 atol = 1e-12
    @test out.tail_ratio < 1e-10
    @test out.kmax_nyquist > 0
    @test out.axis_nyquist == (π / h, π / h, π / h)
    @test out.nmodes == (n + 1, n + 1, n + 1)
    @test out.Δk == (2π / (n * h), 2π / (n * h), 2π / (n * h))
end

@testset "ltkm3dc validation" begin
    sources = [0.0 0.5; 0.0 0.5; 0.0 0.5]
    charges = [1.0, 2.0]
    targets = [0.1 0.2; 0.1 0.2; 0.1 0.2]

    @test_throws ArgumentError ltkm3dc(1e-12, rand(4, 2); charges, targets, pgt = 1, kmax = 10.0)
    @test_throws ArgumentError ltkm3dc(1e-12, sources; charges = [1.0], targets, pgt = 1, kmax = 10.0)
    @test_throws ArgumentError ltkm3dc(1e-12, sources; charges, pgt = 1, kmax = 10.0)
    @test_throws ArgumentError ltkm3dc(1e-12, sources; charges, targets, pg = 3, kmax = 10.0)
    @test_throws ArgumentError ltkm3dc(1e-12, sources; charges, targets, pgt = 3, kmax = 10.0)
end

@testset "ltkm3dc matches analytic Gaussian target gradient" begin
    sigma = 0.2
    center = (0.1, -0.2, 0.3)
    region = sigma * sqrt(2.0 * log(1.0e12))
    sources, charges = make_uniform_gaussian_quadrature(26, region, center, sigma)
    targets = make_targets_3xn([
        (-0.35, -0.10, 0.15),
        (0.00, 0.10, -0.15),
        (0.25, -0.30, 0.05),
        (0.40, 0.20, 0.10),
    ])

    kmax = sqrt(2.0 * log(1.0e12)) / sigma
    out = ltkm3dc(1e-12, sources; charges, targets, pg = 0, pgt = 2, kmax = kmax)

    ref_grad = hcat([gaussian_laplace3d_grad(center, (targets[1, i], targets[2, i], targets[3, i]), sigma) for i in axes(targets, 2)]...)

    @test out.ier == 0
    @test isnothing(out.pot)
    @test isnothing(out.grad)
    @test isnothing(out.pottarg)
    @test size(out.gradtarg) == size(targets)
    @test norm(out.gradtarg .- ref_grad) / norm(ref_grad) < 1e-3
end

@testset "ltkm3dc matches analytic Gaussian potential at 100 random targets" begin
    sigma = 0.2
    center = (0.1, -0.2, 0.3)
    region = sigma * sqrt(2.0 * log(1.0e12))
    sources, charges = make_uniform_gaussian_quadrature(30, region, center, sigma)
    targets = make_random_targets_3xn(100, 1.2 * region; seed = 20260311)

    kmax = sqrt(2.0 * log(1.0e12)) / sigma
    out = ltkm3dc(1e-12, sources; charges, targets, pgt = 1, kmax = kmax)
    ref_pot = [gaussian_laplace3d_pot(center, (targets[1, i], targets[2, i], targets[3, i]), sigma) for i in axes(targets, 2)]

    @test out.ier == 0
    @test isnothing(out.pot)
    @test isnothing(out.grad)
    @test isnothing(out.gradtarg)
    @test length(out.pottarg) == size(targets, 2)
    @test norm(out.pottarg .- ref_pot) / norm(ref_pot) < 1e-7
end

@testset "ltkm3dc estimates kmax from average source spacing" begin
    sigma = 0.18
    center = (-0.05, 0.12, -0.08)
    region = sigma * sqrt(2.0 * log(1.0e12))
    sources, charges = make_uniform_gaussian_quadrature(16, region, center, sigma)
    targets = make_targets_3xn([
        (-0.20, -0.10, 0.10),
        (0.05, 0.15, -0.05),
        (0.25, -0.15, 0.00),
    ])

    out = ltkm3dc(1e-12, sources; charges, targets, pgt = 1)
    ref_pot = [gaussian_laplace3d_pot(center, (targets[1, i], targets[2, i], targets[3, i]), sigma) for i in axes(targets, 2)]

    @test out.ier == 0
    @test isnothing(out.pot)
    @test isnothing(out.grad)
    @test isnothing(out.gradtarg)
    @test length(out.pottarg) == size(targets, 2)
    @test norm(out.pottarg .- ref_pot) / norm(ref_pot) < 2e-2
end

@testset "ltkm3dc source and target outputs" begin
    sigma = 0.16
    center = (0.0, 0.0, 0.0)
    region = sigma * sqrt(2.0 * log(1.0e10))
    sources, charges = make_uniform_gaussian_quadrature(12, region, center, sigma)
    targets = make_targets_3xn([
        (-0.1, 0.2, -0.05),
        (0.15, -0.1, 0.12),
    ])

    out = ltkm3dc(1e-10, sources; charges, targets, pg = 2, pgt = 2, kmax = sqrt(2.0 * log(1.0e10)) / sigma)

    @test out.ier == 0
    @test isnothing(out.pot)
    @test size(out.grad) == size(sources)
    @test isnothing(out.pottarg)
    @test size(out.gradtarg) == size(targets)
end
