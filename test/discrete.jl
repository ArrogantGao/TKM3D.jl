using LinearAlgebra
using Random
using SpecialFunctions: erf
using Statistics: mean

@inline function gaussian_window_fourier_transform(k::Float64, sigma::Float64)
    return exp(-0.25 * sigma^2 * k^2)
end

@inline function gaussian_window_pair_potential(r::Float64, sigma::Float64)
    if r < 10.0 * eps(Float64)
        return 1.0 / (2.0 * pi^(3.0 / 2.0) * sigma)
    end
    return erf(r / sigma) / (4.0 * pi * r)
end

@inline function gaussian_window_pair_gradient(rvec::AbstractVector{Float64}, sigma::Float64)
    r = norm(rvec)
    if r < 10.0 * eps(Float64)
        return zeros(Float64, 3)
    end
    a = 1.0 / sigma
    ddr = ((2a / sqrt(pi)) * exp(-(a * r)^2) * r - erf(a * r)) / (4.0 * pi * r^2)
    return (ddr / r) .* collect(rvec)
end

function gaussian_pairwise_long_range(
    sources::Matrix{Float64},
    charges::Vector{Float64},
    targets::Matrix{Float64},
    sigma::Float64;
    dropdiag::Bool = false,
)
    ns = size(sources, 2)
    nt = size(targets, 2)
    @assert size(sources, 1) == 3
    @assert size(targets, 1) == 3
    @assert length(charges) == ns

    pot = zeros(Float64, nt)
    grad = zeros(Float64, 3, nt)
    for i in 1:nt
        acc = 0.0
        gacc = zeros(Float64, 3)
        yi = @view targets[:, i]
        for j in 1:ns
            xj = @view sources[:, j]
            if dropdiag && nt == ns && i == j
                continue
            end
            rvec = yi .- xj
            acc += charges[j] * gaussian_window_pair_potential(norm(rvec), sigma)
            gacc .+= charges[j] .* gaussian_window_pair_gradient(rvec, sigma)
        end
        pot[i] = acc
        grad[:, i] .= gacc
    end
    return pot, grad
end

function gaussian_kmax_from_tol(sigma::Float64, tail_tol::Float64)
    return 2.0 * sqrt(log(1.0 / tail_tol)) / sigma
end

@testset "spread-only upsampfac selection" begin
    sigma = TKM3D._ltkm3dd_spreadonly_upsampfac((17, 19, 21), 1e-12)
    @test sigma > 1.0
end

@testset "ltkm3dd validation" begin
    sigma = 0.2
    what(k) = gaussian_window_fourier_transform(k, sigma)
    sources = [0.1 0.7; 0.2 0.4; 0.3 0.2]
    charges = [1.0, -0.5]
    targets = [0.4 0.1; 0.4 0.2; 0.4 0.3]

    @test_throws ArgumentError ltkm3dd(1e-12, rand(4, 3); charges, targets, pg = 0, pgt = 1, windowhat = what, lw = 4sigma, kmax = 20.0)
    @test_throws UndefKeywordError ltkm3dd(1e-12, sources; targets, pg = 0, pgt = 1, windowhat = what, lw = 4sigma, kmax = 20.0)
    @test_throws ArgumentError ltkm3dd(1e-12, sources; charges = rand(3), targets, pg = 0, pgt = 1, windowhat = what, lw = 4sigma, kmax = 20.0)
    @test_throws ArgumentError ltkm3dd(1e-12, sources; charges, targets = rand(4, 3), pg = 0, pgt = 1, windowhat = what, lw = 4sigma, kmax = 20.0)
    @test_throws MethodError ltkm3dd(1e-12, sources; charges, targets, pg = 0, pgt = 1, nd = 1, windowhat = what, lw = 4sigma, kmax = 20.0)
end

@testset "ltkm3dd matches analytic Gaussian pot/grad at targets" begin
    rng = MersenneTwister(1234)
    sources = rand(rng, 3, 8)
    charges = randn(rng, 8)
    charges .-= mean(charges)
    targets = rand(rng, 3, 5)

    sigma = 0.12
    what(k) = gaussian_window_fourier_transform(k, sigma)
    l_w = 5.0 * sigma
    k_max = gaussian_kmax_from_tol(sigma, 1e-12)

    out = ltkm3dd(1e-12, sources; charges, targets, pg = 0, pgt = 2, windowhat = what, lw = l_w, kmax = k_max)
    ref_pot, ref_grad = gaussian_pairwise_long_range(sources, charges, targets, sigma)

    @test out.ier == 0
    @test isnothing(out.pot)
    @test isnothing(out.grad)
    @test length(out.pottarg) == size(targets, 2)
    @test size(out.gradtarg) == size(targets)
    @test norm(out.pottarg .- ref_pot) / norm(ref_pot) < 1e-12
    @test norm(out.gradtarg .- ref_grad) / norm(ref_grad) < 1e-11
end

@testset "ltkm3dd matches analytic Gaussian pot/grad at sources without self term" begin
    rng = MersenneTwister(20260310)
    sources = rand(rng, 3, 8)
    charges = randn(rng, 8)
    charges .-= mean(charges)

    sigma = 0.12
    what(k) = gaussian_window_fourier_transform(k, sigma)
    l_w = 5.0 * sigma
    k_max = gaussian_kmax_from_tol(sigma, 1e-12)

    out = ltkm3dd(1e-12, sources; charges, pg = 2, pgt = 0, windowhat = what, lw = l_w, kmax = k_max)
    ref_pot, ref_grad = gaussian_pairwise_long_range(sources, charges, sources, sigma; dropdiag = true)

    @test out.ier == 0
    @test length(out.pot) == size(sources, 2)
    @test size(out.grad) == size(sources)
    @test isnothing(out.pottarg)
    @test isnothing(out.gradtarg)
    @test norm(out.pot .- ref_pot) / norm(ref_pot) < 5e-12
    @test norm(out.grad .- ref_grad) / norm(ref_grad) < 1e-11
end

@testset "ltkm3dd combined source and target pot/grad outputs" begin
    rng = MersenneTwister(99)
    sources = rand(rng, 3, 6)
    charges = randn(rng, 6)
    charges .-= mean(charges)
    targets = rand(rng, 3, 4)

    sigma = 0.18
    what(k) = gaussian_window_fourier_transform(k, sigma)
    l_w = 5.0 * sigma
    k_max = gaussian_kmax_from_tol(sigma, 1e-12)

    out = ltkm3dd(1e-12, sources; charges, targets, pg = 2, pgt = 2, windowhat = what, lw = l_w, kmax = k_max)

    @test out.ier == 0
    @test length(out.pot) == size(sources, 2)
    @test size(out.grad) == size(sources)
    @test length(out.pottarg) == size(targets, 2)
    @test size(out.gradtarg) == size(targets)
end
