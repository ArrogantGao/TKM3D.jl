function _ltkm3dc_mean_positive_gap(values::AbstractVector{<:Real})
    sorted = sort!(collect(Float64.(values)))
    positive_sum = 0.0
    positive_count = 0

    for i in 1:(length(sorted) - 1)
        gap = sorted[i + 1] - sorted[i]
        if gap > 0.0
            positive_sum += gap
            positive_count += 1
        end
    end

    positive_count > 0 || throw(ArgumentError("cannot estimate source spacing from degenerate coordinates; provide kmax explicitly"))
    return positive_sum / positive_count
end

function _ltkm3dc_estimate_kmax(sources::AbstractMatrix{<:Real})
    src = as_3xn(sources)
    dx = _ltkm3dc_mean_positive_gap(vec(view(src, 1, :)))
    dy = _ltkm3dc_mean_positive_gap(vec(view(src, 2, :)))
    dz = _ltkm3dc_mean_positive_gap(vec(view(src, 3, :)))
    return min(pi / dx, pi / dy, pi / dz)
end

function _ltkm3dc_eval(
    sources::AbstractMatrix{<:Real},
    charges::AbstractVector{<:Real},
    targets::AbstractMatrix{<:Real},
    kmax::Real,
    eps::Real;
    need_grad::Bool = false,
)
    kmax > 0 || throw(ArgumentError("kmax must be positive"))
    eps > 0 || throw(ArgumentError("eps must be positive"))

    src0 = as_3xn(sources)
    trg0 = as_3xn(targets)
    length(charges) == size(src0, 2) || throw(ArgumentError("number of charges must match number of sources"))

    T = promote_type(Float64, eltype(src0), eltype(trg0), eltype(charges), typeof(kmax), typeof(eps))
    src = Matrix{T}(src0)
    trg = Matrix{T}(trg0)
    q = Vector{T}(charges)

    lengths, center = combined_box_geometry_3xn(src, trg)
    l_x, l_y, l_z = lengths
    cx, cy, cz = center

    L = sqrt(l_x^2 + l_y^2 + l_z^2)
    L > zero(T) || throw(ArgumentError("source/target box must have positive extent"))

    Δk_x = prevfloat(T(2π) / (l_x + L))
    Δk_y = prevfloat(T(2π) / (l_y + L))
    Δk_z = prevfloat(T(2π) / (l_z + L))

    kx = centered_mode_axis(Δk_x, T(kmax))
    ky = centered_mode_axis(Δk_y, T(kmax))
    kz = centered_mode_axis(Δk_z, T(kmax))

    srcx = Δk_x .* (vec(view(src, 1, :)) .- cx)
    srcy = Δk_y .* (vec(view(src, 2, :)) .- cy)
    srcz = Δk_z .* (vec(view(src, 3, :)) .- cz)
    trgx = Δk_x .* (vec(view(trg, 1, :)) .- cx)
    trgy = Δk_y .* (vec(view(trg, 2, :)) .- cy)
    trgz = Δk_z .* (vec(view(trg, 3, :)) .- cz)

    coeff = nufft3d1(srcx, srcy, srcz, complex.(q), -1, T(eps), length(kx), length(ky), length(kz))

    @inbounds for iz in eachindex(kz), iy in eachindex(ky), ix in eachindex(kx)
        k = sqrt(kx[ix]^2 + ky[iy]^2 + kz[iz]^2)
        if k <= T(kmax)
            coeff[ix, iy, iz] *= truncated_laplace3d_hat(k, L)
        else
            coeff[ix, iy, iz] = zero(eltype(coeff))
        end
    end

    pot_complex = nufft3d2(trgx, trgy, trgz, 1, T(eps), coeff)
    prefactor = (Δk_x * Δk_y * Δk_z) / (T(2π)^3)
    pot = prefactor .* real.(pot_complex)
    grad = nothing

    if need_grad
        grad = Matrix{T}(undef, 3, size(trg, 2))
        grad_coeff = similar(coeff)
        for d in 1:3
            @inbounds for iz in eachindex(kz), iy in eachindex(ky), ix in eachindex(kx)
                kd = d == 1 ? kx[ix] : d == 2 ? ky[iy] : kz[iz]
                grad_coeff[ix, iy, iz] = (im * kd) * coeff[ix, iy, iz]
            end
            grad_complex = nufft3d2(trgx, trgy, trgz, 1, T(eps), grad_coeff)
            grad[d, :] .= prefactor .* real.(grad_complex)
        end
    end

    return pot, grad
end

"""
    ltkm3dc(eps, sources; charges, targets=nothing, pg=0, pgt=0, kmax=nothing)

Continuous free-space Laplace TKM evaluator with arbitrary source points and
preweighted quadrature masses.
"""
function ltkm3dc(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges,
    targets::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    pg::Integer = 0,
    pgt::Integer = 0,
    kmax::Union{Nothing, Real} = nothing,
)
    pg in (0, 1, 2) || throw(ArgumentError("ltkm3dc currently supports only pg = 0, 1, or 2"))
    pgt in (0, 1, 2) || throw(ArgumentError("ltkm3dc currently supports only pgt = 0, 1, or 2"))
    eps > 0 || throw(ArgumentError("eps must be positive"))

    src = as_3xn(sources)
    q = Vector{float(eltype(charges))}(charges)
    length(q) == size(src, 2) || throw(ArgumentError("number of charges must match number of sources"))

    resolved_kmax = isnothing(kmax) ? _ltkm3dc_estimate_kmax(src) : Float64(kmax)
    resolved_kmax > 0 || throw(ArgumentError("kmax must be positive"))

    pot = nothing
    grad = nothing
    pottarg = nothing
    gradtarg = nothing

    if pg > 0
        pot, grad = _ltkm3dc_eval(src, q, src, resolved_kmax, eps; need_grad = (pg == 2))
    end

    if pgt > 0
        isnothing(targets) && throw(ArgumentError("targets are required when pgt > 0"))
        trg = as_3xn(targets)
        pottarg, gradtarg = _ltkm3dc_eval(src, q, trg, resolved_kmax, eps; need_grad = (pgt == 2))
    elseif !isnothing(targets)
        as_3xn(targets)
    end

    return TKMVals(pot, grad, pottarg, gradtarg, 0)
end
