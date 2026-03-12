"""
    _ltkm3dd_eval(sources, charges, targets, windowhat, lw, kmax, eps;
                  need_grad=false, return_selfconst=false)

Evaluate the discrete long-range TKM potential, and optionally the gradient, at
`targets` for `3 x N` source and target layouts.
"""
function _ltkm3dd_eval(
    sources::AbstractMatrix{<:Real},
    charges::AbstractVector{<:Real},
    targets::AbstractMatrix{<:Real},
    windowhat::Function,
    lw::Real,
    kmax::Real,
    eps::Real,
    ;
    compute_pot::Bool = true,
    compute_grad::Bool = false,
    return_selfconst::Bool = false,
)
    lw >= 0 || throw(ArgumentError("lw must be nonnegative"))
    kmax > 0 || throw(ArgumentError("kmax must be positive"))
    eps > 0 || throw(ArgumentError("eps must be positive"))

    src0 = as_3xn(sources)
    trg0 = as_3xn(targets)
    length(charges) == size(src0, 2) || throw(ArgumentError("number of charges must match number of sources"))

    T = promote_type(Float64, eltype(src0), eltype(trg0), eltype(charges), typeof(lw), typeof(kmax), typeof(eps))
    src = Matrix{T}(src0)
    trg = Matrix{T}(trg0)
    q = Vector{T}(charges)

    lengths, center = combined_box_geometry_3xn(src, trg)
    l_x, l_y, l_z = lengths
    cx, cy, cz = center

    L = sqrt(l_x^2 + l_y^2 + l_z^2) + T(lw)
    Δk_x = prevfloat(T(2π) / (l_x + L + T(lw)))
    Δk_y = prevfloat(T(2π) / (l_y + L + T(lw)))
    Δk_z = prevfloat(T(2π) / (l_z + L + T(lw)))

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
    if ndims(coeff) == 4 && size(coeff, 4) == 1
        coeff = dropdims(coeff; dims = 4)
    end

    diag_sum = zero(T)
    @inbounds for iz in eachindex(kz), iy in eachindex(ky), ix in eachindex(kx)
        k = sqrt(kx[ix]^2 + ky[iy]^2 + kz[iz]^2)
        if k <= T(kmax)
            scale = T(windowhat(k)) * truncated_laplace3d_hat(k, L)
            coeff[ix, iy, iz] *= scale
            diag_sum += scale
        else
            coeff[ix, iy, iz] = zero(eltype(coeff))
        end
    end

    prefactor = (Δk_x * Δk_y * Δk_z) / (T(2π)^3)
    pot = nothing
    grad = nothing

    if compute_pot || compute_grad
        plan = _finufft_make_type2_plan_3d(trgx, trgy, trgz, 1, T(eps), (length(kx), length(ky), length(kz)), 1, T)
        try
            if compute_pot
                pot_complex = _finufft_type2_exec_3d(plan, coeff)
                pot = prefactor .* real.(pot_complex)
            end

            if compute_grad
                grad_coeff = _spectral_gradient_coeffs_3d(coeff, kx, ky, kz)
                FINUFFT.finufft_destroy!(plan)
                plan = nothing
                plan = _finufft_make_type2_plan_3d(trgx, trgy, trgz, 1, T(eps), (length(kx), length(ky), length(kz)), 3, T)
                grad_complex = _finufft_type2_exec_3d(plan, grad_coeff)
                grad = prefactor .* permutedims(real.(grad_complex))
            end
        finally
            !isnothing(plan) && FINUFFT.finufft_destroy!(plan)
        end
    end

    selfconst = return_selfconst ? prefactor * diag_sum : nothing
    return pot, grad, selfconst
end

function _ltkm3dd_eval_exact(
    sources::AbstractMatrix{<:Real},
    charges::AbstractVector{<:Real},
    targets::AbstractMatrix{<:Real},
    windowhat::Function,
    lw::Real,
    kmax::Real,
    eps::Real;
    compute_pot::Bool = true,
    compute_grad::Bool = false,
    need_grad::Union{Nothing, Bool} = nothing,
    return_selfconst::Bool = false,
)
    if !isnothing(need_grad)
        compute_pot = true
        compute_grad = need_grad
    end
    return _ltkm3dd_eval(
        sources,
        charges,
        targets,
        windowhat,
        lw,
        kmax,
        eps;
        compute_pot = compute_pot,
        compute_grad = compute_grad,
        return_selfconst = return_selfconst,
    )
end

"""
    ltkm3dd(eps, sources; charges, targets=nothing, pg=0, pgt=0, windowhat, lw, kmax)

FMM3D-style interface for the discrete Laplace TKM long-range evaluator.
Only charge sources are supported; dipole inputs are not implemented.
"""
function ltkm3dd(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges,
    targets::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    pg::Integer = 0,
    pgt::Integer = 0,
    windowhat::Function,
    lw::Real,
    kmax::Real,
)
    pg in (0, 1, 2) || throw(ArgumentError("ltkm3dd currently supports only pg = 0, 1, or 2"))
    pgt in (0, 1, 2) || throw(ArgumentError("ltkm3dd currently supports only pgt = 0, 1, or 2"))

    src = as_3xn(sources)
    q = Vector{float(eltype(charges))}(charges)
    length(q) == size(src, 2) || throw(ArgumentError("number of charges must match number of sources"))

    pot = nothing
    grad = nothing
    pottarg = nothing
    gradtarg = nothing

    if pg > 0
        pot, grad, selfconst = _ltkm3dd_eval_exact(
            src,
            q,
            src,
            windowhat,
            lw,
            kmax,
            eps;
            compute_pot = (pg == 1),
            compute_grad = (pg == 2),
            return_selfconst = (pg == 1),
        )
        if pg == 1
            pot .-= q .* selfconst
        end
    end

    if pgt > 0
        isnothing(targets) && throw(ArgumentError("targets are required when pgt > 0"))
        trg = as_3xn(targets)
        pottarg, gradtarg, _ = _ltkm3dd_eval_exact(
            src,
            q,
            trg,
            windowhat,
            lw,
            kmax,
            eps;
            compute_pot = (pgt == 1),
            compute_grad = (pgt == 2),
        )
    elseif !isnothing(targets)
        as_3xn(targets)
    end

    return TKMVals(pot, grad, pottarg, gradtarg, 0)
end
