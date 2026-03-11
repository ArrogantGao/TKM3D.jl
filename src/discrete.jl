function _ltkm3dd_spreadonly_upsampfac(nmodes::NTuple{3, Int}, eps::Real)
    return 1.0001
end

function _ltkm3dd_make_spreadonly_plan(
    type::Integer,
    nmodes::NTuple{3, Int},
    iflag::Integer,
    ntrans::Integer,
    eps::Real,
    dtype::Type{T};
    kwargs...,
) where {T <: AbstractFloat}
    modes = TKM3D.FINUFFT.BIGINT[nmodes...]
    try
        plan = finufft_makeplan(
            type,
            modes,
            iflag,
            ntrans,
            eps;
            dtype = dtype,
            spreadinterponly = 1,
            upsampfac = 1.0,
            kwargs...,
        )
        return plan, 1.0
    catch err
        if err isa TKM3D.FINUFFT.FINUFFTError && err.errno == TKM3D.FINUFFT.ERR_UPSAMPFAC_TOO_SMALL
            sigma = _ltkm3dd_spreadonly_upsampfac(nmodes, eps)
            plan = finufft_makeplan(
                type,
                modes,
                iflag,
                ntrans,
                eps;
                dtype = dtype,
                spreadinterponly = 1,
                upsampfac = sigma,
                kwargs...,
            )
            return plan, sigma
        end
        rethrow()
    end
end

function _ltkm3dd_make_spreadonly_plan_with_sigma(
    type::Integer,
    nmodes::NTuple{3, Int},
    iflag::Integer,
    ntrans::Integer,
    eps::Real,
    dtype::Type{T},
    sigma::Real;
    kwargs...,
) where {T <: AbstractFloat}
    modes = TKM3D.FINUFFT.BIGINT[nmodes...]
    return finufft_makeplan(
        type,
        modes,
        iflag,
        ntrans,
        eps;
        dtype = dtype,
        spreadinterponly = 1,
        upsampfac = sigma,
        kwargs...,
    )
end

function _ltkm3dd_spreadonly_next235even(n::Integer)
    n <= 2 && return 2
    nplus = isodd(n) ? n : n - 2
    numdiv = 2
    while numdiv > 1
        nplus += 2
        numdiv = nplus
        while numdiv % 2 == 0
            numdiv ÷= 2
        end
        while numdiv % 3 == 0
            numdiv ÷= 3
        end
        while numdiv % 5 == 0
            numdiv ÷= 5
        end
    end
    return nplus
end

function _ltkm3dd_spreadonly_leg_eval(n::Int, x::Float64)
    if n == 0
        return 1.0, 0.0
    elseif n == 1
        return x, 1.0
    end
    p0 = 0.0
    p1 = 1.0
    p2 = x
    for i in 1:(n - 1)
        p0 = p1
        p1 = p2
        p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1)
    end
    return p2, n * (x * p2 - p1) / (x^2 - 1.0)
end

function _ltkm3dd_spreadonly_gaussquad(n::Int)
    xgl = Vector{Float64}(undef, n)
    wgl = Vector{Float64}(undef, n)
    xgl[(n ÷ 2) + 1] = 0.0
    for i in 0:(n ÷ 2 - 1)
        x = cos((2 * i + 1) * π / (2 * n))
        convcount = 0
        while true
            p, dp = _ltkm3dd_spreadonly_leg_eval(n, x)
            dx = -p / dp
            x += dx
            convcount = abs(dx) < 1e-14 ? convcount + 1 : 0
            convcount == 3 && break
        end
        xgl[i + 1] = -x
        xgl[n - i] = x
    end
    for i in 1:(n ÷ 2 + 1)
        _, dp = _ltkm3dd_spreadonly_leg_eval(n, xgl[i])
        p, _ = _ltkm3dd_spreadonly_leg_eval(n + 1, xgl[i])
        wgl[i] = -2.0 / ((n + 1) * dp * p)
        wgl[n - i + 1] = wgl[i]
    end
    return xgl, wgl
end

function _ltkm3dd_spreadonly_kernel_params(tol::Real, sigma::Real; kerformula::Int = 1)
    sigma > 1 || throw(ArgumentError("spread-only upsampfac must be greater than 1"))
    tol_eff = max(Float64(tol), eps(Float64))

    kerformula == 1 || throw(ArgumentError("only ES spread_kerformula = 1 is supported"))

    if sigma == 2.0
        ns = ceil(Int, log10(10.0 / tol_eff))
    else
        ns = ceil(Int, log(1.0 / tol_eff) / (π * sqrt(1.0 - 1.0 / sigma)))
    end
    ns = clamp(ns, 2, 16)

    betaoverns = ns == 2 ? 2.20 : ns == 3 ? 2.26 : ns == 4 ? 2.38 : 2.30
    beta = betaoverns * ns
    if sigma != 2.0
        beta = 0.97 * π * ns * (1.0 - 1.0 / (2.0 * sigma))
    end

    return (nspread = ns, upsampfac = Float64(sigma), beta = beta, kerformula = kerformula)
end

function _ltkm3dd_spreadonly_kernel_definition(params, z::Float64)
    abs(z) > 1.0 && return 0.0
    arg = params.beta * sqrt(max(0.0, 1.0 - z^2))
    return exp(arg - params.beta)
end

function _ltkm3dd_spreadonly_poly_fit(f, n::Int)
    t = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    for k in 0:(n - 1)
        t[k + 1] = cos((2 * k + 1) * π / (2 * n))
        y[k + 1] = f(t[k + 1])
    end

    coef = copy(y)
    for j in 2:n
        for i in n:-1:j
            coef[i] = (coef[i] - coef[i - 1]) / (t[i] - t[i - j + 1])
        end
    end

    function mul_by_linear(p::Vector{Float64}, c::Float64)
        r = zeros(Float64, length(p) + 1)
        for i in eachindex(p)
            r[i] += -c * p[i]
            r[i + 1] += p[i]
        end
        return r
    end

    c = zeros(Float64, n)
    basis = [1.0]
    c[1] += coef[1]
    for j in 2:n
        basis = mul_by_linear(basis, t[j - 1])
        for m in eachindex(basis)
            c[m] += coef[j] * basis[m]
        end
    end
    reverse!(c)
    return c
end

function _ltkm3dd_spreadonly_horner_coeffs(params)
    ns = params.nspread
    nc_fit = min(19, ns + 3)
    coeffs = Matrix{Float64}(undef, nc_fit, ns)
    for j in 1:ns
        xshiftj = 2 * (j - 1) + 1 - ns
        kernel_this_interval = x -> begin
            z = (x + xshiftj) / ns
            _ltkm3dd_spreadonly_kernel_definition(params, z)
        end
        coeffs[:, j] .= _ltkm3dd_spreadonly_poly_fit(kernel_this_interval, nc_fit)
    end
    return coeffs
end

function _ltkm3dd_spreadonly_evaluate_kernel_runtime(x::Float64, coeffs::AbstractMatrix{<:Real}, ns::Int)
    ns2 = ns / 2.0
    res = 0.0
    for i in 1:ns
        if x > -ns2 + (i - 1) && x <= -ns2 + i
            z = muladd(2.0, x - (i - 1), ns - 1)
            for j in axes(coeffs, 1)
                res = muladd(res, z, coeffs[j, i])
            end
            break
        end
    end
    return res
end

function _ltkm3dd_spreadonly_onedim_fseries_kernel(nf::Int, params, coeffs)
    J2 = params.nspread / 2.0
    q = Int(2 + 3.0 * J2)
    z, w = _ltkm3dd_spreadonly_gaussquad(2 * q)
    f = Vector{Float64}(undef, q)
    a = Vector{ComplexF64}(undef, q)
    for n in 1:q
        zn = z[n] * J2
        f[n] = J2 * w[n] * _ltkm3dd_spreadonly_evaluate_kernel_runtime(zn, coeffs, params.nspread)
        a[n] = -exp(2π * im * zn / nf)
    end

    aj = ones(ComplexF64, q)
    fwkerhalf = Vector{Float64}(undef, nf ÷ 2 + 1)
    for j in 0:(nf ÷ 2)
        x = 0.0
        for n in 1:q
            x += f[n] * 2.0 * real(aj[n])
        end
        fwkerhalf[j + 1] = x
        for n in 1:q
            aj[n] *= a[n]
        end
    end
    return fwkerhalf
end

function _ltkm3dd_spreadonly_nfdim(modes::NTuple{3, Int}, params)
    return ntuple(d -> _ltkm3dd_spreadonly_next235even(max(ceil(Int, params.upsampfac * modes[d]), 2 * params.nspread)), 3)
end

@inline function _ltkm3dd_spreadonly_fwindex(k::Int, nf::Int)
    return k >= 0 ? k + 1 : nf + k + 1
end

@inline function _ltkm3dd_spreadonly_fkindex(k::Int, m::Int)
    return k + (m ÷ 2) + 1
end

function _ltkm3dd_spreadonly_deconvolveshuffle3d_dir1(
    fw::AbstractArray{Complex{T}, 3},
    modes::NTuple{3, Int},
    phi1::AbstractVector{T},
    phi2::AbstractVector{T},
    phi3::AbstractVector{T},
) where {T <: AbstractFloat}
    ms, mt, mu = modes
    nf1, nf2, nf3 = size(fw)
    fk = Array{Complex{T}}(undef, ms, mt, mu)
    k1min, k1max = -(ms ÷ 2), (ms - 1) ÷ 2
    k2min, k2max = -(mt ÷ 2), (mt - 1) ÷ 2
    k3min, k3max = -(mu ÷ 2), (mu - 1) ÷ 2
    @inbounds for k3 in k3min:k3max, k2 in k2min:k2max, k1 in k1min:k1max
        fk[_ltkm3dd_spreadonly_fkindex(k1, ms), _ltkm3dd_spreadonly_fkindex(k2, mt), _ltkm3dd_spreadonly_fkindex(k3, mu)] =
            fw[_ltkm3dd_spreadonly_fwindex(k1, nf1), _ltkm3dd_spreadonly_fwindex(k2, nf2), _ltkm3dd_spreadonly_fwindex(k3, nf3)] /
            (phi1[abs(k1) + 1] * phi2[abs(k2) + 1] * phi3[abs(k3) + 1])
    end
    return fk
end

function _ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(
    fk::AbstractArray{Complex{T}, 3},
    nfdim::NTuple{3, Int},
    phi1::AbstractVector{T},
    phi2::AbstractVector{T},
    phi3::AbstractVector{T},
) where {T <: AbstractFloat}
    ms, mt, mu = size(fk)
    nf1, nf2, nf3 = nfdim
    fw = zeros(Complex{T}, nf1, nf2, nf3)
    k1min, k1max = -(ms ÷ 2), (ms - 1) ÷ 2
    k2min, k2max = -(mt ÷ 2), (mt - 1) ÷ 2
    k3min, k3max = -(mu ÷ 2), (mu - 1) ÷ 2
    @inbounds for k3 in k3min:k3max, k2 in k2min:k2max, k1 in k1min:k1max
        fw[_ltkm3dd_spreadonly_fwindex(k1, nf1), _ltkm3dd_spreadonly_fwindex(k2, nf2), _ltkm3dd_spreadonly_fwindex(k3, nf3)] =
            fk[_ltkm3dd_spreadonly_fkindex(k1, ms), _ltkm3dd_spreadonly_fkindex(k2, mt), _ltkm3dd_spreadonly_fkindex(k3, mu)] /
            (phi1[abs(k1) + 1] * phi2[abs(k2) + 1] * phi3[abs(k3) + 1])
    end
    return fw
end

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
    need_grad::Bool = false,
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
    need_grad::Bool = false,
    return_selfconst::Bool = false,
)
    return _ltkm3dd_eval(
        sources,
        charges,
        targets,
        windowhat,
        lw,
        kmax,
        eps;
        need_grad = need_grad,
        return_selfconst = return_selfconst,
    )
end

function _ltkm3dd_eval_spreadonly(
    sources::AbstractMatrix{<:Real},
    charges::AbstractVector{<:Real},
    targets::AbstractMatrix{<:Real},
    windowhat::Function,
    lw::Real,
    kmax::Real,
    eps::Real;
    need_grad::Bool = false,
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
    modes = (length(kx), length(ky), length(kz))

    srcx = Δk_x .* (vec(view(src, 1, :)) .- cx)
    srcy = Δk_y .* (vec(view(src, 2, :)) .- cy)
    srcz = Δk_z .* (vec(view(src, 3, :)) .- cz)
    trgx = Δk_x .* (vec(view(trg, 1, :)) .- cx)
    trgy = Δk_y .* (vec(view(trg, 2, :)) .- cy)
    trgz = Δk_z .* (vec(view(trg, 3, :)) .- cz)

    sigma = T(_ltkm3dd_spreadonly_upsampfac(modes, eps))
    params = _ltkm3dd_spreadonly_kernel_params(T(eps), sigma; kerformula = 1)
    coeffs = _ltkm3dd_spreadonly_horner_coeffs(params)
    nfdim = _ltkm3dd_spreadonly_nfdim(modes, params)
    phi1 = T.(_ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[1], params, coeffs))
    phi2 = T.(_ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[2], params, coeffs))
    phi3 = T.(_ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[3], params, coeffs))

    plan1 = nothing
    plan2 = nothing
    try
        plan1 = _ltkm3dd_make_spreadonly_plan_with_sigma(
            1,
            nfdim,
            -1,
            1,
            T(eps),
            T,
            sigma;
            modeord = 0,
            spread_kerformula = 1,
        )
        TKM3D.FINUFFT.finufft_setpts!(plan1, srcx, srcy, srcz)
        spread = dropdims(TKM3D.FINUFFT.finufft_exec(plan1, complex.(q)); dims = 4)
        fw = FFTW.fft(spread)
        coeff = _ltkm3dd_spreadonly_deconvolveshuffle3d_dir1(fw, modes, phi1, phi2, phi3)

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

        plan2 = _ltkm3dd_make_spreadonly_plan_with_sigma(
            2,
            nfdim,
            1,
            1,
            T(eps),
            T,
            sigma;
            modeord = 0,
            spread_kerformula = 1,
        )
        TKM3D.FINUFFT.finufft_setpts!(plan2, trgx, trgy, trgz)

        prefactor = (Δk_x * Δk_y * Δk_z) / (T(2π)^3)
        pot_fw = _ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(coeff, nfdim, phi1, phi2, phi3)
        pot_grid = FFTW.bfft(pot_fw)
        pot_complex = vec(TKM3D.FINUFFT.finufft_exec(plan2, pot_grid))
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
                grad_fw = _ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(grad_coeff, nfdim, phi1, phi2, phi3)
                grad_grid = FFTW.bfft(grad_fw)
                grad_complex = vec(TKM3D.FINUFFT.finufft_exec(plan2, grad_grid))
                grad[d, :] .= prefactor .* real.(grad_complex)
            end
        end

        selfconst = return_selfconst ? prefactor * diag_sum : nothing
        return pot, grad, selfconst
    finally
        !isnothing(plan1) && TKM3D.FINUFFT.finufft_destroy!(plan1)
        !isnothing(plan2) && TKM3D.FINUFFT.finufft_destroy!(plan2)
    end
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
        pot, grad, selfconst = _ltkm3dd_eval_spreadonly(src, q, src, windowhat, lw, kmax, eps; need_grad = (pg == 2), return_selfconst = true)
        pot .-= q .* selfconst
    end

    if pgt > 0
        isnothing(targets) && throw(ArgumentError("targets are required when pgt > 0"))
        trg = as_3xn(targets)
        pottarg, gradtarg, _ = _ltkm3dd_eval_spreadonly(src, q, trg, windowhat, lw, kmax, eps; need_grad = (pgt == 2))
    elseif !isnothing(targets)
        as_3xn(targets)
    end

    return TKMVals(pot, grad, pottarg, gradtarg, 0)
end
