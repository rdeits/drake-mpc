module SP
    import Base: getindex, endof

    using StaticArrays

    immutable SPolynomial{N, T, S}
        var::Val{S}
        coeffs::SVector{N, T}
    end

    (::Type{SPolynomial{N, T, S}}){N, T, S}(coeffs::NTuple{N, T}) = SPolynomial{N, T, S}(Val{S}(), coeffs)
    SPolynomial{S}(v::Val{S}, coeffs) = SPolynomial(v, SVector(coeffs))

    SPolynomial(var::Symbol, coeffs) = SPolynomial(Val{var}(), SVector(coeffs))

    (p::SPolynomial)(x) = evaluate(p, x)

    evaluate{T, S}(p::SPolynomial{0, T}, x::S) = zero(promote_type(T, S))

    getindex{N, T}(p::SPolynomial{N, T}, i) = (i + 1 > N) ? zero(T) : p.coeffs[i + 1]
    endof(p::SPolynomial) = degree(p)
    degree(p::SPolynomial) = degree(typeof(p))
    degree{N, T, S}(p::Type{SPolynomial{N, T, S}}) = N - 1

    evaluate(p::SPolynomial{1}, args...) = p[0]

    function evaluate{N, T, S}(p::SPolynomial{N, T}, x::S)
        R = promote_type(T, S)
        y = convert(R, p[end])
        for i = (degree(p) - 1):-1:0
            y = p[i] + x * y
        end
        y
    end

    # TODO: do we want this? It kind of makes sense, but it's also somewhat unexpected
    # derivative{T}(p::SPolynomial{2, T}) = p[1]

    @generated function derivative{N, T, S}(p::SPolynomial{N, T, S})
        tup = Expr(:tuple)
        for i in 1:degree(p)
            push!(tup.args, :($i * p[$i]))
        end
        Expr(:call, :(SPolynomial), :(Val{S}()), tup)
    end
end

module PF
    import Base: broadcast

    immutable PiecewiseFunction{Breaks, F} <: Function
        breaks::Breaks
        pieces::Vector{F}

        function PiecewiseFunction(breaks::Breaks, pieces::Vector{F})
            @assert issorted(breaks)
            @assert length(breaks) == length(pieces) + 1
            new(breaks, pieces)
        end
    end

    PiecewiseFunction{Breaks, F}(breaks::Breaks, pieces::Vector{F}) = PiecewiseFunction{Breaks, F}(breaks, pieces)

    (pf::PiecewiseFunction)(t) = from_above(pf, t)

    function from_above(pf::PiecewiseFunction, t)
        i = searchsortedlast(pf.breaks, t)
        if i <= 0 || i >= length(pf.breaks)
            error("Input value $t is out of the allowable range [$(pf.breaks[1]), $(pf.breaks[end]))")
        end
        pf.pieces[i](t - pf.breaks[i])
    end

    function from_below(pf::PiecewiseFunction, t)
        i = searchsortedfirst(pf.breaks, t)
        if i <= 1 || i >= length(pf.breaks) + 1
            error("Input value $t is out of the allowable range ($(pf.breaks[1]), $(pf.breaks[end])]")
        end
        pf.pieces[i - 1](t - pf.breaks[i - 1])
    end

    broadcast(f, pf::PiecewiseFunction) = PiecewiseFunction(pf.breaks, f.(pf.pieces))
end
