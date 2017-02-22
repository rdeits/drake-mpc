module StaticPolynomials
    import Base: getindex, endof

    export SPolynomial, derivative

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