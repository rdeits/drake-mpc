using Base.Test

include("nnls.jl")

function runtests()

    @testset "construct_householder!" begin
        for i in 1:10000
            u = randn(rand(3:10))
            
            u1 = copy(u)
            up1 = NNLS.construct_householder!(u1, 0)
            
            u2 = copy(u)
            up2 = NNLS.h1_reference!(u2)
            @test up1 == up2
            @test u1 == u2
        end
    end

    @testset "apply_householder!" begin
        for i in 1:1000
            u = randn(rand(3:10))
            c = randn(length(u))
            
            u1 = copy(u)
            c1 = copy(c)
            up1 = NNLS.construct_householder!(u1, 0)
            NNLS.apply_householder!(u1, up1, c1)
            
            u2 = copy(u)
            c2 = copy(c)
            up2 = NNLS.h1_reference!(u2)
            NNLS.h2_reference!(u2, up2, c2)
            
            @test up1 == up2
            @test u1 == u2
            @test c1 == c2
        end
    end

    @testset "orthogonal_rotmat" begin
        for i in 1:1000
            a = randn()
            b = randn()
            @test NNLS.orthogonal_rotmat(a, b) == NNLS.g1_reference(a, b)
        end
    end

    @testset "nnls" begin
        srand(1)
        for i in 1:1000
            m = rand(20:100)
            n = rand(20:100)
            A = randn(m, n)
            b = randn(m)

            A1 = copy(A)
            b1 = copy(b)
            work1 = NNLS.NNLSWorkspace(Float64, m, n)
            NNLS.nnls!(work1, A1, b1)

            A2 = copy(A)
            b2 = copy(b)
            work2 = NNLS.NNLSWorkspace(Cdouble, Cint, m, n)
            NNLS.nnls_reference!(work2, A2, b2)

            @test work1.x == work2.x
            @test A1 == A2
            @test b1 == b2
            @test work1.w == work2.w
            @test work1.zz == work2.zz
            @test work1.idx == work2.idx
            @test work1.rnorm[] == work2.rnorm[]
            @test work1.mode[] == work2.mode[]
        end
    end
end

runtests()
Profile.clear_malloc_data()
runtests()