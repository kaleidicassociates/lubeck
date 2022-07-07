/++
$(H1 Lubeck - Linear Algebra)

Authors: Ilya Yaroshenko, Lars Tandle Kyllingstad (SciD author)
+/
module kaleidic.lubeck;

import cblas : Diag;
import mir.blas;
import mir.internal.utility : realType, isComplex;
import mir.lapack;
import mir.math.common;
import mir.ndslice.allocation;
import mir.ndslice.dynamic: transposed;
import mir.ndslice.slice;
import mir.ndslice.topology;
import mir.ndslice.traits : isMatrix;
import mir.utility;
import std.meta;
import std.traits;
import std.typecons: Flag, Yes, No;
import mir.complex: Complex;
public import mir.lapack: lapackint;

template CommonType(A)
{
    alias CommonType = A;
}

template CommonType(A, B)
{
    static if (isComplex!A || isComplex!B)
        alias CommonType = Complex!(CommonType!(realType!A, realType!B));
    else
        alias CommonType = typeof(A.init + B.init);
}

version(LDC)
    import ldc.attributes: fastmath;
else
    enum { fastmath };

private template IterationType(Iterator)
{
    alias T = Unqual!(typeof(Iterator.init[0]));
    static if (isIntegral!T || is(T == real))
        alias IterationType = double;
    else
    static if (is(T == Complex!real))
        alias IterationType = Complex!double;
    else
    {
        static assert(
            is(T == double) ||
            is(T == float) ||
            is(T == Complex!double) ||
            is(T == Complex!float));
        alias IterationType = T;
    }
}

/++
Gets the type that can be used with Blas routines that all types can be implicitly converted to. 

+/
alias BlasType(Iterators...) =
    CommonType!(staticMap!(IterationType, Iterators));

/++
General matrix-matrix multiplication. Allocates result to an uninitialized slice using GC.
Params:
    a = m(rows) x k(cols) matrix
    b = k(rows) x n(cols) matrix
Result: 
    m(rows) x n(cols)
+/
Slice!(BlasType!(IteratorA, IteratorB)*, 2)
    mtimes(IteratorA, SliceKind kindA, IteratorB, SliceKind kindB)(
        Slice!(IteratorA, 2, kindA) a,
        Slice!(IteratorB, 2, kindB) b)
{
    assert(a.length!1 == b.length!0);

    // reallocate data if required
    alias A = BlasType!IteratorA;
    alias B = BlasType!IteratorB;
    alias C = CommonType!(A, B);
    static if (!is(Unqual!IteratorA == C*))
        return .mtimes(a.as!C.slice, b);
    else
    static if (!is(Unqual!IteratorB == C*))
        return .mtimes(a, b.as!C.slice);
    else
    {
        static if (kindA != Contiguous)
            if (a._stride!0 != 1 && a._stride!1 != 1
                || a._stride!0 <= 0
                || a._stride!1 <= 0)
                return .mtimes(a.slice, b);

        static if (kindB != Contiguous)
            if (b._stride!0 != 1 && b._stride!1 != 1
                || b._stride!0 <= 0
                || b._stride!1 <= 0)
                return .mtimes(a, b.slice);

        auto c = uninitSlice!C(a.length!0, b.length!1);

        if (a.length!1 == 1 && b.length!0 == 1)
        {
            c[] = cast(C) 0;
            ger(cast(C)1, a.front!1, b.front, c);
        }
        else
        {
            gemm(cast(C)1, a, b, cast(C)0, c);
        }
        return c;
    }
}

///
unittest
{
    import mir.ndslice;

    auto a =
        [-5,  1,  7, 7, -4,
         -1, -5,  6, 3, -3,
         -5, -2, -3, 6,  0].sliced(3, 5);

    auto b = slice!double(5, 4);
    b[] =
        [[-5, -3,  3,  1],
         [ 4,  3,  6,  4],
         [-4, -2, -2,  2],
         [-1,  9,  4,  8],
         [ 9, 8,  3, -2]];

    assert(mtimes(a, b) ==
        [[-42,  35,  -7, 77],
         [-69, -21, -42, 21],
         [ 23,  69,   3, 29]]
        );
}

/// ger specialized case in mtimes
unittest
{
    import mir.ndslice;

    // from https://github.com/kaleidicassociates/lubeck/issues/8
    {
        auto a = [1.0f, 2.0f].sliced(2, 1);
        auto b = [1.0f, 2.0f].sliced(2, 1);
        assert(mtimes(a, b.transposed) == [[1, 2], [2, 4]]);
    }
    {
        auto a = [1.0, 2.0].sliced(1, 2);
        auto b = [1.0, 2.0].sliced(1, 2);
        assert(mtimes(a.transposed, b) == [[1, 2], [2, 4]]);
    }
}

///
unittest
{
    import mir.ndslice;

    // from https://github.com/kaleidicassociates/lubeck/issues/3
    Slice!(float*, 2) a = slice!float(1, 1);
    Slice!(float*, 2, Universal) b1 = slice!float(16, 1).transposed;
    Slice!(float*, 2) b2 = slice!float(1, 16);

    a[] = 3;
    b1[] = 4;
    b2[] = 4;

    // Confirm that this message does not appear
    // Outputs: ** On entry to SGEMM  parameter number  8 had an illegal value
    assert(a.mtimes(b1) == a.mtimes(b2));
}

/++
General matrix-matrix multiplication. Allocates result to an uninitialized slice using GC.
Params:
    a = m(rows) x k(cols) matrix
    b = k(rows) x 1(cols) vector
Result:
    m(rows) x 1(cols)
+/
Slice!(BlasType!(IteratorA, IteratorB)*)
    mtimes(IteratorA, SliceKind kindA, IteratorB, SliceKind kindB)(
        Slice!(IteratorA, 2, kindA) a,
        Slice!(IteratorB, 1, kindB) b)
{
    assert(a.length!1 == b.length!0);

    // reallocate data if required
    alias A = BlasType!IteratorA;
    alias B = BlasType!IteratorB;
    alias C = CommonType!(A, B);
    static if (!is(Unqual!IteratorA == C*))
        return .mtimes(a.as!C.slice, b);
    else
    static if (!is(Unqual!IteratorB == C*))
        return .mtimes(a, b.as!C.slice);
    else
    {
        static if (kindA != Contiguous)
            if (a._stride!0 != 1 && a._stride!1 != 1
                || a._stride!0 <= 0
                || a._stride!1 <= 0)
                return .mtimes(a.slice, b);

        static if (kindB != Contiguous)
            if (b._stride!1 <= 0)
                return .mtimes(a, b.slice);

        auto c = uninitSlice!C(a.length!0);
        gemv(cast(C)1, a, b, cast(C)0, c);
        return c;
    }
}

/++
General matrix-matrix multiplication.
Params:
    a = 1(rows) x k(cols) vector
    b = k(rows) x n(cols) matrix
Result:
    1(rows) x n(cols)
+/
Slice!(BlasType!(IteratorA, IteratorB)*)
    mtimes(IteratorA, SliceKind kindA, IteratorB, SliceKind kindB)(
        Slice!(IteratorB, 1, kindB) a,
        Slice!(IteratorA, 2, kindA) b,
        )
{
    return .mtimes(b.universal.transposed, a);
}

///
unittest
{
    import mir.ndslice;

    auto a =
        [-5,  1,  7, 7, -4,
         -1, -5,  6, 3, -3,
         -5, -2, -3, 6,  0]
            .sliced(3, 5)
            .universal
            .transposed;

    auto b = slice!double(5);
    b[] = [-5, 4,-4,-1, 9];

    assert(mtimes(b, a) == [-42, -69, 23]);
}

/++
Vector-vector multiplication (dot product).
Params:
    a = 1(rows) x k(cols) vector
    b = k(rows) x 1(cols) matrix
Result:
    scalar
+/
CommonType!(BlasType!IteratorA, BlasType!IteratorB)
    mtimes(IteratorA, SliceKind kindA, IteratorB, SliceKind kindB)(
        Slice!(IteratorB, 1, kindB) a,
        Slice!(IteratorA, 1, kindA) b,
        )
{
    alias A = BlasType!IteratorA;
    alias B = BlasType!IteratorB;
    alias C = CommonType!(A, B);
    static if (is(IteratorB == C*) && is(IteratorA == C*))
    {
        return dot(a, b);
    }
    else
    {
        auto c = cast(typeof(return)) 0;
        import mir.algorithm.iteration: reduce;
        return c.reduce!"a + b * c"(a.as!(typeof(return)), b.as!(typeof(return)));
    }
}

///
unittest
{
    import mir.ndslice;

    auto a = [1, 2, 4].sliced;
    auto b = [3, 4, 2].sliced;
    assert(a.mtimes(b) == 19);
}

/++
Calculates the inverse of a matrix.
+/
auto inv(Iterator, SliceKind kind)(Slice!(Iterator, 2, kind) a)
in
{
    assert (a.length!0 == a.length!1, "matrix must be square");
}
do
{
    alias T = BlasType!Iterator;

    auto m = a.as!T.slice.canonical;
    auto ipiv = m.length.uninitSlice!lapackint;

    auto info = getrf!T(m, ipiv);
    if (info == 0)
    {
        info = getri(m, ipiv, m.getri_wq.uninitSlice!T);
    }

    import std.exception: enforce;
    enforce(info == 0, "inv: matrix is singular");
    return m;
}

///
unittest
{
    import mir.complex;
    import mir.ndslice;

    auto a =  [
        1, 0, 2,
        2, 2, 0,
        0, 1, 1]
        .sliced(3, 3);
    
    enum : double { _13 = 1.0/3.0, _16 = 1.0/6.0, _23 = 2.0/3.0 }
    auto ans = [
        _13, _13, -_23,
        -_13,_16, _23,
        _13, -_16, _13]
        .sliced(a.shape);

    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    assert(equal!((a, b) => a.approxEqual(b, 1e-10L, 1e-10L))(a.inv, ans));
    assert(equal!((a, b) => a.approxEqual(b, 1e-10L, 1e-10L))(a.map!(a => Complex!double(a, 0)).inv.member!"re", ans));
}

///
unittest
{
    import mir.ndslice.topology: iota;

    try
    {
        auto m = [3, 3].iota.inv;
        assert (false, "Matrix should be detected as singular");
    }
    catch (Exception e)
    {
        assert (true);
    }
}

///
struct SvdResult(T)
{
    ///
    Slice!(T*, 2) u;
    ///
    Slice!(realType!T*) sigma;
    ///
    Slice!(T*, 2) vt;
}

/++
Computes the singular value decomposition.

Params:
    matrix = input `M x N` matrix
    slim = If true the first `min(M,N)` columns of `u` and the first
        `min(M,N)` rows of `vt` are returned in the ndslices `u` and `vt`.
Returns: $(LREF SvdResult). Results are allocated by the GC.
+/
auto svd(
        Flag!"allowDestroy" allowDestroy = No.allowDestroy,
        string algorithm = "gesvd",
        SliceKind kind,
        Iterator
    )(
        Slice!(Iterator, 2, kind) matrix,
        Flag!"slim" slim = No.slim,
    )
    if (algorithm == "gesvd" || algorithm == "gesdd")
{
    import lapack;
    alias T = BlasType!Iterator;
    static if (allowDestroy && kind != Universal && is(Iterstor == T*))
        alias a = matrix.canonical;
    else
        auto a = matrix.as!T.slice.canonical;

    auto m = cast(lapackint)a.length!1;
    auto n = cast(lapackint)a.length!0;

    auto s = uninitSlice!(realType!T)(min(m, n));
    auto u = uninitSlice!T(slim ? s.length : m, m);
    auto vt = uninitSlice!T(n, slim ? s.length : n);

    if (m == 0 || n == 0)
    {
        u[] = 0;
        u.diagonal[] = 1;
        vt[] = 0;
        vt.diagonal[] = 1;
    }
    else
    {
        static if (algorithm == "gesvd")
        {
            auto jobu = slim ? 'S' : 'A';
            auto jobvt = slim ? 'S' : 'A';
            auto work = gesvd_wq(jobu, jobvt, a, u.canonical, vt.canonical).uninitSlice!T;

            static if(isComplex!T) {
                auto rwork = uninitSlice!(realType!T)(max(1, 5 * min(m, n)));
                auto info = gesvd!T(jobu, jobvt, a, s, u.canonical, vt.canonical, work, rwork);
            } else {
                auto info = gesvd(jobu, jobvt, a, s, u.canonical, vt.canonical, work);
            }

            enum msg = "svd: DBDSQR did not converge";
        }
        else // gesdd
        {
            auto iwork = uninitSlice!lapackint(s.length * 8);
            auto jobz = slim ? 'S' : 'A';
            auto work = gesdd_wq(jobz, a, u.canonical, vt.canonical).uninitSlice!T;

            static if(isComplex!T) {
                auto mx = max(m, n);
                auto mn = min(m, n);
                auto rwork = uninitSlice!(realType!T)(max(5*mn^^2 + 5*mn, 2*mx*mn + 2*mn^^2 + mn));
                auto info = gesdd!T(jobz, a, s, u.canonical, vt.canonical, work, rwork, iwork);
            } else {
                auto info = gesdd(jobz, a, s, u.canonical, vt.canonical, work, iwork);
            }

            enum msg = "svd: DBDSDC did not converge, updating process failed";
        }
        import std.exception: enforce;
        enforce(info == 0, msg);
    }
    return SvdResult!T(vt, s, u); //transposed
}

///
unittest
{
    import mir.ndslice;

    auto a =  [
         7.52,  -1.10,  -7.95,   1.08,
        -0.76,   0.62,   9.34,  -7.10,
         5.13,   6.62,  -5.66,   0.87,
        -4.75,   8.52,   5.75,   5.30,
         1.33,   4.91,  -5.49,  -3.52,
        -2.40,  -6.77,   2.34,   3.95]
        .sliced(6, 4);

    auto r = a.svd;

    auto sigma = slice!double(a.shape, 0);
    sigma.diagonal[] = r.sigma;
    auto m = r.u.mtimes(sigma).mtimes(r.vt);

    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    assert(equal!((a, b) => a.approxEqual(b, 1e-8, 1e-8))(a, m));
}

///
unittest
{
    import std.typecons: Yes;
    import mir.ndslice;

    auto a =  [
         7.52,  -1.10,  -7.95,   1.08,
        -0.76,   0.62,   9.34,  -7.10,
         5.13,   6.62,  -5.66,   0.87,
        -4.75,   8.52,   5.75,   5.30,
         1.33,   4.91,  -5.49,  -3.52,
        -2.40,  -6.77,   2.34,   3.95]
        .sliced(6, 4);

    auto r = a.svd(Yes.slim);
    assert(r.u.shape == [6, 4]);
    assert(r.vt.shape == [4, 4]);
}

unittest
{
    import mir.algorithm.iteration: all;

    // empty matrix as input means that u or vt is identity matrix
    auto identity = slice!double([4, 4], 0);
    identity.diagonal[] = 1;

    auto a = slice!double(0, 4);
    auto res = a.svd;

    import mir.conv: to;

    assert(res.u.shape == [0, 0]);
    assert(res.vt.shape == [4, 4]);
    assert(res.vt.all!approxEqual(identity), res.vt.to!string);

    auto b = slice!double(4, 0);
    res = b.svd;

    assert(res.u.shape == [4, 4]);
    assert(res.vt.shape == [0, 0]);

    assert(res.u.all!approxEqual(identity), res.u.to!string);
}

unittest
{
    import mir.ndslice;

    alias C = Complex!double;

    auto a =  [
        7.52,  -1.10,  -7.95,   1.08,
        -0.76,   0.62,   9.34,  -7.10,
         5.13,   6.62,  -5.66,   0.87,
        -4.75,   8.52,   5.75,   5.30,
         1.33,   4.91,  -5.49,  -3.52,
        -2.40,  -6.77,   2.34,   3.95]
        .sliced(6, 4).map!(a => C(a)).slice();

    auto r = a.svd;

    auto sigma = slice!C(a.shape, C(0));
    sigma.diagonal[] = r.sigma;
    auto m = r.u.mtimes(sigma).mtimes(r.vt);

    import mir.algorithm.iteration: equal;
    import mir.complex.math: approxEqual;
    assert(equal!((a, b) => a.approxEqual(b, 1e-8, 1e-8))(a, m));
}

/++
Solve systems of linear equations AX = B for X.
Computes minimum-norm solution to a linear least squares problem
if A is not a square matrix.
+/
Slice!(BlasType!(IteratorA, IteratorB)*, 2)
    mldivide
    (IteratorA, SliceKind kindA, IteratorB, SliceKind kindB)(
        Slice!(IteratorA, 2, kindA) a,
        Slice!(IteratorB, 2, kindB) b)
{
    import std.exception: enforce;
    import std.conv: to;

    assert(a.length!0 == b.length!0);

    alias A = BlasType!IteratorA;
    alias B = BlasType!IteratorB;
    alias C = CommonType!(A, B);

    auto a_ = a.universal.transposed.as!C.slice.canonical;
    auto b_ = b.universal.transposed.as!C.slice.canonical;

    if (a.length!0 == a.length!1)
    {
        auto ipiv = a_.length.uninitSlice!lapackint;
        auto info = gesv(a_, ipiv, b_);
    }
    else
    {
        static if(!isComplex!C)
        {
            size_t liwork = void;
            auto lwork = gelsd_wq(a_, b_, liwork);
            auto s = min(a_.length!0, a_.length!1).uninitSlice!C;
            auto work = lwork.uninitSlice!C;
            auto iwork = liwork.uninitSlice!lapackint;
            size_t rank;
            C rcond = -1;

            auto info = gelsd(a_, b_, s, rcond, rank, work, iwork);
        }
        else
        {
            size_t liwork = void;
            size_t lrwork = void;
            auto lwork = gelsd_wq(a_, b_, lrwork, liwork);
            auto s = min(a_.length!0, a_.length!1).uninitSlice!(realType!C);
            auto work = lwork.uninitSlice!C;
            auto iwork = liwork.uninitSlice!lapackint;
            auto rwork = lrwork.uninitSlice!(realType!C);
            size_t rank;
            realType!C rcond = -1;

            auto info = gelsd!C(a_, b_, s, rcond, rank, work, rwork, iwork);
        }

        enforce(info == 0, to!string(info) ~ " off-diagonal elements of an intermediate bidiagonal form did not converge to zero.");
        b_ = b_[0 .. $, 0 .. a_.length!0];
    }

    return b_.universal.transposed.slice;
}

/// ditto
Slice!(BlasType!(IteratorA, IteratorB)*)
    mldivide
    (IteratorA, SliceKind kindA, IteratorB, SliceKind kindB)(
        Slice!(IteratorA, 2, kindA) a,
        Slice!(IteratorB, 1, kindB) b)
{
    return a.mldivide(b.repeat(1).unpack.universal.transposed).front!1.assumeContiguous;
}

/// AX=B
unittest
{
    import mir.complex;
    import std.meta: AliasSeq;
    import mir.ndslice;

    foreach(C; AliasSeq!(double, Complex!double))
    {
        static if(is(C == Complex!double))
            alias transform = a => C(a, 0);
        else
            enum transform = "a";

        auto a = [
            1, -1,  1,
            2,  2, -4,
            -1,  5,  0].sliced(3, 3).map!transform;
        auto b = [
            2.0,  0,
            -6  , -6,
            9  ,  1].sliced(3, 2).map!transform;
        auto t = [
            1.0, -1,
            2  ,  0,
            3  ,  1].sliced(3, 2).map!transform;

        auto x = mldivide(a, b);
        assert(x == t);
    }
}

/// Ax=B
unittest
{
    import mir.complex;
    import std.meta: AliasSeq;
    import mir.ndslice;

    foreach(C; AliasSeq!(double, Complex!double))
    {
        static if(is(C == Complex!double))
            alias transform = a => C(a, 0);
        else
            enum transform = "a";

        auto a = [
            1, -1,  1,
            2,  2, -4,
            -1,  5,  0].sliced(3, 3).map!transform;
        auto b = [
            2.0,
            -6  ,
            9  ].sliced(3).map!transform;
        auto t = [
            1.0,
            2  ,
            3  ].sliced(3).map!transform;

        auto x = mldivide(a, b);
        assert(x == t);
    }
}

/// Least-Squares Solution of Underdetermined System
unittest
{
    import mir.complex;
    import std.meta: AliasSeq;
    import mir.ndslice;

    foreach(C; AliasSeq!(double, )) //Complex!double fails for DMD>=2085
    {
        static if(is(C == Complex!double))
            alias transform = a => C(a, 0);
        else
            enum transform = "a";

        auto a = [
            -0.57,  -1.28,  -0.39,   0.25,
            -1.93,   1.08,  -0.31,  -2.14,
            2.30,   0.24,   0.40,  -0.35,
            -1.93,   0.64,  -0.66,   0.08,
            0.15,   0.30,   0.15,  -2.13,
            -0.02,   1.03,  -1.43,   0.50,
            ].sliced(6, 4).map!transform;

        auto b = [
        -2.67,
        -0.55,
        3.34,
        -0.77,
        0.48,
        4.10,
        ].sliced.map!transform;

        auto x = [
            1.5339,
            1.8707,
            -1.5241,
            0.0392].sliced.map!transform;

        import mir.math.common: approxEqual;
        import mir.algorithm.iteration: all;
        alias appr = all!((a, b) => approxEqual(a, b, 1e-3, 1e-3));
        assert(appr(a.mldivide(b), x));
    }
}

/// Principal component analises result.
struct PcaResult(T)
{
    /// Principal component coefficients, also known as loadings.
    Slice!(T*, 2) coeff;
    /// Principal component scores.
    Slice!(T*, 2) score;
    /// Principal component variances.
    Slice!(T*) latent;
}

/++
Principal component analysis of raw data.

Params:
    matrix = input `M x N` matrix, where 'M (rows)>= N(cols)'
    cc = Flag to centern columns. True by default.
Returns: $(LREF PcaResult)
+/
auto pca(Flag!"allowDestroy" allowDestroy = No.allowDestroy, Iterator, SliceKind kind)(Slice!(Iterator, 2, kind) matrix, in Flag!"centerColumns" cc = Yes.centerColumns)
in
{
    assert(matrix.length!0 >= matrix.length!1);
}
do
{
    import mir.math.sum: sum;
    import mir.algorithm.iteration: maxIndex, eachUploPair;
    import mir.utility: swap;

    alias T = BlasType!Iterator;
    SvdResult!T svdResult;
    if (cc)
    {
        static if (allowDestroy && kind != Universal && is(Iterstor == T*))
            alias m = matrix;
        else
            auto m = matrix.as!T.slice;
        foreach (col; m.universal.transposed)
            col[] -= col.sum!"kb2" / col.length;
        svdResult = m.svd!(Yes.allowDestroy)(Yes.slim);
    }
    else
    {
        svdResult = matrix.svd!(allowDestroy)(Yes.slim);
    }
    with (svdResult)
    {
        foreach (row; u)
            row[] *= sigma;
        T c = max(0, ptrdiff_t(matrix.length) - cast(bool) cc);
        foreach (ref s; sigma)
            s = s * s / c;
        foreach(size_t i; 0 .. sigma.length)
        {
            auto col = vt[i];
            if (col[col.map!fabs.maxIndex] < 0)
            {
                col[] *= -1;
                u[0 .. $, i] *= -1;
            }
        }
        vt.eachUploPair!swap;
        return PcaResult!T(vt, u, sigma);
    }
}

///
unittest
{
    import mir.ndslice;

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;

    auto ingedients = [
         7,  26,   6,  60,
         1,  29,  15,  52,
        11,  56,   8,  20,
        11,  31,   8,  47,
         7,  52,   6,  33,
        11,  55,   9,  22,
         3,  71,  17,   6,
         1,  31,  22,  44,
         2,  54,  18,  22,
        21,  47,   4,  26,
         1,  40,  23,  34,
        11,  66,   9,  12,
        10,  68,   8,  12].sliced(13, 4);

    auto res = ingedients.pca;

    auto coeff = [
        -0.067799985695474,  -0.646018286568728,   0.567314540990512,   0.506179559977705,
        -0.678516235418647,  -0.019993340484099,  -0.543969276583817,   0.493268092159297,
         0.029020832106229,   0.755309622491133,   0.403553469172668,   0.515567418476836,
         0.730873909451461,  -0.108480477171676,  -0.468397518388289,   0.484416225289198,
    ].sliced(4, 4);

    auto score = [
         36.821825999449700,  -6.870878154227367,  -4.590944457629745,   0.396652582713912,
         29.607273420710964,   4.610881963526308,  -2.247578163663940,  -0.395843536696492,
        -12.981775719737618,  -4.204913183175938,   0.902243082694698,  -1.126100587210615,
         23.714725720918022,  -6.634052554708721,   1.854742000806314,  -0.378564808384691,
         -0.553191676624597,  -4.461732123178686,  -6.087412652325177,   0.142384896047281,
        -10.812490833309816,  -3.646571174544059,   0.912970791674604,  -0.134968810314680,
        -32.588166608817929,   8.979846284936063,  -1.606265913996588,   0.081763927599947,
         22.606395499005586,  10.725906457369449,   3.236537714483416,   0.324334774646368,
         -9.262587237675838,   8.985373347478788,  -0.016909578102172,  -0.543746175981799,
         -3.283969329640680, -14.157277337500918,   7.046512994833761,   0.340509860960606,
          9.220031117829379,  12.386080787220454,   3.428342878284624,   0.435152769664895,
        -25.584908517429557,  -2.781693148152386,  -0.386716066864491,   0.446817950545605,
        -26.903161834677597,  -2.930971165042989,  -2.445522630195304,   0.411607156409658,
    ].sliced(13, 4);

    auto latent = [5.177968780739053, 0.674964360487231, 0.124054300480810, 0.002371532651878].sliced;
    latent[] *= 100;

    assert(equal!approxEqual(res.coeff, coeff));
    assert(equal!approxEqual(res.score, score));
    assert(equal!approxEqual(res.latent, latent));
}

/++
Computes Moore-Penrose pseudoinverse of matrix.

Params:
    matrix = Input `M x N` matrix.
    tolerance = The computation is based on SVD and any singular values less than tolerance are treated as zero.
Returns: Moore-Penrose pseudoinverse matrix
+/
Slice!(BlasType!Iterator*, 2)
    pinv(Flag!"allowDestroy" allowDestroy = No.allowDestroy, Iterator, SliceKind kind)(Slice!(Iterator, 2, kind) matrix, double tolerance = double.nan)
{
    import mir.algorithm.iteration: find, each;
    import std.math: nextUp;

    auto svd = matrix.svd!allowDestroy(Yes.slim);
    if (tolerance != tolerance)
    {
        auto n = svd.sigma.front;
        auto eps = n.nextUp - n;
        tolerance = max(matrix.length!0, matrix.length!1) * eps;
    }
    auto st = svd.sigma.find!(a => !(a >= tolerance));
    static if (is(typeof(st) : sizediff_t))
        alias si = st;
    else
        auto si = st[0];
    auto s = svd.sigma[0 .. $ - si];
    s.each!"a = 1 / a";
    svd.vt[0 .. s.length].pack!1.map!"a".zip(s).each!"a.a[] *= a.b";
    auto v = svd.vt[0 .. s.length].universal.transposed;
    auto ut = svd.u.universal.transposed[0 .. s.length];
    return v.mtimes(ut);
}

///
unittest
{
    import mir.ndslice;

    auto a = [
        64,  2,  3, 61, 60,  6,
         9, 55, 54, 12, 13, 51,
        17, 47, 46, 20, 21, 43,
        40, 26, 27, 37, 36, 30,
        32, 34, 35, 29, 28, 38,
        41, 23, 22, 44, 45, 19,
        49, 15, 14, 52, 53, 11,
         8, 58, 59,  5,  4, 62].sliced(8, 6);

    auto b = a.pinv;

    auto result = [
        0.0177, -0.0165, -0.0164,  0.0174,  0.0173, -0.0161, -0.0160,  0.0170,
       -0.0121,  0.0132,  0.0130, -0.0114, -0.0112,  0.0124,  0.0122, -0.0106,
       -0.0055,  0.0064,  0.0060, -0.0043, -0.0040,  0.0049,  0.0045, -0.0028,
       -0.0020,  0.0039,  0.0046, -0.0038, -0.0044,  0.0064,  0.0070, -0.0063,
       -0.0086,  0.0108,  0.0115, -0.0109, -0.0117,  0.0139,  0.0147, -0.0141,
        0.0142, -0.0140, -0.0149,  0.0169,  0.0178, -0.0176, -0.0185,  0.0205].sliced(6, 8);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: all;

    assert(b.all!((a, b) => approxEqual(a, b, 1e-2, 1e-2))(result));
}

/++
Covariance matrix.

Params:
    matrix = matrix whose rows represent observations and whose columns represent random variables.
Returns:
    Normalized by `N-1` covariance matrix.
+/
Slice!(BlasType!Iterator*, 2)
    cov(Iterator, SliceKind kind)(Slice!(Iterator, 2, kind) matrix)
{
    import mir.math.sum: sum;
    import mir.algorithm.iteration: each, eachUploPair;
    alias A = BlasType!Iterator;
    static if (kind == Contiguous)
        auto mc = matrix.canonical;
    else
        alias mc = matrix;
    auto m = mc.shape.uninitSlice!A.canonical;
    auto s = m;

    auto factor = 1 / A(m.length!0 - 1).sqrt;
    while(m.length!1)
    {
        auto shift = - mc.front!1.sum!A / m.length!0;
        m.front!1.each!((ref a, b) { a = (b + shift) * factor; })(mc.front!1);
        mc.popFront!1;
        m.popFront!1;
    }

    auto alpha = cast(A) 1;
    auto beta = cast(A) 0;
    auto c = [s.length!1, s.length!1].uninitSlice!A;
    syrk(Uplo.Upper, alpha, s.universal.transposed, beta, c);

    c.eachUploPair!"b = a";
    return c;
}

///
unittest
{
    import mir.ndslice;

    import std.stdio;
    import mir.ndslice;

    auto c = 8.magic[0..$-1].cov;

    auto result = [
         350.0000, -340.6667, -331.3333,  322.0000,  312.6667, -303.3333, -294.0000,  284.6667,
        -340.6667,  332.4762,  324.2857, -316.0952, -307.9048,  299.7143,  291.5238, -283.3333,
        -331.3333,  324.2857,  317.2381, -310.1905, -303.1429,  296.0952,  289.0476, -282.0000,
         322.0000, -316.0952, -310.1905,  304.2857,  298.3810, -292.4762, -286.5714,  280.6667,
         312.6667, -307.9048, -303.1429,  298.3810,  293.6190, -288.8571, -284.0952,  279.3333,
        -303.3333,  299.7143,  296.0952, -292.4762, -288.8571,  285.2381,  281.6190, -278.0000,
        -294.0000,  291.5238,  289.0476, -286.5714, -284.0952,  281.6190,  279.1429, -276.6667,
         284.6667, -283.3333, -282.0000,  280.6667,  279.3333, -278.0000, -276.6667,  275.3333].sliced(8, 8);
    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: all;
    assert(c.all!((a, b) => approxEqual(a, b, 1e-5, 1e-5))(result));
}

/++
Pearson product-moment correlation coefficients.

Params:
    matrix = matrix whose rows represent observations and whose columns represent random variables.
Returns:
    The correlation coefficient matrix of the variables.
+/
Slice!(BlasType!Iterator*, 2)
    corrcoef(Iterator, SliceKind kind)(Slice!(Iterator, 2, kind) matrix)
{
    import mir.math.common: sqrt, fmin, fmax;
    import core.lifetime: move;
    import mir.algorithm.iteration: eachUploPair;

    auto ret = cov(move(matrix));

    foreach (i; 0 .. ret.length)
    {
        auto isq = 1 / sqrt(ret[i, i]);
        ret[i, i] = 1;
        ret[i, i + 1 .. ret.length] *= isq;
        ret[0 .. i, i] *= isq;
    }

    ret.eachUploPair!((ref a, ref b){b = a = a.fmax(-1).fmin(+1);});
    return ret;
}

///
unittest
{
    import mir.ndslice;

    import std.stdio;
    import mir.ndslice;

    auto m =
      [0.77395605, 0.43887844, 0.85859792,
       0.69736803, 0.09417735, 0.97562235,
       0.7611397 , 0.78606431, 0.12811363].sliced(3, 3);

    auto result =
       [1.        ,  0.99256089, -0.68080986,
        0.99256089,  1.        , -0.76492172,
       -0.68080986, -0.76492172,  1.        ].sliced(3, 3);

    auto corr = m.transposed.corrcoef;

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: all;
    assert(corr.all!((a, b) => approxEqual(a, b, 1e-5, 1e-5))(result));
}

/++
Matrix determinant.
+/
auto detSymmetric(Iterator, SliceKind kind)(char store, Slice!(Iterator, 2, kind) a)
in
{
    assert(store == 'U' || store == 'L');
    assert (a.length!0 == a.length!1, "matrix must be square");
    assert (a.length!0, "matrix must not be empty");
}
do
{
    import mir.algorithm.iteration: each;
    import mir.ndslice.topology: diagonal;
    import mir.math.numeric: ProdAccumulator;

    alias T = BlasType!Iterator;

    auto packed = uninitSlice!T(a.length * (a.length + 1) / 2);
    auto ipiv = a.length.uninitSlice!lapackint;
    int sign;
    ProdAccumulator!T prod;
    if (store == 'L')
    {
        auto pck = packed.stairs!"+"(a.length);
        auto gen = a.stairs!"+";
        pck.each!"a[] = b"(gen);
        auto info = sptrf(pck, ipiv);
        if (info > 0)
            return cast(T) 0;
        for (size_t j; j < ipiv.length; j++)
        {
            auto i = ipiv[j];
            // 1x1 block at m[k,k]
            if (i > 0)
            {
                prod.put(pck[j].back);
                sign ^= i != j + 1; // i.e. row interchanged with another
            }
            else
            {
                i = -i;
                auto offDiag = pck[j + 1][$ - 2];
                auto blockDet = pck[j].back * pck[j + 1].back - offDiag * offDiag;
                prod.put(blockDet);
                sign ^= i != j + 1 && i != j + 2; // row interchanged with other
                j++;
            }
        }
    }
    else
    {
        auto pck = packed.stairs!"-"(a.length);
        auto gen = a.stairs!"-";
        pck.each!"a[] = b"(gen);
        auto info = sptrf(pck, ipiv);
        if (info > 0)
            return cast(T) 0;
        for (size_t j; j < ipiv.length; j++)
        {
            auto i = ipiv[j];
            // 1x1 block at m[k,k]
            if (i > 0)
            {
                prod.put(pck[j].front);
                sign ^= i != j + 1; // i.e. row interchanged with another
            }
            else
            {
                i = -i;
                auto offDiag = pck[j][1];
                auto blockDet = pck[j].front * pck[j + 1].front - offDiag * offDiag;
                prod.put(blockDet);
                sign ^= i != j + 1 && i != j + 2; // row interchanged with other
                j++;
            }
        }
    }
    if(sign & 1)
        prod.x = -prod.x;
    return prod.prod;
}

/// ditto
auto det(Iterator, SliceKind kind)(Slice!(Iterator, 2, kind) a)
in
{
    assert (a.length!0 == a.length!1, "matrix must be square");
}
do
{
    import mir.ndslice.topology: diagonal, zip, iota;
    import mir.math.numeric: ProdAccumulator;

    alias T = BlasType!Iterator;

    auto m = a.as!T.slice.canonical;
    auto ipiv = a.length.uninitSlice!lapackint;

    // LU factorization
    auto info = m.getrf(ipiv);

    // If matrix is singular, determinant is zero.
    if (info > 0)
    {
        return cast(T) 0;
    }

    // The determinant is the product of the diagonal entries
    // of the upper triangular matrix. The array ipiv contains
    // the pivots.
    int sign;
    ProdAccumulator!T prod;
    foreach (tup; m.diagonal.zip(ipiv, [ipiv.length].iota(1)))
    {
        prod.put(tup.a);
        sign ^= tup.b != tup.c; // i.e. row interchanged with another
    }
    if(sign & 1)
        prod.x = -prod.x;
    return prod.prod;
}

///
unittest
{
    import mir.ndslice;
    import mir.math;

    // Check for zero-determinant shortcut.
    auto ssing = [4, 2, 2, 1].sliced(2, 2);
    auto ssingd = det(ssing);
    assert (det(ssing) == 0);
    assert (detSymmetric('L', ssing) == 0);

    // check determinant of empty matrix
    assert(slice!double(0, 0).det == 1);
    // check determinant of zero matrix
    assert(repeat(0, 9).sliced(3, 3).det == 0);

    // General dense matrix.
    int dn = 101;
    auto d = uninitSlice!double(dn, dn);
    foreach (k; 0 .. dn)
    foreach (l; 0 .. dn)
        d[k,l] = 0.5 * (k == l ? (k + 1) * (k + 1) + 1 : 2 * (k + 1) * (l + 1));

    auto dd = det(d);
    import mir.math.common: approxEqual;
    assert (approxEqual(dd, 3.539152633479803e289, double.epsilon.sqrt));

    // Symmetric packed matrix
    auto spa = [ 1.0, -2, 3, 4, 5, -6, -7, -8, -9, 10].sliced.stairs!"+"(4);
    auto sp = [spa.length, spa.length].uninitSlice!double;
    import mir.algorithm.iteration: each;
    sp.stairs!"+".each!"a[] = b"(spa);
    assert (detSymmetric('L', sp).approxEqual(5874.0, double.epsilon.sqrt));
    assert (detSymmetric('U', sp.universal.transposed).approxEqual(5874.0, double.epsilon.sqrt));
}

/++
Eigenvalues and eigenvectors POD.

See_also: $(LREF eigSymmetric).
+/
struct EigSymmetricResult(T)
{
    /// Eigenvalues
    Slice!(T*) values;
    /// Eigenvectors stored in rows
    Slice!(T*, 2) vectors;
}

/++
Eigenvalues and eigenvectors of symmetric matrix.

Returns:
    $(LREF EigSymmetricResult)
+/
auto eigSymmetric(Flag!"computeVectors" cv = Yes.computeVectors, Iterator, SliceKind kind)(char store, Slice!(Iterator, 2, kind) a)
in
{
    assert(store == 'U' || store == 'L');
    assert (a.length!0 == a.length!1, "matrix must be square");
    assert (a.length!0, "matrix must not be empty");
}
do
{
    import mir.algorithm.iteration: each;
    import mir.ndslice.topology: diagonal;
    import mir.math.numeric: ProdAccumulator;

    alias T = BlasType!Iterator;

    auto packed = [a.length * (a.length + 1) / 2].uninitSlice!T;
    auto w = [a.length].uninitSlice!T;
    static if (cv)
    {
        auto z = [a.length, a.length].uninitSlice!T;
    }
    else
    {
        T[1] _vData = void;
        auto z = _vData[].sliced(1, 1);
    }
    auto work = [a.length * 3].uninitSlice!T;
    size_t info = void;
    auto jobz = cv ? 'V' : 'N';
    if (store == 'L')
    {
        auto pck = packed.stairs!"+"(a.length);
        auto gen = a.stairs!"+";
        pck.each!"a[] = b"(gen);
        info = spev!T(jobz, pck, w, z.canonical, work);
    }
    else
    {
        auto pck = packed.stairs!"-"(a.length);
        auto gen = a.stairs!"-";
        pck.each!"a[] = b"(gen);
        info = spev!T(jobz, pck, w, z.canonical, work);
    }
    import std.exception: enforce;
    import std.format: format;
    enforce (info == 0, format("The algorithm failed to converge." ~
        "%s off-diagonal elements of an intermediate tridiagonal form did not converge to zero.", info));
    static if (cv)
    {
        return EigSymmetricResult!T(w, z);
    }
    else
    {
        return EigSymmetricResult!T(w);
    }
}

///
unittest
{
    import mir.ndslice;

    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed;
    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: all;

    auto a = [
        1.0000, 0.5000, 0.3333, 0.2500,
        0.5000, 1.0000, 0.6667, 0.5000,
        0.3333, 0.6667, 1.0000, 0.7500,
        0.2500, 0.5000, 0.7500, 1.0000].sliced(4, 4);

    auto eigr = eigSymmetric('L', a);

    alias appr = all!((a, b) => approxEqual(a, b, 1e-3, 1e-3));

    assert(appr(eigr.values, [0.2078,0.4078,0.8482,2.5362]));

    auto test = [
         0.0693, -0.4422, -0.8105, 0.3778,
        -0.3618,  0.7420, -0.1877, 0.5322,
         0.7694,  0.0486,  0.3010, 0.5614,
        -0.5219, -0.5014,  0.4662, 0.5088].sliced(4, 4).transposed;


    foreach (i; 0 .. 4)
        assert(appr(eigr.vectors[i], test[i]) || appr(eigr.vectors[i].map!"-a", test[i]));
}

version (unittest)
{
/++
Swaps rows of input matrix
Params
    ipiv = pivot points
Returns:
    a = shifted matrix
+/
    private void moveRows(Iterator, SliceKind kind)
                        (Slice!(Iterator, 2, kind) a,
                        Slice!(lapackint*) ipiv)
    {
        import mir.algorithm.iteration: each;
        foreach_reverse(i;0..ipiv.length)
        {
            if(ipiv[i] == i + 1)
                continue;
            each!swap(a[i], a[ipiv[i] - 1]);
        }
    }
}

unittest
{
    import mir.ndslice;

    auto A = 
           [ 9,  9,  9,
             8,  8,  8,
             7,  7,  7,
             6,  6,  6,
             5,  5,  5,
             4,  4,  4,
             3,  3,  3,
             2,  2,  2,
             1,  1,  1,
             0,  0,  0 ]
             .sliced(10, 3)
             .as!double.slice;
    auto ipiv = [ lapackint(10), 9, 8, 7, 6, 6, 7, 8, 9, 10 ].sliced(10);
    moveRows(A, ipiv);

    auto B = 
           [ 0,  0,  0,
             1,  1,  1,
             2,  2,  2,
             3,  3,  3,
             4,  4,  4,
             5,  5,  5,
             6,  6,  6,
             7,  7,  7,
             8,  8,  8,
             9,  9,  9 ]
             .sliced(10, 3)
             .as!double.slice;

    import mir.algorithm.iteration: equal;
    assert(equal!((a, b) => fabs(a - b) < 1e-12)(B, A));
}

unittest
{
    auto A = 
           [ 1,  1,  1,
             2,  2,  2,
             3,  3,  3,
             4,  4,  4,
             5,  5,  5,
             6,  6,  6,
             7,  7,  7,
             8,  8,  8,
             9,  9,  9,
             0,  0,  0 ]
             .sliced(10, 3)
             .as!double.slice;
    auto ipiv = [ lapackint(2), 3, 4, 5, 6, 7, 8, 9, 10, 10 ].sliced(10);
    moveRows(A, ipiv);
    
    auto B = 
           [ 0,  0,  0,
             1,  1,  1,
             2,  2,  2,
             3,  3,  3,
             4,  4,  4,
             5,  5,  5,
             6,  6,  6,
             7,  7,  7,
             8,  8,  8,
             9,  9,  9 ]
             .sliced(10, 3)
             .as!double.slice;

    import mir.algorithm.iteration: equal;
    assert(equal!((a, b) => fabs(a - b) < 1e-12)(B, A));
}

///LUResult consist lu factorization.
struct LUResult(T)
{
    /++
    Matrix in witch lower triangular is L part of factorization
    (diagonal elements of L are not stored), upper triangular
    is U part of factorization.
    +/
    Slice!(T*, 2, Canonical) lut;
    /++
    The pivot indices, for 1 <= i <= min(M,N), row i of the matrix
    was interchanged with row ipiv(i).
    +/
    Slice!(lapackint*) ipiv;
    ///L part of the factorization.
    auto l() @property
    {
        import mir.algorithm.iteration: eachUpper;
        auto l = lut.transposed[0..lut.length!1, 0..min(lut.length!0, lut.length!1)].slice.canonical;
        l.eachUpper!"a = 0";
        l.diagonal[] = 1;
        return l;
    }
    ///U part of the factorization.
    auto u() @property
    {
        import mir.algorithm.iteration: eachLower;
        auto u = lut.transposed[0..min(lut.length!0, lut.length!1), 0..lut.length!0].slice.canonical;
        u.eachLower!"a = 0";
        return u;
    }

    ///
    auto solve(Flag!"allowDestroy" allowDestroy = No.allowDestroy, Iterator, size_t N, SliceKind kind)(
        char trans,
        Slice!(Iterator, N, kind) b)
    {
        return luSolve!(allowDestroy)(trans, lut, ipiv, b);
    }
}

/++
Computes LU factorization of a general 'M x N' matrix 'A' using partial
pivoting with row interchanges.
The factorization has the form:
    \A = P * L * U
Where P is a permutation matrix, L is lower triangular with unit
diagonal elements (lower trapezoidal if m > n), and U is upper
triangular (upper trapezoidal if m < n).
Params:
    allowDestroy = flag to delete the source matrix.
    a = input 'M x N' matrix for factorization.
Returns: $(LREF LUResalt)
+/
auto luDecomp(Flag!"allowDestroy" allowDestroy = No.allowDestroy,
              Iterator, SliceKind kind)
             (Slice!(Iterator, 2, kind) a)
{
    alias T = BlasType!Iterator;
    auto ipiv = uninitSlice!lapackint(min(a.length!0, a.length!1));
    auto b = a.transposed;
    auto m = (allowDestroy && b._stride!1 == 1) ? b.assumeCanonical : a.transposed.as!T.slice.canonical;
    
    getrf(m, ipiv);
    return LUResult!T(m, ipiv);
}

/++
Solves a system of linear equations
    \A * X = B, or
    \A**T * X = B
with a general 'N x N' matrix 'A' using the LU factorization computed by luDecomp.
Params:
    allowDestroy = flag to delete the source matrix.
    lut = factorization of matrix 'A', A = P * L * U.
    ipiv = the pivot indices from luDecomp.
    b = the right hand side matrix B.
    trans = specifies the form of the system of equations:
          'N': A * X = B (No transpose)
          'T': A**T * X = B (Transpose)
          'C': A**T * X = B (Conjugate transpose = Transpose)
Returns:
    Return solve of the system linear equations.
+/
auto luSolve(Flag!"allowDestroy" allowDestroy = No.allowDestroy, SliceKind kindB, size_t N, IteratorB, IteratorLU)(
    char trans,
    Slice!(IteratorLU, 2, Canonical) lut,
    Slice!(lapackint*) ipiv,
    Slice!(IteratorB, N, kindB) b,
    )
in
{
    assert(lut.length!0 == lut.length!1, "matrix must be squared");
    assert(ipiv.length == lut.length, "size of ipiv must be equal to the number of rows a");
    assert(lut.length!1 == b.length!0, "number of columns a should be equal to the number of rows b");
}
do
{
    alias LU = BlasType!IteratorLU;
    alias B = BlasType!IteratorB;
    alias T = CommonType!(LU, B);
    static if(is(T* == IteratorLU))
        auto lut_ = lut;
    else
        auto lut_ = lut.as!T.slice.canonical;
    
    //convect vector to matrix.
    static if(N == 1)
        auto k = b.sliced(1, b.length);
    else
        auto k = b.transposed;

    static if(is(IteratorB == T*))
        auto m = (allowDestroy && k._stride!1 == 1) ? k.assumeCanonical : k.as!T.slice.canonical;
    else
        auto m = k.as!T.slice.canonical;
    getrs!T(trans, lut_, m, ipiv);
    return m.transposed;
}

unittest
{
    auto A =
        [ 1,  4, -3,  5,  6,
         -2,  8,  5,  7,  8,
          3,  4,  7,  9,  1,
          2,  4,  6,  3,  2,
          6,  8,  3,  5,  2 ]
            .sliced(5, 5)
            .as!double.slice
            .canonical;
    auto B =
        [ 1,  3,  4,  8,  0,  0,  0,
          2,  1,  7,  1,  0,  0,  0,
          3,  5,  7,  7,  0,  0,  0,
          4,  4,  9,  8,  0,  0,  0,
          5,  5,  8,  1,  0,  0,  0 ]
            .sliced(5, 7)
            .as!double.slice
            .universal;

    auto B_ = B[0..$, 0..4];
    auto LU = A.luDecomp();
    auto m = luSolve('N', LU.lut, LU.ipiv, B_);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, m), B_));
}

///
unittest
{
    import mir.math;
    import mir.ndslice;

    auto A =
        [ 1,  4, -3,  5,  6,
         -2,  8,  5,  7,  8,
          3,  4,  7,  9,  1,
          2,  4,  6,  3,  2,
          6,  8,  3,  5,  2 ]
            .sliced(5, 5)
            .as!double.slice
            .canonical;
    
    import mir.random.variable;
    import mir.random.algorithm;
    auto B = randomSlice(uniformVar(-100, 100), 5, 100);
    
    auto LU = A.luDecomp();
    auto X = LU.solve('N', B);

    import mir.algorithm.iteration: equal;
    assert(equal!((a, b) => fabs(a - b) < 1e-12)(mtimes(A, X), B));
}

unittest
{
    auto A =
        [ 1,  4, -3,  5,  6,
         -2,  8,  5,  7,  8,
          3,  4,  7,  9,  1,
          2,  4,  6,  3,  2,
          6,  8,  3,  5,  2 ]
            .sliced(5, 5)
            .as!double.slice
            .canonical;
    auto B =
        [ 2,  8,  3,  5,  8,
          8,  1,  4,  9,  86,
          1,  6,  7,  1,  67,
          6,  1,  5,  4,  45,
          1,  2,  3,  1,  11 ]
            .sliced(5, 5)
            .as!double.slice
            .universal;
    auto C = B.slice;

    auto LU = A.luDecomp();
    auto m = luSolve!(Yes.allowDestroy)('N', LU.lut, LU.ipiv, B.transposed);
    auto m2 = LU.solve('N', C);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, m), C.transposed));
    assert(equal!approxEqual(mtimes(A, m2), C));
}

unittest
{
    auto A =
        [ 1,  4, -3,  5,  6,
         -2,  8,  5,  7,  8,
          3,  4,  7,  9,  1,
          2,  4,  6,  3,  2,
          6,  8,  3,  5,  2 ]
            .sliced(5, 5)
            .as!float.slice
            .canonical;
    auto B = [ 1,  2,  3,  4,  5 ].sliced(5).as!double.slice;
    auto C = B.slice.sliced(5, 1);

    auto LU = A.luDecomp();
    auto m = luSolve!(Yes.allowDestroy)('N', LU.lut, LU.ipiv, B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, m), C));
}

unittest
{
    auto A =
        [ 1,  4, -3,  5,  6,
         -2,  8,  5,  7,  8,
          3,  4,  7,  9,  1,
          2,  4,  6,  3,  2,
          6,  8,  3,  5,  2 ]
            .sliced(5, 5)
            .as!double.slice
            .canonical;
    auto B = [ 1,  15,  4,  5,  8,
               3,  20,  1,  9,  11 ].sliced(5, 2).as!float.slice;

    auto LU = A.luDecomp();
    auto m = luSolve('N', LU.lut, LU.ipiv, B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, m), B));
}

unittest
{
    auto A =
        [ 11,  14, -31,  53,  62,
         -92,  83,  52,  74,  83,
          31,  45,  73,  96,  17,
          23,  14,  65,  35,  26,
          62,  28,  34,  51,  25 ]
            .sliced(5, 5)
            .as!float.slice
            .universal;
    auto B =
        [ 6,  1,  3,  1,  11,
         12,  5,  7,  6,  78,
          8,  4,  1,  5,  54,
          3,  1,  8,  1,  45,
          1,  6,  8,  6,  312 ]
            .sliced(5, 5)
            .as!double.slice;
    auto B2 = B.slice;
    auto C = B.slice;

    auto LU = luDecomp(A.transposed);
    auto m = luSolve!(Yes.allowDestroy)('T', LU.lut, LU.ipiv, B);
    auto m2 = luSolve!(Yes.allowDestroy)('N', LU.lut, LU.ipiv, B2);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));

    assert(appr(mtimes(A, m), C));
    assert(appr(mtimes(A.transposed, m2), C));
}

unittest
{
    auto A =
        [ 54,  93,  14,  44,  33,
          51,  85,  28,  81,  75,
          89,  17,  15,  44,  58,
          75,  80,  18,  35,  14,
          21,  48,  72,  21,  88 ]
            .sliced(5, 5)
            .as!double.slice
            .universal;
    auto B =
        [ 5,  7,  8,  3,  78,
          1,  2,  5,  4,  5,
          2,  4,  1,  5,  15,
          1,  1,  4,  1,  154,
          1,  3,  1,  8,  17 ]
            .sliced(5, 5)
            .as!float.slice
            .canonical;

    auto LU = A.luDecomp();
    auto m = luSolve('N', LU.lut, LU.ipiv, B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, m), B));
}

unittest
{
    
    auto B =
        [ 1,  4, -3,
         -2,  8,  5,
          3,  4,  7,
          2,  4,  6 ]
            .sliced(4, 3)
            .as!double.slice;

    auto LU = B.luDecomp!(Yes.allowDestroy)();
    LU.l; LU.u;
    auto res = mtimes(LU.l, LU.u);
    moveRows(res, LU.ipiv);

    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    assert(res.equal!approxEqual(B));
}

unittest
{
    import mir.ndslice;

    auto B =
        [ 3, -7, -2,  2,
         -3,  5,  1,  0,
          6, -4,  0, -5,
         -9,  5, -5, 12 ]
            .sliced(4, 4)
            .as!double.slice
            .canonical;
    auto C = B.transposed.slice;

    auto LU = B.transposed.luDecomp!(Yes.allowDestroy)();
    auto res = mtimes(LU.l, LU.u);
    moveRows(res, LU.ipiv);

    import mir.algorithm.iteration: equal;
    import mir.math.common: approxEqual;
    assert(res.equal!approxEqual(C));
}

///Consist LDL factorization;
struct LDLResult(T)
{
    /++
    uplo = 'U': Upper triangle is stored;
         'L': lower triangle is stored.
    +/
    char uplo;
    /++
    Matrix in witch lower triangular matrix is 'L' part of
    factorization, diagonal is 'D' part.
    +/
    Slice!(T*, 2, Canonical) matrix;
    /++
    The pivot indices.
    If ipiv(k) > 0, then rows and columns k and ipiv(k) were
    interchanged and D(k, k) is a '1 x 1' diagonal block.
    If ipiv(k) = ipiv(k + 1) < 0, then rows and columns k+1 and
    -ipiv(k) were interchanged and D(k:k+1, k:k+1) is a '2 x 2'
    diagonal block.
    +/
    Slice!(lapackint*) ipiv;
    /++
    Return solves a system of linear equations
        \A * X = B,
    using LDL factorization.
    +/
    auto solve(Flag!"allowDestroy" allowDestroy = No.allowDestroy, SliceKind kindB, size_t N, IteratorB)(
        Slice!(IteratorB, N, kindB) b)
    {
        return ldlSolve!(allowDestroy)(uplo, matrix, ipiv, b);
    }
}

/++
Computes the factorization of a real symmetric matrix A using the
Bunch-Kaufman diagonal pivoting method.
The for of the factorization is:
    \A = L*D*L**T
Where L is product if permutation and unit lower triangular matrices,
and D is symmetric and block diagonal with '1 x 1' and '2 x 2'
diagonal blocks.
Params:
    allowDestroy = flag to delete the source matrix.
    a = input symmetric 'n x n' matrix for factorization.
    uplo = 'U': Upper triangle is stored;
         'L': lower triangle is stored.
Returns:$(LREF LDLResult)
+/
auto ldlDecomp(Flag!"allowDestroy" allowDestroy = No.allowDestroy, Iterator, SliceKind kind)(
    char uplo,
    Slice!(Iterator, 2, kind) a)
in
{
    assert(a.length!0 == a.length!1, "matrix must be squared");
}
do
{
    alias T = BlasType!Iterator;
    auto work = [T.sizeof * a.length].uninitSlice!T;
    auto ipiv = a.length.uninitSlice!lapackint;
    auto m = (allowDestroy && a._stride!1 == 1) ? a.assumeCanonical : a.transposed.as!T.slice.canonical;

    sytrf!T(uplo, m, ipiv, work);
    return LDLResult!T(uplo, m, ipiv);
}

/++
Solves a system of linear equations \A * X = B with symmetric matrix 'A' using the
factorization
\A = U * D * U**T, or
\A = L * D * L**T
computed by ldlDecomp.
Params:
    allowDestroy = flag to delete the source matrix.
    a = 'LD' or 'UD' matrix computed by ldlDecomp.
    ipiv = details of the interchanges and the block structure of D as determined by ldlDecomp.
    b = the right hand side matrix.
    uplo = specifies whether the details of the factorization are stored as an upper or
           lower triangular matrix:
         'U': Upper triangular, form is \A = U * D * U**T;
         'L': Lower triangular, form is \A = L * D * L**T.
Returns:
    The solution matrix.
+/
auto ldlSolve(Flag!"allowDestroy" allowDestroy = No.allowDestroy, SliceKind kindB, size_t N, IteratorB, IteratorA)(
    char uplo,
    Slice!(IteratorA, 2, Canonical) a,
    Slice!(lapackint*) ipiv,
    Slice!(IteratorB, N, kindB) b)
in
{
    assert(a.length!0 == a.length!1, "matrix must be squared");
    assert(ipiv.length == a.length, "size of ipiv must be equal to the number of rows a");
    assert(a.length!1 == b.length!0, "number of columns a should be equal to the number of rows b");
}
do
{
    alias A = BlasType!IteratorA;
    alias B = BlasType!IteratorB;
    alias T = CommonType!(A, B);
    static if(is(T* == IteratorA))
        auto a_ = a;
    else
        auto a_ = a.as!T.slice.canonical;
    
    //convect vector to matrix.
    static if(N == 1)
        auto k = b.sliced(1, b.length);
    else
        auto k = b.transposed;

    auto work = [T.sizeof * a.length].uninitSlice!T;
    static if(is(IteratorB == T*))
        auto m = (allowDestroy && k._stride!1 == 1) ? k.assumeCanonical : k.as!T.slice.canonical;
    else
        auto m = k.as!T.slice.canonical;
    sytrs2!T(a_, m, ipiv, work, uplo);
    return m.transposed;
}

///
unittest
{
    import mir.ndslice;

    auto A =
        [ 2.07,  3.87,  4.20, -1.15,
          3.87, -0.21,  1.87,  0.63,
          4.20,  1.87,  1.15,  2.06,
         -1.15,  0.63,  2.06, -1.81 ]
            .sliced(4, 4)
            .as!double.slice
            .canonical;

    import mir.random.variable;
    import mir.random.algorithm;
    auto B = randomSlice(uniformVar(-100, 100), 4, 100);

    auto LDL = ldlDecomp('L', A);
    auto X = LDL.solve(B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, X), B));
}

unittest
{
    auto A =
        [ 9, -1,  2,
         -1,  8, -5,
          2, -5,  7 ]
            .sliced(3, 3)
            .as!float.slice
            .canonical;
    auto A_ = A.slice;

    auto B =
        [ 5,  7,  1,
          1,  8,  5,
          9,  3,  2 ]
          .sliced(3, 3)
          .as!double.slice
          .canonical;
    auto B_ = B.slice;

    auto LDL = ldlDecomp!(Yes.allowDestroy)('L', A);
    auto X = ldlSolve!(Yes.allowDestroy)(LDL.uplo, A, LDL.ipiv, B.transposed);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A_, X), B_.transposed));
}

unittest
{
    auto A =
        [10, 20, 30,
         20, 45, 80,
         30, 80, 171 ]
            .sliced(3, 3)
            .as!double.slice
            .canonical;
    auto B = [ 1, 4, 7 ].sliced(3).as!float.slice.canonical;
    auto B_ = B.sliced(3, 1);

    auto LDL = ldlDecomp('L', A);
    auto X = LDL.solve(B);
    
    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, X), B_));
}

struct choleskyResult(T)
{
    /++
    If uplo = 'L': lower triangle of 'matrix' is stored.
    If uplo = 'U': upper triangle of 'matrix' is stored.
    +/
    char uplo;
    /++
    if uplo = 'L', the leading 'N x N' lower triangular part of A
    contains the lower triangular part of the matrix A, and the
    strictly upper triangular part of A is zeroed.
    if uplo = 'U', the leading 'N x N' upper triangular part of A
    contains the upper triangular part of the matrix A, and the
    strictly lower triangular part of A is zeroed.
    +/
    Slice!(T*, 2, Canonical) matrix;
    /++
    Return solves a system of linear equations
        \A * X = B,
    using Cholesky factorization.
    +/
    auto solve(Flag!"allowDestroy" allowDestroy = No.allowDestroy,
               Iterator, size_t N, SliceKind kind)
              (Slice!(Iterator, N, kind) b)
    {
        return choleskySolve!(allowDestroy)(uplo, matrix, b);
    }
}
/++
Computs Cholesky decomposition of symmetric positive definite matrix 'A'.
The factorization has the form:
    \A = U**T * U, if UPLO = 'U', or
    \A = L * L**T, if UPLO = 'L'
Where U is an upper triangular matrix and L is lower triangular.
Params:
    allowDestroy = flag to delete the source matrix.
    a = symmetric 'N x N' matrix.
    uplo = if uplo is Upper, then upper triangle of A is stored, else
    lower.
Returns: $(LREF choleskyResult)
+/
auto choleskyDecomp( Flag!"allowDestroy" allowDestroy = No.allowDestroy, Iterator, SliceKind kind)(
    char uplo,
    Slice!(Iterator, 2, kind) a)
in
{
    assert(uplo == 'L' || uplo == 'U');
    assert(a.length!0 == a.length!1, "matrix must be squared");
}
do
{
    import mir.exception: MirException;
    alias T = BlasType!Iterator;
    static if(is(Iterator == T*))
        auto m = (allowDestroy && a._stride!1 == 1) ? a.assumeCanonical : a.as!T.slice.canonical;
    else
        auto m = a.as!T.slice.canonical;
    if (auto info = potrf!T(uplo == 'U' ? 'L' : 'U', m))
        throw new MirException("Leading minor of order ", info, " is not positive definite, and the factorization could not be completed.");
    import mir.algorithm.iteration: eachUploPair;
    auto d = m.universal;
    if (uplo == 'U')
        d = d.transposed;
    d.eachUploPair!("a = 0");
    return choleskyResult!T(uplo, m);
}

///
unittest
{
    import mir.algorithm.iteration: equal, eachUploPair;
    import mir.ndslice;
    import mir.random.algorithm;
    import mir.random.variable;
    import mir.math.common: approxEqual;

    auto A =
           [ 25, double.nan, double.nan,
             15, 18,  double.nan,
             -5,  0, 11 ]
             .sliced(3, 3);
    
    auto B = randomSlice(uniformVar(-100, 100), 3, 100);

    auto C = choleskyDecomp('L', A);
    auto X = C.solve(B);

    A.eachUploPair!"a = b";
    assert(equal!approxEqual(mtimes(A, X), B));
}

/++
    Solves a system of linear equations A * X = B with a symmetric matrix A using the
    Cholesky factorization:
    \A = U**T * U or
    \A = L * L**T
    computed by choleskyDecomp.
Params:
    allowDestroy = flag to delete the source matrix.
    c = the triangular factor 'U' or 'L' from the Cholesky factorization
        \A = U**T * U or
        \A = L * L**T,
    as computed by choleskyDecomp.
    b = the right hand side matrix.
    uplo = 'U': Upper triangle of A is stored;
         'L': Lower triangle of A is stored.
Returns:
    The solution matrix X.
+/
auto choleskySolve(Flag!"allowDestroy" allowDestroy = No.allowDestroy, SliceKind kindB, size_t N, IteratorB, IteratorC)(
    char uplo,
    Slice!(IteratorC, 2, Canonical) c,
    Slice!(IteratorB, N, kindB) b)
in
{
    assert(uplo == 'L' || uplo == 'U');
    assert(c.length!0 == c.length!1, "matrix must be squared");
    assert(c.length!1 == b.length!0, "number of columns a should be equal to the number of rows b");
}
do
{
    uplo = uplo == 'U' ? 'L' : 'U';
    alias B = BlasType!IteratorB;
    alias C = BlasType!IteratorC;
    alias T = CommonType!(B, C);
    static if(is(T* == IteratorC))
        auto c_ = c;
    else
        auto c_ = c.as!T.slice.canonical;

    //convect vector to matrix.
    static if(N == 1)
        auto k = b.sliced(1, b.length);
    else
        auto k = b.transposed;

    static if(is(IteratorB == T*))
        auto m = (allowDestroy && k._stride!1 == 1) ? k.assumeCanonical : k.as!T.slice.canonical;
    else
        auto m = k.as!T.slice.canonical;
    potrs!T(uplo, c_, m);
    return m.transposed;
}

///
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: as;
    import std.typecons: Flag, Yes;

    auto A =
            [ 1.0,  1,  3,
              1  ,  5,  5,
              3  ,  5, 19 ].sliced(3, 3);

    auto B = [ 10.0,  157,  80 ].sliced;
    auto C_ = B.dup.sliced(3, 1);

    auto C = choleskyDecomp('U', A);
    auto X = choleskySolve!(Yes.allowDestroy)(C.uplo, C.matrix, B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, X), C_));
}

unittest
{
    auto A =
            [6.0f,  15,  55,
             15,  55, 225,
             55, 225, 979 ].sliced(3, 3).canonical;
    auto B =
            [ 7.0,  3,
              2,  1,
              1,  8 ].sliced(3, 2).universal;

    auto C = choleskyDecomp('L', A);
    auto X = choleskySolve(C.uplo, C.matrix, B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));

    assert(appr(mtimes(A, X), B));
}


///
struct QRResult(T)
{
    /++
    Matrix in witch the elements on and above the diagonal of the array contain the min(M, N) x N
    upper trapezoidal matrix 'R' (R is upper triangular if m >= n). The elements below the
    diagonal, with the array tau, represent the orthogonal matrix 'Q' as product of min(m, n).
    +/
    Slice!(T*, 2, Canonical) matrix;
    ///The scalar factors of the elementary reflectors
    Slice!(T*) tau;
    /++
    Solve the least squares problem:
        \min ||A * X - B||
    Using the QR factorization:
        \A = Q * R
    computed by qrDecomp.
    +/
    auto solve(Flag!"allowDestroy" allowDestroy = No.allowDestroy,
               Iterator, size_t N, SliceKind kind)
              (Slice!(Iterator, N, kind) b)
    {
        return qrSolve!(allowDestroy)(matrix, tau, b);
    }
    
    /++
    Extract the Q matrix
    +/
    auto Q(Flag!"allowDestroy" allowDestroy = No.allowDestroy)
    {
        auto work = [matrix.length].uninitSlice!T;

        auto m = (allowDestroy && matrix._stride!1 == 1) ? matrix.assumeCanonical : matrix.as!T.slice.canonical;

        static if(is(T == double) || is(T == float))
            orgqr!T(m, tau, work);
        else
            ungqr!T(m, tau, work);
        return m.transposed;
    }
    
    /++
    Extract the R matrix
    +/
    auto R()
    {
        import mir.algorithm.iteration: eachLower;

        auto r = [tau.length, tau.length].uninitSlice!T;
        if (matrix.shape[0] == matrix.shape[1]) {
            r[] = matrix.transposed.slice;
        } else {
            r[] = matrix[0..tau.length, 0..tau.length].transposed.slice;
        }
        r.eachLower!((ref a) {a = cast(T)0;});
        return r.universal;
    }

    /++
    Reconstruct the original matrix given a QR decomposition
    +/
    auto reconstruct()
    {
        auto q = Q();
        auto r = R();
        return reconstruct(q, r);
    }

    /++
    Reconstruct the original matrix given a QR decomposition
    +/
    auto reconstruct(T, U)(T q, U r)
        if (isMatrix!T && isMatrix!U)
    {
        return mtimes(q, r).universal;
    }
}

///
unittest
{
    import mir.ndslice;

    auto A =
            [ 1,  1,  0,
              1,  0,  1,
              0,  1,  1 ]
              .sliced(3, 3)
              .as!double.slice;

    auto Q_test =
            [ -0.7071068,  0.4082483,  -0.5773503,
              -0.7071068, -0.4082483,   0.5773503,
                       0,  0.8164966,   0.5773503]
              .sliced(3, 3)
              .as!double.slice;

    auto R_test =
            [ -1.414214,  -0.7071068,   -0.7071068,
                      0,   1.2247449,    0.4082483,
                      0,           0,    1.1547005]
              .sliced(3, 3)
              .as!double.slice;

    auto val = qrDecomp(A);

    //saving these values to doublecheck they don't change later
    auto val_matrix = val.matrix.slice;
    auto val_tau = val.tau.slice;

    import mir.math.common: approxEqual;
    import mir.ndslice : equal;
    
    auto r = val.R;
    assert(equal!approxEqual(val.R, R_test));

    auto q = val.Q;
    assert(equal!approxEqual(val.Q, Q_test));

    //double-checking values do not change
    assert(equal!approxEqual(val_matrix, val.matrix));
    assert(equal!approxEqual(val_tau, val.tau));

    auto a = val.reconstruct;
    assert(equal!approxEqual(A, a));
}

unittest
{
    auto A =
            [  3,  -6,
               4,  -8,
               0,   1]
              .sliced(3, 2)
              .as!double.slice;

    auto Q_check =
            [ -0.6,  0,
              -0.8,  0,
               0.0, -1]
              .sliced(3, 2)
              .as!double.slice;

    auto R_check =
            [ -5,  10,
               0,  -1]
              .sliced(2, 2)
              .as!double.slice;

    auto C = qrDecomp(A);
    auto q = C.Q;
    auto r = C.R;
    auto A_reconstructed = C.reconstruct(q, r);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(q, Q_check));
    assert(equal!approxEqual(r, R_check));
    assert(equal!approxEqual(A_reconstructed, A));
}

/++
Computes a QR factorization of matrix 'a'.
Params:
    allowDestroy = flag to delete the source matrix.
    a = initial matrix
Returns: $(LREF QRResult)
+/
auto qrDecomp(Flag!"allowDestroy" allowDestroy = No.allowDestroy,
              Iterator, SliceKind kind)
             (Slice!(Iterator, 2, kind) a)
{
    alias T = BlasType!Iterator;
    auto work = [T.sizeof * a.length].uninitSlice!T;
    auto tau = (cast(int) min(a.length!0, a.length!1)).uninitSlice!T;
    auto m = (allowDestroy && a._stride!1 == 1) ? a.assumeCanonical : a.transposed.as!T.slice.canonical;

    geqrf!T(m, tau, work);
    return QRResult!T(m, tau);
}

/++
Solve the least squares problem:
    \min ||A * X - B||
Using the QR factorization:
    \A = Q * R
computed by qrDecomp.
Params:
    allowDestroy = flag to delete the source matrix.
    a = detalis of the QR factorization of the original matrix as returned by qrDecomp.
    tau = details of the orhtogonal matrix Q.
    b = right hand side matrix.
Returns: solution matrix.
+/
auto qrSolve(Flag!"allowDestroy" allowDestroy = No.allowDestroy,
             SliceKind kindB, size_t N, IteratorB, IteratorA, IteratorT)
            (Slice!(IteratorA, 2, Canonical) a,
             Slice!(IteratorT) tau,
             Slice!(IteratorB, N, kindB) b
            )
in
{
    assert(a.length!1 == b.length!0, "number of columns a should be equal to the number of rows b");
}
do
{
    alias A = BlasType!IteratorA;
    alias B = BlasType!IteratorB;
    alias T = CommonType!(A, B);
    static if(is(T* == IteratorA))
        auto a_ = a;
    else
        auto a_ = a.as!T.slice.canonical;
    static if(is(T* == IteratorT))
        auto tau_ = tau;
    else
        auto tau_ = tau.as!T.slice.canonical;

    //convect vector to matrix.
    static if(N == 1)
        auto k = b.sliced(1, b.length);
    else
        auto k = b.transposed;

    static if(is(IteratorB == T*))
        auto m = (allowDestroy && k._stride!1 == 1) ? k.assumeCanonical : k.as!T.slice.canonical;
    else
        auto m = k.as!T.slice.canonical;
    auto work = [m.length!0].uninitSlice!T;
    static if(is(T == double) || is(T == float))
        ormqr!T('L', 'T', a_, tau_, m, work);
    else
        unmqr!T('L', 'C', a_, tau_, m, work);

    if (a_.length!0 != a_.length!1) {
        a_ = a_[0..tau.length, 0..tau.length];
        m = m.selectFront!1(tau.length);
    }
    trsm!T(Side.Right, Uplo.Lower, Diag.NonUnit, cast(T) 1.0, a_, m);

    return m.transposed;
}

///
unittest
{
    import mir.ndslice;

    auto A =
            [ 3,  1, -1,  2,
             -5,  1,  3, -4,
              2,  0,  1, -1,
              1, -5,  3, -3 ]
              .sliced(4, 4)
              .as!double.slice;

    import mir.random.variable;
    import mir.random.algorithm;
    auto B = randomSlice(uniformVar(-100, 100), 4, 100);

    auto C = qrDecomp(A);
    auto X = C.solve(B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, X), B));
}

unittest
{
    auto A =
            [ 3,  1, -1,  2,
             -5,  1,  3, -4,
              2,  0,  1, -1,
              1, -5,  3, -3 ]
              .sliced(4, 4)
              .as!float.slice;
    auto B = [ 6, -12, 1, 3 ].sliced(4).as!double.slice.canonical;
    auto B_ = B.slice.sliced(B.length, 1);
    auto C = qrDecomp(A);
    auto X = qrSolve(C.matrix, C.tau, B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(mtimes(A, X), B_));
}

unittest
{
    auto A =
            [ 1,  1,  0,
              1,  0,  1,
              0,  1,  1 ]
              .sliced(3, 3)
              .as!double.slice;

    auto B =
            [ 7,  6,  98,
              4,  8,  17,
              5,  3,  24 ]
              .sliced(3, 3)
              .as!float.slice;
    auto C = qrDecomp(A);
    auto X = qrSolve(C.matrix, C.tau, B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    
    assert(equal!approxEqual(mtimes(A, X), B));
}

unittest
{
    import mir.complex;
    auto A =
            [ 1,  1,  0,
              1,  0,  1,
              0,  1,  1 ]
              .sliced(3, 3)
              .as!(Complex!double).slice;

    auto B =
            [ 15,  78,  11,
              21,  47,  71,
              81,  11,  81 ]
              .sliced(3, 3)
              .as!(Complex!float).slice;
    auto C = qrDecomp(A);
    auto X = qrSolve(C.matrix, C.tau, B);
    auto res = mtimes(A, X);

    import mir.algorithm.iteration: equal;
    assert(equal!((a, b) => fabs(a - b) < 1e-12)(res, B));
}

unittest
{
    auto A =
            [  3,  -6,
               4,  -8,
               0,   1]
              .sliced(3, 2)
              .as!double.slice;

    auto B = [-1, 7, 2]
               .sliced(3)
               .as!double.slice.canonical;

    auto X_check = [5, 2]
                    .sliced(2, 1)
                    .as!double.slice;

    auto C = qrDecomp(A);
    auto X = qrSolve(C.matrix, C.tau, B);

    import mir.math.common: approxEqual;
    import mir.algorithm.iteration: equal;
    alias appr = equal!((a, b) => approxEqual(a, b, 1e-5, 1e-5));
    assert(appr(X, X_check));
}
