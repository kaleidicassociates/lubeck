/++
$(H1 Lubeck - Linear Algebra)

Authors: Ilya Yaroshenko, Lars Tandle Kyllingstad (SciD author)
+/
module lubeck;

import mir.ndslice.slice;
import mir.ndslice.dynamic: transposed;
import mir.ndslice.topology;
import mir.ndslice.allocation;
import mir.utility;
import mir.math.common;
import std.traits;
import std.meta;
import std.typecons: Flag, Yes, No;

import mir.blas;
import mir.lapack;

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
    static if (is(T == creal))
        alias IterationType = cdouble;
    else
    {
        static assert(
            is(T == double) ||
            is(T == float) ||
            is(T == cdouble) ||
            is(T == cfloat));
        alias IterationType = T;
    }
}

private alias BlasType(Iterators...) =
    CommonType!(staticMap!(IterationType, Iterators));

/++
General matrix-matrix multiplication.
Params:
    a = m(rows) x k(cols) matrix
    b = k(rows) x n(cols) matrix
Result:
    m(rows) x n(cols)
+/
Slice!(Contiguous, [2], BlasType!(IteratorA, IteratorB)*)
    mtimes(SliceKind kindA, IteratorA, SliceKind kindB, IteratorB)(
        Slice!(kindA, [2], IteratorA) a,
        Slice!(kindB, [2], IteratorB) b)
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
            c[] = 0;
            ger(cast(C)1, a[0..$,0], b[0,0..$], c);
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
    // from https://github.com/kaleidicassociates/lubeck/issues/3
    Slice!(cast(SliceKind)2, [2LU], float*) a = slice!float(1, 1);
    Slice!(cast(SliceKind)0, [2LU], float*) b1 = slice!float(16, 1).transposed;
    Slice!(cast(SliceKind)2, [2LU], float*) b2 = slice!float(1, 16);

    a[] = 3;
    b1[] = 4;
    b2[] = 4;

    // Confirm that this message does not appear
    // Outputs: ** On entry to SGEMM  parameter number  8 had an illegal value
    assert(a.mtimes(b1) == a.mtimes(b2));
}

/++
General matrix-vector multiplication.
Params:
    a = m(rows) x k(cols) matrix
    b = k(rows) x 1(cols) vector
Result:
    m(rows) x 1(cols)
+/
Slice!(Contiguous, [1], BlasType!(IteratorA, IteratorB)*)
    mtimes(SliceKind kindA, IteratorA, SliceKind kindB, IteratorB)(
        Slice!(kindA, [2], IteratorA) a,
        Slice!(kindB, [1], IteratorB) b)
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
General vector-matrix multiplication.
Params:
    a = 1(rows) x k(cols) vector
    b = k(rows) x n(cols) matrix
Result:
    1(rows) x n(cols)
+/
Slice!(Contiguous, [1], BlasType!(IteratorA, IteratorB)*)
    mtimes(SliceKind kindA, IteratorA, SliceKind kindB, IteratorB)(
        Slice!(kindB, [1], IteratorB) a,
        Slice!(kindA, [2], IteratorA) b,
        )
{
    return .mtimes(b.universal.transposed, a);
}

///
unittest
{
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
    mtimes(SliceKind kindA, IteratorA, SliceKind kindB, IteratorB)(
        Slice!(kindB, [1], IteratorB) a,
        Slice!(kindA, [1], IteratorA) b,
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
        import mir.ndslice.algorithm: reduce;
        return c.reduce!"a + b * c"(a.as!(typeof(return)), b.as!(typeof(return)));
    }
}

///
unittest
{
    auto a = [1, 2, 4].sliced;
    auto b = [3, 4, 2].sliced;
    assert(a.mtimes(b) == 19);
}

/++
Calculates the inverse of a matrix.
+/
auto inv(SliceKind kind, Iterator)(Slice!(kind, [2], Iterator) a)
in
{
    assert (a.length!0 == a.length!1, "matrix must be square");
}
body
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

    import mir.ndslice.algorithm: all;
    import std.math: approxEqual;
    assert(all!((a, b) => a.approxEqual(b, 1e-10L, 1e-10L))(a.inv, ans));
    assert(all!((a, b) => a.approxEqual(b, 1e-10L, 1e-10L))(a.as!cdouble.inv.as!double, ans));
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
    Slice!(Contiguous, [2], T*) u;
    ///
    Slice!(Contiguous, [1], T*) sigma;
    ///
    Slice!(Contiguous, [2], T*) vt;
}

/++
Computes the singular value decomposition.

Params:
    matrix = input `M x N` matrix
    slim = If true the first `min(M,N)` columns of `u` and the first
        `min(M,N)` rows of `vt` are returned in the ndslices `u` and `vt`.
Returns: $(LREF SvdResult)
+/
auto svd(
        Flag!"allowDestroy" allowDestroy = No.allowDestroy,
        string algorithm = "gesvd",
        SliceKind kind,
        Iterator
    )(
        Slice!(kind, [2], Iterator) matrix,
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

    auto s = uninitSlice!T(min(m, n));
    auto u = uninitSlice!T(slim ? s.length : m, m);
    auto vt = uninitSlice!T(n, slim ? s.length : n);

    static if (algorithm == "gesvd")
    {
        auto jobu = slim ? 'S' : 'A';
        auto jobvt = slim ? 'S' : 'A';
        auto work = gesvd_wq(jobu, jobvt, a, u.canonical, vt.canonical).uninitSlice!T;
        auto info = gesvd(jobu, jobvt, a, s, u.canonical, vt.canonical, work);
        enum msg = "svd: DBDSQR did not converge";
    }
    else // gesdd
    {
        auto iwork = uninitSlice!lapackint(s.length * 8);
        auto jobz = slim ? 'S' : 'A';
        auto work = gesdd_wq(jobz, a, u.canonical, vt.canonical).uninitSlice!T;
        auto info = gesdd(jobz, a, s, u.canonical, vt.canonical, work, iwork);
        enum msg = "svd: DBDSDC did not converge, updating process failed";
    }

    import std.exception: enforce;
    enforce(info == 0, msg);
    return SvdResult!T(vt, s, u); //transposed
}

///
unittest
{
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

    import mir.ndslice.algorithm: all;
    import std.math: approxEqual;
    assert(all!((a, b) => a.approxEqual(b, 1e-8, 1e-8))(a, m));
}

///
unittest
{
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

/++
Solve systems of linear equations AX = B for X.
Computes minimum-norm solution to a linear least squares problem
if A is not a square matrix.
+/
Slice!(Contiguous, [2], BlasType!(IteratorA, IteratorB)*)
    mldivide
    (SliceKind kindA, IteratorA, SliceKind kindB, IteratorB)(
        Slice!(kindA, [2], IteratorA) a,
        Slice!(kindB, [2], IteratorB) b)
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
        size_t liwork = void;
        auto lwork = gelsd_wq(a_, b_, liwork);
        auto s = min(a_.length!0, a_.length!1).uninitSlice!C;
        auto work = lwork.uninitSlice!C;
        auto iwork = liwork.uninitSlice!lapackint;
        size_t rank;
        C rcond = -1;
        
        auto info = gelsd(a_, b_, s, rcond, rank, work, iwork);

        enforce(info == 0, to!string(info) ~ " off-diagonal elements of an intermediate bidiagonal form did not converge to zero.");
        b_ = b_[0 .. $, 0 .. a_.length!0];
    }
    return b_.universal.transposed.slice;
}

/// ditto
Slice!(Contiguous, [1], BlasType!(IteratorA, IteratorB)*)
    mldivide
    (SliceKind kindA, IteratorA, SliceKind kindB, IteratorB)(
        Slice!(kindA, [2], IteratorA) a,
        Slice!(kindB, [1], IteratorB) b)
{
    return a.mldivide(b.repeat(1).unpack.universal.transposed).front!1.assumeContiguous;
}

/// AX=B
unittest
{
    auto a = [
         1, -1,  1,
         2,  2, -4,
        -1,  5,  0].sliced(3, 3);
    auto b = [
         2.0,  0,
        -6  , -6,
         9  ,  1].sliced(3, 2);
    auto t = [
         1.0, -1,
         2  ,  0,
         3  ,  1].sliced(3, 2);

    auto x = mldivide(a, b);
    assert(x == t);
}

/// Ax=B
unittest
{
    auto a = [
         1, -1,  1,
         2,  2, -4,
        -1,  5,  0].sliced(3, 3);
    auto b = [
         2.0,
        -6  ,
         9  ].sliced(3);
    auto t = [
         1.0,
         2  ,
         3  ].sliced(3);

    auto x = mldivide(a, b);
    assert(x == t);
}

/// Least-Squares Solution of Underdetermined System
unittest
{
    auto a = [
        -0.57,  -1.28,  -0.39,   0.25,
        -1.93,   1.08,  -0.31,  -2.14,
         2.30,   0.24,   0.40,  -0.35,
        -1.93,   0.64,  -0.66,   0.08,
         0.15,   0.30,   0.15,  -2.13,
        -0.02,   1.03,  -1.43,   0.50,
        ].sliced(6, 4);

    auto b = [
     -2.67,
     -0.55,
      3.34,
     -0.77,
      0.48,
      4.10,
    ].sliced;

    auto x = [
        1.5339,
        1.8707,
       -1.5241,
        0.0392].sliced;

    import std.math: approxEqual;
    assert(a.mldivide(b).approxEqual(x));
}

/// Principal component analises result.
struct PcaResult(T)
{
    /// Principal component coefficients, also known as loadings.
    Slice!(Contiguous, [2], T*) coeff;
    /// Principal component scores.
    Slice!(Contiguous, [2], T*) score;
    /// Principal component variances.
    Slice!(Contiguous, [1], T*) latent;
}

/++
Principal component analysis of raw data.

Params:
    matrix = input `M x N` matrix, where 'M (rows)>= N(cols)'
    centerColumns = Flag to centern columns. True by default.
Returns: $(LREF PcaResult)
+/
auto pca(Flag!"allowDestroy" allowDestroy = No.allowDestroy, SliceKind kind, Iterator)(Slice!(kind, [2], Iterator) matrix, in Flag!"centerColumns" cc = Yes.centerColumns)
in
{
    assert(matrix.length!0 >= matrix.length!1);
}
body
{
    import mir.math.sum: sum;
    import mir.ndslice.algorithm: maxIndex, eachUploPair;
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
    import std.math: approxEqual;
    import mir.ndslice.algorithm: all;

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

    assert(all!approxEqual(res.coeff, coeff));
    assert(all!approxEqual(res.score, score));
    assert(all!approxEqual(res.latent, latent));
}

/++
Computes Moore-Penrose pseudoinverse of matrix.

Params:
    matrix = Input `M x N` matrix.
    tolerance = The computation is based on AVD and any singular values less than tolerance are treated as zero.
Returns: Moore-Penrose pseudoinverse matrix
+/
Slice!(Contiguous, [2], BlasType!Iterator*)
    pinv(Flag!"allowDestroy" allowDestroy = No.allowDestroy, SliceKind kind, Iterator)(Slice!(kind, [2], Iterator) matrix, BlasType!Iterator tolerance = BlasType!Iterator.nan)
{
    import mir.ndslice.algorithm: find, each;
    import std.math: nextUp;

    auto svd = matrix.svd!allowDestroy(Yes.slim);
    if (tolerance != tolerance)
    {
        auto n = svd.sigma.front;
        auto eps = n.nextUp - n;
        tolerance = max(matrix.length!0, matrix.length!1) * eps;
    }
    auto s = svd.sigma[0 .. $ - svd.sigma.find!(a => !(a >= tolerance))[0]];
    s.each!"a = 1 / a";
    svd.vt[0 .. s.length].pack!1.map!"a".zip(s).each!"a.a[] *= a.b";
    auto v = svd.vt[0 .. s.length].universal.transposed;
    auto ut = svd.u.universal.transposed[0 .. s.length];
    return v.mtimes(ut);
}

///
unittest
{
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
        0.0142, -0.0140, -0.0149,  0.0169,  0.0178, -0.0176, -0.0185,  0.0205];

    import std.math: approxEqual;

    assert(b.field.approxEqual(result, 1e-2, 1e-2));
}

/++
Covariance matrix.

Params:
    matrix = matrix whose rows represent observations and whose columns represent random variables.
Reuturns:
    Normalized by `N-1` covariance matrix.
+/
Slice!(Contiguous, [2], BlasType!Iterator*)
    cov(SliceKind kind, Iterator)(Slice!(kind, [2], Iterator) matrix)
{
    import mir.math.sum: sum;
    import mir.ndslice.algorithm: each, eachUploPair;
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
    import std.stdio;
    import mir.ndslice;

    auto c = 8.magic[0..$-1].cov;

    import std.math: approxEqual;
    assert(c.field.approxEqual([
         350.0000, -340.6667, -331.3333,  322.0000,  312.6667, -303.3333, -294.0000,  284.6667,
        -340.6667,  332.4762,  324.2857, -316.0952, -307.9048,  299.7143,  291.5238, -283.3333,
        -331.3333,  324.2857,  317.2381, -310.1905, -303.1429,  296.0952,  289.0476, -282.0000,
         322.0000, -316.0952, -310.1905,  304.2857,  298.3810, -292.4762, -286.5714,  280.6667,
         312.6667, -307.9048, -303.1429,  298.3810,  293.6190, -288.8571, -284.0952,  279.3333,
        -303.3333,  299.7143,  296.0952, -292.4762, -288.8571,  285.2381,  281.6190, -278.0000,
        -294.0000,  291.5238,  289.0476, -286.5714, -284.0952,  281.6190,  279.1429, -276.6667,
         284.6667, -283.3333, -282.0000,  280.6667,  279.3333, -278.0000, -276.6667,  275.3333]));
}

/++
Matrix determinant.
+/
auto detSymmetric(SliceKind kind, Iterator)(Slice!(kind, [2], Iterator) a, char store = 'L')
in
{
    assert(store == 'U' || store == 'L');
    assert (a.length!0 == a.length!1, "matrix must be square");
    assert (a.length!0, "matrix must not be empty");
}
body
{
    import mir.ndslice.algorithm: each;
    import mir.ndslice.topology: diagonal;
    import mir.math.numeric: Prod;

    alias T = BlasType!Iterator;

    auto packed = uninitSlice!T(a.length * (a.length + 1) / 2);
    auto ipiv = a.length.uninitSlice!lapackint;
    int sign;
    Prod!T prod;
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
    return prod.value;
}

/// ditto
auto det(SliceKind kind, Iterator)(Slice!(kind, [2], Iterator) a)
in
{
    assert (a.length!0 == a.length!1, "matrix must be square");
}
body
{
    import mir.ndslice.topology: diagonal, zip, iota;
    import mir.math.numeric: Prod;

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
    Prod!T prod;
    foreach (tup; m.diagonal.zip(ipiv, [ipiv.length].iota(1)))
    {
        prod.put(tup.a);
        sign ^= tup.b != tup.c; // i.e. row interchanged with another
    }
    if(sign & 1)
        prod.x = -prod.x;
    return prod.value;
}

///
unittest
{
    // Check for zero-determinant shortcut.
    auto ssing = [4, 2, 2, 1].sliced(2, 2);
    auto ssingd = det(ssing);
    assert (det(ssing) == 0);
    assert (detSymmetric(ssing) == 0);
    assert (detSymmetric(ssing, 'L') == 0);

    //import std.stdio;


    // General dense matrix.
    int dn = 101;
    auto d = uninitSlice!double(dn, dn);
    foreach (k; 0 .. dn)
    foreach (l; 0 .. dn)
        d[k,l] = 0.5 * (k == l ? (k + 1) * (k + 1) + 1 : 2 * (k + 1) * (l + 1));

    auto dd = det(d);
    import std.math: ldexp, approxEqual;
    assert (approxEqual(dd, ldexp(8.972817920259982e319L, -dn), double.epsilon.sqrt));

    // Symmetric packed matrix
    auto spa = [ 1.0, -2, 3, 4, 5, -6, -7, -8, -9, 10].sliced.stairs!"+"(4);
    auto sp = [spa.length, spa.length].uninitSlice!double;
    import mir.ndslice.algorithm: each;
    sp.stairs!"+".each!"a[] = b"(spa);
    assert (sp.detSymmetric('L').approxEqual(5874.0, double.epsilon.sqrt));
    assert (sp.universal.transposed.detSymmetric('U').approxEqual(5874.0, double.epsilon.sqrt));
}

/++
Eigenvalues and eigenvectors POD.

See_also: $(LREF eigSymmetric).
+/
struct EigSymmetricResult(T)
{
    /// Eigenvalues
    Slice!(Contiguous, [1], T*) values;
    /// Eigenvectors stored in rows
    Slice!(Contiguous, [2], T*) vectors;
}

/++
Eigenvalues and eigenvectors of symmetric matrix.

Returns:
    $(LREF EigSymmetricResult)
+/
auto eigSymmetric(Flag!"computeVectors" cv = Yes.computeVectors, SliceKind kind, Iterator)(Slice!(kind, [2], Iterator) a, char store = 'L')
in
{
    assert(store == 'U' || store == 'L');
    assert (a.length!0 == a.length!1, "matrix must be square");
    assert (a.length!0, "matrix must not be empty");
}
body
{
    import mir.ndslice.algorithm: each;
    import mir.ndslice.topology: diagonal;
    import mir.math.numeric: Prod;

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
        auto z = _vData.sliced(1, 1);
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
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal, map;
    import mir.ndslice.dynamic: transposed;
    import std.math: approxEqual;

    auto a = [
        1.0000, 0.5000, 0.3333, 0.2500,
        0.5000, 1.0000, 0.6667, 0.5000,
        0.3333, 0.6667, 1.0000, 0.7500,
        0.2500, 0.5000, 0.7500, 1.0000].sliced(4, 4);

    auto eigr = a.eigSymmetric;

    assert(eigr.values.approxEqual([0.2078,0.4078,0.8482,2.5362]));

    auto test = [
         0.0693, -0.4422, -0.8105, 0.3778,
        -0.3618,  0.7420, -0.1877, 0.5322,
         0.7694,  0.0486,  0.3010, 0.5614,
        -0.5219, -0.5014,  0.4662, 0.5088]
            .sliced(4, 4)
            .universal
            .transposed;

    foreach (i; 0 .. 4)
        assert(eigr.vectors[i].approxEqual(test[i]) ||
            eigr.vectors[i].map!"-a".approxEqual(test[i]));
}
