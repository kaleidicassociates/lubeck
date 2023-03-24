/++
$(H1 Lubeck 2 - `@nogc` Linear Algebra)
+/
module kaleidic.lubeck2;

import mir.algorithm.iteration: equal, each;
import mir.blas;
import mir.exception;
import mir.internal.utility: isComplex, realType;
import mir.lapack;
import mir.math.common: sqrt, approxEqual, pow, fabs;
import mir.ndslice;
import mir.rc.array;
import mir.utility: min, max;
import std.traits: isFloatingPoint, Unqual;
import std.typecons: Flag, Yes, No;
import mir.complex: Complex;

/++
Identity matrix.

Params:
n = number of columns
m = optional number of rows, default n
Results:
Matrix which is 1 on the diagonal and 0 elsewhere
+/
@safe pure nothrow @nogc
Slice!(RCI!T, 2) eye(T = double)(
    size_t n,
    size_t m = 0
)
    if (isFloatingPoint!T || isComplex!T)
in
{
    assert(n>0);
    assert(m>=0);
}
out (i)
{
    assert(i.length!0==n);
    assert(i.length!1== (m==0 ? n : m));
}
do
{
    auto c = rcslice!T([n, (m==0?n:m)], cast(T)0);
    c.diagonal[] = cast(T)1;
    return c;
}

/// Real numbers
@safe pure nothrow
unittest
{
    import mir.ndslice;
    import mir.math;

    assert(eye(1)== [
        [1]]);
    assert(eye(2)== [
        [1, 0],
        [0, 1]]);
    assert(eye(3)== [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]);
    assert(eye(1,2) == [
        [1,0]]);
    assert(eye(2,1) == [
        [1],
        [0]]);
}

///
enum AssumedMatrix
{
    general,
    selfAdjoint,
    selfAdjointRight
}

/++
Matrix multiplication. Allocates result to using Mir refcounted arrays.

This function has multiple overloads that include the following functionality:
- a.mtimes(b) where `a` and `b` are both two-dimensional slices. The result is a
two-dimensional slice.
- a.mtimes(b) where `a` is a two-dimensional slice and `b` is a one-dimensional
slice. The result is a one-dimensional slice. In this case, `b` can be thought
of as a column vector.
- b.mtimes(a) where `a` is a two-dimensional slice and `b` is a one-dimensional
slice. The result is a one-dimensional slice. In this case, `b` can be thought
of as a row vector.

The template parameter `assumedMatrix` controls the BLAS algorithms that are
used to perform the matrix multiplication.
- AssumedMatrix.general (default): Assumes matrix `a` is a general matrix and
uses `gemm` for matrix-matrix operations and `gemv` for matrix-vector operations.
- AssumedMatrix.selfAdjoint: Assumes matrix `a` is a self-adjoint matrix
(symmetric in the real case, hermetian in the complex case, but not currently
implemented) and uses `symm`/`hemm` for matrix-matrix operations and
`symv`/`hemv` for matrix-vector operations.
- AssumedMatrix.selfAdjointRight: Assumes matrix `b` is a self-adjoint matrix
(symmetric in the real case, hermetian in the complex case, but not currently
implemented). For vector case, follows same approach as `AssumeMatrix.selfAdjoint`.

+/
@safe pure nothrow @nogc
template mtimes(AssumedMatrix assumedMatrix = AssumedMatrix.general)
{
    /++
    Params:
        a = m(rows) x k(cols) matrix
        b = k(rows) x n(cols) matrix
    Result:
        m(rows) x n(cols)
    +/
    Slice!(RCI!T, 2) mtimes(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 2, kindB) b
    )
        if (isFloatingPoint!T || isComplex!T)
    in
    {
        assert(a.length!1 == b.length!0);
        static if (assumedMatrix == AssumedMatrix.selfAdjoint) {
            assert(a.length!0 == a.length!1);
            static if (isFloatingPoint!T)
                assert(isSymmetric(a));
        } else static if (assumedMatrix == AssumedMatrix.selfAdjointRight) {
            assert(b.length!0 == b.length!1);
            static if (isFloatingPoint!T)
                assert(isSymmetric(b));
        }
    }
    out (c)
    {
        assert(c.length!0 == a.length!0);
        assert(c.length!1 == b.length!1);
    }
    do
    {
        // more optimisations for special cases can be added in the future
        auto c = mininitRcslice!T(a.length!0, b.length!1);
        static if (assumedMatrix == AssumedMatrix.general) {
            gemm(cast(T)1, a, b, cast(T)0, c.lightScope);
        } else static if (assumedMatrix == AssumedMatrix.selfAdjoint ||
                          assumedMatrix == AssumedMatrix.selfAdjointRight) {
            import cblas: Uplo, Side;
            static if (isFloatingPoint!T) {
                static if (assumedMatrix == AssumedMatrix.selfAdjoint) {
                    symm(Side.Left, Uplo.Upper, cast(T)1, a, b, cast(T)0, c.lightScope);
                } else {
                    static if (kindA == Universal) {
                        assert(a._stride!1 == 1 && c.universal._stride!1 == 1,
                               "stride of `a` and output do not match, most likely the result of a transpose operation, try allocating `a` before passing it to the function, such as `a.rcslice.mtimes(b)`");
                    }
                    symm(Side.Right, Uplo.Upper, cast(T)1, b, a, cast(T)0, c.lightScope);
                }
            } else {
                static assert(0, "Complex version not implemented");
            }
        } else {
            static assert(0, "Not implemented");
        }
        return c;
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0);
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimes!assumedMatrix(scopeA, scopeB);
    }

    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0);
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimes!assumedMatrix(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 2) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 2, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length!0);
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimes!assumedMatrix(a, scopeB);
    }

    /++
    Params:
        a = m(rows) x n(cols) matrix
        b = n(rows) x 1(cols) vector
    Result:
        m(rows) x 1(cols)
    +/
    @safe pure nothrow @nogc
    Slice!(RCI!T, 1) mtimes(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 1, kindB) b
    )
        if (isFloatingPoint!T || isComplex!T)
    in
    {
        assert(a.length!1 == b.length);
        static if (assumedMatrix == AssumedMatrix.selfAdjoint ||
                   assumedMatrix == AssumedMatrix.selfAdjointRight) {
            assert(a.length!0 == a.length!1);
            static if (isFloatingPoint!T)
                assert(isSymmetric(a));
        }
    }
    out (c)
    {
        assert(c.length == a.length);
    }
    do
    {
        auto c = mininitRcslice!T(a.length!0);
        static if (assumedMatrix == AssumedMatrix.general) {
            gemv(cast(T)1, a, b, cast(T)0, c.lightScope);
        } else static if (assumedMatrix == AssumedMatrix.selfAdjoint ||
                          assumedMatrix == AssumedMatrix.selfAdjointRight) {
            import cblas: Uplo;
            static if (isFloatingPoint!T)
                symv(Uplo.Upper, cast(T)1, a, b, cast(T)0, c.lightScope);
            else
                static assert(0, "Complex version not implemented");
        } else {
            static assert(0, "Not implemented");
        }
        return c;
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length);
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimes!assumedMatrix(scopeA, scopeB);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!A, 2, kindA) a,
        Slice!(const(B)*, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length);
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimes!assumedMatrix(scopeA, b);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(A)*, 2, kindA) a,
        auto ref const Slice!(RCI!B, 1, kindB) b
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!1 == b.length);
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimes!assumedMatrix(a, scopeB);
    }

    /++
    Params:
        b = 1(rows) x n(cols) vector
        a = n(rows) x m(cols) matrix
    Result:
        1(rows) x m(cols)
    +/
    @safe pure nothrow @nogc
    Slice!(RCI!T, 1) mtimes(T, SliceKind kindA, SliceKind kindB)(
        Slice!(const(T)*, 1, kindB) b,
        Slice!(const(T)*, 2, kindA) a
    )
        if (isFloatingPoint!T || isComplex!T)
    in
    {
        assert(a.length!0 == b.length);
        static if (assumedMatrix == AssumedMatrix.selfAdjoint ||
                   assumedMatrix == AssumedMatrix.selfAdjointRight) {
            assert(a.length!0 == a.length!1);
            static if (isFloatingPoint!T)
                assert(isSymmetric(a));
        }
    }
    out (c)
    {
        assert(c.length == a.length!1);
    }
    do
    {
        static if (assumedMatrix == AssumedMatrix.general) {
            auto c = mininitRcslice!T(a.length!1);
            gemv(cast(T)1, a.transposed, b, cast(T)0, c.lightScope);
            return c;
        } else static if (assumedMatrix == AssumedMatrix.selfAdjoint ||
                          assumedMatrix == AssumedMatrix.selfAdjointRight) {
            return .mtimes!assumedMatrix(a, b);
        } else {
            static assert(0, "Not implemented");
        }
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!B, 1, kindB) b,
        auto ref const Slice!(RCI!A, 2, kindA) a
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!0 == b.length);
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        auto scopeB = b.lightScope.lightConst;
        return .mtimes!assumedMatrix(scopeB, scopeA);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        Slice!(const(B)*, 1, kindB) b,
        auto ref const Slice!(RCI!A, 2, kindA) a
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!0 == b.length);
    }
    do
    {
        auto scopeA = a.lightScope.lightConst;
        return .mtimes!assumedMatrix(b, scopeA);
    }

    /// ditto
    @safe pure nothrow @nogc
    Slice!(RCI!(Unqual!A), 1) mtimes(A, B, SliceKind kindA, SliceKind kindB)(
        auto ref const Slice!(RCI!B, 1, kindB) b,
        Slice!(const(A)*, 2, kindA) a
    )
        if (is(Unqual!A == Unqual!B))
    in
    {
        assert(a.length!0 == b.length);
    }
    do
    {
        auto scopeB = b.lightScope.lightConst;
        return .mtimes!assumedMatrix(scopeB, a);
    }
}

/++
Params:
    a = 1(rows) x n(cols) vector
    b = n(rows) x 1(cols) vector
Result:
    dot product
+/
@safe pure nothrow @nogc
Unqual!A mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 1, kindA) a,
    Slice!(const(B)*, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length == b.length);
}
do
{
    import mir.blas: dot;

    return dot(a, b);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 1, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length == b.length);
}
do
{
    auto scopeA = a.lightScope.lightConst;
    auto scopeB = b.lightScope.lightConst;
    return mtimes(scopeA, scopeB);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 1, kindA) a,
    Slice!(const(B)*, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length == b.length);
}
do
{
    auto scopeA = a.lightScope.lightConst;
    return mtimes(scopeA, b);
}

/// ditto
@safe pure nothrow @nogc
Unqual!A mtimes(A, B, SliceKind kindA, SliceKind kindB)(
    Slice!(const(A)*, 1, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
    if (is(Unqual!A == Unqual!B))
in
{
    assert(a.length == b.length);
}
do
{
    auto scopeB = b.lightScope.lightConst;
    return mtimes(a, scopeB);
}

/// ditto
template mtimes(string assumedMatrix)
{
    mixin("alias mtimes = .mtimes!(AssumedMatrix." ~ assumedMatrix ~ ");");
}

/// Matrix-matrix multiplication (real)
@safe pure nothrow
unittest
{
    import mir.ndslice;
    import mir.math;

    auto a = mininitRcslice!double(3, 5);
    auto b = mininitRcslice!double(5, 4);

    a[] =
    [[-5,  1,  7, 7, -4],
     [-1, -5,  6, 3, -3],
     [-5, -2, -3, 6,  0]];

    b[] =
    [[-5, -3,  3,  1],
     [ 4,  3,  6,  4],
     [-4, -2, -2,  2],
     [-1,  9,  4,  8],
     [ 9,  8,  3, -2]];

    assert(mtimes(a, b) ==
        [[-42,  35,  -7, 77],
         [-69, -21, -42, 21],
         [ 23,  69,   3, 29]]);

    assert(mtimes(b.transposed, a.transposed) ==
        [[-42, -69, 23],
         [ 35, -21, 69],
         [ -7, -42,  3],
         [ 77,  21, 29]]);
}

// test mixed strides
@safe pure nothrow
unittest
{
    import mir.ndslice;
    import mir.math;

    auto a = mininitRcslice!double(3, 5);
    auto b = mininitRcslice!double(5, 4);

    a[] =
    [[-5,  1,  7, 7, -4],
     [-1, -5,  6, 3, -3],
     [-5, -2, -3, 6,  0]];

    b[] =
    [[-5, -3,  3,  1],
     [ 4,  3,  6,  4],
     [-4, -2, -2,  2],
     [-1,  9,  4,  8],
     [ 9,  8,  3, -2]];
         
    auto at = mininitRcslice!double(5, 3);
    at[] =
    [[-5, -1, -5],
     [ 1, -5, -2],
     [ 7,  6, -3],
     [ 7,  3,  6],
     [-4, -3,  0]];
     assert(mtimes(b.transposed, at) ==
        [[-42, -69, 23],
         [ 35, -21, 69],
         [ -7, -42,  3],
         [ 77,  21, 29]]);

    auto bt = mininitRcslice!double(4, 5);
    bt[] =
    [[-5, 4, -4, -1,  9],
     [-3, 3, -2,  9,  8],
     [ 3, 6, -2,  4,  3],
     [ 1, 4,  2,  8, -2]];
     assert(mtimes(bt, a.transposed) ==
        [[-42, -69, 23],
         [ 35, -21, 69],
         [ -7, -42,  3],
         [ 77,  21, 29]]);
}

/// Matrix-matrix multiplication (complex)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.ndslice;
    import mir.math;

    auto a = mininitRcslice!(Complex!double)(3, 5);
    auto b = mininitRcslice!(Complex!double)(5, 4);

    a[] =
    [[-5,  1,  7, 7, -4],
     [-1, -5,  6, 3, -3],
     [-5, -2, -3, 6,  0]];

    b[] =
    [[-5, -3,  3,  1],
     [ 4,  3,  6,  4],
     [-4, -2, -2,  2],
     [-1,  9,  4,  8],
     [ 9, 8,  3, -2]];

    assert(mtimes(a, b) ==
        [[-42,  35,  -7, 77],
         [-69, -21, -42, 21],
         [ 23,  69,   3, 29]]);
}

/// Matrix-matrix multiplication, specialization for MxN times Nx1
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[3.0, 5, 2, -3], [-2.0, 2, 3, 10], [0.0, 2, 1, 1]];
    static immutable b = [2.0, 3, 4, 5];
    static immutable c = [14.0, 64, 15];

    auto X = mininitRcslice!double(3, 4);
    auto y = mininitRcslice!double(4);
    auto result = mininitRcslice!double(3);

    X[] = a;
    y[] = b;
    result[] = c;

    auto Xy = X.mtimes(y);
    assert(Xy.equal(result));
    auto yXT = y.mtimes(X.transposed);
    assert(yXT.equal(result));
}

/// Reference-counted dot product
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [-5.0,  1,  7,  7, -4];
    static immutable b = [ 4.0, -4, -2, 10,  4];

    auto x = mininitRcslice!double(5);
    auto y = mininitRcslice!double(5);

    x[] = a;
    y[] = b;

    assert(x.mtimes(y) == 16);
}

/// Mix slice & RC dot product
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.slice: sliced;

    static immutable a = [-5.0,  1,  7,  7, -4];
    static immutable b = [ 4.0, -4, -2, 10,  4];

    auto x = mininitRcslice!double(5);
    auto y = b.sliced;

    x[] = a;

    assert(x.mtimes(y) == 16);
    assert(y.mtimes(x) == 16);
}

/// Symmetric Matrix-Matrix multiplication (real)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.ndslice.topology: universal;

    static immutable a = [[3.0, 5, 2], [5.0, 2, 3], [2.0, 3, 1]];
    static immutable b = [[2.0, 3], [4.0, 3], [0.0, -5]];
    static immutable bt = [[2.0, 4, 0], [3.0, 3, -5]];
    static immutable c = [[26.0, 14], [18.0, 6], [16.0, 10]];

    auto X = mininitRcslice!double(3, 3);
    auto Y = mininitRcslice!double(3, 2);
    auto YT = mininitRcslice!double(2, 3);
    auto result = mininitRcslice!double(3, 2);

    X[] = a;
    Y[] = b;
    YT[] = bt;
    result[] = c;

    auto XY = X.mtimes!"selfAdjoint"(Y);
    assert(XY.equal(result));
    auto YTX = YT.mtimes!"selfAdjointRight"(X);
    assert(YTX.equal(result.transposed));
    // may need to allocate transposed LHS
    auto YtransX = Y.transposed.rcslice.mtimes!"selfAdjointRight"(X);
    assert(YtransX.equal(result.transposed));
}

/// Symmetric Matrix, specialization for MxN times Nx1
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: mininitRcslice;

    static immutable a = [[3.0, 5, 2], [5.0, 2, 3], [2.0, 3, 1]];
    static immutable b = [2.0, 3, 4];
    static immutable c = [29, 28, 17];

    auto X = mininitRcslice!double(3, 3);
    auto y = mininitRcslice!double(3);
    auto result = mininitRcslice!double(3);

    X[] = a;
    y[] = b;
    result[] = c;

    auto Xy = X.mtimes!"selfAdjoint"(y);
    assert(Xy.equal(result));
    auto yXT = y.mtimes!"selfAdjoint"(X);
    assert(yXT.equal(result));
}

/++
Solve systems of linear equations AX = B for X.
Computes minimum-norm solution to a linear least squares problem
if A is not a square matrix.
+/
@safe pure @nogc
Slice!(RCI!T, 2) mldivide (T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b,
)
    if (isFloatingPoint!T || isComplex!T)
{
    enforce!"mldivide: parameter shapes mismatch"(a.length!0 == b.length!0);

    auto rcat = a.transposed.as!T.rcslice;
    auto at = rcat.lightScope.canonical;
    auto rcbt = b.transposed.as!T.rcslice;
    auto bt = rcbt.lightScope.canonical;
    size_t info;
    if (a.length!0 == a.length!1)
    {
        auto rcipiv = at.length.mininitRcslice!lapackint;
        auto ipiv = rcipiv.lightScope;
        foreach(i; 0 .. ipiv.length)
            ipiv[i] = 0;
        info = gesv!T(at, ipiv, bt);
        //info > 0 means some diagonla elem of U is 0 so no solution
    }
    else
    {
        static if(!isComplex!T)
        {
            size_t liwork = void;
            auto lwork = gelsd_wq(at, bt, liwork);
            auto rcs = min(at.length!0, at.length!1).mininitRcslice!T;
            auto s = rcs.lightScope;
            auto rcwork = lwork.rcslice!T;
            auto work = rcwork.lightScope;
            auto rciwork = liwork.rcslice!lapackint;
            auto iwork = rciwork.lightScope;
            size_t rank = void;
            T rcond = -1;

            info = gelsd!T(at, bt, s, rcond, rank, work, iwork);
            //info > 0 means that many components failed to converge
        }
        else
        {
            size_t liwork = void;
            size_t lrwork = void;
            auto lwork = gelsd_wq(at, bt, lrwork, liwork);
            auto rcs = min(at.length!0, at.length!1).mininitRcslice!(realType!T);
            auto s = rcs.lightScope;
            auto rcwork = lwork.rcslice!T;
            auto work = rcwork.lightScope;
            auto rciwork = liwork.rcslice!lapackint;
            auto iwork = rciwork.lightScope;
            auto rcrwork = lrwork.rcslice!(realType!T);
            auto rwork = rcrwork.lightScope;
            size_t rank = void;
            realType!T rcond = -1;

            info = gelsd!T(at, bt, s, rcond, rank, work, rwork, iwork);
            //info > 0 means that many components failed to converge
        }
        bt = bt[0 .. $, 0 .. at.length!0];
    }
    enforce!"mldivide: some off-diagonal elements of an intermediate bidiagonal form did not converge to zero."(!info);
    return bt.transposed.as!T.rcslice;
}

/// ditto
@safe pure @nogc
Slice!(RCI!(Unqual!A), 2)
mldivide
(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 2, kindB) b
)
do
{
    auto al = a.lightScope.lightConst;
    auto bl = b.lightScope.lightConst;
    return mldivide(al, bl);
}

/// ditto
@safe pure @nogc
Slice!(RCI!(Unqual!A), 1)
mldivide
(A, B, SliceKind kindA, SliceKind kindB)(
    auto ref const Slice!(RCI!A, 2, kindA) a,
    auto ref const Slice!(RCI!B, 1, kindB) b
)
do
{
    auto al = a.lightScope.lightConst;
    auto bl = b.lightScope.lightConst;
    return mldivide(al, bl);
}

/// ditto
@safe pure @nogc
Slice!(RCI!T, 1) mldivide (T, SliceKind kindA, SliceKind kindB)(
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 1, kindB) b,
)
    if (isFloatingPoint!T || isComplex!T)
{
    import mir.ndslice.topology: flattened;
    return mldivide(a, b.sliced(b.length, 1)).flattened;
}

pure unittest
{
    import mir.complex;
    auto a = mininitRcslice!double(2, 2);
    a[] = [[2,3],
           [1, 4]];
    auto res = mldivide(eye(2), a);
    assert(equal!approxEqual(res, a));
    auto b = mininitRcslice!(Complex!double)(2, 2);
    b[] = [[Complex!double(2, 1),Complex!double(3, 2)],
           [Complex!double(1, 3), Complex!double(4, 4)]];
    auto cres = mldivide(eye!(Complex!double)(2), b);
    assert(cres == b);
    auto c = mininitRcslice!double(2, 2);
    c[] = [[5,3],
           [2,6]];
    auto d = mininitRcslice!double(2,1);
    d[] = [[4],
           [1]];
    auto e = mininitRcslice!double(2,1);
    e[] = [[23],
           [14]];
    res = mldivide(c, e);
    assert(equal!approxEqual(res, d));
}

pure unittest
{
    import mir.complex;
    import mir.ndslice;
    import mir.math;

    auto a =  mininitRcslice!double(6, 4);
    a[] = [
        -0.57,  -1.28,  -0.39,   0.25,
        -1.93,   1.08,  -0.31,  -2.14,
        2.30,   0.24,   0.40,  -0.35,
        -1.93,   0.64,  -0.66,   0.08,
        0.15,   0.30,   0.15,  -2.13,
        -0.02,   1.03,  -1.43,   0.50,
    ].sliced(6, 4);

    auto b = mininitRcslice!double(6,1);
    b[] = [
        -2.67,
        -0.55,
        3.34,
        -0.77,
        0.48,
        4.10,
    ].sliced(6,1);

    auto x = mininitRcslice!double(4,1);
    x[] = [
        1.5339,
        1.8707,
        -1.5241,
        0.0392
    ].sliced(4,1);

    auto res = mldivide(a, b);
    assert(equal!((a, b) => fabs(a - b) < 5e-5)(res, x));

    auto ca =  mininitRcslice!(Complex!double)(6, 4);
    ca[] = [
        -0.57,  -1.28,  -0.39,   0.25,
        -1.93,   1.08,  -0.31,  -2.14,
        2.30,   0.24,   0.40,  -0.35,
        -1.93,   0.64,  -0.66,   0.08,
        0.15,   0.30,   0.15,  -2.13,
        -0.02,   1.03,  -1.43,   0.50,
    ].sliced(6, 4);

    auto cb = mininitRcslice!(Complex!double)(6,1);
    cb[] = [
        -2.67,
        -0.55,
        3.34,
        -0.77,
        0.48,
        4.10,
    ].sliced(6,1);

    auto cx = mininitRcslice!(Complex!double)(4,1);
    cx[] = [
        1.5339,
        1.8707,
        -1.5241,
        0.0392
    ].sliced(4,1);

    auto cres = mldivide(ca, cb);
    assert(equal!((a, b) => fabs(a - b) < 5e-5)(cres, x));
}

/++
Solve systems of linear equations AX = I for X, where I is the identity.
X is the right inverse of A if it exists, it's also a (Moore-Penrose) Pseudoinverse if A is invertible then X is the inverse.
Computes minimum-norm solution to a linear least squares problem
if A is not a square matrix.
+/
@safe pure @nogc Slice!(RCI!A, 2) mlinverse(A, SliceKind kindA)(
    auto ref Slice!(RCI!A, 2, kindA) a
)
{
    auto aScope = a.lightScope.lightConst;
    return mlinverse!A(aScope);
}

@safe pure @nogc Slice!(RCI!A, 2) mlinverse(A, SliceKind kindA)(
    Slice!(const(A)*, 2, kindA) a
)
{
    auto a_i = a.as!A.rcslice;
    auto a_i_light = a_i.lightScope.canonical;
    auto rcipiv = min(a_i.length!0, a_i.length!1).mininitRcslice!lapackint;
    auto ipiv = rcipiv.lightScope;
    auto info = getrf!A(a_i_light, ipiv);
    if (info == 0)
    {
        auto rcwork = getri_wq!A(a_i_light).mininitRcslice!A;
        auto work = rcwork.lightScope;
        info = getri!A(a_i_light, ipiv, work);
    }
    enforce!"Matrix is not invertible as has zero determinant"(!info);
    return a_i;
}

pure unittest
{
    import mir.ndslice;
    import mir.math;

    auto a = mininitRcslice!double(2, 2);
    a[] = [[1,0],
           [0,-1]];
    auto ans = mlinverse!double(a);
    assert(equal!approxEqual(ans, a));
}

pure 
unittest
{
    import mir.ndslice;
    import mir.math;

    auto a = mininitRcslice!double(2, 2);
    a[] = [[ 0, 1],
           [-1, 0]];
    auto aInv = mininitRcslice!double(2, 2);
    aInv[] = [[0, -1],
              [1,  0]];
    auto ans = a.mlinverse;
    assert(equal!approxEqual(ans, aInv));
}

///Singuar value decomposition result
struct SvdResult(T)
{
    ///
    Slice!(RCI!T, 2) u;
    ///Singular Values
    Slice!(RCI!(realType!T)) sigma;
    ///
    Slice!(RCI!T, 2) vt;
}

/++
Computes the singular value decomposition.
Params:
    a = input `M x N` matrix
    slim = If true the first `min(M,N)` columns of `u` and the first
        `min(M,N)` rows of `vt` are returned in the ndslices `u` and `vt`.
out result = $(LREF SvdResult). ]
Returns: error code from CBlas
+/
@safe pure @nogc SvdResult!T svd
(
    T,
    string algorithm = "gesvd",
    SliceKind kind,
)(
    Slice!(const(T)*, 2, kind) a,
    Flag!"slim" slim = No.slim,
)
    if (algorithm == "gesvd" || algorithm == "gesdd")
{
    auto m = cast(lapackint)a.length!1;
    auto n = cast(lapackint)a.length!0;

    auto s = mininitRcslice!(realType!T)(min(m, n));
    auto u = mininitRcslice!T(slim ? s.length : m, m);
    auto vt = mininitRcslice!T(n, slim ? s.length : n);

    if (m == 0 || n == 0)
    {
        u.lightScope[] = 0;
        u.lightScope.diagonal[] = 1;
        vt.lightScope[] = 0;
        vt.lightScope.diagonal[] = 1;
    }
    else
    {
        static if (algorithm == "gesvd")
        {
            auto jobu = slim ? 'S' : 'A';
            auto jobvt = slim ? 'S' : 'A';
            auto rca_sliced = a.as!T.rcslice;
            auto rca = rca_sliced.lightScope.canonical;
            auto rcwork = gesvd_wq(jobu, jobvt, rca, u.lightScope.canonical, vt.lightScope.canonical).mininitRcslice!T;
            auto work = rcwork.lightScope;

            static if(isComplex!T) {
                auto rwork = mininitRcslice!(realType!T)(max(1, 5 * min(m, n)));
                auto info = gesvd!T(jobu, jobvt, rca, s.lightScope, u.lightScope.canonical, vt.lightScope.canonical, work, rwork.lightScope);
            } else {
                auto info = gesvd!T(jobu, jobvt, rca, s.lightScope, u.lightScope.canonical, vt.lightScope.canonical, work);
            }
        }
        else // gesdd
        {
            auto rciwork = mininitRcslice!lapackint(s.length * 8);
            auto iwork = rciwork.lightScope;
            auto jobz = slim ? 'S' : 'A';
            auto rca_sliced = a.as!T.rcslice;
            auto rca = rca_sliced.lightScope.canonical;
            auto rcwork = gesdd_wq(jobz, rca, u.lightScope, vt.lightScope).minitRcslice!T;
            auto work = rcwork.lightScope;

            static if(isComplex!T) {
                auto mx = max(m, n);
                auto mn = min(m, n);
                auto rwork = mininitRcslice!(realType!T)(max(5*mn^^2 + 5*mn, 2*mx*mn + 2*mn^^2 + mn));
                auto info = gesdd!T(jobz, rca, s.lightScope, u.lightScope, vt.lightScope, work, rwork.lightScope, iwork);
            } else {
                auto info = gesdd!T(jobz, rca, s.lightScope, u.lightScope, vt.lightScope, work, iwork);
            }
        }
        enum msg = (algorithm == "gesvd" ? "svd: DBDSDC did not converge, updating process failed" : "svd: DBDSQR did not converge");
        enforce!("svd: " ~ msg)(!info);
    }
    return SvdResult!T(vt, s, u); //transposed
}

@safe pure SvdResult!T svd
(
    T,
    string algorithm = "gesvd",
    SliceKind kind,
)(
    auto ref scope const Slice!(RCI!T,2,kind) matrix,
    Flag!"slim" slim = No.slim
)
{
    auto matrixScope = matrix.lightScope.lightConst;
    return svd!(T, algorithm)(matrixScope, slim); 
}

pure unittest
{
    import mir.ndslice;
    import mir.math;

    auto a = mininitRcslice!double(6, 4);
    a[] = [[7.52,  -1.10,  -7.95,   1.08],
           [-0.76,   0.62,   9.34,  -7.10],
           [5.13,   6.62,  -5.66,   0.87],
           [-4.75,   8.52,   5.75,   5.30],
           [1.33,   4.91,  -5.49,  -3.52],
           [-2.40,  -6.77,   2.34,   3.95]];

    auto r1 = a.svd;

    auto sigma1 = rcslice!double(a.shape, 0);
    sigma1.diagonal[] = r1.sigma;
    auto m1 = (r1.u).mtimes(sigma1).mtimes(r1.vt);
    assert(equal!((a, b) => fabs(a-b)< 1e-8)(a, m1));

    auto r2 = a.svd;

    auto sigma2 = rcslice!double(a.shape, 0.0);
    sigma2.diagonal[] = r2.sigma;
    auto m2 = r2.u.mtimes(sigma2).mtimes(r2.vt);

    assert(equal!((a, b) => fabs(a-b)< 1e-8)(a, m2));

    auto r = a.svd(Yes.slim);
    assert(r.u.shape == [6, 4]);
    assert(r.vt.shape == [4, 4]);
}

pure unittest
{
    import mir.ndslice;
    import mir.math;

    auto a = mininitRcslice!double(6, 4);
    a[] =   [[7.52,  -1.10,  -7.95,   1.08],
            [-0.76,   0.62,   9.34,  -7.10],
            [5.13,   6.62,  -5.66,   0.87],
            [-4.75,   8.52,   5.75,   5.30],
            [1.33,   4.91,  -5.49,  -3.52],
            [-2.40,  -6.77,   2.34,   3.95]];

    auto r1 = a.svd(No.slim);

    auto sigma1 = rcslice!double(a.shape, 0.0);
    sigma1.diagonal[] = r1.sigma;
    auto m1 = r1.u.mtimes(sigma1).mtimes(r1.vt);

    assert(equal!((a, b) => fabs(a-b)< 1e-8)(a, m1));
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

pure unittest
{
    import mir.ndslice;
    import mir.math;
    import mir.complex : cabs;

    alias C = Complex!double;

    auto a = mininitRcslice!C(6, 4);
    a[] = [[7.52,  -1.10,  -7.95,   1.08],
           [-0.76,   0.62,   9.34,  -7.10],
           [5.13,   6.62,  -5.66,   0.87],
           [-4.75,   8.52,   5.75,   5.30],
           [1.33,   4.91,  -5.49,  -3.52],
           [-2.40,  -6.77,   2.34,   3.95]];

    auto r1 = a.svd;

    auto sigma1 = rcslice!C(a.shape, C(0));
    sigma1.diagonal[] = r1.sigma;
    auto m1 = (r1.u).mtimes(sigma1).mtimes(r1.vt);
    assert(equal!((a, b) => cabs(a-b)< 1e-8)(a, m1));

    auto r2 = a.svd;

    auto sigma2 = rcslice!C(a.shape, C(0));
    sigma2.diagonal[] = r2.sigma;
    auto m2 = r2.u.mtimes(sigma2).mtimes(r2.vt);

    assert(equal!((a, b) => cabs(a-b)< 1e-8)(a, m2));

    auto r = a.svd(Yes.slim);
    assert(r.u.shape == [6, 4]);
    assert(r.vt.shape == [4, 4]);
}

struct EigenResult(T)
{
    Slice!(RCI!(complexType!T), 2) eigenvectors;
    Slice!(RCI!(complexType!T)) eigenvalues;
}

struct QRResult(T)
{
    Slice!(RCI!T,2) Q;
    Slice!(RCI!T,2) R;

    @safe this
    (
        SliceKind kindA,
        SliceKind kindTau
    )(
        Slice!(RCI!T, 2, kindA) a,
        Slice!(RCI!T, 1, kindTau) tau
    )
    {
        auto aScope = a.lightScope.lightConst;
        auto tauScope = tau.lightScope.lightConst;
        this(aScope, tauScope);
    }

    @safe this
    (
        SliceKind kindA,
        SliceKind kindTau
    )(
        Slice!(const(T)*, 2, kindA) a,
        Slice!(const(T)*, 1, kindTau) tau
    )
    {
        R = mininitRcslice!T(a.shape);
        foreach (i; 0 .. R.length!0)
        {
            foreach (j; 0 .. R.length!1)
            {
                if (i >= j)
                {
                    R[j, i] = a[i, j];
                }
                else
                {
                    R[j ,i] = cast(T)0;
                }
            }
        }
        auto rcwork = mininitRcslice!T(a.length!0);
        auto work = rcwork.lightScope;
        auto aSliced = a.as!T.rcslice;
        auto aScope = aSliced.lightScope.canonical;
        auto tauSliced = tau.as!T.rcslice;
        auto tauScope = tauSliced.lightScope;

        static if(is(T == double) || is(T == float))
            orgqr!T(aScope, tauScope, work);
        else
            ungqr!T(aScope, tauScope, work);

        Q = aScope.transposed.as!T.rcslice;
    }
}

@safe pure QRResult!T qr(T, SliceKind kind)(
    auto ref const Slice!(RCI!T, 2, kind) matrix
)
{
    auto matrixScope = matrix.lightScope.lightConst;
    return qr!(T, kind)(matrixScope);
}

@safe pure QRResult!T qr(T, SliceKind kind)(
    auto ref const Slice!(const(T)*, 2, kind) matrix
)
{
    auto a = matrix.transposed.as!T.rcslice;
    auto tau = mininitRcslice!T(a.length!0);
    auto rcwork = mininitRcslice!T(a.length!0);
    auto work = rcwork.lightScope;
    auto aScope = a.lightScope.canonical;
    auto tauScope = tau.lightScope;
    geqrf!T(aScope, tauScope, work);
    return QRResult!T(aScope, tauScope);
}

pure nothrow
unittest
{
    import mir.ndslice;
    import mir.math;

    auto data = mininitRcslice!double(3, 3);
    data[] = [[12, -51,   4],
              [ 6, 167, -68],
              [-4,  24, -41]];

    auto res = qr(data);
    auto q = mininitRcslice!double(3, 3);
    q[] = [[-6.0/7.0,   69.0/175.0, 58.0/175.0],
           [-3.0/7.0, -158.0/175.0, -6.0/175.0],
           [ 2.0/7.0,   -6.0/35.0 , 33.0/35.0 ]];
    auto aE = function (double x, double y) => approxEqual(x, y, 0.00005, 0.00005);
    auto r = mininitRcslice!double(3, 3);
    r[] = [[-14,  -21,  14],
           [  0, -175,  70],
           [  0,    0, -35]];
    assert(equal!approxEqual(mtimes(res.Q, res.R), data));
}

pure nothrow
unittest
{
    import mir.ndslice;
    import mir.math;
    import mir.complex.math: approxEqual;

    auto data = mininitRcslice!(Complex!double)(3, 3);
    data[] = [[12, -51,   4],
              [ 6, 167, -68],
              [-4,  24, -41]];

    auto res = qr(data);
    auto q = mininitRcslice!(Complex!double)(3, 3);
    q[] = [[-6.0/7.0,   69.0/175.0, 58.0/175.0],
           [-3.0/7.0, -158.0/175.0, -6.0/175.0],
           [ 2.0/7.0,   -6.0/35.0 , 33.0/35.0 ]];
    auto aE = function (Complex!double x, Complex!double y) => approxEqual(x, y, 0.00005, 0.00005);
    auto r = mininitRcslice!(Complex!double)(3, 3);
    r[] = [[-14,  -21,  14],
           [  0, -175,  70],
           [  0,    0, -35]];
    assert(equal!approxEqual(mtimes(res.Q, res.R), data));
}


@safe pure @nogc
EigenResult!(realType!T) eigen(T, SliceKind kind)(
    auto ref Slice!(const(T)*,2, kind) a,
)
{
    enforce!"eigen: input matrix must be square"(a.length!0 == a.length!1);
    const n = a.length;
    auto rcw = n.mininitRcslice!(complexType!T);
    auto w = rcw.lightScope;
    auto rcwork = mininitRcslice!T(16 * n);
    auto work = rcwork.lightScope;
    auto z = [n, n].mininitRcslice!(complexType!T);//vl (left eigenvectors)
    auto rca = a.transposed.as!T.rcslice;
    auto as = rca.lightScope;
    static if (isComplex!T)
    {
        auto rcrwork = [2 * n].mininitRcslice!(realType!T);
        auto rwork = rcrwork.lightScope;
        auto info = geev!(T, realType!T)('N', 'V', as.canonical, w, Slice!(T*, 2, Canonical).init, z.lightScope.canonical, work, rwork);
        enforce!"eigen failed"(!info);
    }
    else
    {
        alias C = complexType!T;
        auto wr = sliced((cast(T[]) w.field)[0 .. n]);
        auto wi = sliced((cast(T[]) w.field)[n .. n * 2]);
        auto rczr = [n, n].mininitRcslice!T;
        auto zr = rczr.lightScope;
        auto info = geev!T('N', 'V', as.canonical, wr, wi, Slice!(T*, 2, Canonical).init, zr.canonical, work);
        enforce!"eigen failed"(!info);
        work[0 .. n] = wr;
        work[n .. n * 2] = wi;
        foreach (i, ref e; w.field)
        {
            e = C(work[i], work[n + i]);
            auto zi = z.lightScope[i];
            if (e.im > 0)
                zi[] = zip(zr[i], zr[i + 1]).map!((a, b) => C(a, b));
            else
            if (e.im < 0)
                zi[] = zip(zr[i - 1], zr[i]).map!((a, b) => C(a, -b));
            else
                zi[] = zr[i].as!C;
        }
    }
    return typeof(return)(z, rcw);
}

//@safe pure @nogc
EigenResult!(realType!T) eigen(T, SliceKind kind)(
    auto ref Slice!(RCI!T,2, kind) a
)
{
    auto as = a.lightScope.lightConst;
    return eigen(as);
}

///
// pure
unittest
{
    import mir.blas;
    import mir.complex;
    import mir.ndslice;
    import mir.math;
    import mir.complex.math: approxEqual;

    auto data = 
        [[ 0, 1],
         [-1, 0]].fuse.as!double.rcslice;
    auto eigenvalues = [Complex!double(0, 1), Complex!double(0, -1)].sliced;
    auto eigenvectors =
        [[Complex!double(0, -1), Complex!double(1, 0)],
         [Complex!double(0, 1), Complex!double(1, 0)]].fuse;

    auto res = data.eigen;

    assert(res.eigenvalues.equal!approxEqual(eigenvalues));

    import mir.math.common: approxEqual;
    foreach (i; 0 .. eigenvectors.length)
        assert((res.eigenvectors.lightScope[i] / eigenvectors[i]).diff.slice.nrm2.approxEqual(0));
}


///
@safe pure
unittest
{
    import mir.complex;
    import mir.ndslice;
    import mir.math;
    import mir.blas;
    import mir.math.common: fapproxEqual = approxEqual;
    import mir.complex.math: approxEqual;

    auto data =
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]].fuse.as!double.rcslice;
    auto c = 3.0.sqrt;
    auto eigenvalues = [Complex!double(-1, c) / 2, Complex!double(-1, - c) / 2, Complex!double(1, 0)];

    auto eigenvectors =
        [[Complex!double(-1, +c), Complex!double(-1, -c) , Complex!double(2, 0)],
         [Complex!double(-1, -c), Complex!double(-1, +c) , Complex!double(2, 0)],
         [Complex!double(1, 0), Complex!double(1, 0), Complex!double(1, 0)]].fuse;

    auto res = data.eigen;

    assert(res.eigenvalues.equal!approxEqual(eigenvalues));
    foreach (i; 0 .. eigenvectors.length)
        assert((res.eigenvectors.lightScope[i] / eigenvectors[i]).diff.slice.nrm2.fapproxEqual(0));

    auto cdata = data.lightScope.as!(Complex!double).rcslice;
    res = cdata.eigen;

    assert(res.eigenvalues.equal!approxEqual(eigenvalues));
    foreach (i; 0 .. eigenvectors.length)
        assert((res.eigenvectors.lightScope[i] / eigenvectors[i]).diff.slice.nrm2.fapproxEqual(0));
}


/// Principal component analysis result.
struct PcaResult(T)
{
    /// Eigenvectors (Eigenvectors[i] is the ith eigenvector)
    Slice!(RCI!T,2) eigenvectors;
    /// Principal component scores. (Input matrix rotated to basis of eigenvectors)
    Slice!(RCI!T, 2) scores;
    /// Principal component variances. (Eigenvalues)
    Slice!(RCI!T) eigenvalues;
    /// The means of each data column (0 if not centred)
    Slice!(RCI!T) mean;
    /// The standard deviations of each column (1 if not normalized)
    Slice!(RCI!T) stdDev;
    /// Principal component Loadings vectors (Eigenvectors times sqrt(Eigenvalue))
    Slice!(RCI!T, 2) loadings()
    {
        auto result = mininitRcslice!T(eigenvectors.shape);
        for (size_t i = 0; i < eigenvectors.length!0 && i < eigenvalues.length; i++){
            if(eigenvalues[i] != 0)
                foreach (j; 0 .. eigenvectors.length!1)
                    result[i, j] = eigenvectors[i, j] * sqrt(eigenvalues[i]);
            else
                result[i][] = eigenvectors[i][];
        }
        return result;
    }

    Slice!(RCI!T, 2) loadingsScores() {// normalized data in basis of {sqrt(eval)*evect}
        //if row i is a vector p, the original data point is mean[] + p[] * stdDev[]
        auto result = mininitRcslice!T(scores.shape);
        foreach (i; 0 .. scores.length!0){
            for (size_t j=0; j < scores.length!1 && j < eigenvalues.length; j++)
            {
                if(eigenvalues[j] != 0)
                    result[i, j] = scores[i, j] / sqrt(eigenvalues[j]);
                else
                    result[i, j] = scores[i, j];
            }
        }
        return result;
    }

    Slice!(RCI!T) explainedVariance() {
        import mir.math.sum: sum;
        //auto totalVariance = eigenvalues.sum!"kb2";
        auto totalVariance = 0.0;
        foreach (val; eigenvalues)
            totalVariance += val;
        auto result = mininitRcslice!double(eigenvalues.shape);
        foreach (i; 0 .. result.length!0)
            result[i] = eigenvalues[i] / totalVariance;
        return result;
    }

    Slice!(RCI!T,2) q()
    {
        return eigenvectors;
    }

    //returns matrix to transform into basis of 'n' most significatn eigenvectors
    Slice!(RCI!(T),2) q(size_t n)
    in {
        assert(n <= eigenvectors.length!0);
    }
    do {
        auto res = mininitRcslice!T(eigenvectors.shape);
        res[] = eigenvectors;
        for ( ; n < eigenvectors.length!0; n++)
            res[n][] = 0;
        return res;
    }
    //returns matrix to transform into basis of eigenvectors with value larger than delta
    Slice!(RCI!(T),2) q(T delta)
    do
    {
        auto res = mininitRcslice!T(eigenvectors.shape);
        res[] = eigenvectors;
        foreach (i; 0 .. eigenvectors.length!0)
            if (fabs(eigenvalues[i]) <= delta)
                res[i][] = 0;
        return res;
    }

    //transforms data into eigenbasis
    Slice!(RCI!(T),2) transform(Slice!(RCI!(T),2) data)
    {
        return q.mlinverse.mtimes(data);
    }

    //transforms data into eigenbasis
    Slice!(RCI!(T),2) transform(Slice!(RCI!(T),2) data, size_t n)
    {
        return q(n).mlinverse.mtimes(data);
    }
    //transforms data into eigenbasis
    Slice!(RCI!(T),2) transform(Slice!(RCI!(T),2) data, T delta)
    {
        return q(delta).mlinverse.mtimes(data);
    }
}


/++
Principal component analysis of raw data.
Template:
correlation = Flag to use correlation matrix instead of covariance
Params:
    data = input `M x N` matrix, where 'M (rows)>= N(cols)'
    devEst =
    meanEst =
    fixEigenvectorDirections =
Returns: $(LREF PcaResult)
+/
@safe pure @nogc
PcaResult!T pca(
    SliceKind kind,
    T
 )(
    Slice!(const(T)*, 2, kind) data,
    DeviationEstimator devEst = DeviationEstimator.sample,
    MeanEstimator meanEst = MeanEstimator.average,
    Flag!"fixEigenvectorDirections" fixEigenvectorDirections = Yes.fixEigenvectorDirections,
)
in
{
    assert(data.length!0 >= data.length!1);
}
do
{
    real n = (data.length!0 <= 1 ? 1 : data.length!0 -1 );//num observations
    auto mean = rcslice!(T,1)([data.length!1], cast(T)0);
    auto stdDev = rcslice!(T,1)([data.length!1], cast(T)1);
    auto centeredData = centerColumns!T(data, mean, meanEst);
    //this part gets the eigenvectors of the sample covariance without explicitly calculating the covariance matrix
    //to implement a minimum covariance deteriminant this block would need to be redone
    //firstly one would calculate the MCD then call eigen to get it's eigenvectors
    auto processedData = normalizeColumns!T(centeredData, stdDev, devEst);
    auto svdResult = processedData.svd(Yes.slim);
    with (svdResult)
    {
        //u[i][] is the ith eigenvector
        foreach (i; 0 .. u.length!0){
            foreach (j; 0 .. u.length!1){
                u[i, j] *= sigma[j];
            }
        }
        auto eigenvalues = mininitRcslice!double(sigma.shape);
        for (size_t i = 0; i < sigma.length && i < eigenvalues.length; i++){
            eigenvalues[i] = sigma[i] * sigma[i] / n; //square singular values to get eigenvalues
        }
        if (fixEigenvectorDirections)
        {
            foreach(size_t i; 0 .. sigma.length)
            {
                //these are directed so the 0th component is +ve
                if (vt[i, 0] < 0)
                {
                    vt[i][] *= -1;
                    u[0 .. $, i] *= -1;
                }
            }
        }
        PcaResult!T result;
        result.scores = u;
        result.eigenvectors = vt;
        result.eigenvalues = eigenvalues;
        result.mean = mean;
        result.stdDev = stdDev;
        return result;
    }
}

@safe pure @nogc
PcaResult!T pca(T, SliceKind kind)
(
    auto ref const Slice!(RCI!T, 2, kind) data,
    DeviationEstimator devEst = DeviationEstimator.sample,
    MeanEstimator meanEst = MeanEstimator.average,
    Flag!"fixEigenvectorDirections" fixEigenvectorDirections = Yes.fixEigenvectorDirections,
)
do
{
    auto d = data.lightScope.lightConst;
    return d.pca(devEst, meanEst, fixEigenvectorDirections);
}

pure
unittest
{
    import mir.ndslice;
    import mir.math;

    auto data = mininitRcslice!double(3, 2);
    data[] = [[ 1, -1],
              [ 0,  1],
    [-1,  0]];
    //cov =0.5 * [[ 2, -1],
    //            [-1,  2]]
    const auto const_data = data;
    auto mean = mininitRcslice!double(2);
    assert(data == centerColumns(const_data, mean));
    assert(mean == [0,0]);
    PcaResult!double res = const_data.pca;

    auto evs = mininitRcslice!double(2, 2);
    evs[] = [[1, -1],
             [1,  1]];
    evs[] /= sqrt(2.0);
    assert(equal!approxEqual(res.eigenvectors, evs));

    auto score = mininitRcslice!double(3, 2);
    score[] = [[ 1.0,  0.0],
               [-0.5,  0.5],
    [-0.5, -0.5]];
    score[] *= sqrt(2.0);
    assert(equal!approxEqual(res.scores, score));

    auto evals = mininitRcslice!double(2);
    evals[] = [1.5, 0.5];
    assert(equal!approxEqual(res.eigenvalues, evals));
}

pure
unittest
{
    import mir.ndslice;
    import mir.math;

    auto data = mininitRcslice!double(10, 3);
    data[] = [[7, 4, 3],
              [4, 1, 8],
              [6, 3, 5],
              [8, 6, 1],
              [8, 5, 7],
              [7, 2, 9],
              [5, 3, 3],
              [9, 5, 8],
              [7, 4, 5],
              [8, 2, 2]];

    auto m1 = 69.0/10.0;
    auto m2 = 35.0/10.0;
    auto m3 = 51.0/10.0;

    auto centeredData = mininitRcslice!double(10, 3);
    centeredData[] =   [[7-m1, 4-m2, 3-m3],
                        [4-m1, 1-m2, 8-m3],
                        [6-m1, 3-m2, 5-m3],
                        [8-m1, 6-m2, 1-m3],
                        [8-m1, 5-m2, 7-m3],
                        [7-m1, 2-m2, 9-m3],
                        [5-m1, 3-m2, 3-m3],
                        [9-m1, 5-m2, 8-m3],
                        [7-m1, 4-m2, 5-m3],
                        [8-m1, 2-m2, 2-m3]];
    auto mean = mininitRcslice!double(3);
    auto cenRes = centerColumns(data, mean);

    assert(equal!approxEqual(centeredData, cenRes));
    assert(equal!approxEqual(mean, [m1,m2,m3]));

    auto res = data.pca;
    auto coeff = mininitRcslice!double(3, 3);
    coeff[] = [[0.6420046 ,  0.6863616 , -0.3416692 ],
               [0.38467229,  0.09713033,  0.91792861], 
               [0.6632174 , -0.7207450 , -0.2016662 ]];

    auto score = mininitRcslice!double(10, 3);
    score[] = [[ 0.5148128, -0.63083556, -0.03351152],
               [-2.6600105,  0.06280922, -0.33089322],
               [-0.5840389, -0.29060575, -0.15658900],
               [ 2.0477577, -0.90963475, -0.36627323],
               [ 0.8832739,  0.99120250, -0.34153847],
               [-1.0837642,  1.20857108,  0.44706241],
               [-0.7618703, -1.19712391, -0.44810271],
               [ 1.1828371,  1.57067601,  0.02182598],
               [ 0.2713493,  0.02325373, -0.17721300],
               [ 0.1896531, -0.82831257,  1.38523276]];

    auto stdDev = mininitRcslice!double(3);
    stdDev[] = [1.3299527, 0.9628478, 0.5514979];

    auto eigenvalues = mininitRcslice!double(3);
    eigenvalues[] = [1.768774, 0.9270759, 0.3041499];

    assert(equal!approxEqual(res.eigenvectors, coeff));
    assert(equal!approxEqual(res.scores, score));
    assert(equal!approxEqual(res.eigenvalues, eigenvalues));
}

pure
unittest
{
    import mir.ndslice;
    import mir.math;

    auto data = mininitRcslice!double(13, 4);
    data[] =[[ 7,  26,   6,  60],
             [ 1,  29,  15,  52],
             [11,  56,   8,  20],
             [11,  31,   8,  47],
             [ 7,  52,   6,  33],
             [11,  55,   9,  22],
             [ 3,  71,  17,   6],
             [ 1,  31,  22,  44],
             [ 2,  54,  18,  22],
             [21,  47,   4,  26],
             [ 1,  40,  23,  34],
             [11,  66,   9,  12],
             [10,  68,   8,  12]];

    auto m1 = 97.0/13.0;
    auto m2 = 626.0/13.0;
    auto m3 = 153.0/13.0;
    auto m4 = 390.0/13.0;

    auto centeredData = mininitRcslice!double(13, 4);
    centeredData[] = [[ 7-m1, 26-m2,  6-m3, 60-m4],
                      [ 1-m1, 29-m2, 15-m3, 52-m4],
                      [11-m1, 56-m2,  8-m3, 20-m4],
                      [11-m1, 31-m2,  8-m3, 47-m4],
                      [ 7-m1, 52-m2,  6-m3, 33-m4],
                      [11-m1, 55-m2,  9-m3, 22-m4],
                      [ 3-m1, 71-m2, 17-m3,  6-m4],
                      [ 1-m1, 31-m2, 22-m3, 44-m4],
                      [ 2-m1, 54-m2, 18-m3, 22-m4],
                      [21-m1, 47-m2,  4-m3, 26-m4],
                      [ 1-m1, 40-m2, 23-m3, 34-m4],
                      [11-m1, 66-m2,  9-m3, 12-m4],
                      [10-m1, 68-m2  ,8-m3, 12-m4]];
    auto mean = mininitRcslice!double(4);
    auto cenRes = centerColumns(data, mean);

    assert(equal!approxEqual(centeredData, cenRes));
    assert(mean == [m1,m2,m3,m4]);

    auto res = data.pca(DeviationEstimator.none);
    auto coeff = mininitRcslice!double(4, 4);
    coeff[] = [[0.067799985695474,  0.678516235418647, -0.029020832106229, -0.730873909451461],
               [0.646018286568728,  0.019993340484099, -0.755309622491133,  0.108480477171676],
               [0.567314540990512, -0.543969276583817,  0.403553469172668, -0.468397518388289],
               [0.506179559977705,  0.493268092159297,  0.515567418476836,  0.484416225289198]];

    auto score = mininitRcslice!double(13, 4);
    score[] = [[-36.821825999449700,   6.870878154227367,  -4.590944457629745,   0.396652582713912],
               [-29.607273420710964,  -4.610881963526308,  -2.247578163663940,  -0.395843536696492],
               [ 12.981775719737618,   4.204913183175938,   0.902243082694698,  -1.126100587210615],
               [-23.714725720918022,   6.634052554708721,   1.854742000806314,  -0.378564808384691],
               [  0.553191676624597,   4.461732123178686,  -6.087412652325177,   0.142384896047281],
               [ 10.812490833309816,   3.646571174544059,   0.912970791674604,  -0.134968810314680],
               [ 32.588166608817929,  -8.979846284936063,  -1.606265913996588,   0.081763927599947],
               [-22.606395499005586, -10.725906457369449,   3.236537714483416,   0.324334774646368],
               [  9.262587237675838,  -8.985373347478788,  -0.016909578102172,  -0.543746175981799],
               [  3.283969329640680,  14.157277337500918,   7.046512994833761,   0.340509860960606],
               [ -9.220031117829379, -12.386080787220454,   3.428342878284624,   0.435152769664895],
               [ 25.584908517429557,   2.781693148152386,  -0.386716066864491,   0.446817950545605],
               [ 26.903161834677597,   2.930971165042989,  -2.445522630195304,   0.411607156409658]];

    auto eigenvalues = mininitRcslice!double(4);


    eigenvalues[] = [517.7968780739053, 67.4964360487231, 12.4054300480810, 0.2371532651878];

    assert(equal!approxEqual(res.eigenvectors, coeff));
    assert(equal!approxEqual(res.scores, score));
    assert(equal!approxEqual(res.eigenvalues, eigenvalues));
}

///complex extensions

private T conj(T)(
    const T z
)
    if (isComplex!T)
{
    return z.re - (1fi* z.im);
}

private template complexType(C)
{
    import mir.complex: Complex;
    static if (isComplex!C)
        alias complexType = Unqual!C;
    else static if (is(Unqual!C == double))
        alias complexType = Complex!(Unqual!C);
}

///
enum MeanEstimator
{
    ///
    none,
    ///
    average,
    ///
    median
}

///
@safe pure nothrow @nogc
T median(T)(auto ref Slice!(RCI!T) data)
{
    auto dataScope = data.lightScope.lightConst;
    return median!T(dataScope);
}

/// ditto
@safe pure nothrow @nogc
T median(T)(Slice!(const(T)*) data)
{
    import mir.ndslice.sorting: sort;
    size_t len = cast(int) data.length;
    size_t n = len / 2;
    auto temp = data.as!T.rcslice;
    temp.lightScope.sort();
    return len % 2 ? temp[n] : 0.5f * (temp[n - 1] + temp[n]);
}

///
@safe pure
unittest
{
    import mir.ndslice;
    import mir.math;

    auto a = mininitRcslice!double(3);
    a[] = [3, 1, 7];
    auto med = median!double(a.flattened);
    assert(med == 3.0);
    assert(a == [3, 1, 7]);//add in stddev out param
    double aDev;
    auto aCenter = centerColumns(a, aDev, MeanEstimator.median);
    assert(aCenter == [0.0, -2.0, 4.0]);
    assert(aDev == 3.0);
    auto b = mininitRcslice!double(4);
    b[] = [4,2,5,1];
    auto medB = median!double(b.flattened);
    assert(medB == 3.0);
    assert(b == [4,2,5,1]);
    double bDev;
    auto bCenter = centerColumns(b, bDev, MeanEstimator.median);
    assert(bCenter == [1.0, -1.0, 2.0, -2.0]);
    assert(bDev == 3.0);
}

/++
Mean Centring of raw data.
Params:
    matrix = input `M x N` matrix
    mean = column means
    est = mean estimation method
Returns:
`M x N` matrix with each column translated by the column mean
+/
@safe pure nothrow @nogc
Slice!(RCI!T,2) centerColumns(T, SliceKind kind)
(
    Slice!(const(T)*, 2, kind) matrix,
    out Slice!(RCI!T) mean,
    MeanEstimator est = MeanEstimator.average,
)
{
    mean = rcslice!T([matrix.length!1], cast(T)0);
    if (est == MeanEstimator.none)
    {
        return matrix.as!T.rcslice;
    }
    auto at = matrix.transposed.as!T.rcslice;
    auto len = at.length!1;
    foreach (i; 0 .. at.length!0)
    {
        if (est == MeanEstimator.average)
        {
            foreach(j; 0 .. at.length!1)
                mean[i] += (at[i][j]/len);
        }
        else // (est == MeanEstimator.median)
        {
            mean[i] = median(at[i].flattened);
        }
        at[i][] -= mean[i];
    }
    auto atSliced = at.transposed.as!T.rcslice;
    return atSliced;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!T) centerColumns(T)
(
    Slice!(const(T)*) col,
    out T mean,
    MeanEstimator est = MeanEstimator.average,
)
{
    mean = cast(T)0;
    if (est == MeanEstimator.none)
    {
        return col.as!T.rcslice;
    }
    auto len = col.length;
    if (est == MeanEstimator.average)
    {
        foreach(j; 0 .. len)
            mean += (col[j]/len);
    }
    else // (est == MeanEstimator.median)
    {
        mean = median(col);
    }
    auto result = mininitRcslice!T(len);
    foreach (j; 0 .. len)
        result[j] = col[j] - mean;
    return result;
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!T) centerColumns(T)
(
   auto ref const Slice!(RCI!T) col,
   out T mean,
    MeanEstimator est = MeanEstimator.average,
)
{
    auto colScope = col.lightScope;
    return centerColumns!(T)(colScope, mean, est);
}

/// ditto
@safe pure nothrow @nogc
Slice!(RCI!T,2) centerColumns(T, SliceKind kind)
(
    auto ref const Slice!(RCI!T,2,kind) matrix,
    out Slice!(RCI!T) mean,
    MeanEstimator est = MeanEstimator.average,
)
{
    auto matrixScope = matrix.lightScope;
    return centerColumns(matrixScope, mean, est);
}

///
@safe pure nothrow
unittest
{
    import mir.ndslice;
    import mir.math;

    auto data = mininitRcslice!double(2,1);
    data[] = [[1],
              [3]];
    auto mean = mininitRcslice!double(1);
    auto res = centerColumns(data, mean, MeanEstimator.average);
    assert(mean[0] == 2);
    assert(res == [[-1],[1]]);
}

///
enum DeviationEstimator
{
    ///
    none,
    ///
    sample,
    /// median absolute deviation
    mad
}

/++
Normalization of raw data.
Params:
    matrix = input `M x N` matrix, each row an observation and each column mean centred
    stdDev = column standard deviation
    devEst = estimation method
Returns:
    `M x N` matrix with each column divided by it's standard deviation
+/
@safe pure nothrow @nogc Slice!(RCI!T,2) normalizeColumns(T, SliceKind kind)(
    auto ref const Slice!(const(T)*,2,kind) matrix,
    out Slice!(RCI!T) stdDev,
    DeviationEstimator devEst = DeviationEstimator.sample
)
{
    stdDev = rcslice!T([matrix.length!1], cast(T)0);
    if (devEst == DeviationEstimator.none)
    {
        auto matrixSliced = matrix.as!T.rcslice;
        return matrixSliced;
    }
    else
    {   
        import mir.math.sum: sum;
        auto mTSliced = matrix.transposed.as!T.rcslice;
        auto mT = mTSliced.lightScope.canonical;
        foreach (i; 0 .. mT.length!0)
        {
            auto processedRow = mininitRcslice!T(mT.length!1);
            if (devEst == DeviationEstimator.sample)
            {
                foreach (j; 0 .. mT.length!1)
                    processedRow[j] = mT[i, j] * mT[i, j];
                stdDev[i] = sqrt(processedRow.sum!"kb2" / (mT[i].length - 1));
            }
            else if (devEst == DeviationEstimator.mad)
            {
                foreach (j; 0 .. mT.length!1)
                    processedRow[j] = fabs(mT[i,j]);
                stdDev[i] = median!T(processedRow);
            }
            mT[i][] /= stdDev[i];
        }
        auto mSliced = mT.transposed.rcslice;
        return mSliced;
    }
}

/// ditto
@safe pure nothrow @nogc Slice!(RCI!T,2) normalizeColumns(T, SliceKind kind)(
    auto ref const Slice!(RCI!T,2,kind) matrix,
    out Slice!(RCI!T) stdDev,
    DeviationEstimator devEst = DeviationEstimator.sample,
)
{
    auto matrixScope = matrix.lightScope.lightConst;
    return normalizeColumns!(T, kind)(matrixScope, stdDev, devEst);
}

///
@safe pure nothrow unittest
{
    import mir.ndslice;
    import mir.math;

    auto data = mininitRcslice!double(2,2);
    data[] = [[ 2, -1],
              [-2,  1]];
    //sd1 = 2 * sqrt(2.0);
    //sd2 = sqrt(2.0);
    auto x = 1.0 / sqrt(2.0);
    auto scaled = mininitRcslice!double(2,2);
    scaled[] = [[ x, -x],
                [-x,  x]];
    auto stdDev = mininitRcslice!double(2);
    assert(normalizeColumns(data, stdDev) == scaled);
    assert(stdDev == [2*sqrt(2.0), sqrt(2.0)]);
}
