
[![Gitter](https://img.shields.io/gitter/room/libmir/public.svg)](https://gitter.im/libmir/public)
[![Build Status](https://www.travis-ci.org/kaleidicassociates/lubeck.svg?branch=master)](https://www.travis-ci.org/kaleidicassociates/lubeck)
[![Dub downloads](https://img.shields.io/dub/dt/lubeck.svg)](http://code.dlang.org/packages/lubeck)
[![Dub downloads](https://img.shields.io/dub/dm/lubeck.svg)](http://code.dlang.org/packages/lubeck)
[![License](https://img.shields.io/dub/l/lubeck.svg)](http://code.dlang.org/packages/lubeck)
[![Latest version](https://img.shields.io/dub/v/lubeck.svg)](http://code.dlang.org/packages/lubeck)

# Lubeck
High level linear algebra library for Dlang

## Required system libraries

See [wiki: Link with CBLAS & LAPACK](https://github.com/libmir/mir-lapack/wiki/Link-with-CBLAS-&-LAPACK).

## API
 - `mtimes` - General matrix-matrix, row-matrix, matrix-column, and row-column multiplications.
 - `mldivide` - Solve systems of linear equations AX = B for X. Computes minimum-norm solution to a linear least squares problem
if A is not a square matrix.
 - `inv` - Inverse of matrix.
 - `svd` - Singular value decomposition.
 - `pca` - Principal component analysis of raw data.
 - `pinv` - Moore-Penrose pseudoinverse of matrix.
 - `det`/`detSymmetric` - General/symmetric matrix determinant.
 - `eigSymmetric` - Eigenvalues and eigenvectors of symmetric matrix.
 - Qr decomposition: `qrDecomp` with `solve` method
 - Cholesky: `choleskyDecomp` with `solve` method
 - LU decomposition: `luDecomp` with `solve` method
 - LDL decomposition: `ldlDecomp` with `solve` method

## Example

```d
/+dub.sdl:
dependency "lubeck" version="~>0.1"
libs "lapack" "blas"
+/
// or libs "openblas"
import std.stdio;
import mir.ndslice: magic, repeat, as, slice;
import kaleidic.lubeck: mtimes;

void main()
{
    auto n = 5;
    // Magic Square
    auto matrix = n.magic.as!double.slice;
    // [1 1 1 1 1]
    auto vec = 1.repeat(n).as!double.slice;
    // Uses CBLAS for multiplication
    matrix.mtimes(vec).writeln;
    matrix.mtimes(matrix).writeln;
}
```

[![Open on run.dlang.io](https://img.shields.io/badge/run.dlang.io-open-blue.svg)](https://run.dlang.io/is/RQRMoo)

### Related packages
 - [mir-algorithm](https://github.com/libmir/mir-algorithm)
 - [mir-lapack](https://github.com/libmir/mir-lapack)
 - [mir-blas](https://github.com/libmir/mir-blas)
 - [lapack](https://github.com/libmir/lapack)
 - [cblas](https://github.com/DlangScience/cblas)

---------------

This work has been sponsored by [Symmetry Investments](http://symmetryinvestments.com) and [Kaleidic Associates](https://github.com/kaleidicassociates).


About Kaleidic Associates
-------------------------
We are a boutique consultancy that advises a small number of hedge fund clients.  We are
not accepting new clients currently, but if you are interested in working either remotely
or locally in London or Hong Kong, and if you are a talented hacker with a moral compass
who aspires to excellence then feel free to drop me a line: laeeth at kaleidic.io

We work with our partner Symmetry Investments, and some background on the firm can be
found here:

http://symmetryinvestments.com/about-us/
