[![Build Status](https://www.travis-ci.org/kaleidicassociates/lubeck.svg?branch=master)](https://www.travis-ci.org/kaleidicassociates/lubeck)

# lubeck
High level linear algebra library for Dlang

## Required static libraries
 - `blas`/`cblas` - CBLAS API
 - `lapack` - FORTRAN 77 LAPACK API

See the `dub.sdl` in the example folder.

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

## Example

```d
/+dub.sdl:
dependency "lubeck" version="~>0.0.4"
libs "lapack" "blas"
+/
// or libs "openblas"
import std.stdio;
import mir.ndslice: magic, repeat, as, slice;
import lubeck: mtimes;

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

[![Open on run.dlang.io](https://img.shields.io/badge/run.dlang.io-open-blue.svg)](https://run.dlang.io/is/vzhvo5)

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
