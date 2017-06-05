# lubeck
High level linear algebra library for Dlang

## Required static libraries
 - `blas`/`cblas` - CBLAS API
 - `lapack` - FORTRAN 77 LAPACK API

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
