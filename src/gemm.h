#ifndef _GEMM_H_
#define _GEMM_H_

#include "includes.h"

/* TA: transpose for A;    TB: transpose for B
 * A(M*K) X B(K*N) = C(M*N)
 * lda ldb and ldc are the lengths of the last dimension of A B and C
 * e.g.  gemm(0, 0, m, n, k, A, k, B, n, C, n)
 *       gemm(0, 1, m, n, k, A, k, B, k, C, n)
 *       gemm(1, 0, m, n, k, A, m, B, n, C, n)
 *       gemm(1, 1, m, n, k, A, m, B, k, C, n)
 */
void gemm(int TA, int TB, int M, int N, int K, DataType *A, int lda, DataType *B, int ldb, DataType *C, int ldc);

#if GPU == 1
/* TA: transpose for A;    TB: transpose for B
 * A(M*K) X B(K*N) = C(M*N)
 * lda ldb and ldc are the lengths of the last dimension of A B and C
 * e.g.  gemm(0, 0, m, n, k, A, k, B, n, C, n)
 *       gemm(0, 1, m, n, k, A, k, B, k, C, n)
 *       gemm(1, 0, m, n, k, A, m, B, n, C, n)
 *       gemm(1, 1, m, n, k, A, m, B, k, C, n)
 */
void gemm_gpu(int TA, int TB, int M, int N, int K, DataType *A_gpu, int lda, DataType *B_gpu, int ldb, DataType *C_gpu, int ldc);
#endif

#endif
