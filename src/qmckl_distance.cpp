#include "../include/qmckl_distance.hpp"

using namespace sycl;

qmckl_exit_code_device qmckl_distance_devices(const qmckl_context_device context, const char transa,
                                              const char transb, const int64_t m, const int64_t n,
                                              const double *A, const int64_t lda, const double *B,
                                              const int64_t ldb, double* const C, const int64_t ldc)
{
    int transab;

    qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

    if (context == QMCKL_NULL_CONTEXT_DEVICE)
    {
        info = QMCKL_INVALID_CONTEXT_DEVICE;
        return info;
    }

    if (m <= 0)
    {
        info = QMCKL_INVALID_ARG_4_DEVICE;
        return info;
    }

    if (n <= 0)
    {
        info = QMCKL_INVALID_ARG_5_DEVICE;
        return info;
    }

    if (transa == 'N' || transa == 'n')
    {
        transab = 0;
    }
    else if (transa == 'T' || transa == 't')
    {
        transab = 1;
    }
    else
    {
        transab = -100;
    }

    if (transb == 'N' || transb == 'n')
    {
    }
    else if (transb == 'T' || transb == 't')
    {
        transab = transab + 2;
    }
    else
    {
        transab = -100;
    }

    if (transab < 0)
    {
        info = QMCKL_INVALID_ARG_1_DEVICE;
        return info;
    }

    // check for LDA
    if ((transab & 1) == 0 && lda < 3)
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if ((transab & 1) == 1 && lda < m)
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if ((transab & 2) == 0 && lda < 3)
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if ((transab & 2) == 2 && lda < m)
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    // check for LDB
    if ((transab & 1) == 0 && ldb < 3)
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 1) == 1 && ldb < n)
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 2) == 0 && ldb < 3)
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 2) == 2 && ldb < n)
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    // check for LDC
    if (ldc < m)
    {
        info = QMCKL_INVALID_ARG_11_DEVICE;
        return info;
    }

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device *)context;

    queue q = ctx->q;

    switch (transab)
    {
    case 0:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
            int j = idx[0];
            int i = idx[1];
            double x = A[0 + i * lda] - B[0 + j * ldb];
            double y = A[1 + i * lda] - B[1 + j * ldb];
            double z = A[2 + i * lda] - B[2 + j * ldb];
            C[i + j * ldc] = x * x + y * y + z * z;
            for (int k = 0; k < ldc; k++) {
                C[k + j * ldc] = sqrt(C[k + j * ldc]);
            } });
        q.wait();
        break;
    case 1:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
            int j = idx[0];
            int i = idx[1];
            double x = A[i + 0 * lda] - B[0 + j * ldb];
            double y = A[i + 1 * lda] - B[1 + j * ldb];
            double z = A[i + 2 * lda] - B[2 + j * ldb];
            C[i + j * ldc] = x * x + y * y + z * z;
            for (int k = 0; k < j; k++) {
                C[k + j * ldc] = sqrt(C[k + j * ldc]);
            } });
        q.wait();
        break;
    case 2:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
            int j = idx[0];
            int i = idx[1];
            double x = A[0 + i * lda] - B[j + 0 * ldb];
            double y = A[1 + i * lda] - B[j + 1 * ldb];
            double z = A[2 + i * lda] - B[j + 2 * ldb];
            C[i + j * ldc] = x * x + y * y + z * z;
            for (int k = 0; k < ldc; k++) {
                C[k + j * ldc] = sqrt(C[k + j * ldc]);
            } });
        q.wait();
        break;
    case 3:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
            int j = idx[0];
            int i = idx[1];
            double x = A[i + 0 * lda] - B[j + 0 * ldb];
            double y = A[i + 1 * lda] - B[j + 1 * ldb];
            double z = A[i + 2 * lda] - B[j + 2 * ldb];
            C[i + j * ldc] = x * x + y * y + z * z;
            for (int k = 0; k < ldc; k++) {
                C[k + j * ldc] = sqrt(C[k + j * ldc]);
            } });
        q.wait();
        break;
    }

    return QMCKL_SUCCESS_DEVICE;
}
/*
qmckl_exit_code_device qmckl_distance_rescaled_device(const qmckl_context_device context, const char transa, const char transb, const int64_t m, const int64_t n, const double *A, const int64_t lda, const double *B, const int64_t ldb, double *const C, const int64_t ldc, const double rescale_factor_kappa, queue &q)
{

    int i, j, transab;
    double rescale_factor_kappa_inv;

    rescale_factor_kappa_inv = 1.0 / rescale_factor_kappa;

    qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

    if (context == QMCKL_NULL_CONTEXT_DEVICE)
    {
        info = QMCKL_INVALID_CONTEXT_DEVICE;
        return info;
    }

    if (m <= 0)
    {
        info = QMCKL_INVALID_ARG_4_DEVICE;
        return info;
    }

    if (n <= 0)
    {
        info = QMCKL_INVALID_ARG_5_DEVICE;
        return info;
    }

    if (transa == 'N' || transa == 'n')
    {
        transab = 0;
    }
    else if (transa == 'T' || transa == 't')
    {
        transab = 1;
    }
    else
    {
        transab = -100;
    }

    if (transb == 'N' || transb == 'n')
    {
    }
    else if (transb == 'T' || transb == 't')
    {
        transab = transab + 2;
    }
    else
    {
        transab = -100;
    }

    // check for LDA
    if (transab < 0)
    {
        info = QMCKL_INVALID_ARG_1_DEVICE;
        return info;
    }

    if ((transab & 1) == 0 && lda < 3)
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if ((transab & 1) == 1 && lda < m)
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if ((transab & 2) == 0 && lda < 3)
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if ((transab & 2) == 2 && lda < m)
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    // check for LDB
    if ((transab & 1) == 0 && ldb < 3)
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 1) == 1 && ldb < n)
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 2) == 0 && ldb < 3)
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 2) == 2 && ldb < n)
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    // check for LDC
    if (ldc < m)
    {
        info = QMCKL_INVALID_ARG_11_DEVICE;
        return info;
    }

    switch (transab)
    {

    case 0:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
                int j = idx[0];
                int i = idx[1];
                double x = A[0 + i * lda] - B[0 + j * ldb];
                double y = A[1 + i * lda] - B[1 + j * ldb];
                double z = A[2 + i * lda] - B[2 + j * ldb];
                double dist = sqrt(x * x + y * y + z * z);
                C[i + j * ldc] = (1.0 - exp(-rescale_factor_kappa * dist)) * rescale_factor_kappa_inv; });
        q.wait();
        break;

    case 1:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
                int j = idx[0];
                int i = idx[1];
                double x = A[i + 0 * lda] - B[0 + j * ldb];
				double y = A[i + 1 * lda] - B[1 + j * ldb];
				double z = A[i + 2 * lda] - B[2 + j * ldb];
				double dist = sqrt(x * x + y * y + z * z);
				C[i + j * ldc] = (1.0 - exp(-rescale_factor_kappa * dist)) * rescale_factor_kappa_inv; })
            .wait();
        break;

    case 2:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
                int j = idx[0];
                int i = idx[1];
                double x = A[0 + i * lda] - B[j + 0 * ldb];
				double y = A[1 + i * lda] - B[j + 1 * ldb];
				double z = A[2 + i * lda] - B[j + 2 * ldb];
				double dist = sqrt(x * x + y * y + z * z);
				C[i + j * ldc] = (1.0 - exp(-rescale_factor_kappa * dist)) * rescale_factor_kappa_inv; });
        q.wait();
        break;

    case 3:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
                int j = idx[0];
                int i = idx[1];
                double x = A[i + 0 * lda] - B[j + 0 * ldb];
				double y = A[i + 1 * lda] - B[j + 1 * ldb];
				double z = A[i + 2 * lda] - B[j + 2 * ldb];
				double dist = sqrt(x * x + y * y + z * z);
				C[i + j * ldc] = (1.0 - exp(-rescale_factor_kappa * dist)) * rescale_factor_kappa_inv; });
        q.wait();
        break;
    }

    return info;
}

qmckl_exit_code_device qmckl_distance_rescaled_deriv_e_device(qmckl_context_device context, char transa, char transb, int m, int n, double *A, int lda, double *B, int ldb, double *C, int ldc, double rescale_factor_kappa)
{

    double rescale_factor_kappa_inv;
    int transab;

    rescale_factor_kappa_inv = 1.0 / rescale_factor_kappa;

    qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

    if (context == QMCKL_NULL_CONTEXT_DEVICE)
    {
        info = QMCKL_INVALID_CONTEXT_DEVICE;
        return info;
    }

    if (m <= 0)
    {
        info = QMCKL_INVALID_ARG_4_DEVICE;
        return info;
    }

    if (n <= 0)
    {
        info = QMCKL_INVALID_ARG_5_DEVICE;
        return info;
    }

    if (transa == 'N' || transa == 'n')
    {
        transab = 0;
    }
    else if (transa == 'T' || transa == 't')
    {
        transab = 1;
    }
    else
    {
        transab = -100;
    }

    if (transb == 'N' || transb == 'n')
    {
    }
    else if (transb == 'T' || transb == 't')
    {
        transab = transab + 2;
    }
    else
    {
        transab = -100;
    }

    // check for LDA
    if (transab < 0)
    {
        info = QMCKL_INVALID_ARG_1_DEVICE;
        return info;
    }

    if (((transab & 1) == 0) && (lda < 3))
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if (((transab & 1) == 1) && (lda < m))
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if (((transab & 2) == 0) && (lda < 3))
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    if (((transab & 2) == 2) && (lda < m))
    {
        info = QMCKL_INVALID_ARG_7_DEVICE;
        return info;
    }

    // check for LDB
    if ((transab & 1) == 0 && (ldb < 3))
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 1) == 1 && (ldb < n))
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 2) == 0 && (ldb < 3))
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    if ((transab & 2) == 2 && (ldb < n))
    {
        info = QMCKL_INVALID_ARG_9_DEVICE;
        return info;
    }

    // check for LDC
    if (ldc < m)
    {
        info = QMCKL_INVALID_ARG_11_DEVICE;
        return info;
    }

    queue q;
    switch (transab)
    {

    case 0:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
            int j = idx[0];
            int i = idx[1];
            double x = A[0 + i * lda] - B[0 + j * ldb];
            double y = A[1 + i * lda] - B[1 + j * ldb];
            double z = A[2 + i * lda] - B[2 + j * ldb];
            double dist = sqrt(x * x + y * y + z * z);
            // Avoid floating-point exception
            if (dist == 0.) {
                dist = 69./rescale_factor_kappa;
            }
            double dist_inv = 1.0 / dist;
            double rij = (1.0 - exp(-rescale_factor_kappa * dist)) * rescale_factor_kappa_inv;
            C[0 + i * 4 + j * 4 * ldc] = x * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[1 + i * 4 + j * 4 * ldc] = y * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[2 + i * 4 + j * 4 * ldc] = z * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[3 + i * 4 + j * 4 * ldc] = (2.0 * dist_inv - rescale_factor_kappa_inv) * (1.0 - rescale_factor_kappa_inv * rij); });
        q.wait();
        break;

    case 1:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
            int j = idx[0];
            int i = idx[1];
            double x = A[i + 0 * lda] - B[0 + j * ldb];
            double y = A[i + 1 * lda] - B[1 + j * ldb];
            double z = A[i + 2 * lda] - B[2 + j * ldb];
            double dist = sqrt(x * x + y * y + z * z);
            // Avoid floating-point exception
            if (dist == 0.) {
                dist = 69./rescale_factor_kappa;
            }
            double dist_inv = 1.0 / dist;
            double rij = (1.0 - exp(-rescale_factor_kappa * dist)) * rescale_factor_kappa_inv;
            C[0 + i + j] = x * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[1 + i + j] = y * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[2 + i + j] = z * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[3 + i + j] = (2.0 * dist_inv - rescale_factor_kappa_inv) * (1.0 - rescale_factor_kappa_inv * rij); });
        q.wait();
        break;

    case 2:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
            int j = idx[0];
            int i = idx[1];
            double x = A[0 + i * lda] - B[j + 0 * ldb];
            double y = A[1 + i * lda] - B[j + 1 * ldb];
            double z = A[2 + i * lda] - B[j + 2 * ldb];
            double dist = sqrt(x * x + y * y + z * z);
            // Avoid floating-point exception
            if (dist == 0.) {
                dist = 69./rescale_factor_kappa;
            }
            double dist_inv = 1.0 / dist;
            double rij = (1.0 - exp(-rescale_factor_kappa * dist)) * rescale_factor_kappa_inv;
            C[0 + i * 4 + j * 4 * ldc] = x * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[1 + i * 4 + j * 4 * ldc] = y * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[2 + i * 4 + j * 4 * ldc] = z * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[3 + i * 4 + j * ldc] = (2.0 * dist_inv - rescale_factor_kappa_inv) * (1.0 - rescale_factor_kappa_inv * rij); });
        q.wait();
        break;

    case 3:
        q.parallel_for(range<2>(n, m), [=](id<2> idx)
                       {
            int j = idx[0];
            int i = idx[1];
            double x = A[i + 0 * lda] - B[j + 0 * ldb];
            double y = A[i + 1 * lda] - B[j + 1 * ldb];
            double z = A[i + 2 * lda] - B[j + 2 * ldb];
            double dist = sqrt(x * x + y * y + z * z);
            // Avoid floating-point exception
            if (dist == 0.)
            {
                dist = 69. / rescale_factor_kappa;
            }
            double dist_inv = 1.0 / dist;
            double rij = (1.0 - exp(-rescale_factor_kappa * dist)) * rescale_factor_kappa_inv;
            C[0 + i * 4 + j * 4 * ldc] = x * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[1 + i * 4 + j * 4 * ldc] = y * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[2 + i * 4 + j * 4 * ldc] = z * dist_inv * (1.0 - rescale_factor_kappa_inv * rij);
            C[3 + i * 4 + j * 4 * ldc] = (2.0 * dist_inv - rescale_factor_kappa_inv) * (1.0 - rescale_factor_kappa_inv * rij); });
        q.wait();
        break;
    }
    return info;
}
*/