#include "../include/qmckl_jastrow.hpp"
#include "../include/qmckl_distance.hpp"

#include <CL/sycl.hpp>

using namespace sycl;
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
//**********
// COMPUTES
//**********

// Finalize computes
qmckl_exit_code_device qmckl_compute_jastrow_asymp_jasa_device(
	const qmckl_context_device context, const int64_t aord_num,
	const int64_t type_nucl_num, const double *a_vector,
	const double *rescale_factor_en, double *const asymp_jasa) {

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	int i, j, p;
	float kappa_inv, x, asym_one;
	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (aord_num < 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}
    sycl::buffer<float, 1> buffer_kappa_inv(&kappa_inv, range<1>(1));
    sycl::buffer<float, 1> buffer_x(&x, range<1>(1));
    
	q.submit([&](handler &h) {
		sycl::accessor acc_kappa_inv(buffer_kappa_inv, h, read_write);
		sycl::accessor acc_x(buffer_x, h, read_write);
		h.parallel_for(sycl::range<1>(type_nucl_num), [=](sycl::id<1> idx) {
			auto i = idx[0];
			
			acc_kappa_inv[0] = 1.0 / rescale_factor_en[i];

			asymp_jasa[i] = a_vector[0 + i * (aord_num + 1)] * acc_kappa_inv[0] /
						(1.0 + a_vector[1 + i * (aord_num + 1)] * acc_kappa_inv[0]);

			acc_x[0] = acc_kappa_inv[0];
			for (int p = 1; p < aord_num; p++) {
				acc_x[0] = acc_x[0] * acc_kappa_inv[0];
				asymp_jasa[i] =
					asymp_jasa[i] + a_vector[p + 1 + i * (aord_num + 1)] * acc_x[0];
			}
		});
	}).wait();
	
	buffer_kappa_inv.get_host_access();
	buffer_x.get_host_access();

	return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device qmckl_compute_jastrow_asymp_jasb_device(
	const qmckl_context_device context, const int64_t bord_num,
	const double *b_vector, const double rescale_factor_ee,
	double *const asymp_jasb) {
    
    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	double asym_one, x;
	double kappa_inv = 1.0 / rescale_factor_ee;

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (bord_num < 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

    sycl::buffer<double, 1> buffer_asym_one(&asym_one, range<1>(1));
    sycl::buffer<double, 1> buffer_x(&x, range<1>(1));
    sycl::buffer<double, 1> buffer_kappa_inv(&kappa_inv, range<1>(1));
	
    q.submit([&](handler &h) {
        sycl::accessor acc_asym_one(buffer_asym_one, h, read_write);
        sycl::accessor acc_kappa_inv(buffer_kappa_inv, h, read_write);
        h.single_task([=]() {
            acc_asym_one[0] = b_vector[0] * acc_kappa_inv[0] / (1.0 + b_vector[1] * acc_kappa_inv[0]);
            asymp_jasb[0] = acc_asym_one[0];
            asymp_jasb[1] = 0.5 * acc_asym_one[0];
        });
    }).wait();

    q.submit([&](handler &h) {
        sycl::accessor acc_x(buffer_x, h, read_write);
        sycl::accessor acc_kappa_inv(buffer_kappa_inv, h, read_write);
        h.parallel_for(sycl::range<1>(2), [=](sycl::id<1> idx) {
            auto i = idx[0];
            
            acc_x[0] = acc_kappa_inv[0];
            for (int p = 1; p < bord_num; p++) {
                acc_x[0] = acc_x[0] * acc_kappa_inv[0];
                asymp_jasb[i] = asymp_jasb[i] + b_vector[p + 1] * acc_x[0];
            }
        });
    }).wait();

	buffer_asym_one.get_host_access();
	buffer_x.get_host_access();
	buffer_kappa_inv.get_host_access();

	return QMCKL_SUCCESS_DEVICE;
}

// Total Jastrow
qmckl_exit_code_device
qmckl_compute_jastrow_value_device(const qmckl_context_device context,
								   const int64_t walk_num, const double *f_ee,
								   const double *f_en, const double *f_een,
								   double *const value) {

	if (context == QMCKL_NULL_CONTEXT_DEVICE)
		return QMCKL_INVALID_CONTEXT_DEVICE;
	if (walk_num <= 0)
		return QMCKL_INVALID_ARG_2_DEVICE;
	if (f_ee == NULL)
		return QMCKL_INVALID_ARG_3_DEVICE;
	if (f_en == NULL)
		return QMCKL_INVALID_ARG_4_DEVICE;
	if (f_een == NULL)
		return QMCKL_INVALID_ARG_5_DEVICE;
	if (value == NULL)
		return QMCKL_INVALID_ARG_6_DEVICE;

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	q.parallel_for(sycl::range<1>(walk_num), [=](sycl::id<1> idx) {
		auto i = idx[0];

		double arg = f_ee[i] + f_en[i] + f_een[i];
		if (arg < -100) {
			value[i] = 0;
		} else {
			value[i] = exp(arg);
		}
	}).wait();

	return QMCKL_SUCCESS_DEVICE;
}


qmckl_exit_code_device qmckl_compute_jastrow_gl_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const double *value, const double *gl_ee,
	const double *gl_en, const double *gl_een, double *const gl) {

	if (context == QMCKL_NULL_CONTEXT_DEVICE)
		return QMCKL_INVALID_CONTEXT_DEVICE;
	if (walk_num <= 0)
		return QMCKL_INVALID_ARG_2_DEVICE;
	if (elec_num <= 0)
		return QMCKL_INVALID_ARG_3_DEVICE;
	if (value == NULL)
		return QMCKL_INVALID_ARG_4_DEVICE;
	if (gl_ee == NULL)
		return QMCKL_INVALID_ARG_5_DEVICE;
	if (gl_en == NULL)
		return QMCKL_INVALID_ARG_6_DEVICE;
	if (gl_een == NULL)
		return QMCKL_INVALID_ARG_7_DEVICE;
	if (gl == NULL)
		return QMCKL_INVALID_ARG_8_DEVICE;
    
    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;
    
	q.parallel_for(sycl::range<1>(walk_num), [=](sycl::id<1> idx) {
		auto k = idx[0];

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < elec_num; i++) {
				gl[i + j * elec_num + k * elec_num * 4] =
					gl_ee[i + j * elec_num + k * elec_num * 4] +
					gl_en[i + j * elec_num + k * elec_num * 4] +
					gl_een[i + j * elec_num + k * elec_num * 4];
			}
		}

		for (int i = 0; i < elec_num; i++) {
			gl[i + 3 * elec_num + k * elec_num * 4] =
				gl[i + 3 * elec_num + k * elec_num * 4] +
				gl[i + 0 * elec_num + k * elec_num * 4] *
					gl[i + 0 * elec_num + k * elec_num * 4] +
				gl[i + 1 * elec_num + k * elec_num * 4] *
					gl[i + 1 * elec_num + k * elec_num * 4] +
				gl[i + 2 * elec_num + k * elec_num * 4] *
					gl[i + 2 * elec_num + k * elec_num * 4];
		}

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < elec_num; i++) {
				gl[i + j * elec_num + k * elec_num * 4] =
					gl[i + j * elec_num + k * elec_num * 4] * value[k];
			}
		}
	}).wait();

	return QMCKL_SUCCESS_DEVICE;
}


// Electron/electron component
qmckl_exit_code_device qmckl_compute_jastrow_factor_ee_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t up_num, const int64_t bord_num,
	const double *b_vector, const double *ee_distance_rescaled,
	const double *asymp_jasb, double *const factor_ee) {

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (walk_num <= 0) {
		return QMCKL_INVALID_ARG_2_DEVICE;
	}

	if (elec_num <= 0) {
		return QMCKL_INVALID_ARG_3_DEVICE;
	}

	if (bord_num < 0) {
		return QMCKL_INVALID_ARG_4_DEVICE;
	}

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;
    
	q.parallel_for(sycl::range<1>(walk_num), [=](sycl::id<1> idx) {
		auto nw = idx[0];
		
		factor_ee[nw] = 0.0; // put init array here.
		size_t ishift = nw * elec_num * elec_num;
		for (int i = 0; i < elec_num; ++i) {
			for (int j = 0; j < i; ++j) {
				double x = ee_distance_rescaled[j + i * elec_num + ishift];
				const double x1 = x;
				double power_ser = 0.0;
				double spin_fact = 1.0;
				int ipar = 0; // index of asymp_jasb

				for (int p = 1; p < bord_num; ++p) {
					x = x * x1;
					power_ser += b_vector[p + 1] * x;
				}

				if (i < up_num || j >= up_num) {
					spin_fact = 0.5;
					ipar = 1;
				}

				factor_ee[nw] +=
					spin_fact * b_vector[0] * x1 / (1.0 + b_vector[1] * x1) -
					asymp_jasb[ipar] + power_ser;
			}
		}          
	}).wait();

	return QMCKL_SUCCESS_DEVICE;
}


qmckl_exit_code_device qmckl_compute_jastrow_factor_ee_deriv_e_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t up_num, const int64_t bord_num,
	const double *b_vector, const double *ee_distance_rescaled,
	const double *ee_distance_rescaled_deriv_e,
	double *const factor_ee_deriv_e) {

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (walk_num <= 0) {
		return QMCKL_INVALID_ARG_2_DEVICE;
	}

	if (elec_num <= 0) {
		return QMCKL_INVALID_ARG_3_DEVICE;
	}

	if (bord_num < 0) {
		return QMCKL_INVALID_ARG_4_DEVICE;
	}

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;
    
	q.parallel_for(sycl::range<3>(walk_num, 4, elec_num), [=](sycl::id<3> idx) {
		auto nw = idx[0];
		auto ii = idx[1];
		auto j = idx[2];
		factor_ee_deriv_e[j + ii * elec_num + nw * elec_num * 4] = 0.0;
	}).wait();

	const double third = 1.0 / 3.0;
    
	q.parallel_for(sycl::range<3>(walk_num, elec_num, elec_num), [=](sycl::id<3> idx) {
		auto nw = idx[0];
		auto i = idx[1];
		auto j = idx[2];
		const double x0 =
					ee_distance_rescaled[j + i * elec_num +
											nw * elec_num * elec_num];
		for (int i = 0; i < 0; i++) {
			if (fabs(x0) < 1.0e-18)
				continue;
		}	
		
		double spin_fact = 1.0;
		const double den = 1.0 + b_vector[1] * x0;
		const double invden = 1.0 / den;
		const double invden2 = invden * invden;
		const double invden3 = invden2 * invden;
		const double xinv = 1.0 / (x0 + 1.0e-18);

		double dx[4];
		dx[0] = ee_distance_rescaled_deriv_e[0 + j * 4 +
												i * 4 * elec_num +
												nw * 4 * elec_num *
													elec_num];
		dx[1] = ee_distance_rescaled_deriv_e[1 + j * 4 +
												i * 4 * elec_num +
												nw * 4 * elec_num *
													elec_num];
		dx[2] = ee_distance_rescaled_deriv_e[2 + j * 4 +
												i * 4 * elec_num +
												nw * 4 * elec_num *
													elec_num];
		dx[3] = ee_distance_rescaled_deriv_e[3 + j * 4 +
												i * 4 * elec_num +
												nw * 4 * elec_num *
													elec_num];

		if ((i <= (up_num - 1) && j <= (up_num - 1)) ||
			(i > (up_num - 1) && j > (up_num - 1))) {
			spin_fact = 0.5;
		}

		double lap1 = 0.0;
		double lap2 = 0.0;
		double lap3 = 0.0;
		double pow_ser_g[3] = {0., 0., 0.};
		for (int ii = 0; ii < 3; ++ii) {
			double x = x0;
			if (fabs(x) < 1.0e-18)
				continue;
			for (int p = 2; p < bord_num + 1; ++p) {
				const double y = p * b_vector[(p - 1) + 1] * x;
				pow_ser_g[ii] = pow_ser_g[ii] + y * dx[ii];
				lap1 = lap1 + (p - 1) * y * xinv * dx[ii] * dx[ii];
				lap2 = lap2 + y;
				x = x *
					ee_distance_rescaled[j + i * elec_num +
											nw * elec_num * elec_num];
			}

			lap3 = lap3 - 2.0 * b_vector[1] * dx[ii] * dx[ii];

			factor_ee_deriv_e[i + ii * elec_num +
								nw * elec_num * 4] +=
				+spin_fact * b_vector[0] * dx[ii] * invden2 +
				pow_ser_g[ii];
		}

		int ii = 3;
		lap2 = lap2 * dx[ii] * third;
		lap3 = lap3 + den * dx[ii];
		lap3 = lap3 * (spin_fact * b_vector[0] * invden3);
		factor_ee_deriv_e[i + ii * elec_num + nw * elec_num * 4] +=
			lap1 + lap2 + lap3;       
	}).wait();

	return QMCKL_SUCCESS_DEVICE;
}


// Electron/nucleus component
qmckl_exit_code_device qmckl_compute_jastrow_factor_en_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t nucl_num, const int64_t type_nucl_num,
	const int64_t *type_nucl_vector, const int64_t aord_num,
	const double *a_vector, const double *en_distance_rescaled,
	const double *asymp_jasa, double *const factor_en) {

	int i, a, p, nw;
	double x, power_ser;
	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (walk_num <= 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (elec_num <= 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	if (nucl_num <= 0) {
		info = QMCKL_INVALID_ARG_4_DEVICE;
		return info;
	}

	if (type_nucl_num <= 0) {
		info = QMCKL_INVALID_ARG_4_DEVICE;
		return info;
	}

	if (aord_num < 0) {
		info = QMCKL_INVALID_ARG_7_DEVICE;
		return info;
	}
    
    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

    sycl::buffer<int, 1> buff_i(&i, range<1>(1));
    sycl::buffer<int, 1> buff_a(&a, range<1>(1));
    sycl::buffer<int, 1> buff_p(&p, range<1>(1));
    sycl::buffer<int, 1> buff_nw(&nw, range<1>(1));
    sycl::buffer<double, 1> buff_x(&x, range<1>(1));
	
    q.submit([&](handler &h) {
        sycl::accessor acc_i(buff_i, h, read_write);
        sycl::accessor acc_a(buff_a, h, read_write);
        sycl::accessor acc_p(buff_p, h, read_write);
        sycl::accessor acc_nw(buff_nw, h, read_write);
        sycl::accessor acc_x(buff_x, h, read_write);
        
        h.parallel_for(sycl::range<1>(walk_num), [=](sycl::id<1> idx) {
            acc_nw[0] = idx[0];
            
            factor_en[acc_nw[0]] = 0.0;
            for (acc_a[0] = 0; acc_a[0] < nucl_num; acc_a[0]++) {
                for (acc_i[0] = 0; acc_i[0] < elec_num; acc_i[0]++) {
                    acc_x[0] = en_distance_rescaled[acc_i[0] + acc_a[0] * elec_num +
                                                acc_nw[0] * elec_num * nucl_num];

                    factor_en[acc_nw[0]] =
                        factor_en[acc_nw[0]] +
                        a_vector[0 +
                                    type_nucl_vector[acc_a[0]] * (aord_num + 1)] *
                            acc_x[0] /
                            (1.0 + a_vector[1 + type_nucl_vector[acc_a[0]] *
                                                    (aord_num + 1)] *
                                        acc_x[0]) -
                        asymp_jasa[type_nucl_vector[acc_a[0]]];

                    for (acc_p[0] = 1; acc_p[0] < aord_num; acc_p[0]++) {
                        acc_x[0] = acc_x[0] * en_distance_rescaled[acc_i[0] + acc_a[0] * elec_num +
                                                        acc_nw[0] * elec_num * nucl_num];
                        factor_en[acc_nw[0]] =
                            factor_en[acc_nw[0]] + a_vector[acc_p[0] + 1 +
                                                        type_nucl_vector[acc_a[0]] *
                                                            (aord_num + 1)] *
                                                acc_x[0];
                    }
                }
            }
        });
    }).wait();

	buff_i.get_host_access();
	buff_a.get_host_access();
	buff_p.get_host_access();
	buff_nw.get_host_access();
	buff_x.get_host_access();

	return info;
}


qmckl_exit_code_device qmckl_compute_jastrow_factor_en_deriv_e_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t nucl_num, const int64_t type_nucl_num,
	const int64_t *type_nucl_vector, const int64_t aord_num,
	const double *a_vector, const double *en_distance_rescaled,
	const double *en_distance_rescaled_deriv_e,
	double *const factor_en_deriv_e) {

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (walk_num <= 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (elec_num <= 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	if (nucl_num <= 0) {
		info = QMCKL_INVALID_ARG_4_DEVICE;
		return info;
	}

	if (aord_num < 0) {
		info = QMCKL_INVALID_ARG_7_DEVICE;
		return info;
	}

	int i, a, p, ipar, nw, ii;
	double x, den, invden, invden2, invden3, xinv;
	double y, lap1, lap2, lap3, third;

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	q.parallel_for(sycl::range<1>(elec_num * 4 * walk_num), [=](sycl::id<1> idx) {
		auto i = idx[0];

		factor_en_deriv_e[i] = 0.0;         
	}).wait();

	third = 1.0 / 3.0;

    sycl::buffer<int, 1> buff_i(&i, range<1>(1));
    sycl::buffer<int, 1> buff_a(&a, range<1>(1));
    sycl::buffer<int, 1> buff_p(&p, range<1>(1));
    sycl::buffer<int, 1> buff_nw(&nw, range<1>(1));
    sycl::buffer<int, 1> buff_ii(&ii, range<1>(1));
    sycl::buffer<double, 1> buff_x(&x, range<1>(1));
    sycl::buffer<double, 1> buff_den(&den, range<1>(1));
    sycl::buffer<double, 1> buff_invden(&invden, range<1>(1));
    sycl::buffer<double, 1> buff_invden2(&invden2, range<1>(1));
    sycl::buffer<double, 1> buff_invden3(&invden3, range<1>(1));
    sycl::buffer<double, 1> buff_xinv(&xinv, range<1>(1));
    sycl::buffer<double, 1> buff_y(&y, range<1>(1));
    sycl::buffer<double, 1> buff_lap1(&lap1, range<1>(1));
    sycl::buffer<double, 1> buff_lap2(&lap2, range<1>(1));
    sycl::buffer<double, 1> buff_lap3(&lap3, range<1>(1));
    sycl::buffer<double, 1> buff_third(&third, range<1>(1));
    q.submit([&](handler &h) {
        sycl::accessor acc_a(buff_a, h, read_write);
        sycl::accessor acc_nw(buff_nw, h, read_write);
        sycl::accessor acc_x(buff_x, h, read_write);
        sycl::accessor acc_i(buff_i, h, read_write);
        sycl::accessor acc_p(buff_p, h, read_write);
        sycl::accessor acc_ii(buff_ii, h, read_write);
        sycl::accessor acc_den(buff_den, h, read_write);
        sycl::accessor acc_invden(buff_invden, h, read_write);
        sycl::accessor acc_invden2(buff_invden2, h, read_write);
        sycl::accessor acc_invden3(buff_invden3, h, read_write);
        sycl::accessor acc_xinv(buff_xinv, h, read_write);
        sycl::accessor acc_y(buff_y, h, read_write);
        sycl::accessor acc_lap1(buff_lap1, h, read_write);
        sycl::accessor acc_lap2(buff_lap2, h, read_write);
        sycl::accessor acc_lap3(buff_lap3, h, read_write);
        sycl::accessor acc_third(buff_third, h, read_write);
        q.parallel_for(sycl::range<3>(walk_num, nucl_num, elec_num), [=](sycl::id<3> idx) {
            acc_nw[0] = idx[0];
            acc_a[0] = idx[1];
            acc_i[0] = idx[2]; 

            double power_ser_g[3];
            double dx[4];

            acc_x[0] = en_distance_rescaled[acc_i[0] + acc_a[0] * elec_num +
                                        acc_nw[0] * elec_num * nucl_num];
            for (int z = 0; z < 0; z++) {
                if (fabs(acc_x[0]) < 1.0e-18) {
                    continue;
                }
            }
            power_ser_g[0] = 0.0;
            power_ser_g[1] = 0.0;
            power_ser_g[2] = 0.0;
            acc_den[0] =
                1.0 +
                a_vector[1 + (type_nucl_vector[acc_a[0]]+1) * (aord_num + 1)] * x;
            acc_invden[0] = 1.0 / den;
            acc_invden2[0] = acc_invden[0] * acc_invden[0];
            acc_invden3[0] = acc_invden2[0] * invden;
            acc_xinv[0] = 1.0 / acc_x[0];

            for (acc_ii[0] = 0; acc_ii[0] < 4; acc_ii[0]++) {
                dx[acc_ii[0]] =
                    en_distance_rescaled_deriv_e[acc_ii[0] + acc_i[0] * 4 +
                                                    acc_a[0] * 4 * elec_num +
                                                    acc_nw[0] * 4 * elec_num *
                                                        nucl_num];
            }

            acc_lap1[0] = 0.0;
            acc_lap2[0] = 0.0;
            acc_lap3[0] = 0.0;
            for (acc_ii[0] = 0; acc_ii[0] < 3; acc_ii[0]++) {
                acc_x[0] = en_distance_rescaled[acc_i[0] + acc_a[0] * elec_num +
                                            acc_nw[0] * elec_num * nucl_num];

                for (acc_p[0] = 1; acc_p[0] < aord_num; acc_p[0]++) {
                    acc_y[0] = (acc_p[0] + 1) *
                        a_vector[(acc_p[0] + 1) + type_nucl_vector[acc_a[0]] *
                                                (aord_num + 1)] *
                        acc_x[0];
                    power_ser_g[acc_ii[0]] = power_ser_g[acc_ii[0]] + acc_y[0] * dx[acc_ii[0]];
                    acc_lap1[0] = acc_lap1[0] + acc_p[0] * acc_y[0] * acc_xinv[0] * dx[acc_ii[0]] * dx[acc_ii[0]];
                    acc_lap2[0] = acc_lap2[0] + acc_y[0];
                    acc_x[0] = acc_x[0] *
                        en_distance_rescaled[acc_i[0] + acc_a[0] * elec_num +
                                                acc_nw[0] * elec_num * nucl_num];
                }

                acc_lap3[0] =
                    acc_lap3[0] - 2.0 *
                                a_vector[1 + type_nucl_vector[acc_a[0]] *
                                                (aord_num + 1)] *
                                dx[acc_ii[0]] * dx[acc_ii[0]];

                factor_en_deriv_e[acc_i[0] + acc_ii[0] * elec_num +
                                    acc_nw[0] * elec_num * 4] =
                    factor_en_deriv_e[acc_i[0] + acc_ii[0] * elec_num +
                                        acc_nw[0] * elec_num * 4] +
                    a_vector[0 + type_nucl_vector[acc_a[0]] *
                                        (aord_num + 1)] *
                        dx[acc_ii[0]] * acc_invden2[0] +
                    power_ser_g[acc_ii[0]];
            }

            acc_ii[0] = 3;
            acc_lap2[0] = acc_lap2[0] * dx[acc_ii[0]] * acc_third[0];
            acc_lap3[0] = acc_lap3[0] + acc_den[0] * dx[acc_ii[0]];
            acc_lap3[0] = acc_lap3[0] *
                    a_vector[0 + type_nucl_vector[acc_a[0]] *
                                    (aord_num + 1)] *
                    acc_invden3[0];
            factor_en_deriv_e[acc_i[0] + acc_ii[0] * elec_num + acc_nw[0] * elec_num * 4] =
                factor_en_deriv_e[acc_i[0] + acc_ii[0] * elec_num +
                                    acc_nw[0] * elec_num * 4] +
                acc_lap1[0] + acc_lap2[0] + acc_lap3[0];
        });
    }).wait();

	buff_i.get_host_access();
	buff_a.get_host_access();
	buff_p.get_host_access();
	buff_nw.get_host_access();
	buff_x.get_host_access();
	buff_ii.get_host_access();
	buff_den.get_host_access();
	buff_invden.get_host_access();
	buff_invden2.get_host_access();
	buff_invden3.get_host_access();
	buff_xinv.get_host_access();
	buff_y.get_host_access();
	buff_lap1.get_host_access();
	buff_lap2.get_host_access();
	buff_lap3.get_host_access();
	buff_third.get_host_access();

	return info;
}


qmckl_exit_code_device qmckl_compute_en_distance_rescaled_deriv_e_device(
	const qmckl_context_device context, const int64_t elec_num,
	const int64_t nucl_num, const int64_t type_nucl_num,
	int64_t *const type_nucl_vector, const double *rescale_factor_en,
	const int64_t walk_num, const double *elec_coord, const double *nucl_coord,
	double *const en_distance_rescaled_deriv_e) {
	int i, k;
	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	int64_t *type_nucl_vector_h = reinterpret_cast<int64_t *>(malloc(nucl_num * sizeof(int64_t)));
	qmckl_memcpy_D2H(context, type_nucl_vector_h, type_nucl_vector,
					 nucl_num * sizeof(int64_t));

	double *rescale_factor_en_h = reinterpret_cast<double *>(malloc(nucl_num * sizeof(int64_t)));
	qmckl_memcpy_D2H(context, reinterpret_cast<void *const>(const_cast<double *>(rescale_factor_en_h)), reinterpret_cast<void *const>(const_cast<double *>(rescale_factor_en)),
					 type_nucl_num * sizeof(double));

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (elec_num <= 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (nucl_num <= 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	if (walk_num <= 0) {
		info = QMCKL_INVALID_ARG_5_DEVICE;
		return info;
	}

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	double *coord = reinterpret_cast<double *>(qmckl_malloc_device(context, 3 * sizeof(double)));
	
    for (int i = 0; i < nucl_num; i++) {
        q.single_task([=]() {
            coord[0] = nucl_coord[i + nucl_num * 0];
			coord[1] = nucl_coord[i + nucl_num * 1];
			coord[2] = nucl_coord[i + nucl_num * 2];
	    });
		for (k = 0; k < walk_num; k++) {
			info = qmckl_distance_rescaled_deriv_e_device(
				context, 'T', 'T', elec_num, 1, const_cast<double *>(elec_coord + (k * elec_num)),
				elec_num * walk_num, coord, 1,
				en_distance_rescaled_deriv_e + (0 + 0 * 4 + i * 4 * elec_num +
												k * 4 * elec_num * nucl_num),
				elec_num, rescale_factor_en_h[type_nucl_vector_h[i]]);
			if (info != QMCKL_SUCCESS_DEVICE) {
				qmckl_free_device(context, coord);
				return info;
			}
		}
	}

	free(type_nucl_vector_h);
	free(rescale_factor_en_h);
	qmckl_free_device(context, coord);

	return info;
}


qmckl_exit_code_device qmckl_compute_jastrow_champ_factor_en_deriv_e(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t nucl_num, const int64_t type_nucl_num,
	const int64_t *type_nucl_vector, const int64_t aord_num,
	const double *a_vector, const double *en_distance_rescaled,
	const double *en_distance_rescaled_deriv_e,
	double *const factor_en_deriv_e) {

	int i, a, p, ipar, nw, ii;
	double x, den, invden, invden2, invden3, xinv;
	double y, lap1, lap2, lap3, third;

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (walk_num <= 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (elec_num <= 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	if (nucl_num <= 0) {
		info = QMCKL_INVALID_ARG_4_DEVICE;
		return info;
	}

	if (aord_num < 0) {
		info = QMCKL_INVALID_ARG_7_DEVICE;
		return info;
	}

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	sycl::buffer<int, 1> buff_i(&i, range<1>(1));
	sycl::buffer<int, 1> buff_a(&a, range<1>(1));
	sycl::buffer<int, 1> buff_nw(&nw, range<1>(1));
	sycl::buffer<int, 1> buff_ii(&a, range<1>(1));
	sycl::buffer<double, 1> buff_x(&x, range<1>(1));
	sycl::buffer<double, 1> buff_den(&den, range<1>(1));
	sycl::buffer<double, 1> buff_invden(&invden, range<1>(1));
	sycl::buffer<double, 1> buff_invden2(&invden2, range<1>(1));
	sycl::buffer<double, 1> buff_invden3(&invden3, range<1>(1));
	sycl::buffer<double, 1> buff_xinv(&xinv, range<1>(1));
	sycl::buffer<double, 1> buff_y(&y, range<1>(1));
	sycl::buffer<double, 1> buff_lap1(&lap1, range<1>(1));
	sycl::buffer<double, 1> buff_lap2(&lap2, range<1>(1));
	sycl::buffer<double, 1> buff_lap3(&lap3, range<1>(1));
	sycl::buffer<double, 1> buff_third(&third, range<1>(1));
	q.submit([&](handler &h) {
		sycl::accessor acc_i(buff_i, h, read_write);
		q.parallel_for(sycl::range<1>(elec_num * 4 * walk_num), [=](sycl::id<1> idx) {
			acc_i[0] = idx[0];

			factor_en_deriv_e[i] = 0.0;
		});
	}).wait();

	third = 1.0 / 3.0;

	q.submit([&](handler &h) {
		sycl::accessor acc_i(buff_i, h, read_write);
		sycl::accessor acc_a(buff_a, h, read_write);
		sycl::accessor acc_nw(buff_nw, h, read_write);
		sycl::accessor acc_ii(buff_ii, h, read_write);
		sycl::accessor acc_x(buff_x, h, read_write);
		sycl::accessor acc_den(buff_den, h, read_write);
		sycl::accessor acc_invden(buff_invden, h, read_write);
		sycl::accessor acc_invden2(buff_invden2, h, read_write);
		sycl::accessor acc_invden3(buff_invden3, h, read_write);
		sycl::accessor acc_xinv(buff_xinv, h, read_write);
		sycl::accessor acc_y(buff_y, h, read_write);
		sycl::accessor acc_lap1(buff_lap1, h, read_write);
		sycl::accessor acc_lap2(buff_lap2, h, read_write);
		sycl::accessor acc_lap3(buff_lap3, h, read_write);
		sycl::accessor acc_third(buff_third, h, read_write);
		q.parallel_for(sycl::range<3>(walk_num, nucl_num, elec_num), [=](sycl::id<3> idx) {
			acc_nw[0] = idx[0];
			acc_a[0] = idx[1];
			acc_i[0] = idx[2];

			double power_ser_g[3];
			double dx[4];

			acc_x[0] = en_distance_rescaled[acc_i[0] + acc_a[0] * elec_num +
									acc_nw[0] * elec_num * nucl_num];
			for (int z = 0; z < 0; z++) {
				if (fabs(acc_x[0]) < 1.0e-18)
					continue;
			}
			power_ser_g[0] = 0.0;
			power_ser_g[1] = 0.0;
			power_ser_g[2] = 0.0;
			acc_den[0] =
				1.0 +
				a_vector[1 + (type_nucl_vector[acc_a[0]]+1) * (aord_num + 1)] * x;
			acc_invden[0] = 1.0 / acc_den[0];
			acc_invden2[0] = acc_invden[0] * acc_invden[0];
			acc_invden3[0] = acc_invden2[0] * invden;
			acc_xinv[0] = 1.0 / acc_x[0];

			for (acc_ii[0] = 0; acc_ii[0] < 4; acc_ii[0]++) {
				dx[acc_ii[0]] =
					en_distance_rescaled_deriv_e[acc_ii[0] + acc_i[0] * 4 +
												acc_a[0] * 4 * elec_num +
												acc_nw[0] * 4 * elec_num *
													nucl_num];
			}

			acc_lap1[0] = 0.0;
			acc_lap2[0] = 0.0;
			acc_lap3[0] = 0.0;
			for (acc_ii[0] = 0; acc_ii[0] < 3; acc_ii[0]++) {
				acc_x[0] = en_distance_rescaled[acc_i[0] + acc_a[0] * elec_num +
										acc_nw[0] * elec_num * nucl_num];
				for (int p = 1; p < aord_num; p++) {
					acc_y[0] = p *
						a_vector[p +
								(type_nucl_vector[acc_a[0]]+1) * (aord_num + 1)] *
						acc_x[0];
					power_ser_g[acc_ii[0]] = power_ser_g[acc_ii[0]] + acc_y[0] * dx[acc_ii[0]];
					acc_lap1[0] = acc_lap1[0] + (p - 1) * acc_y[0] * acc_xinv[0] * dx[acc_ii[0]] * dx[acc_ii[0]];
					acc_lap2[0] = acc_lap2[0] + acc_y[0];
					acc_x[0] = acc_x[0] *
						en_distance_rescaled[acc_i[0] + acc_a[0] * elec_num +
											acc_nw[0] * elec_num * nucl_num];
				}

				acc_lap3[0] = acc_lap3[0] - 2.0 *
								a_vector[1 + (type_nucl_vector[acc_a[0]]+1) *
												(aord_num + 1)] *
								dx[acc_ii[0]] * dx[acc_ii[0]];

				factor_en_deriv_e[i + acc_ii[0] * elec_num +
								acc_nw[0] * elec_num * 4] =
					factor_en_deriv_e[acc_i[0] + acc_ii[0] * elec_num * +acc_nw[0] *
											elec_num * 4] +
					a_vector[0 + (type_nucl_vector[a]+1) * (aord_num + 1)] *
						dx[acc_ii[0]] * acc_invden2[0] +
					power_ser_g[acc_ii[0]];
			}

			acc_ii[0] = 3;
			acc_lap2[0] = acc_lap2[0] * dx[acc_ii[0]] * acc_third[0];
			acc_lap3[0] = acc_lap3[0] + acc_den[0] * dx[acc_ii[0]];
			acc_lap3[0] = acc_lap3[0] *
				a_vector[0 + (type_nucl_vector[acc_a[0]]+1) * (aord_num + 1)] *
				acc_invden3[0];
			factor_en_deriv_e[acc_i[0] + acc_ii[0] * elec_num + acc_nw[0] * elec_num * 4] =
				factor_en_deriv_e[acc_i[0] + acc_ii[0] * elec_num +
								acc_nw[0] * elec_num * 4] +
				acc_lap1[0] + acc_lap2[0] + acc_lap3[0];       
		});
	}).wait();

	buff_i.get_host_access();
	buff_a.get_host_access();
	buff_nw.get_host_access();
	buff_ii.get_host_access();
	buff_x.get_host_access();
	buff_den.get_host_access();
	buff_invden.get_host_access();
	buff_invden2.get_host_access();
	buff_invden3.get_host_access();
	buff_xinv.get_host_access();
	buff_y.get_host_access();
	buff_lap1.get_host_access();
	buff_lap2.get_host_access();
	buff_lap3.get_host_access();
	buff_third.get_host_access();

	return info;
}

// Electron/electron/nucleus component
qmckl_exit_code_device qmckl_compute_jastrow_factor_een_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t nucl_num, const int64_t cord_num,
	const int64_t dim_c_vector, const double *c_vector_full,
	const int64_t *lkpm_combined_index, const double *tmp_c,
	const double *een_rescaled_n, double *const factor_een) {

	int i, a, j, l, k, p, m, n, nw;
	double accu, accu2, cn;

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (walk_num <= 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (elec_num <= 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	if (nucl_num <= 0) {
		info = QMCKL_INVALID_ARG_4_DEVICE;
		return info;
	}

	if (cord_num < 0) {
		info = QMCKL_INVALID_ARG_5_DEVICE;
		return info;
	}

	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	sycl::buffer<int, 1> buff_a(&a, range<1>(1));
	sycl::buffer<int, 1> buff_l(&l, range<1>(1));
	sycl::buffer<int, 1> buff_k(&k, range<1>(1));
	sycl::buffer<int, 1> buff_p(&p, range<1>(1));
	sycl::buffer<int, 1> buff_m(&m, range<1>(1));
	sycl::buffer<int, 1> buff_n(&n, range<1>(1));
	sycl::buffer<int, 1> buff_nw(&nw, range<1>(1));
	sycl::buffer<double, 1> buff_accu(&accu, range<1>(1));
	sycl::buffer<double, 1> buff_cn(&cn, range<1>(1));
	q.submit([&](handler &h) {
		sycl::accessor acc_a(buff_a, h, read_write);
		sycl::accessor acc_l(buff_l, h, read_write);
		sycl::accessor acc_k(buff_k, h, read_write);
		sycl::accessor acc_p(buff_p, h, read_write);
		sycl::accessor acc_m(buff_m, h, read_write);
		sycl::accessor acc_n(buff_n, h, read_write);
		sycl::accessor acc_nw(buff_nw, h, read_write);
		sycl::accessor acc_accu(buff_accu, h, read_write);
		sycl::accessor acc_cn(buff_cn, h, read_write);

		q.parallel_for(sycl::range<1>(walk_num), [=](sycl::id<1> idx) {
			acc_nw[0] = idx[0];
			factor_een[acc_nw[0]] = 0.0;
			for (acc_n[0] = 0; acc_n[0] < dim_c_vector; acc_n[0]++) {
				acc_l[0] = lkpm_combined_index[acc_n[0]];
				acc_k[0] = lkpm_combined_index[acc_n[0] + dim_c_vector];
				acc_p[0] = lkpm_combined_index[acc_n[0] + 2 * dim_c_vector];
				acc_m[0] = lkpm_combined_index[acc_n[0] + 3 * dim_c_vector];

				for (acc_a[0] = 0; acc_a[0] < nucl_num; acc_a[0]++) {
					acc_cn[0] = c_vector_full[acc_a[0] + acc_n[0] * nucl_num];
					if (acc_cn[0] == 0.0)
						continue;

					acc_accu[0] = 0.0;
					for (int j = 0; j < elec_num; j++) {
						acc_accu[0] =
							acc_accu[0] +
							een_rescaled_n[j + acc_a[0] * elec_num +
											acc_m[0] * elec_num * nucl_num +
											acc_nw[0] * elec_num * nucl_num *
												(cord_num + 1)] *
								tmp_c[j + acc_a[0] * elec_num +
										(acc_m[0] + acc_l[0]) * elec_num * nucl_num +
										acc_k[0] * elec_num * nucl_num * (cord_num + 1) +
										acc_nw[0] * elec_num * nucl_num *
											(cord_num + 1) * cord_num];
					}
					factor_een[acc_nw[0]] = factor_een[acc_nw[0]] + acc_accu[0] * acc_cn[0];
				}
			}
		});
	}).wait();

	buff_a.get_host_access();
	buff_l.get_host_access();
	buff_k.get_host_access();
	buff_p.get_host_access();
	buff_m.get_host_access();
	buff_n.get_host_access();
	buff_nw.get_host_access();
	buff_accu.get_host_access();
	buff_cn.get_host_access();

	return info;
}


qmckl_exit_code_device qmckl_compute_jastrow_factor_een_deriv_e_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t nucl_num, const int64_t cord_num,
	const int64_t dim_c_vector, const double *c_vector_full,
	const int64_t *lkpm_combined_index, const double *tmp_c,
	const double *dtmp_c, const double *een_rescaled_n,
	const double *een_rescaled_n_deriv_e, double *const factor_een_deriv_e) {

	int64_t info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE)
		return QMCKL_INVALID_CONTEXT_DEVICE;
	if (walk_num <= 0)
		return QMCKL_INVALID_ARG_2_DEVICE;
	if (elec_num <= 0)
		return QMCKL_INVALID_ARG_3_DEVICE;
	if (nucl_num <= 0)
		return QMCKL_INVALID_ARG_4_DEVICE;
	if (cord_num < 0)
		return QMCKL_INVALID_ARG_5_DEVICE;

	double *tmp3 = reinterpret_cast<double *>(qmckl_malloc_device(context, elec_num * sizeof(double)));
	
	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

    
	q.parallel_for(sycl::range<1>(elec_num * 4 * walk_num), [=](sycl::id<1> idx) {
		auto i = idx[0];
		factor_een_deriv_e[i] = 0.;
	}).wait();
	
	const size_t elec_num2 = elec_num << 1;
	const size_t elec_num3 = elec_num * 3;

	sycl::buffer<size_t, 1> buff_elec_num2(&elec_num2, range<1>(1));
	sycl::buffer<size_t, 1> buff_elec_num3(&elec_num3, range<1>(1));
    
	q.submit([&](handler &h) {
		sycl::accessor acc_elec_num2(buff_elec_num2, h, read_only);
		sycl::accessor acc_elec_num3(buff_elec_num3, h, read_only);
		h.parallel_for(sycl::range<1>((size_t)walk_num), [=](sycl::id<1> idx) {
			auto nw = idx[0];

			double *const __restrict__ factor_een_deriv_e_0nw =
				&(factor_een_deriv_e[elec_num * 4 * nw]);
			for (size_t n = 0; n < (size_t)dim_c_vector; ++n) {
				const size_t l = lkpm_combined_index[n];
				const size_t k = lkpm_combined_index[n + dim_c_vector];
				const size_t m = lkpm_combined_index[n + 3 * dim_c_vector];

				const size_t en = elec_num * nucl_num;
				const size_t len = l * en;
				const size_t len4 = len << 2;
				const size_t cn = cord_num * nw;
				const size_t c1 = cord_num + 1;
				const size_t addr0 = en * (m + c1 * (k + cn));
				const size_t addr1 = en * (m + cn);

				const double *__restrict__ tmp_c_mkn = tmp_c + addr0;
				const double *__restrict__ tmp_c_mlkn = tmp_c_mkn + len;
				const double *__restrict__ een_rescaled_n_mnw =
					een_rescaled_n + addr1;
				const double *__restrict__ een_rescaled_n_mlnw =
					een_rescaled_n_mnw + len;
				const double *__restrict__ dtmp_c_mknw = &(dtmp_c[addr0 << 2]);
				const double *__restrict__ dtmp_c_mlknw = dtmp_c_mknw + len4;
				const double *__restrict__ een_rescaled_n_deriv_e_mnw =
					een_rescaled_n_deriv_e + (addr1 << 2);
				const double *__restrict__ een_rescaled_n_deriv_e_mlnw =
					een_rescaled_n_deriv_e_mnw + len4;

				for (size_t a = 0; a < (size_t)nucl_num; a++) {
					double cn = c_vector_full[a + n * nucl_num];
					if (cn == 0.0)
						continue;

					const size_t ishift = elec_num * a;
					const size_t ishift4 = ishift << 2;

					const double *__restrict__ tmp_c_amlkn = tmp_c_mlkn + ishift;
					const double *__restrict__ tmp_c_amkn = tmp_c_mkn + ishift;
					const double *__restrict__ een_rescaled_n_amnw =
						een_rescaled_n_mnw + ishift;
					const double *__restrict__ een_rescaled_n_amlnw =
						een_rescaled_n_mlnw + ishift;
					const double *__restrict__ dtmp_c_0amknw =
						dtmp_c_mknw + ishift4;
					const double *__restrict__ dtmp_c_0amlknw =
						dtmp_c_mlknw + ishift4;
					const double *__restrict__ een_rescaled_n_deriv_e_0amnw =
						een_rescaled_n_deriv_e_mnw + ishift4;
					const double *__restrict__ een_rescaled_n_deriv_e_0amlnw =
						een_rescaled_n_deriv_e_mlnw + ishift4;

					const double *__restrict__ dtmp_c_1amknw =
						dtmp_c_0amknw + elec_num;
					const double *__restrict__ dtmp_c_1amlknw =
						dtmp_c_0amlknw + elec_num;
					const double *__restrict__ dtmp_c_2amknw =
						dtmp_c_0amknw + acc_elec_num2[0];
					const double *__restrict__ dtmp_c_2amlknw =
						dtmp_c_0amlknw + acc_elec_num2[0];
					const double *__restrict__ dtmp_c_3amknw =
						dtmp_c_0amknw + acc_elec_num3[0];
					const double *__restrict__ dtmp_c_3amlknw =
						dtmp_c_0amlknw + acc_elec_num3[0];
					const double *__restrict__ een_rescaled_n_deriv_e_1amnw =
						een_rescaled_n_deriv_e_0amnw + elec_num;
					const double *__restrict__ een_rescaled_n_deriv_e_1amlnw =
						een_rescaled_n_deriv_e_0amlnw + elec_num;
					const double *__restrict__ een_rescaled_n_deriv_e_2amnw =
						een_rescaled_n_deriv_e_0amnw + acc_elec_num2[0];
					const double *__restrict__ een_rescaled_n_deriv_e_2amlnw =
						een_rescaled_n_deriv_e_0amlnw + acc_elec_num2[0];
					const double *__restrict__ een_rescaled_n_deriv_e_3amnw =
						een_rescaled_n_deriv_e_0amnw + acc_elec_num3[0];
					const double *__restrict__ een_rescaled_n_deriv_e_3amlnw =
						een_rescaled_n_deriv_e_0amlnw + acc_elec_num3[0];
					double *const __restrict__ factor_een_deriv_e_1nw =
						factor_een_deriv_e_0nw + elec_num;
					double *const __restrict__ factor_een_deriv_e_2nw =
						factor_een_deriv_e_0nw + acc_elec_num2[0];
					double *const __restrict__ factor_een_deriv_e_3nw =
						factor_een_deriv_e_0nw + acc_elec_num3[0];

					for (size_t j = 0; j < (size_t)elec_num; ++j) {
						factor_een_deriv_e_0nw[j] +=
							cn *
							(tmp_c_amkn[j] * een_rescaled_n_deriv_e_0amlnw[j] +
								dtmp_c_0amknw[j] * een_rescaled_n_amlnw[j] +
								dtmp_c_0amlknw[j] * een_rescaled_n_amnw[j] +
								tmp_c_amlkn[j] * een_rescaled_n_deriv_e_0amnw[j]);
						tmp3[j] =
							dtmp_c_0amknw[j] *
								een_rescaled_n_deriv_e_0amlnw[j] +
							dtmp_c_0amlknw[j] * een_rescaled_n_deriv_e_0amnw[j];
					}

					for (size_t j = 0; j < (size_t)elec_num; ++j) {
						factor_een_deriv_e_1nw[j] +=
							cn *
							(tmp_c_amkn[j] * een_rescaled_n_deriv_e_1amlnw[j] +
								dtmp_c_1amknw[j] * een_rescaled_n_amlnw[j] +
								dtmp_c_1amlknw[j] * een_rescaled_n_amnw[j] +
								tmp_c_amlkn[j] * een_rescaled_n_deriv_e_1amnw[j]);
						tmp3[j] +=
							dtmp_c_1amknw[j] *
								een_rescaled_n_deriv_e_1amlnw[j] +
							dtmp_c_1amlknw[j] * een_rescaled_n_deriv_e_1amnw[j];
					}

					for (size_t j = 0; j < (size_t)elec_num; ++j) {
						factor_een_deriv_e_2nw[j] +=
							cn *
							(tmp_c_amkn[j] * een_rescaled_n_deriv_e_2amlnw[j] +
								dtmp_c_2amknw[j] * een_rescaled_n_amlnw[j] +
								dtmp_c_2amlknw[j] * een_rescaled_n_amnw[j] +
								tmp_c_amlkn[j] * een_rescaled_n_deriv_e_2amnw[j]);
						tmp3[j] +=
							dtmp_c_2amknw[j] *
								een_rescaled_n_deriv_e_2amlnw[j] +
							dtmp_c_2amlknw[j] * een_rescaled_n_deriv_e_2amnw[j];
					}

					for (size_t j = 0; j < (size_t)elec_num; ++j) {
						factor_een_deriv_e_3nw[j] +=
							cn *
							(tmp_c_amkn[j] * een_rescaled_n_deriv_e_3amlnw[j] +
								dtmp_c_3amknw[j] * een_rescaled_n_amlnw[j] +
								dtmp_c_3amlknw[j] * een_rescaled_n_amnw[j] +
								tmp_c_amlkn[j] * een_rescaled_n_deriv_e_3amnw[j] +
								tmp3[j] * 2.0);
					}
				}
			}
		});
	}).wait();

	buff_elec_num2.get_host_access();
	buff_elec_num3.get_host_access();

	qmckl_free_device(context, tmp3);

	return info;
}


// Electron/electron/nucleus deriv
qmckl_exit_code_device
qmckl_compute_jastrow_factor_een_rescaled_e_deriv_e_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t cord_num,
	const double rescale_factor_ee, const double *coord_ee,
	const double *ee_distance, const double *een_rescaled_e,
	double *const een_rescaled_e_deriv_e) {

	double x, rij_inv, kappa_l;
	int i, j, k, l, nw, ii;

	double *elec_dist_deriv_e =
		reinterpret_cast<double *>(qmckl_malloc_device(context, 4 * elec_num * elec_num * sizeof(double)));

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (walk_num <= 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (elec_num <= 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	if (cord_num < 0) {
		info = QMCKL_INVALID_ARG_4_DEVICE;
		return info;
	}

	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	sycl::buffer<double, 1> buff_rij_inv(&rij_inv, range<1>(1));
	sycl::buffer<double, 1> buff_kappa_l(&kappa_l, range<1>(1));

	// Prepare table of exponentiated distances raised to appropriate power   
	q.parallel_for(sycl::range<1>(elec_num * 4 * elec_num * (cord_num + 1) * walk_num), [=](sycl::id<1> idx) {
		auto i = idx[0];
		een_rescaled_e_deriv_e[i] = 0.0;
	}).wait();

	q.submit([&](handler &h) {
		sycl::accessor acc_rij_inv(buff_rij_inv, h, read_write);
		sycl::accessor acc_kappa_l(buff_kappa_l, h, read_write);
		h.parallel_for(sycl::range<1>(walk_num), [=](sycl::id<1> idx) {
			auto nw = idx[0];
			for (int j = 0; j < elec_num; j++) {
				for (int i = 0; i < elec_num; i++) {
					acc_rij_inv[0] = 1.0 / ee_distance[i + j * elec_num +
												nw * elec_num * elec_num];
					for (int ii = 0; ii < 3; ii++) {
						elec_dist_deriv_e[ii + i * 4 + j * 4 * elec_num] =
							(coord_ee[i + ii * elec_num + nw * elec_num * 3] -
								coord_ee[j + ii * elec_num + nw * elec_num * 3]) *
							acc_rij_inv[0];
					}
					elec_dist_deriv_e[3 + i * 4 + j * 4 * elec_num] =
						2.0 * acc_rij_inv[0];
				}

				for (int ii = 0; ii < 4; ii++) {
					elec_dist_deriv_e[ii + j * 4 + j * 4 * elec_num] = 0.0;
				}
			}

			// prepare the actual een table
			for (int l = 0; l < cord_num; l++) {
				acc_kappa_l[0] = -l * rescale_factor_ee;
				for (int j = 0; j < elec_num; j++) {
					for (int i = 0; i < elec_num; i++) {
						een_rescaled_e_deriv_e[i + 0 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							acc_kappa_l[0] *
							elec_dist_deriv_e[0 + i * 4 + j * 4 * elec_num];
						een_rescaled_e_deriv_e[i + 1 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							acc_kappa_l[0] *
							elec_dist_deriv_e[1 + i * 4 + j * 4 * elec_num];

						een_rescaled_e_deriv_e[i + 2 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							acc_kappa_l[0] *
							elec_dist_deriv_e[2 + i * 4 + j * 4 * elec_num];

						een_rescaled_e_deriv_e[i + 3 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							acc_kappa_l[0] *
							elec_dist_deriv_e[3 + i * 4 + j * 4 * elec_num];

						een_rescaled_e_deriv_e[i + 3 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							een_rescaled_e_deriv_e[i + 3 * elec_num +
													j * elec_num * 4 +
													l * elec_num * 4 * elec_num +
													nw * elec_num * 4 *
														elec_num *
														(cord_num + 1)] +
							een_rescaled_e_deriv_e[i + 0 * elec_num +
													j * elec_num * 4 +
													l * elec_num * 4 * elec_num +
													nw * elec_num * 4 *
														elec_num *
														(cord_num + 1)] *
								een_rescaled_e_deriv_e
									[i + 0 * elec_num + j * elec_num * 4 +
										l * elec_num * 4 * elec_num +
										nw * elec_num * 4 * elec_num *
											(cord_num + 1)] +
							een_rescaled_e_deriv_e[i + 1 * elec_num +
													j * elec_num * 4 +
													l * elec_num * 4 * elec_num +
													nw * elec_num * 4 *
														elec_num *
														(cord_num + 1)] *
								een_rescaled_e_deriv_e
									[i + 1 * elec_num + j * elec_num * 4 +
										l * elec_num * 4 * elec_num +
										nw * elec_num * 4 * elec_num *
											(cord_num + 1)] +
							een_rescaled_e_deriv_e[i + 2 * elec_num +
													j * elec_num * 4 +
													l * elec_num * 4 * elec_num +
													nw * elec_num * 4 *
														elec_num *
														(cord_num + 1)] *
								een_rescaled_e_deriv_e
									[i + 2 * elec_num + j * elec_num * 4 +
										l * elec_num * 4 * elec_num +
										nw * elec_num * 4 * elec_num *
											(cord_num + 1)];

						een_rescaled_e_deriv_e[i + 0 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							een_rescaled_e_deriv_e[i + 0 * elec_num +
													j * elec_num * 4 +
													l * elec_num * 4 * elec_num +
													nw * elec_num * 4 *
														elec_num *
														(cord_num + 1)] *
							een_rescaled_e[i + j * elec_num +
											l * elec_num * elec_num +
											nw * elec_num * elec_num *
												(cord_num + 1)];

						een_rescaled_e_deriv_e[i + 1 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							een_rescaled_e_deriv_e[i + 1 * elec_num +
													j * elec_num * 4 +
													l * elec_num * 4 * elec_num +
													nw * elec_num * 4 *
														elec_num *
														(cord_num + 1)] *
							een_rescaled_e[i + j * elec_num +
											l * elec_num * elec_num +
											nw * elec_num * elec_num *
												(cord_num + 1)];

						een_rescaled_e_deriv_e[i + 2 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							een_rescaled_e_deriv_e[i + 2 * elec_num +
													j * elec_num * 4 +
													l * elec_num * 4 * elec_num +
													nw * elec_num * 4 *
														elec_num *
														(cord_num + 1)] *
							een_rescaled_e[i + j * elec_num +
											l * elec_num * elec_num +
											nw * elec_num * elec_num *
												(cord_num + 1)];

						een_rescaled_e_deriv_e[i + 3 * elec_num +
												j * elec_num * 4 +
												l * elec_num * 4 * elec_num +
												nw * elec_num * 4 * elec_num *
													(cord_num + 1)] =
							een_rescaled_e_deriv_e[i + 3 * elec_num +
													j * elec_num * 4 +
													l * elec_num * 4 * elec_num +
													nw * elec_num * 4 *
														elec_num *
														(cord_num + 1)] *
							een_rescaled_e[i + j * elec_num +
											l * elec_num * elec_num +
											nw * elec_num * elec_num *
												(cord_num + 1)];
					}
				}
			}
		});
	}).wait();

	buff_rij_inv.get_host_access();
	buff_kappa_l.get_host_access();

	qmckl_free_device(context, elec_dist_deriv_e);
	return info;
}


qmckl_exit_code_device
qmckl_compute_jastrow_factor_een_rescaled_n_deriv_e_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t nucl_num, const int64_t type_nucl_num,
	int64_t *const type_nucl_vector, const int64_t cord_num,
	const double *rescale_factor_en, const double *coord_ee,
	const double *coord_en, const double *en_distance,
	const double *een_rescaled_n, double *const een_rescaled_n_deriv_e) {

	double *elnuc_dist_deriv_e =
		reinterpret_cast<double *>(qmckl_malloc_device(context, 4 * elec_num * nucl_num * sizeof(double)));

	double x, ria_inv, kappa_l;
	int i, a, k, l, nw, ii;

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (walk_num <= 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (elec_num <= 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	if (nucl_num <= 0) {
		info = QMCKL_INVALID_ARG_4_DEVICE;
		return info;
	}

	if (cord_num < 0) {
		info = QMCKL_INVALID_ARG_5_DEVICE;
		return info;
	}
	
	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	sycl::buffer<double, 1> buff_ria_inv(&ria_inv, range<1>(1));
	sycl::buffer<double, 1> buff_kappa_l(&kappa_l, range<1>(1));

 	// Prepare table of exponentiated distances raised to appropriate power   
	q.parallel_for(sycl::range<1>(elec_num * 4 * nucl_num * (cord_num + 1) * walk_num), [=](sycl::id<1> idx) {
		auto i = idx[0];
		een_rescaled_n_deriv_e[i] = 0.0;
	}).wait();

	q.submit([&](handler &h) {
		sycl::accessor acc_ria_inv(buff_ria_inv, h, read_write);
		sycl::accessor acc_kappa_l(buff_kappa_l, h, read_write);
		h.parallel_for(sycl::range<1>(walk_num), [=](sycl::id<1> idx) {
			auto nw = idx[0];
			// Prepare the actual een table
			for (int a = 0; a < nucl_num; a++) {
				for (int i = 0; i < elec_num; i++) {
					acc_ria_inv[0] = 1.0 / en_distance[a + i * nucl_num +
												nw * nucl_num * elec_num];
					for (int ii = 0; ii < 3; ii++) {
						elnuc_dist_deriv_e[ii * nucl_num * elec_num +
											i * nucl_num + a] =
							(coord_ee[i + ii * elec_num + nw * elec_num * 4] -
								coord_en[a + ii * nucl_num]) *
							acc_ria_inv[0];
					}
					elnuc_dist_deriv_e[3 * nucl_num * elec_num + i * nucl_num +
										a] = 2.0 * acc_ria_inv[0];
				}
			}

			for (int l = 0; l < cord_num; l++) {
				// NOTE In CPU, bound is up to (nucl_num+1), but seems
				for (int a = 0; a < nucl_num; a++) {
					acc_kappa_l[0] = -((double)l) *
								rescale_factor_en[type_nucl_vector[a]];
					for (int i = 0; i < elec_num; i++) {

						een_rescaled_n_deriv_e[i + 0 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							acc_kappa_l[0] *
							elnuc_dist_deriv_e[0 * nucl_num * elec_num +
												i * nucl_num + a];
						een_rescaled_n_deriv_e[i + 1 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							acc_kappa_l[0] *
							elnuc_dist_deriv_e[1 * nucl_num * elec_num +
												i * nucl_num + a];
						een_rescaled_n_deriv_e[i + 2 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							acc_kappa_l[0] *
							elnuc_dist_deriv_e[2 * nucl_num * elec_num +
												i * nucl_num + a];
						een_rescaled_n_deriv_e[i + 3 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							acc_kappa_l[0] *
							elnuc_dist_deriv_e[3 * nucl_num * elec_num +
												i * nucl_num + a];

						double een_1_squared = een_rescaled_n_deriv_e
							[i + 0 * elec_num + a * elec_num * 4 +
								l * elec_num * 4 * nucl_num +
								nw * elec_num * 4 * nucl_num * (cord_num + 1)];
						een_1_squared = een_1_squared * een_1_squared;
						double een_2_squared = een_rescaled_n_deriv_e
							[i + 1 * elec_num + a * elec_num * 4 +
								l * elec_num * 4 * nucl_num +
								nw * elec_num * 4 * nucl_num * (cord_num + 1)];
						een_2_squared = een_2_squared * een_2_squared;
						double een_3_squared = een_rescaled_n_deriv_e
							[i + 2 * elec_num + a * elec_num * 4 +
								l * elec_num * 4 * nucl_num +
								nw * elec_num * 4 * nucl_num * (cord_num + 1)];
						een_3_squared = een_3_squared * een_3_squared;

						een_rescaled_n_deriv_e[i + 3 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							een_rescaled_n_deriv_e[i + 3 * elec_num +
													a * elec_num * 4 +
													l * elec_num * 4 * nucl_num +
													nw * elec_num * 4 *
														nucl_num *
														(cord_num + 1)] +
							een_1_squared + een_2_squared + een_3_squared;

						een_rescaled_n_deriv_e[i + 0 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							een_rescaled_n_deriv_e[i + 0 * elec_num +
													a * elec_num * 4 +
													l * elec_num * 4 * nucl_num +
													nw * elec_num * 4 *
														nucl_num *
														(cord_num + 1)] *
							een_rescaled_n[i + a * elec_num +
											l * elec_num * nucl_num +
											nw * elec_num * nucl_num *
												(cord_num + 1)];
						een_rescaled_n_deriv_e[i + 1 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							een_rescaled_n_deriv_e[i + 1 * elec_num +
													a * elec_num * 4 +
													l * elec_num * 4 * nucl_num +
													nw * elec_num * 4 *
														nucl_num *
														(cord_num + 1)] *
							een_rescaled_n[i + a * elec_num +
											l * elec_num * nucl_num +
											nw * elec_num * nucl_num *
												(cord_num + 1)];
						een_rescaled_n_deriv_e[i + 2 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							een_rescaled_n_deriv_e[i + 2 * elec_num +
													a * elec_num * 4 +
													l * elec_num * 4 * nucl_num +
													nw * elec_num * 4 *
														nucl_num *
														(cord_num + 1)] *
							een_rescaled_n[i + a * elec_num +
											l * elec_num * nucl_num +
											nw * elec_num * nucl_num *
												(cord_num + 1)];
						een_rescaled_n_deriv_e[i + 3 * elec_num +
												a * elec_num * 4 +
												l * elec_num * 4 * nucl_num +
												nw * elec_num * 4 * nucl_num *
													(cord_num + 1)] =
							een_rescaled_n_deriv_e[i + 3 * elec_num +
													a * elec_num * 4 +
													l * elec_num * 4 * nucl_num +
													nw * elec_num * 4 *
														nucl_num *
														(cord_num + 1)] *
							een_rescaled_n[i + a * elec_num +
											l * elec_num * nucl_num +
											nw * elec_num * nucl_num *
												(cord_num + 1)];
					}
				}
			}
		});
	}).wait();

	buff_ria_inv.get_host_access();
	buff_kappa_l.get_host_access();

	qmckl_free_device(context, elnuc_dist_deriv_e);
	return QMCKL_SUCCESS_DEVICE;
}


// Distances
qmckl_exit_code_device qmckl_compute_en_distance_rescaled_device(
	const qmckl_context_device context, const int64_t elec_num,
	const int64_t nucl_num, const int64_t type_nucl_num,
	int64_t *const type_nucl_vector, const double *rescale_factor_en,
	const int64_t walk_num, const double *elec_coord, const double *nucl_coord,
	double *const en_distance_rescaled) {

	int i, k;
	double *coord = reinterpret_cast<double *>(qmckl_malloc_device(context, 3 * nucl_num * sizeof(double)));

	int64_t *type_nucl_vector_h = reinterpret_cast<int64_t *>(malloc(nucl_num * sizeof(int64_t)));
	qmckl_memcpy_D2H(context, type_nucl_vector_h, type_nucl_vector,
					 nucl_num * sizeof(int64_t));

	double *rescale_factor_en_h = reinterpret_cast<double *>(malloc(nucl_num * sizeof(int64_t)));
	qmckl_memcpy_D2H(context, rescale_factor_en_h, const_cast<double *>(rescale_factor_en),
					 type_nucl_num * sizeof(double));

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (elec_num <= 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (nucl_num <= 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	if (walk_num <= 0) {
		info = QMCKL_INVALID_ARG_5_DEVICE;
		return info;
	}

	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	sycl::buffer<int, 1> buff_i(&i, range<1>(1));
    
		for (i = 0; i < nucl_num; i++) {
			q.submit([&](handler &h) {
				sycl::accessor acc_i(buff_i, h, read_write);
				h.single_task([=]() {
					coord[0] = nucl_coord[acc_i[0] + 0 * nucl_num];
					coord[1] = nucl_coord[acc_i[0] + 1 * nucl_num];
					coord[2] = nucl_coord[acc_i[0] + 2 * nucl_num];
				});
			}).wait();
			buff_i.get_host_access();

			for (k = 0; k < walk_num; k++) {
				info = qmckl_distance_rescaled_device(
					context, 'T', 'T', elec_num, 1, elec_coord + k * elec_num,
					elec_num * walk_num, coord, 1,
					en_distance_rescaled + i * elec_num + k * elec_num * nucl_num,
					elec_num, rescale_factor_en_h[type_nucl_vector_h[i]], q);
				if (info != QMCKL_SUCCESS_DEVICE) {
					break;
				}
			}
		}

	qmckl_free_device(context, coord);
	free(type_nucl_vector_h);
	free(rescale_factor_en_h);
	return info;
}

qmckl_exit_code_device qmckl_compute_een_rescaled_e_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t cord_num,
	const double rescale_factor_ee, const double *ee_distance,
	double *const een_rescaled_e) {
	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (walk_num <= 0) {
		return QMCKL_INVALID_ARG_2_DEVICE;
	}

	if (elec_num <= 0) {
		return QMCKL_INVALID_ARG_3_DEVICE;
	}

	if (cord_num < 0) {
		return QMCKL_INVALID_ARG_4_DEVICE;
	}

	const size_t elec_pairs = (size_t)(elec_num * (elec_num - 1)) / 2;
	const size_t len_een_ij = (size_t)elec_pairs * (cord_num + 1);
	double *een_rescaled_e_ij =
		reinterpret_cast<double *>(qmckl_malloc_device(context, len_een_ij * sizeof(double)));

	// number of elements for the een_rescaled_e_ij[N_e*(N_e-1)/2][cord+1]
	// probably in C is better [cord+1, Ne*(Ne-1)/2]
	// elec_pairs = (elec_num * (elec_num - 1)) / 2;
	// len_een_ij = elec_pairs * (cord_num + 1);
	const size_t e2 = elec_num * elec_num;
	
	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	sycl::buffer<size_t, 1> buff_elec_pairs(&elec_pairs, range<1>(1));
	sycl::buffer<size_t, 1> buff_len_een_ij(&len_een_ij, range<1>(1));
	sycl::buffer<size_t, 1> buff_e2(&e2, range<1>(1));

	// Prepare table of exponentiated distances raised to appropriate power
	// init   
	q.parallel_for(sycl::range<1>(walk_num * (cord_num + 1) * elec_num * elec_num), [=](sycl::id<1> idx) {
		auto i = idx[0];
		een_rescaled_e[i] = 0;
	}).wait();
	
	q.submit([&](handler &h) {
		sycl::accessor acc_elec_pairs(buff_elec_pairs, h, read_write);
		sycl::accessor acc_len_een_ij(buff_len_een_ij, h, read_write);
		sycl::accessor acc_e2(buff_e2, h, read_write);
		q.parallel_for(sycl::range<1>((size_t)walk_num), [=](sycl::id<1> idx) {
			auto nw = idx[0];
			for (size_t kk = 0; kk < acc_elec_pairs[0]; ++kk) {
				een_rescaled_e_ij[kk] = 0.0;
			}
			for (size_t kk = 0; kk < acc_len_een_ij[0]; ++kk) {
				een_rescaled_e_ij[kk] = 1.0;
			}

			size_t kk = 0;
			for (size_t i = 0; i < (size_t)elec_num; ++i) {
				for (size_t j = 0; j < i; ++j) {
					een_rescaled_e_ij[j + kk + acc_elec_pairs[0]] =
						-rescale_factor_ee *
						ee_distance[j + i * elec_num + nw * acc_e2[0]];
				}
				kk += i;
			}

			for (size_t k = acc_elec_pairs[0]; k < 2 * acc_elec_pairs[0]; ++k) {
				een_rescaled_e_ij[k] = exp(een_rescaled_e_ij[k]);
			}

			for (size_t l = 2; l < (size_t)(cord_num + 1); ++l) {
				for (size_t k = 0; k < acc_elec_pairs[0]; ++k) {
					// een_rescaled_e_ij(k, l + 1) = een_rescaled_e_ij(k, l + 1
					// - 1)
					// * een_rescaled_e_ij(k, 2)
					een_rescaled_e_ij[k + l * acc_elec_pairs[0]] =
						een_rescaled_e_ij[k + (l - 1) * acc_elec_pairs[0]] *
						een_rescaled_e_ij[k + acc_elec_pairs[0]];
				}
			}

			double *const een_rescaled_e_ =
				&(een_rescaled_e[nw * (cord_num + 1) * acc_e2[0]]);
			// prepare the actual een table
			for (size_t i = 0; i < acc_e2[0]; ++i) {
				een_rescaled_e_[i] = 1.0;
			}

			for (size_t l = 1; l < (size_t)(cord_num + 1); ++l) {
				double *x = een_rescaled_e_ij + l * acc_elec_pairs[0];
				double *const een_rescaled_e__ = &(een_rescaled_e_[l * acc_e2[0]]);
				double *een_rescaled_e_i = een_rescaled_e__;
				for (size_t i = 0; i < (size_t)elec_num; ++i) {
					for (size_t j = 0; j < i; ++j) {
						een_rescaled_e_i[j] = *x;
						een_rescaled_e__[i + j * elec_num] = *x;
						x += 1;
					}
					een_rescaled_e_i += elec_num;
				}
			}

			double *const x0 = &(een_rescaled_e[nw * acc_e2[0] * (cord_num + 1)]);
			for (size_t l = 0; l < (size_t)(cord_num + 1); ++l) {
				double *x1 = &(x0[l * acc_e2[0]]);
				for (size_t j = 0; j < (size_t)elec_num; ++j) {
					*x1 = 0.0;
					x1 += 1 + elec_num;
				}
			}
		});
	}).wait();

	buff_elec_pairs.get_host_access();
	buff_len_een_ij.get_host_access();
	buff_e2.get_host_access();

	qmckl_free_device(context, een_rescaled_e_ij);
	return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device qmckl_compute_een_rescaled_n_device(
	const qmckl_context_device context, const int64_t walk_num,
	const int64_t elec_num, const int64_t nucl_num, const int64_t type_nucl_num,
	int64_t *const type_nucl_vector, const int64_t cord_num,
	const double *rescale_factor_en, const double *en_distance,
	double *const een_rescaled_n) {
	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (walk_num <= 0) {
		return QMCKL_INVALID_ARG_2_DEVICE;
	}

	if (elec_num <= 0) {
		return QMCKL_INVALID_ARG_3_DEVICE;
	}

	if (nucl_num <= 0) {
		return QMCKL_INVALID_ARG_4_DEVICE;
	}

	if (cord_num < 0) {
		return QMCKL_INVALID_ARG_5_DEVICE;
	}
	
	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	// Prepare table of exponentiated distances raised to appropriate power
	q.parallel_for(sycl::range<1>(walk_num * (cord_num + 1) * nucl_num * elec_num), [=](sycl::id<1> idx) {
		auto i = idx[0];
		een_rescaled_n[i] = 0.0;
	}).wait();
	
	q.parallel_for(sycl::range<1>(walk_num), [=](sycl::id<1> idx) {
		auto nw = idx[0];
		
		// prepare the actual een table
		for (int a = 0; a < nucl_num; ++a) {
			for (int i = 0; i < elec_num; ++i) {
				een_rescaled_n[i + a * elec_num +
								nw * elec_num * nucl_num * (cord_num + 1)] =
					1.0;
				een_rescaled_n[i + a * elec_num + elec_num * nucl_num +
								nw * elec_num * nucl_num * (cord_num + 1)] =
					exp(-rescale_factor_en[type_nucl_vector[a]] *
						en_distance[a + i * nucl_num +
									nw * elec_num * nucl_num]);
			}
		}

		for (int l = 2; l < (cord_num + 1); ++l) {
			for (int a = 0; a < nucl_num; ++a) {
				for (int i = 0; i < elec_num; ++i) {
					een_rescaled_n[i + a * elec_num +
									l * elec_num * nucl_num +
									nw * elec_num * nucl_num *
										(cord_num + 1)] =
						een_rescaled_n[i + a * elec_num +
										(l - 1) * elec_num * nucl_num +
										nw * elec_num * nucl_num *
											(cord_num + 1)] *
						een_rescaled_n[i + a * elec_num +
										elec_num * nucl_num +
										nw * elec_num * nucl_num *
											(cord_num + 1)];
				}
			}
		}	
	}).wait();

	return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device qmckl_compute_c_vector_full_device(
	const qmckl_context_device context, const int64_t nucl_num,
	const int64_t dim_c_vector, const int64_t type_nucl_num,
	const int64_t *type_nucl_vector, const double *c_vector,
	double *const c_vector_full) {
	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (nucl_num <= 0) {
		return QMCKL_INVALID_ARG_2_DEVICE;
	}

	if (type_nucl_num <= 0) {
		return QMCKL_INVALID_ARG_4_DEVICE;
	}

	if (dim_c_vector < 0) {
		return QMCKL_INVALID_ARG_5_DEVICE;
	}
	
	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	q.parallel_for(sycl::range<1>(dim_c_vector), [=](sycl::id<1> idx) {
		auto i = idx[0];
		for (int a = 0; a < nucl_num; ++a) {
			c_vector_full[a + i * nucl_num] =
				c_vector[ i + type_nucl_vector[a] * dim_c_vector];
		}
	}).wait();

	return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device qmckl_compute_lkpm_combined_index_device(
	const qmckl_context_device context, const int64_t cord_num,
	const int64_t dim_c_vector, int64_t *const lkpm_combined_index) {

	double x;
	int i, a, k, l, kk, p, lmax, m;

	qmckl_context_device info = QMCKL_SUCCESS_DEVICE;

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		info = QMCKL_INVALID_CONTEXT_DEVICE;
		return info;
	}

	if (cord_num < 0) {
		info = QMCKL_INVALID_ARG_2_DEVICE;
		return info;
	}

	if (dim_c_vector < 0) {
		info = QMCKL_INVALID_ARG_3_DEVICE;
		return info;
	}

	kk = 0;
	
	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	sycl::buffer<int, 1> buff_kk(&kk, range<1>(1));
	sycl::buffer<int, 1> buff_lmax(&lmax, range<1>(1));
	sycl::buffer<int, 1> buff_m(&m, range<1>(1));

	q.submit([&](handler &h) {
		sycl::accessor acc_kk(buff_kk, h, read_write);
		sycl::accessor acc_lmax(buff_lmax, h, read_write);
		sycl::accessor acc_m(buff_m, h, read_write);
		q.single_task([=]() {
			for (int p = 2; p <= cord_num; ++p) {
				for (int k = (p - 1); k >= 0; --k) {
					if (k != 0) {
						acc_lmax[0] = p - k;
					} else {
						acc_lmax[0] = p - k - 2;
					}
					for (int l = acc_lmax[0]; l >= 0; --l) {
						if (((p - k - l) & 1) == 1)
							continue;
						acc_m[0] = (p - k - l) / 2;
						lkpm_combined_index[acc_kk[0]] = l;
						lkpm_combined_index[acc_kk[0] + dim_c_vector] = k;
						lkpm_combined_index[acc_kk[0] + 2 * dim_c_vector] = p;
						lkpm_combined_index[acc_kk[0] + 3 * dim_c_vector] = acc_m[0];
						acc_kk[0] = acc_kk[0] + 1;
					}
				}
			}
		});
	}).wait();

	buff_kk.get_host_access();
	buff_lmax.get_host_access();
	buff_m.get_host_access();

	return QMCKL_SUCCESS_DEVICE;
}


qmckl_exit_code_device
qmckl_compute_tmp_c_device(const qmckl_context_device context,
						   const int64_t cord_num, const int64_t elec_num,
						   const int64_t nucl_num, const int64_t walk_num,
						   const double *een_rescaled_e,
						   const double *een_rescaled_n, double *const tmp_c) {

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (cord_num < 0) {
		return QMCKL_INVALID_ARG_2_DEVICE;
	}

	if (elec_num <= 0) {
		return QMCKL_INVALID_ARG_3_DEVICE;
	}

	if (nucl_num <= 0) {
		return QMCKL_INVALID_ARG_4_DEVICE;
	}

	if (walk_num <= 0) {
		return QMCKL_INVALID_ARG_5_DEVICE;
	}

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	const char TransA = 'N';
	const char TransB = 'N';
	const double alpha = 1.0;
	const double beta = 0.0;

	const int64_t M = elec_num;
	const int64_t N = nucl_num * (cord_num + 1);
	const int64_t K = elec_num;

	const int64_t LDA = elec_num;
	const int64_t LDB = elec_num;
	const int64_t LDC = elec_num;

	const int64_t af = elec_num * elec_num;
	const int64_t bf = elec_num * nucl_num * (cord_num + 1);
	const int64_t cf = bf;
	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

    
#ifdef HAVE_LIBGPUBLAS

	gpu_dgemm('N', 'N', M, N, K, alpha, een_rescaled_e, LDA, een_rescaled_n, LDB, beta, tmp_c, LDC);

#elif HAVE_CUBLAS
	
	q.submit([&](handler &h) {
		q.single_task([=]() {
			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasStatus_t error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, een_rescaled_e, LDA, een_rescaled_n, LDB, &beta, tmp_c, LDC ); 
			printf("%s\n",cublasGetStatusString(error));
		});
	}).wait();

#else
	sycl::buffer<int64_t, 1> buff_M(&M, range<1>(1));
	sycl::buffer<int64_t, 1> buff_N(&N, range<1>(1));
	sycl::buffer<int64_t, 1> buff_K(&K, range<1>(1));
	sycl::buffer<int64_t, 1> buff_LDA(&LDA, range<1>(1));
	sycl::buffer<int64_t, 1> buff_LDB(&LDB, range<1>(1));
	sycl::buffer<int64_t, 1> buff_LDC(&LDC, range<1>(1));
	sycl::buffer<int64_t, 1> buff_af(&af, range<1>(1));
	sycl::buffer<int64_t, 1> buff_bf(&bf, range<1>(1));
	sycl::buffer<int64_t, 1> buff_cf(&cf, range<1>(1));
	q.submit([&](handler &h) {
		sycl::accessor acc_M(buff_M, h, read_only);
		sycl::accessor acc_N(buff_N, h, read_only);
		sycl::accessor acc_LDA(buff_LDA, h, read_only);
		sycl::accessor acc_LDB(buff_LDB, h, read_only);
		sycl::accessor acc_LDC(buff_LDC, h, read_only);
		sycl::accessor acc_af(buff_af, h, read_only);
		sycl::accessor acc_bf(buff_bf, h, read_only);
		sycl::accessor acc_cf(buff_cf, h, read_only);
		sycl::accessor acc_K(buff_K, h, read_only);
		q.parallel_for(sycl::range<2>(walk_num, cord_num), [=](sycl::id<2> idx)  {
			auto nw = idx[0];
			auto i = idx[1];

			// Single DGEMM
			double *A = const_cast<double *>(een_rescaled_e + (acc_af[0] * (i + nw * (cord_num + 1))));
			double *B = const_cast<double *>(een_rescaled_n + (acc_bf[0] * nw));
			double *C = tmp_c + (acc_cf[0] * (i + nw * cord_num));

			// Row of A
			for (int i = 0; i < acc_M[0]; i++) {
				// Cols of B
				for (int j = 0; j < acc_N[0]; j++) {

					// Compute C(i,j)
					C[i + LDC * j] = 0.;
					for (int k = 0; k < acc_K[0]; k++) {
						C[i + LDC * j] += A[i + k * acc_LDA[0]] * B[k + j * acc_LDB[0]];
					}
				}
			}
		});
	}).wait();


#endif
	return info;
}


qmckl_exit_code_device qmckl_compute_dtmp_c_device(
	const qmckl_context_device context, const int64_t cord_num,
	const int64_t elec_num, const int64_t nucl_num, const int64_t walk_num,
	const double *een_rescaled_e_deriv_e, const double *een_rescaled_n,
	double *const dtmp_c) {

	if (context == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (cord_num < 0) {
		return QMCKL_INVALID_ARG_2_DEVICE;
	}

	if (elec_num <= 0) {
		return QMCKL_INVALID_ARG_3_DEVICE;
	}

	if (nucl_num <= 0) {
		return QMCKL_INVALID_ARG_4_DEVICE;
	}

	if (walk_num <= 0) {
		return QMCKL_INVALID_ARG_5_DEVICE;
	}

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;

	const char TransA = 'N';
	const char TransB = 'N';
	const double alpha = 1.0;
	const double beta = 0.0;

	const int64_t M = 4 * elec_num;
	const int64_t N = nucl_num * (cord_num + 1);
	const int64_t K = elec_num;

	const int64_t LDA = 4 * elec_num;
	const int64_t LDB = elec_num;
	const int64_t LDC = 4 * elec_num;

	const int64_t af = elec_num * elec_num * 4;
	const int64_t bf = elec_num * nucl_num * (cord_num + 1);
	const int64_t cf = elec_num * 4 * nucl_num * (cord_num + 1);

	// TODO Alternative versions with call to DGEMM / batched DGEMM ?
	
	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue q = ctx->q;

	sycl::buffer<int64_t, 1> buff_M(&M, range<1>(1));
	sycl::buffer<int64_t, 1> buff_N(&N, range<1>(1));
	sycl::buffer<int64_t, 1> buff_K(&K, range<1>(1));
	sycl::buffer<int64_t, 1> buff_LDC(&LDC, range<1>(1));
	sycl::buffer<int64_t, 1> buff_af(&af, range<1>(1));
	sycl::buffer<int64_t, 1> buff_bf(&bf, range<1>(1));
	sycl::buffer<int64_t, 1> buff_cf(&cf, range<1>(1));

	q.submit([&](handler &h) {
		sycl::accessor acc_M(buff_M, h, read_only);
		sycl::accessor acc_N(buff_N, h, read_only);
		sycl::accessor acc_K(buff_K, h, read_only);
		sycl::accessor acc_LDC(buff_LDC, h, read_only);
		sycl::accessor acc_af(buff_af, h, read_only);
		sycl::accessor acc_bf(buff_bf, h, read_only);
		sycl::accessor acc_cf(buff_cf, h, read_only);
		q.parallel_for(sycl::range<2>(walk_num, cord_num), [=](sycl::id<2> idx) {
			auto nw = idx[0];
			auto i = idx[1];
			
			// Single DGEMM
			double *A =
				const_cast<double *>(een_rescaled_e_deriv_e + (acc_af[0] * (i + nw * (cord_num + 1))));
			double *B = const_cast<double *>(een_rescaled_n + (acc_bf[0] * nw));
			double *C = dtmp_c + (acc_cf[0] * (i + nw * cord_num));

			// Row of A
			for (int i = 0; i < acc_M[0]; i++) {
				// Cols of B
				for (int j = 0; j < acc_N[0]; j++) {

					// Compute C(i,j)
					C[i + acc_LDC[0] * j] = 0.;
					for (int k = 0; k < acc_K[0]; k++) {
						C[i + LDC * j] += A[i + k * LDA] * B[k + j * LDB];
					}
				}
			}
		});
	}).wait();

	buff_M.get_host_access();
	buff_N.get_host_access();
	buff_K.get_host_access();
	buff_LDC.get_host_access();
	buff_af.get_host_access();
	buff_bf.get_host_access();
	buff_cf.get_host_access();

	return info;
}


//**********
// SETTERS (requiring offload)
//**********

qmckl_exit_code_device
qmckl_set_jastrow_rescale_factor_en_device(qmckl_context_device context,
										   const double *rescale_factor_en,
										   const int64_t size_max) {
	int32_t mask = 1 << 9;

	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_NULL_CONTEXT_DEVICE;
	}

	qmckl_context_struct_device *const ctx =
		(qmckl_context_struct_device *)context;

	if (mask != 0 && !(ctx->jastrow.uninitialized & mask)) {
		return qmckl_failwith_device(context, QMCKL_ALREADY_SET_DEVICE,
									 "qmckl_set_jastrow_*", NULL);
	}

	if (ctx->jastrow.type_nucl_num <= 0) {
		return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
									 "qmckl_set_jastrow_rescale_factor_en",
									 "type_nucl_num not set");
	}

	if (rescale_factor_en == NULL) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_set_jastrow_rescale_factor_en",
									 "Null pointer");
	}

	if (size_max < ctx->jastrow.type_nucl_num) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_set_jastrow_rescale_factor_en",
									 "Array too small");
	}

	if (ctx->jastrow.rescale_factor_en != NULL) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_set_jastrow_rescale_factor_en",
									 "Already set");
	}
    
    sycl::queue queue = ctx->q;

	qmckl_memory_info_struct_device mem_info =
		qmckl_memory_info_struct_zero_device;
	mem_info.size = ctx->jastrow.type_nucl_num * sizeof(double);
	ctx->jastrow.rescale_factor_en =
		reinterpret_cast<double *>(qmckl_malloc_device(context, mem_info.size));

	double *ctx_rescale_factor_en = ctx->jastrow.rescale_factor_en;
	bool wrongval = false;
	sycl::buffer<bool, 1> buffer_wrongval(&wrongval, range<1>(1));
	int64_t ctx_type_nucl_num = ctx->jastrow.type_nucl_num;

	queue.submit([&](handler &h) {
		sycl::accessor acc_wrongval(buffer_wrongval, h, write_only);
		h.single_task([=]() {
			for (int64_t i = 0; i < ctx_type_nucl_num; ++i) {
				if (rescale_factor_en[i] <= 0.0) {
					acc_wrongval[0] = true;
				}
				ctx_rescale_factor_en[i] = rescale_factor_en[i];
			}
		});
	});
	if (wrongval) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_set_jastrow_rescale_factor_en",
									 "rescale_factor_en <= 0.0");
	}

	ctx->jastrow.uninitialized &= ~mask;
	ctx->jastrow.provided = (ctx->jastrow.uninitialized == 0);
	if (ctx->jastrow.provided) {
		qmckl_exit_code_device rc_ = qmckl_finalize_jastrow_device(context);
		if (rc_ != QMCKL_SUCCESS_DEVICE)
			return rc_;
	}

	return QMCKL_SUCCESS_DEVICE;
}

