#include "include/qmckl_mo.hpp"

#include <CL/sycl.hpp>

using namespace sycl;

//**********
// COMPUTE
//**********

/* mo_vgl */

qmckl_exit_code_device qmckl_compute_mo_basis_mo_vgl_device(
	qmckl_context_device context, int64_t ao_num, int64_t mo_num,
	int64_t point_num, double *__restrict__ coefficient_t, double *__restrict__ ao_vgl,
	double *__restrict__ mo_vgl) {

	assert(context != QMCKL_NULL_CONTEXT_DEVICE);

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue queue = ctx->q;

    queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(point_num), [=](sycl::id<1> j) {
            for (int k = 0; k < 5; ++k) {
				for (int l = 0; l < mo_num; l++) {
					mo_vgl[l + mo_num * k + mo_num * 5 * j] = 0.;
				}
			}

			for (int64_t k = 0; k < ao_num; k++) {
				for (int l = 0; l < 5; l++) {
					if (ao_vgl[k + ao_num * 5 * j] != 0.) {
						double c1 = ao_vgl[k + ao_num * l + ao_num * 5 * j];
						for (int i = 0; i < mo_num; i++) {
							mo_vgl[i + mo_num * l + mo_num * 5 * j] =
								mo_vgl[i + mo_num * l + mo_num * 5 * j] +
								coefficient_t[i + mo_num * k] * c1;
						}
					}
				}
			}
        });
    });

	// End of GPU region

	return QMCKL_SUCCESS_DEVICE;

}

/* mo_value */

qmckl_exit_code_device qmckl_compute_mo_basis_mo_value_device(
	qmckl_context_device context, int64_t ao_num, int64_t mo_num,
	int64_t point_num, double *__restrict__ coefficient_t,
	double *__restrict__ ao_value, double *__restrict__ mo_value) {
	assert(context != QMCKL_NULL_CONTEXT_DEVICE);

 	qmckl_context_struct_device *const ctx = (qmckl_context_struct_device*)(context);
    
    sycl::queue queue = ctx->q;

	double *av1_shared =
		reinterpret_cast<double *>(qmckl_malloc_device(context, point_num * ao_num * sizeof(double)));
	int64_t *idx_shared =
		reinterpret_cast<int64_t *>(qmckl_malloc_device(context, point_num * ao_num * sizeof(int64_t)));

    queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(point_num), [=](sycl::id<1> ipoint) {
			
            double *av1 = av1_shared + ipoint * ao_num;
			int64_t *idx = idx_shared + ipoint * ao_num;

			double *vgl1 = mo_value + ipoint * mo_num;
			double *avgl1 = ao_value + ipoint * ao_num;

			for (int64_t i = 0; i < mo_num; ++i) {
				vgl1[i] = 0.;
			}

			int64_t nidx = 0;
			for (int64_t k = 0; k < ao_num; ++k) {
				if (avgl1[k] != 0.) {
					idx[nidx] = k;
					av1[nidx] = avgl1[k];
					++nidx;
				}
			}

			int64_t n = 0;

			for (n = 0; n < nidx - 4; n += 4) {
				double *__restrict__ ck1 = coefficient_t + idx[n] * mo_num;
				double *__restrict__ ck2 = coefficient_t + idx[n + 1] * mo_num;
				double *__restrict__ ck3 = coefficient_t + idx[n + 2] * mo_num;
				double *__restrict__ ck4 = coefficient_t + idx[n + 3] * mo_num;

				double a11 = av1[n];
				double a21 = av1[n + 1];
				double a31 = av1[n + 2];
				double a41 = av1[n + 3];

				for (int64_t i = 0; i < mo_num; ++i) {
					vgl1[i] = vgl1[i] + ck1[i] * a11 + ck2[i] * a21 +
							  ck3[i] * a31 + ck4[i] * a41;
				}
			}

			for (int64_t m = n; m < nidx; m += 1) {
				double *__restrict__ ck = coefficient_t + idx[m] * mo_num;
				double a1 = av1[m];

				for (int64_t i = 0; i < mo_num; ++i) {
					vgl1[i] += ck[i] * a1;
				}
			}
		});
	});

	return QMCKL_SUCCESS_DEVICE;
}

//**********
// FINALIZE MO BASIS
//**********

qmckl_exit_code_device
qmckl_finalize_mo_basis_device(qmckl_context_device context) {

	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE) {
		return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
									 "qmckl_finalize_mo_basis_device", NULL);
	}

	qmckl_context_struct_device *ctx = (qmckl_context_struct_device *)context;
	assert(ctx != NULL);
    
    sycl::queue queue = ctx->q;

	double *new_array = (double *)qmckl_malloc_device(
		context, ctx->ao_basis.ao_num * ctx->mo_basis.mo_num * sizeof(double));
	if (new_array == NULL) {
		return qmckl_failwith_device(context, QMCKL_ALLOCATION_FAILED_DEVICE,
									 "qmckl_finalize_mo_basis_device", NULL);
	}

	assert(ctx->mo_basis.coefficient != NULL);

	if (ctx->mo_basis.coefficient_t != NULL) {
		qmckl_exit_code_device rc =
			qmckl_free_device(context, ctx->mo_basis.coefficient_t);
		if (rc != QMCKL_SUCCESS_DEVICE) {
			return qmckl_failwith_device(
				context, rc, "qmckl_finalize_mo_basis_device", NULL);
		}
	}

	double *coefficient = ctx->mo_basis.coefficient;

	int64_t ao_num = ctx->ao_basis.ao_num;
	int64_t mo_num = ctx->mo_basis.mo_num;

    queue.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<2>(ao_num, mo_num), [=](sycl::id<2> idx) {
            auto i = idx[0];
            auto j = idx[1];

			new_array[i * mo_num + j] = coefficient[j * ao_num + i];
        });
	});

	ctx->mo_basis.coefficient_t = new_array;
	qmckl_exit_code_device rc = QMCKL_SUCCESS_DEVICE;
	return rc;
}
