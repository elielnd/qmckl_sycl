#include "../include/qmckl_nucleus.hpp"
#include "../include/qmckl_blas.hpp"

using namespace sycl;

qmckl_exit_code_device qmckl_finalize_nucleus_basis_hpc_device(qmckl_context_device context) {

	qmckl_context_struct_device *ctx = (qmckl_context_struct_device *)context;
	qmckl_memory_info_struct_device mem_info =
		qmckl_memory_info_struct_zero_device;

	ctx->ao_basis.prim_num_per_nucleus = (int32_t *)qmckl_malloc_device(
		context, ctx->nucleus.num * sizeof(int32_t));

	/* Find max number of primitives per nucleus */

	// Extract arrays from context
	int64_t *nucleus_shell_num = ctx->ao_basis.nucleus_shell_num;
	int64_t *nucleus_index = ctx->ao_basis.nucleus_index;
	int64_t *shell_prim_num = ctx->ao_basis.shell_prim_num;
	int32_t *prim_num_per_nucleus = ctx->ao_basis.prim_num_per_nucleus;

	int64_t shell_max = 0;
	int64_t prim_max = 0;
	int64_t nucl_num = ctx->nucleus.num;

	int64_t *shell_max_ptr = &shell_max;
	int64_t *prim_max_ptr = &prim_max;

    queue q;
    q.submit([&](handler &h) {
        auto shell_max_ptr_device = shell_max_ptr;
        auto prim_max_ptr_device = prim_max_ptr;

        h.parallel_for<class find_max_primitives_kernel>(range<1>(nucl_num), [=](id<1> inucl) {
            int64_t i = inucl[0];

            shell_max_ptr_device[0] = nucleus_shell_num[i] > shell_max_ptr_device[0] ? nucleus_shell_num[i] : shell_max_ptr_device[0];
            
            int64_t prim_num = 0;
            for (int64_t ishell = nucleus_index[i]; ishell < nucleus_index[i] + nucleus_shell_num[i]; ++ishell) {
                prim_num += shell_prim_num[ishell];
            }

            prim_max_ptr_device[0] = prim_num > prim_max_ptr_device[0] ? prim_num : prim_max_ptr_device[0];
            prim_num_per_nucleus[i] = prim_num;
           
        });
    }).wait();

    int64_t size[3] = {prim_max, shell_max, nucl_num};
	ctx->ao_basis.coef_per_nucleus = qmckl_tensor_alloc_device(context, 3, size);
	ctx->ao_basis.coef_per_nucleus = qmckl_tensor_set_device(ctx->ao_basis.coef_per_nucleus, 0.);
	ctx->ao_basis.expo_per_nucleus = qmckl_matrix_alloc_device(context, prim_max, nucl_num);
	ctx->ao_basis.expo_per_nucleus = qmckl_matrix_set_device(ctx->ao_basis.expo_per_nucleus, 0.);


    // To avoid offloading structures, expo is split in two arrays :
	// struct combined expo[prim_max];
	// ... gets replaced by :
	double *expo_expo = (double *)qmckl_malloc_device(context, prim_max * sizeof(double));
	int64_t *expo_index = (int64_t*)qmckl_malloc_device(context, prim_max * sizeof(double));

	double *coef = (double *)qmckl_malloc_device(context, shell_max * prim_max * sizeof(double));
	double *newcoef = (double *)qmckl_malloc_device(context, prim_max * sizeof(double));

	int64_t *newidx = (int64_t *)qmckl_malloc_device(context, prim_max * sizeof(int64_t));

	int64_t *shell_prim_index = ctx->ao_basis.shell_prim_index;
	double *exponent = ctx->ao_basis.exponent;
	double *coefficient_normalized = ctx->ao_basis.coefficient_normalized;

	double *expo_per_nucleus_data = ctx->ao_basis.expo_per_nucleus.data;
	int expo_per_nucleus_s0 = ctx->ao_basis.expo_per_nucleus.size[0];

	double *coef_per_nucleus_data = ctx->ao_basis.coef_per_nucleus.data;
	int coef_per_nucleus_s0 = ctx->ao_basis.coef_per_nucleus.size[0];
	int coef_per_nucleus_s1 = ctx->ao_basis.coef_per_nucleus.size[1];

     q.submit([&](handler &h) {
        auto expo_expo_device = expo_expo;
        auto expo_index_device = expo_index;
        auto coef_device = coef;
        auto newcoef_device = newcoef;
        auto newidx_device = newidx;
        auto nucleus_index_device = nucleus_index;
        auto shell_prim_index_device = shell_prim_index;
        auto nucleus_shell_num_device = nucleus_shell_num;
        auto exponent_device = exponent;
        auto coefficient_normalized_device = coefficient_normalized;
        auto expo_per_nucleus_data_device = expo_per_nucleus_data;
        auto coef_per_nucleus_data_device = coef_per_nucleus_data;
        auto prim_num_per_nucleus_device = prim_num_per_nucleus;

        h.parallel_for<class process_data_kernel>(range<1>(nucl_num), [=](id<1> inucl) {
            int64_t inucl_val = inucl[0];

            for (int i = 0; i < prim_max; i++) {
                expo_expo_device[i] = 0.;
                expo_index_device[i] = 0;
            }
            for (int i = 0; i < shell_max * prim_max; i++) {
                coef_device[i] = 0.;
            }

            int64_t idx = 0;
            int64_t ishell_start = nucleus_index_device[inucl_val];
            int64_t ishell_end = nucleus_index_device[inucl_val] + nucleus_shell_num_device[inucl_val];

            for (int64_t ishell = ishell_start; ishell < ishell_end; ++ishell) {
                int64_t iprim_start = shell_prim_index_device[ishell];
                int64_t iprim_end = shell_prim_index_device[ishell] + shell_prim_num[ishell];

                for (int64_t iprim = iprim_start; iprim < iprim_end; ++iprim) {
                    expo_expo_device[idx] = exponent_device[iprim];
                    expo_index_device[idx] = idx;
                    idx += 1;
                }
            }

            // Sort exponents
            double tmp;
            for (int i = 0; i < idx - 1; i++) {
                for (int j = 0; j < idx - i - 1; j++) {
                    if (expo_expo_device[j + 1] < expo_expo_device[j]) {
                        tmp = expo_expo_device[j + 1];
                        expo_expo_device[j + 1] = expo_expo_device[j];
                        expo_expo_device[j] = tmp;

                        tmp = expo_index_device[j + 1];
                        expo_index_device[j + 1] = expo_index_device[j];
                        expo_index_device[j] = tmp;
                    }
                }
            }

            idx = 0;
            int64_t idx2 = 0;
            for (int64_t ishell = ishell_start; ishell < ishell_end; ++ishell) {
                for (int i = 0; i < prim_max; i++) {
                    newcoef_device[i] = 0;
                }

                int64_t iprim_start = shell_prim_index_device[ishell];
                int64_t iprim_end = shell_prim_index_device[ishell] + shell_prim_num[ishell];

                for (int64_t iprim = iprim_start; iprim < iprim_end; ++iprim) {
                    newcoef_device[idx] = coefficient_normalized_device[iprim];
                    idx += 1;
                }

                for (int32_t i = 0; i < prim_num_per_nucleus_device[inucl_val]; ++i) {
                    idx2 = expo_index_device[i];
                    coef_device[(ishell - ishell_start) * prim_max + i] = newcoef_device[idx2];
                }
            }
            // Apply ordering to coefficients

            // Remove duplicates
            int64_t idxmax = 0;
            idx = 0;
            newidx_device[0] = 0;

            for (int32_t i = 1; i < prim_num_per_nucleus_device[inucl_val]; ++i) {
                if (expo_expo_device[i] != expo_expo_device[i - 1]) {
                    idx += 1;
                }
                newidx_device[i] = idx;
            }
            idxmax = idx;

            for (int32_t j = 0; j < ishell_end - ishell_start; ++j) {
                for (int i = 0; i < prim_max; i++) {
                    newcoef_device[i] = 0.;
                }
                for (int32_t i = 0; i < prim_num_per_nucleus_device[inucl_val]; ++i) {
                    newcoef_device[newidx_device[i]] += coef_device[j * prim_max +i];
                }
                for (int32_t i = 0; i < prim_num_per_nucleus_device[inucl_val]; ++i) {
					coef_device[j * prim_max + i] = newcoef_device[i];
				}
            }
            for (int32_t i = 0; i < prim_num_per_nucleus_device[inucl_val]; ++i) {
				expo_expo_device[newidx_device[i]] = expo_expo_device[i];
			}

            prim_num_per_nucleus_device[inucl_val] = (int32_t)idxmax + 1;

            for (int32_t i = 0; i < prim_num_per_nucleus_device[inucl_val]; ++i) {
				expo_per_nucleus_data_device[i + inucl_val * expo_per_nucleus_s0] = expo_expo_device[i];
			}

            for (int32_t j = 0; j < ishell_end - ishell_start; ++j) {
				for (int32_t i = 0; i < prim_num_per_nucleus_device[inucl_val]; ++i) {
					coef_per_nucleus_data_device[(i) + coef_per_nucleus_s0 * ((j) + coef_per_nucleus_s1 * (inucl))] = coef[j * prim_max + i];
				}
			}
        });
    }).wait();

    qmckl_free_device(context, expo_expo);
	qmckl_free_device(context, expo_index);
	qmckl_free_device(context, coef);
	qmckl_free_device(context, newcoef);
	qmckl_free_device(context, newidx);

	return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device qmckl_finalize_nucleus_basis_device(qmckl_context_device context) {

    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE) {
        return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
                                     "qmckl_finalize_nucleus_basis_device",
                                     NULL);
    }

    qmckl_context_struct_device *ctx = (qmckl_context_struct_device *)context;
	assert(ctx != NULL);
	int device_id = qmckl_get_device_id(context);

	int64_t nucl_num = 0;

	qmckl_exit_code_device rc = qmckl_get_nucleus_num_device(context, &nucl_num);
	if (rc != QMCKL_SUCCESS_DEVICE)
		return rc;

	/* nucleus_prim_index */
	{
		ctx->ao_basis.nucleus_prim_index = (int64_t *)qmckl_malloc_device(context, (ctx->nucleus.num + (int64_t)1) * sizeof(int64_t));

		if (ctx->ao_basis.nucleus_prim_index == NULL) {
			return qmckl_failwith_device(context,
										 QMCKL_ALLOCATION_FAILED_DEVICE,
										 "ao_basis.nucleus_prim_index", NULL);
		}

		// Extract arrays from context
		int64_t *nucleus_index = ctx->ao_basis.nucleus_index;
		int64_t *nucleus_prim_index = ctx->ao_basis.nucleus_prim_index;
		int64_t *shell_prim_index = ctx->ao_basis.shell_prim_index;

		int prim_num = ctx->ao_basis.prim_num;

        // DPC++ parallel_for loop to compute nucleus_prim_index on device
        queue q;
        q.parallel_for(nucl_num, [=](id<1> i) {
            int64_t shell_idx = nucleus_index[i];
            nucleus_prim_index[i] = shell_prim_index[shell_idx];
        });

        nucleus_prim_index[nucl_num] = prim_num;
    }

    /* Normalize coefficients */
	{
		ctx->ao_basis.coefficient_normalized = (double *)qmckl_malloc_device(context, ctx->ao_basis.prim_num * sizeof(double));

		if (ctx->ao_basis.coefficient_normalized == NULL) {
			return qmckl_failwith_device(
				context, QMCKL_ALLOCATION_FAILED_DEVICE,
				"ao_basis.coefficient_normalized", NULL);
		}

		// Extract arrays from context
		int64_t *shell_prim_index = ctx->ao_basis.shell_prim_index;
		int64_t *shell_prim_num = ctx->ao_basis.shell_prim_num;
		double *coefficient_normalized = ctx->ao_basis.coefficient_normalized;
		double *coefficient = ctx->ao_basis.coefficient;
		double *prim_factor = ctx->ao_basis.prim_factor;
		double *shell_factor = ctx->ao_basis.shell_factor;

		int shell_num = ctx->ao_basis.shell_num;

        // DPC++ parallel_for loop to compute coefficient_normalized on device
        queue q;
        q.parallel_for(shell_num, [=](id<1> ishell) {
            for (int64_t iprim = shell_prim_index[ishell]; iprim < shell_prim_index[ishell] + shell_prim_num[ishell]; ++iprim) {
                coefficient_normalized[iprim] = coefficient[iprim] * prim_factor[iprim] * shell_factor[ishell];
            }
        });
    }

    /* Find max angular momentum on each nucleus */
	{
		ctx->ao_basis.nucleus_max_ang_mom = (int32_t *)qmckl_malloc_device(context, ctx->nucleus.num * sizeof(int32_t));

		if (ctx->ao_basis.nucleus_max_ang_mom == NULL) {
			return qmckl_failwith_device(context,
										 QMCKL_ALLOCATION_FAILED_DEVICE,
										 "ao_basis.nucleus_max_ang_mom", NULL);
		}

		// Extract arrays from context
		int32_t *nucleus_max_ang_mom = ctx->ao_basis.nucleus_max_ang_mom;
		int64_t *nucleus_index = ctx->ao_basis.nucleus_index;
		int64_t *nucleus_shell_num = ctx->ao_basis.nucleus_shell_num;
		int32_t *shell_ang_mom = ctx->ao_basis.shell_ang_mom;

        // DPC++ parallel_for loop to compute nucleus_max_ang_mom on device
        queue q;
        q.parallel_for(nucl_num, [=](id<1> i) {
            nucleus_max_ang_mom[i] = 0;
            for (int64_t ishell = nucleus_index[i]; ishell < nucleus_index[i] + nucleus_shell_num[i]; ++ishell) {
                nucleus_max_ang_mom[i] = nucleus_max_ang_mom[i] > shell_ang_mom[ishell] ? nucleus_max_ang_mom[i] : shell_ang_mom[ishell];
            }
        });
    }

    /* Find distance beyond which all AOs are zero.
	   The distance is obtained by sqrt(log(cutoff)*range) */
	{
		if (ctx->ao_basis.type == 'G') {
			ctx->ao_basis.nucleus_range = (double *)qmckl_malloc_device(context, ctx->nucleus.num * sizeof(double));

			if (ctx->ao_basis.nucleus_range == NULL) {
				return qmckl_failwith_device(context,
											 QMCKL_ALLOCATION_FAILED_DEVICE,
											 "ao_basis.nucleus_range", NULL);
			}

			// Extract arrays from context
			double *nucleus_range = ctx->ao_basis.nucleus_range;
			int64_t *nucleus_index = ctx->ao_basis.nucleus_index;
			int64_t *nucleus_shell_num = ctx->ao_basis.nucleus_shell_num;
			int64_t *shell_prim_index = ctx->ao_basis.shell_prim_index;
			int64_t *shell_prim_num = ctx->ao_basis.shell_prim_num;
			double *exponent = ctx->ao_basis.exponent;

			int nucleus_num = ctx->nucleus.num;

            // DPC++ parallel_for loop to compute nucleus_range on device
            queue q;
            q.parallel_for(nucleus_num, [=](id<1> inucl) {
                nucleus_range[inucl] = 0.;
                for (int64_t ishell = nucleus_index[inucl]; ishell < nucleus_index[inucl] + nucleus_shell_num[inucl]; ++ishell) {
                    for (int64_t iprim = shell_prim_index[ishell]; iprim < shell_prim_index[ishell] + shell_prim_num[ishell]; ++iprim) {
                        double range = 1. / exponent[iprim];
                        nucleus_range[inucl] = nucleus_range[inucl] > range ? nucleus_range[inucl] : range;
                    }
                }
            });
        }
    }

    rc = qmckl_finalize_nucleus_basis_hpc_device(context);

    return rc;
}



//**********
// GETTERS
//**********

qmckl_exit_code_device qmckl_get_nucleus_num_device(const qmckl_context_device context,
							 int64_t *const num) {

	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (num == NULL) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_get_nucleus_num",
									 "num is a null pointer");
	}

	qmckl_context_struct_device *const ctx =
		(qmckl_context_struct_device *)context;
	assert(ctx != NULL);

	int32_t mask = 1 << 0;

	if ((ctx->nucleus.uninitialized & mask) != 0) {
		*num = (int64_t)0;
		return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
									 "qmckl_get_nucleus_num",
									 "nucleus data is not provided");
	}

	assert(ctx->nucleus.num >= (int64_t)0);
	*num = ctx->nucleus.num;

	return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device qmckl_get_nucleus_coord_device(const qmckl_context_device context,
							   const char transp, double *const coord,
							   const int64_t size_max) {

	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (transp != 'N' && transp != 'T') {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_get_nucleus_coord_device",
									 "transp should be 'N' or 'T'");
	}

	if (coord == NULL) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_get_nucleus_coord_device",
									 "coord is a null pointer");
	}

	qmckl_context_struct_device *const ctx =
		(qmckl_context_struct_device *)context;
	assert(ctx != NULL);

	int32_t mask = 1 << 2;

	if ((ctx->nucleus.uninitialized & mask) != 0) {
		return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
									 "qmckl_get_nucleus_coord_device",
									 "nucleus data is not provided");
	}

	assert(ctx->nucleus.coord.data != NULL);

	qmckl_exit_code_device rc;

	if (transp == 'N') {
		qmckl_matrix_device At =
			qmckl_matrix_alloc_device(context, 3, ctx->nucleus.coord.size[0]);
		rc = qmckl_transpose_device(context, ctx->nucleus.coord, At);
		if (rc != QMCKL_SUCCESS_DEVICE)
			return rc;

		// Copy content of At onto coord
		// rc = qmckl_double_of_matrix_device(context, At, coord, size_max);
		qmckl_memcpy_D2D(context, coord, At.data,
						 At.size[0] * At.size[1] * sizeof(double));
		qmckl_matrix_free_device(context, &At);
	} else {
		// Copy content of ctx->nucleus.coord onto coord
		// rc = qmckl_double_of_matrix_device(context, ctx->nucleus.coord,
		// coord, 							size_max);
		qmckl_memcpy_D2D(context, coord, ctx->nucleus.coord.data,
						 ctx->nucleus.coord.size[0] *
							 ctx->nucleus.coord.size[1] * sizeof(double));
	}

	return rc;
}

qmckl_exit_code_device
qmckl_get_nucleus_charge_device(const qmckl_context_device context,
								double *const charge, const int64_t size_max) {

	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE) {
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (charge == NULL) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_get_nucleus_charge",
									 "charge is a null pointer");
	}

	qmckl_context_struct_device *const ctx =
		(qmckl_context_struct_device *)context;
	assert(ctx != NULL);

	int32_t mask = 1 << 1;

	if ((ctx->nucleus.uninitialized & mask) != 0) {
		return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
									 "qmckl_get_nucleus_charge_device",
									 "nucleus data is not provided");
	}

	assert(ctx->nucleus.charge.data != NULL);

	if (ctx->nucleus.num > size_max) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_get_nucleus_charge",
									 "Array too small");
	}

	// Copy content of ctx->nucleus.charge onto charge
	// rc = qmckl_double_of_vector_device(context, ctx->nucleus.charge, charge,
	// size_max);
	qmckl_memcpy_D2D(context, charge, ctx->nucleus.charge.data,
					 ctx->nucleus.charge.size * sizeof(double));

	return QMCKL_SUCCESS_DEVICE;
}
