#pragma once
#include <CL/sycl.hpp>

#include "qmckl_types.hpp"
#include "qmckl_basic_functions.hpp"
#include "qmckl_context.hpp"
#include "qmckl_memory.hpp"
#include "qmckl_blas.hpp"


bool qmckl_nucleus_provided_device(qmckl_context_device context);

qmckl_exit_code_device qmckl_get_nucleus_num_device(qmckl_context_device context, int64_t *num);

qmckl_exit_code_device qmckl_set_nucleus_num_device(qmckl_context_device context, int64_t num);

qmckl_exit_code_device qmckl_set_nucleus_charge_device(qmckl_context_device context, double *charge,
								int64_t size_max);
qmckl_exit_code_device qmckl_set_nucleus_coord_device(qmckl_context_device context, char transp,
							   double *coord, int64_t size_max);

qmckl_exit_code_device qmckl_finalize_nucleus_basis_hpc_device(qmckl_context_device context);
qmckl_exit_code_device qmckl_finalize_nucleus_basis_device(qmckl_context_device context);

qmckl_exit_code_device qmckl_get_nucleus_coord_device(const qmckl_context_device context,
							   const char transp, double *const coord,
							   const int64_t size_max);

qmckl_exit_code_device qmckl_get_nucleus_charge_device(const qmckl_context_device context,
								double *const charge, const int64_t size_max);
