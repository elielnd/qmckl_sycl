#pragma once
// This file contains prototypes for device context related functions,
// as well as the definition of qmckl_context_device and
// qmckl_context_device_struct, the device specific datatypes for context

#include <cassert>
#include <cerrno>
#include <cmath>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <mutex>

#include "qmckl_types.hpp"
#include "qmckl_basic_functions.hpp"
#include "qmckl_memory.hpp"
#include "qmckl_blas.hpp"

qmckl_exit_code_device qmckl_context_touch_device(const qmckl_context_device context);

qmckl_exit_code_device qmckl_init_point_device(qmckl_context_device context);
qmckl_exit_code_device qmckl_init_ao_basis_device(qmckl_context_device context);
qmckl_exit_code_device qmckl_init_mo_basis_device(qmckl_context_device context);
qmckl_exit_code_device
qmckl_init_determinant_device(qmckl_context_device context);
qmckl_exit_code_device qmckl_init_jastrow_device(qmckl_context_device context);

qmckl_context_device qmckl_context_create_device(int device_id);
qmckl_exit_code_device
qmckl_context_destroy_device(sycl::queue queue, const qmckl_context_device context);

static inline size_t qmckl_get_device_id(qmckl_context_device context)
{
	qmckl_context_struct_device *const ctx =
		(qmckl_context_struct_device *)context;
	return ctx->device_id;
}

qmckl_context_device qmckl_context_check_device(const qmckl_context_device context);

void qmckl_lock_device(qmckl_context_device context);
void qmckl_unlock_device(qmckl_context_device context);