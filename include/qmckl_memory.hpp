#pragma once

// This file contains functions prototypes for memory management functions

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <CL/sycl.hpp>

#include "qmckl_types.hpp"
#include "qmckl_context.hpp"

/* Allocs & frees */
void *qmckl_malloc_host(qmckl_context_device context,
						const qmckl_memory_info_struct_device info);

void *qmckl_malloc_device(sycl::queue &queue, qmckl_context_device context, size_t size);

qmckl_exit_code_device qmckl_free_host(qmckl_context_device context,
									   void *const ptr);

qmckl_exit_code_device qmckl_free_device(qmckl_context_device context,
										 void *const ptr);

/* Memcpys */

qmckl_exit_code_device qmckl_memcpy_H2D(qmckl_context_device context,
										void *const dest, void *const src,
										size_t size);
qmckl_exit_code_device qmckl_memcpy_D2H(qmckl_context_device context,
										void *const dest, void *const src,
										size_t size);
qmckl_exit_code_device qmckl_memcpy_D2D(qmckl_context_device context,
										void *const dest, void *const src,
										size_t size);