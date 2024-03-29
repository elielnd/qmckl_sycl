#pragma once
// This file contains functions prototypes for BLAS related functions
// (mostly manipulation of the device, matrix and tensor types)
#include <CL/sycl.hpp>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "qmckl_types.hpp"
#include "qmckl_memory.hpp"
#include "qmckl_context.hpp"
#include "qmckl_nucleus.hpp"

//**********
// VECTOR
//**********
qmckl_vector_device qmckl_vector_alloc_device(qmckl_context_device context, const int64_t size);
qmckl_exit_code_device qmckl_vector_free_device(qmckl_context_device context, qmckl_vector_device *vector);
qmckl_exit_code_device qmckl_vector_of_double_device(const qmckl_context_device context, const double *target, const int64_t size_max, qmckl_vector_device *vector_out);

//**********
// MATRIX
//**********
qmckl_matrix_device qmckl_matrix_alloc_device(qmckl_context_device context, const int64_t size1, const int64_t size2);
qmckl_exit_code_device qmckl_matrix_of_double_device(const qmckl_context_device context, const double *target, const int64_t size_max, qmckl_matrix_device *matrix_out);
qmckl_exit_code_device qmckl_matrix_free_device(qmckl_context_device context, qmckl_matrix_device *matrix);
qmckl_matrix_device qmckl_matrix_set_device(qmckl_matrix_device &matrix, double value, sycl::queue &q);
qmckl_exit_code_device qmckl_transpose_device(qmckl_context_device context, const qmckl_matrix_device &A, qmckl_matrix_device &At);

//**********
// TENSOR
//**********
qmckl_tensor_device qmckl_tensor_alloc_device(qmckl_context_device context, const int64_t order, const int64_t *size);
qmckl_exit_code_device qmckl_tensor_free_device(qmckl_context_device context, qmckl_tensor_device *tensor);
qmckl_tensor_device qmckl_tensor_set_device(qmckl_tensor_device tensor, double value, sycl::queue &q);
