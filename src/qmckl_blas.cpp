#include "../include/qmckl_blas.hpp"

using namespace sycl;

// This file provides SYCL implementations of BLAS functions (mostly
// initialization and manipulation of vector, matrix, ... types). All functions
// accept device pointers.
// (funtions requiring OpenMP pragmas only)

//**********
// MATRIX
//**********

qmckl_matrix_device qmckl_matrix_alloc_device(qmckl_context_device context,
											  const int64_t size1,
											  const int64_t size2) {
	/* Should always be true by contruction */
	assert(size1 * size2 > (int64_t)0);

	qmckl_matrix_device result;

	result.size[0] = size1;
	result.size[1] = size2;

	result.data =
		(double *)qmckl_malloc_device(context, size1 * size2 * sizeof(double));

	if (result.data == NULL) {
		result.size[0] = (int64_t)0;
		result.size[1] = (int64_t)0;
	}
	double *data = result.data;

	return result;
}

qmckl_exit_code_device qmckl_matrix_free_device(qmckl_context_device context,
												qmckl_matrix_device *matrix) {
	/* Always true */
	assert(matrix->data != NULL);

	qmckl_exit_code_device rc;

	rc = qmckl_free_device(context, matrix->data);
	if (rc != QMCKL_SUCCESS_DEVICE) {
		return rc;
	}
	matrix->data = NULL;
	matrix->size[0] = (int64_t)0;
	matrix->size[1] = (int64_t)0;

	return QMCKL_SUCCESS_DEVICE;
}

qmckl_matrix_device qmckl_matrix_set_device(qmckl_matrix_device matrix,
											double value)
{
	// Recompute array size
	int prod_size = matrix.size[0] * matrix.size[1];

	double *data = matrix.data;

	// Create a queue with default device selector
	queue q;

	// Submit a command group to the queue
	q.submit([&](handler &h)
			 {
        // Access data on the device
        auto data_device = data;

        // Use parallel_for to parallelize the loop on the device
        h.parallel_for<class set_matrix_kernel>(range<1>(prod_size), [=](id<1> idx) {
            data_device[idx] = value;
        }); })
		.wait(); // Wait for the command group to finish

	return matrix;
}

qmckl_exit_code_device qmckl_transpose_device(qmckl_context_device context,
											  const qmckl_matrix_device A,
											  qmckl_matrix_device At)
{
	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
	{
		return QMCKL_INVALID_CONTEXT_DEVICE;
	}

	if (A.size[0] < 1)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_transpose_device",
									 "Invalid size for A");
	}

	if (At.data == NULL)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_transpose_device",
									 "Output matrix not allocated");
	}

	if (At.size[0] != A.size[1] || At.size[1] != A.size[0])
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_transpose_device",
									 "Invalid size for At");
	}

	double *A_data = A.data;
	int A_s0 = A.size[0];

	double *At_data = At.data;
	int At_s0 = At.size[0];
	int At_s1 = At.size[1];

	// Create a queue with default device selector
	queue q;

	// Submit a command group to the queue
	q.submit([&](handler &h)
			 {
        // Access data on the device
        auto A_data_device = A_data;
        auto At_data_device = At_data;

        // Use parallel_for to parallelize the nested loop on the device
        h.parallel_for<class transpose_kernel>(range<2>(At_s0, At_s1), [=](id<2> idx) {
            int64_t i = idx[0];
            int64_t j = idx[1];
			At_data[i + j * At_s0] = A_data[j + i * A_s0];
        }); })
		.wait(); // Wait for the command group to finish

	return QMCKL_SUCCESS_DEVICE;
}
//**********
// TENSOR
//**********

qmckl_tensor_device qmckl_tensor_alloc_device(qmckl_context_device context,
											  const int64_t order,
											  const int64_t *size) {
	/* Should always be true by construction */
	assert(order > 0);
	assert(order <= QMCKL_TENSOR_ORDER_MAX_DEVICE);
	assert(size != NULL);

	qmckl_tensor_device result;
	result.order = order;

	int64_t prod_size = (int64_t)1;
	for (int64_t i = 0; i < order; ++i) {
		assert(size[i] > (int64_t)0);
		result.size[i] = size[i];
		prod_size *= size[i];
	}

	result.data =
		(double *)qmckl_malloc_device(context, prod_size * sizeof(double));

	if (result.data == NULL) {
		memset(&result, 0, sizeof(qmckl_tensor_device));
	}

	return result;
}

qmckl_exit_code_device qmckl_tensor_free_device(qmckl_context_device context,
												qmckl_tensor_device *tensor) {
	/* Always true */
	assert(tensor->data != NULL);

	qmckl_exit_code_device rc;

	rc = qmckl_free_device(context, tensor->data);
	if (rc != QMCKL_SUCCESS_DEVICE) {
		return rc;
	}

	// TODO Memset to 0
	// memset(tensor, 0, sizeof(qmckl_tensor_device));

	return QMCKL_SUCCESS_DEVICE;
}


qmckl_tensor_device qmckl_tensor_set_device(qmckl_tensor_device tensor,
											double value)
{
	// Recompute array size
	int prod_size = 1;

	for (int i = 0; i < tensor.order; i++)
	{
		prod_size *= tensor.size[i];
	}

	double *data = tensor.data;

	// Create a queue with default device selector
	queue q;

	// Submit a command group to the queue
	q.submit([&](handler &h)
			 {
        // Access data on the device
        auto data_device = data;

        // Use parallel_for to parallelize
        h.parallel_for<class set_kernel>(range<1>(prod_size), [=](id<1> idx) {
            data_device[idx] = value;
        }); })
		.wait();

	return tensor;
}


