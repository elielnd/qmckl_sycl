#include "../include/qmckl_memory.hpp"
#include <assert.h>

// This file contains functions prototypes for context memory management
// functions (on device only, we expect most if not all of the context
// memory to be allocated on device in most cases)
// (SYCL implémentations)


//**********
// ALLOCS / FREES
//**********
#include "../include/qmckl_memory.hpp"
#include <assert.h>

// This file contains functions prototypes for context memory management
// functions (on device only, we expect most if not all of the context
// memory to be allocated on device in most cases)
// (OpenMP implementations)

//**********
// ALLOCS / FREES
//**********

qmckl_exit_code_device qmckl_free_device(qmckl_context_device context,
										 void *const ptr) {

	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE) {
		return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
									 "qmckl_free_device", NULL);
	}

	if (ptr == NULL) {
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_free_device", "NULL pointer");
	}

	qmckl_context_struct_device *const ctx =
		(qmckl_context_struct_device *)context;
	int device_id = qmckl_get_device_id(context);

	qmckl_lock_device(context);

	{
		/* Find pointer in array of saved pointers */
		size_t pos = (size_t)0;
		while (pos < ctx->memory_device.array_size &&
			   ctx->memory_device.element[pos].pointer != ptr) {
			pos += (size_t)1;
		}

		if (pos >= ctx->memory_device.array_size) {
			/* Not found */
			qmckl_unlock_device(context);
			return qmckl_failwith_device(context, QMCKL_FAILURE_DEVICE,
										 "qmckl_free_device",
										 "Pointer not found in context");
		}


		memset(&(ctx->memory_device.element[pos]), 0,
			   sizeof(qmckl_memory_info_struct_device));
		ctx->memory_device.n_allocated -= (size_t)1;
	}
	qmckl_unlock_device(context);

	return QMCKL_SUCCESS_DEVICE;
}

//**********
// MEMCPYS
//**********