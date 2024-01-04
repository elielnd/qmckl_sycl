#include "../include/qmckl_memory.hpp"
#include <assert.h>
#include <CL/sycl.hpp>

// This file contains functions prototypes for context memory management
// functions (on device only, we expect most if not all of the context
// memory to be allocated on device in most cases)
// (OpenMP implementations)

//**********
// ALLOCS / FREES
//**********

void *qmckl_malloc_device(sycl::queue &queue, qmckl_context_device context, size_t size)
{
	assert(qmckl_context_check_device(context) != QMCKL_NULL_CONTEXT_DEVICE);

	qmckl_context_struct_device *const ctx =
		reinterpret_cast<qmckl_context_struct_device *>(context);
	int device_id = qmckl_get_device_id(context);

	// Allocate memory and zero it using USM
	void *pointer = static_cast<void *>(sycl::malloc_shared(size, queue));
	if (pointer == nullptr)
	{
		return nullptr;
	}
	
	qmckl_lock_device(context);
	{
		// If qmckl_memory_struct is full, reallocate a larger one
		if (ctx->memory_device.n_allocated == ctx->memory_device.array_size)
		{
			const size_t old_size = ctx->memory_device.array_size;
			qmckl_memory_info_struct_device *new_array = static_cast<qmckl_memory_info_struct_device *>(
				sycl::malloc_device(2L * old_size * sizeof(qmckl_memory_info_struct_device), queue));

			if (new_array == nullptr)
			{
				qmckl_unlock_device(context);
				return nullptr;
			}

			memset(&(new_array[old_size]), 0, old_size * sizeof(qmckl_memory_info_struct_device));
			queue.memcpy(new_array, ctx->memory_device.element,
						 old_size * sizeof(qmckl_memory_info_struct_device))
				.wait_and_throw();

			ctx->memory_device.element = new_array;
			ctx->memory_device.array_size = 2L * old_size;
		}

		// Find first NULL entry
		size_t pos = (size_t)0;
		while (pos < ctx->memory_device.array_size &&
			   ctx->memory_device.element[pos].size > (size_t)0)
		{
			pos += (size_t)1;
		}
		assert(ctx->memory_device.element[pos].size == (size_t)0);

		// Copy info at the new location
		ctx->memory_device.element[pos].size = size;
		ctx->memory_device.element[pos].pointer = pointer;
		ctx->memory_device.n_allocated += (size_t)1;
	}
	qmckl_unlock_device(context);

	return pointer;
}

qmckl_exit_code_device qmckl_free_device(sycl::queue &queue, qmckl_context_device context, void *const ptr)
{
	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
									 "qmckl_free_device", nullptr);
	}

	if (ptr == nullptr)
	{
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
			   ctx->memory_device.element[pos].pointer != ptr)
		{
			pos += (size_t)1;
		}

		if (pos >= ctx->memory_device.array_size)
		{
			/* Not found */
			qmckl_unlock_device(context);
			return qmckl_failwith_device(context, QMCKL_FAILURE_DEVICE,
										 "qmckl_free_device",
										 "Pointer not found in context");
		}

		sycl::free(ptr, queue);

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

qmckl_exit_code_device qmckl_memcpy_H2D(qmckl_context_device context,
										void *const dest, void *const src,
										size_t size)
{
	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
									 "qmckl_memcpy_H2D", nullptr);
	}

	if (dest == nullptr)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_memcpu_H2D", "NULL dest pointer");
	}

	if (src == nullptr)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_memcpu_H2D", "NULL src pointer");
	}

	qmckl_lock_device(context);
	{
		try
		{
			// Get the SYCL queue associated with the context
			sycl::queue q;

			// Use USM for memory management
			// Memory allocation and data copy to device
			void *dest_device = malloc_device(size, q);
			void *src_device = malloc_device(size, q);
			q.memcpy(dest_device, dest, size);
			q.memcpy(src_device, src, size);

			// Perform data transfer using SYCL command group
			q.submit([&](sycl::handler &h)
					 { h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i)
									  { reinterpret_cast<char *>(dest_device)[i] =
											reinterpret_cast<char *>(src_device)[i]; }); });

		}
		catch (sycl::exception const &e)
		{
			// Handle exceptions, if any
			std::cerr << "SYCL Exception: " << e.what() << std::endl;
			return qmckl_failwith_device(context, QMCKL_FAILURE_DEVICE,
										 "qmckl_memcpy_H2D", "SYCL exception");
		}
	}
	qmckl_unlock_device(context);

	return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device qmckl_memcpy_D2H(qmckl_context_device context,
										void *const dest, void *const src,
										size_t size)
{
	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
									 "qmckl_memcpy_D2H", nullptr);
	}

	if (dest == nullptr)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_memcpu_D2H", "NULL dest pointer");
	}

	if (src == nullptr)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_memcpu_D2H", "NULL src pointer");
	}

	qmckl_lock_device(context);
	{
		try
		{
			// Get the SYCL queue associated with the context
			sycl::queue q;

			// Use USM for memory management
			// Memory allocation and data copy to device
			void *dest_device = malloc_device(size, q);
			void *src_device = malloc_device(size, q);
			q.memcpy(dest_device, dest, size);
			q.memcpy(src_device, src, size);

			// Perform data transfer using SYCL command group
			q.submit([&](sycl::handler &h)
					 { h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i)
									  { reinterpret_cast<char *>(dest_device)[i] =
											reinterpret_cast<char *>(src_device)[i]; }); });

		}
		catch (sycl::exception const &e)
		{
			// Handle exceptions, if any
			std::cerr << "SYCL Exception: " << e.what() << std::endl;
			return qmckl_failwith_device(context, QMCKL_FAILURE_DEVICE,
										 "qmckl_memcpy_D2H", "SYCL exception");
		}
	}
	qmckl_unlock_device(context);

	return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device qmckl_memcpy_D2D(qmckl_context_device context,
										void *const dest, void *const src,
										size_t size)
{
	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
									 "qmckl_memcpy_D2D", nullptr);
	}

	if (dest == nullptr)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
									 "qmckl_memcpu_D2D", "NULL dest pointer");
	}

	if (src == nullptr)
	{
		return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
									 "qmckl_memcpu_D2D", "NULL src pointer");
	}

	qmckl_lock_device(context);
	{
		try
		{
			// Get the SYCL queue associated with the context
			sycl::queue q;

			// Use USM for memory management
			// Memory allocation and data copy to device
			void *dest_device = malloc_device(size, q);
			void *src_device = malloc_device(size, q);
			q.memcpy(dest_device, dest, size);
			q.memcpy(src_device, src, size);

			// Perform data transfer using SYCL command group
			q.submit([&](sycl::handler &h)
					 { h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i)
									  { reinterpret_cast<char *>(dest_device)[i] =
											reinterpret_cast<char *>(src_device)[i]; }); });

		}
		catch (sycl::exception const &e)
		{
			// Handle exceptions, if any
			std::cerr << "SYCL Exception: " << e.what() << std::endl;
			return qmckl_failwith_device(context, QMCKL_FAILURE_DEVICE,
										 "qmckl_memcpy_D2D", "SYCL exception");
		}
	}
	qmckl_unlock_device(context);

	return QMCKL_SUCCESS_DEVICE;
}