#include <CL/sycl.hpp>
#include "../include/qmckl_memory.hpp"
#include "../include/qmckl_context.hpp"

//namespace sycl = cl::sycl;

// This file contains functions to manage host memory in a
// qmckl_context_device (we will likely use these functions in exceptional cases
// only)

//**********
// HOST MEMORY
//**********

void *qmckl_malloc_host(sycl::queue queue, qmckl_context_device context,
                        const qmckl_memory_info_struct_device info)
{

    assert(qmckl_context_check_device(context) != QMCKL_NULL_CONTEXT_DEVICE);

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;

    // Allocation of memory using USM
    void *pointer = sycl::malloc_shared(info.size, queue);
    if (pointer == nullptr)
    {
        return nullptr;
    }

    qmckl_lock_device(context);

    try
    {
        // Lock the device
        queue.wait_and_throw();

        /* If qmckl_memory_struct is full, reallocate a larger one */
        if (ctx->memory.n_allocated == ctx->memory.array_size)
        {
            const size_t old_size = ctx->memory.array_size;
            qmckl_memory_info_struct_device *new_array = sycl::malloc_shared<qmckl_memory_info_struct_device>(
                2L * old_size, queue);

            if (new_array == nullptr)
            {
                sycl::free(pointer, queue);
                return nullptr;
            }

            // Free the old array
            sycl::free(ctx->memory.element, queue);

            // Update the context
            ctx->memory.element = new_array;
            ctx->memory.array_size = 2L * old_size;
        }

        // Find the first NULL entry
        size_t pos = 0;
        while (pos < ctx->memory.array_size && ctx->memory.element[pos].size > 0)
        {
            pos += 1;
        }
        assert(ctx->memory.element[pos].size == 0);

        // Copy info at the new location
        //sycl::memcpy(&ctx->memory.element[pos], &info, sizeof(qmckl_memory_info_struct_device), queue).wait();
        queue.memcpy(&ctx->memory.element[pos], &info, sizeof(qmckl_memory_info_struct_device)).wait();
        ctx->memory.element[pos].pointer = pointer;
        ctx->memory.n_allocated += 1;

        // Unlock the device
        queue.wait_and_throw();
    }
    catch(const sycl::exception e)
    {
        qmckl_unlock_device(context);
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return nullptr;
    }

    qmckl_unlock_device(context);

    return pointer;
}

qmckl_exit_code_device qmckl_free_host(sycl::queue &queue, qmckl_context_device context,
                                       void *const ptr)
{

    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
                                     "qmckl_free_host", NULL);
    }

    if (ptr == NULL)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
                                     "qmckl_free_host", "NULL pointer");
    }

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;

    qmckl_lock_device(context);
    {
        /* Find pointer in array of saved pointers */
        size_t pos = (size_t)0;
        while (pos < ctx->memory.array_size &&
               ctx->memory.element[pos].pointer != ptr)
        {
            pos += (size_t)1;
        }

        if (pos >= ctx->memory.array_size)
        {
            /* Not found */
            qmckl_unlock_device(context);
            return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
                                         "qmckl_free_host",
                                         "Pointer not found in context");
        }

        // Free using SYCL's free
        sycl::free(ptr, queue);

        memset(&(ctx->memory.element[pos]), 0,
               sizeof(qmckl_memory_info_struct_device));
        ctx->memory.n_allocated -= (size_t)1;
    }
    qmckl_unlock_device(context);

    return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device
qmckl_get_malloc_info_host(sycl::queue &queue, qmckl_context_device context, const void *ptr,
                           qmckl_memory_info_struct_device *info)
{

    assert(qmckl_context_check_device(context) != QMCKL_NULL_CONTEXT_DEVICE);

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;

    if (ptr == NULL)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
                                     "qmckl_get_malloc_info_host",
                                     "Null pointer");
    }

    if (info == NULL)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
                                     "qmckl_get_malloc_info_host",
                                     "Null pointer");
    }

    qmckl_lock_device(context);
    {
        /* Find the pointer entry */
        size_t pos = (size_t)0;
        while (pos < ctx->memory.array_size &&
               ctx->memory.element[pos].pointer != ptr)
        {
            pos += (size_t)1;
        }

        if (pos >= ctx->memory.array_size)
        {
            /* Not found */
            qmckl_unlock_device(context);
            return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
                                         "qmckl_get_malloc_info_host",
                                         "Pointer not found in context");
        }

        // Copy info using SYCL's memcpy
        //sycl::memcpy(info, &(ctx->memory.element[pos]), sizeof(qmckl_memory_info_struct_device), queue).wait();
        queue.memcpy(&info, &ctx->memory.element[pos], sizeof(qmckl_memory_info_struct_device)).wait();
    }
    qmckl_unlock_device(context);

    return QMCKL_SUCCESS_DEVICE;
}

