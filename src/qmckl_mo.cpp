#include "include/qmckl_mo.hpp"
#include "include/qmckl_ao.hpp"

#include <CL/sycl.hpp>

using namespace sycl;

/* Provided check  */

bool qmckl_mo_basis_provided_device(qmckl_context_device context)
{

    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return false;
    }

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    return ctx->mo_basis.provided;
}

/* mo_select */

// Forward declare this, as its needed by select_mo
qmckl_exit_code_device
qmckl_finalize_mo_basis_device(qmckl_context_device context);

bool qmckl_mo_basis_select_mo_device(qmckl_context_device context,
                                     int32_t *keep, int64_t size_max)
{
    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return qmckl_failwith_device(context, QMCKL_NULL_CONTEXT_DEVICE,
                                     "qmckl_get_mo_basis_select_mo_device",
                                     NULL);
    }

    // WARNING Here, we are expecting a CPU array (instead of a GPU array
    // usually), because it will not be used as a data to be stored in the
    // context. Thus, it makes more sense (and is actually more efficient) to
    // use a CPU array.

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    if (!(qmckl_mo_basis_provided_device(context)))
    {
        return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
                                     "qmckl_get_mo_basis_select_mo_device",
                                     NULL);
    }

    if (keep == NULL)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
                                     "qmckl_get_mo_basis_select_mo_device",
                                     "NULL pointer");
    }

    const int64_t mo_num = ctx->mo_basis.mo_num;
    const int64_t ao_num = ctx->ao_basis.ao_num;

    if (size_max < mo_num)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
                                     "qmckl_get_mo_basis_select_mo",
                                     "Array too small: expected mo_num.");
    }

    int64_t mo_num_new = 0;
    for (int64_t i = 0; i < mo_num; ++i)
    {
        if (keep[i] != 0)
            ++mo_num_new;
    }

    double *__restrict__ coefficient = reinterpret_cast<double *>(qmckl_malloc_device(
        context, ao_num * mo_num_new * sizeof(double)));

    int64_t k = 0;
    for (int64_t i = 0; i < mo_num; ++i)
    {
        if (keep[i] != 0)
        {
            qmckl_memcpy_D2D(context, &(coefficient[k * ao_num]),
                             &(ctx->mo_basis.coefficient[i * ao_num]),
                             ao_num * sizeof(double));
            ++k;
        }
    }

    qmckl_exit_code_device rc =
        qmckl_free_device(context, ctx->mo_basis.coefficient);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return rc;

    ctx->mo_basis.coefficient = coefficient;
    ctx->mo_basis.mo_num = mo_num_new;

    rc = qmckl_finalize_mo_basis_device(context);
    return rc;
}

//**********
// PROVIDE
//**********

/* mo_vgl */

qmckl_exit_code_device
qmckl_provide_mo_basis_mo_vgl_device(qmckl_context_device context)
{

    qmckl_exit_code_device rc = QMCKL_SUCCESS_DEVICE;

    if (qmckl_context_check_device((qmckl_context_device)context) ==
        QMCKL_NULL_CONTEXT_DEVICE)
    {
        return qmckl_failwith_device(context, QMCKL_NULL_CONTEXT_DEVICE,
                                     "qmckl_provide_mo_basis_mo_vgl_device",
                                     NULL);
    }

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    if (!ctx->mo_basis.provided)
    {
        return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
                                     "qmckl_provide_mo_basis_mo_vgl_device",
                                     NULL);
    }

    /* Compute if necessary */
    if (ctx->point.date > ctx->mo_basis.mo_vgl_date)
    {

        /* Allocate array */
        if (ctx->mo_basis.mo_vgl == NULL)
        {

            double *mo_vgl = reinterpret_cast<double *>(qmckl_malloc_device(
                context,
                5 * ctx->mo_basis.mo_num * ctx->point.num * sizeof(double)));

            if (mo_vgl == NULL)
            {
                return qmckl_failwith_device(context,
                                             QMCKL_ALLOCATION_FAILED_DEVICE,
                                             "qmckl_mo_basis_mo_vgl", NULL);
            }
            ctx->mo_basis.mo_vgl = mo_vgl;
        }

        rc = qmckl_provide_ao_basis_ao_vgl_device(context);
        if (rc != QMCKL_SUCCESS_DEVICE)
        {
            return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
                                         "qmckl_ao_basis", NULL);
        }

        rc = qmckl_compute_mo_basis_mo_vgl_device(
            context, ctx->ao_basis.ao_num, ctx->mo_basis.mo_num, ctx->point.num,
            ctx->mo_basis.coefficient_t, ctx->ao_basis.ao_vgl,
            ctx->mo_basis.mo_vgl);

        if (rc != QMCKL_SUCCESS_DEVICE)
        {
            return rc;
        }

        ctx->mo_basis.mo_vgl_date = ctx->date;
    }

    return QMCKL_SUCCESS_DEVICE;
}

/* mo_value */

qmckl_exit_code_device
qmckl_provide_mo_basis_mo_value_device(qmckl_context_device context)
{

    qmckl_exit_code_device rc = QMCKL_SUCCESS_DEVICE;

    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return qmckl_failwith_device(context, QMCKL_NULL_CONTEXT_DEVICE,
                                     "qmckl_provide_mo_basis_mo_value_device",
                                     NULL);
    }

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    sycl::queue q = ctx->q;

    if (!ctx->mo_basis.provided)
    {
        return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
                                     "qmckl_provide_mo_basis_mo_value_device",
                                     NULL);
    }

    /* Compute if necessary */
    if (ctx->point.date > ctx->mo_basis.mo_value_date)
    {
        qmckl_exit_code_device rc;

        /* Allocate array */
        if (ctx->mo_basis.mo_value == NULL)
        {
            double *mo_value = reinterpret_cast<double *>(qmckl_malloc_device(
                context,
                ctx->mo_basis.mo_num * ctx->point.num * sizeof(double)));

            if (mo_value == NULL)
            {
                return qmckl_failwith_device(context,
                                             QMCKL_ALLOCATION_FAILED_DEVICE,
                                             "qmckl_mo_basis_mo_value", NULL);
            }
            ctx->mo_basis.mo_value = mo_value;
        }

        if (ctx->mo_basis.mo_vgl_date == ctx->point.date)
        {

            // mo_vgl has been computed at this step: Just copy the data.

            double *v = &(ctx->mo_basis.mo_value[0]);
            double *vgl = &(ctx->mo_basis.mo_vgl[0]);

            // buffer<double, 1> v_buffer(v, range<1>(ctx->mo_basis.mo_num * ctx->point.num));
            // buffer<double, 1> vgl_buffer(vgl, range<1>(ctx->mo_basis.mo_num * ctx->point.num * 5));

            q.submit([&](sycl::handler &h)
                     {
                // Accessors for buffers
                // auto v_acc = v_buffer.get_access<sycl::access::mode::write>(h);
                // auto vgl_acc = vgl_buffer.get_access<sycl::access::mode::read>(h);

                // Kernel
                h.parallel_for<class myKernel>(sycl::range<1>(ctx->point.num), [=](sycl::id<1> idx)
                                               {
                    for (int k = 0; k < ctx->mo_basis.mo_num; ++k) 
                    {
                        v[idx * ctx->mo_basis.mo_num + k] = vgl[idx * ctx->mo_basis.mo_num * 5 + k];
                    } 
                }); });
            q.wait();
            // v_buffer.get_host_access();
            // vgl_buffer.get_host_access();
        }
        else
        {
            rc = qmckl_provide_ao_basis_ao_value_device(context);
            if (rc != QMCKL_SUCCESS_DEVICE)
            {
                return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
                                             "qmckl_ao_basis_ao_value_device",
                                             NULL);
            }

            rc = qmckl_compute_mo_basis_mo_value_device(
                context, ctx->ao_basis.ao_num, ctx->mo_basis.mo_num,
                ctx->point.num, ctx->mo_basis.coefficient_t,
                ctx->ao_basis.ao_value, ctx->mo_basis.mo_value);
        }

        if (rc != QMCKL_SUCCESS_DEVICE)
        {
            return rc;
        }

        ctx->mo_basis.mo_value_date = ctx->date;
    }

    return QMCKL_SUCCESS_DEVICE;
}

//**********
// GET
//**********

/* mo_vgl */

qmckl_exit_code_device
qmckl_get_mo_basis_mo_vgl_device(qmckl_context_device context,
                                 double *const mo_vgl, const int64_t size_max)
{

    if (qmckl_context_check_device((qmckl_context_device)context) ==
        QMCKL_NULL_CONTEXT_DEVICE)
    {
        return QMCKL_NULL_CONTEXT_DEVICE;
    }

    qmckl_exit_code_device rc;

    rc = qmckl_provide_mo_basis_mo_vgl_device(context);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return rc;

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    int64_t sze = 5 * ctx->point.num * ctx->mo_basis.mo_num;
    if (size_max < sze)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
                                     "qmckl_get_mo_basis_mo_vgl",
                                     "input array too small");
    }
    qmckl_memcpy_D2D(context, mo_vgl, ctx->mo_basis.mo_vgl,
                     sze * sizeof(double));

    return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device
qmckl_get_mo_basis_mo_vgl_inplace_device(qmckl_context_device context,
                                         double *const mo_vgl,
                                         const int64_t size_max)
{

    if (qmckl_context_check_device((qmckl_context_device)context) ==
        QMCKL_NULL_CONTEXT_DEVICE)
    {
        return qmckl_failwith_device((qmckl_context_device)context,
                                     QMCKL_NULL_CONTEXT_DEVICE,
                                     "qmckl_get_mo_basis_mo_vgl_device", NULL);
    }

    qmckl_exit_code_device rc;

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    const int64_t sze = 5 * ctx->mo_basis.mo_num * ctx->point.num;
    if (size_max < sze)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
                                     "qmckl_get_mo_basis_mo_vgl_device",
                                     "input array too small");
    }

    rc = qmckl_context_touch_device(context);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return rc;

    double *old_array = ctx->mo_basis.mo_vgl;

    ctx->mo_basis.mo_vgl = mo_vgl;

    rc = qmckl_provide_mo_basis_mo_vgl_device(context);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return rc;

    ctx->mo_basis.mo_vgl = old_array;

    return QMCKL_SUCCESS_DEVICE;
}

/* mo_value */

qmckl_exit_code_device
qmckl_get_mo_basis_mo_value_device(qmckl_context_device context,
                                   double *const mo_value,
                                   const int64_t size_max)
{

    if (qmckl_context_check_device((qmckl_context_device)context) ==
        QMCKL_NULL_CONTEXT_DEVICE)
    {
        return QMCKL_NULL_CONTEXT_DEVICE;
    }

    qmckl_exit_code_device rc;

    rc = qmckl_provide_mo_basis_mo_value_device(context);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return rc;

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    const int64_t sze = ctx->point.num * ctx->mo_basis.mo_num;
    if (size_max < sze)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
                                     "qmckl_get_mo_basis_mo_value",
                                     "input array too small");
    }
    qmckl_memcpy_D2D(context, mo_value, ctx->mo_basis.mo_value,
                     sze * sizeof(double));

    return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device
qmckl_get_mo_basis_mo_value_inplace_device(qmckl_context_device context,
                                           double *const mo_value,
                                           const int64_t size_max)
{

    if (qmckl_context_check_device((qmckl_context_device)context) ==
        QMCKL_NULL_CONTEXT_DEVICE)
    {
        return qmckl_failwith_device(
            (qmckl_context_device)context, QMCKL_NULL_CONTEXT_DEVICE,
            "qmckl_get_mo_basis_mo_value_device", NULL);
    }

    qmckl_exit_code_device rc;

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    const int64_t sze = ctx->mo_basis.mo_num * ctx->point.num;
    if (size_max < sze)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
                                     "qmckl_get_mo_basis_mo_value_device",
                                     "input array too small");
    }

    rc = qmckl_context_touch_device(context);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return rc;

    double *old_array = ctx->mo_basis.mo_value;

    ctx->mo_basis.mo_value = mo_value;

    rc = qmckl_provide_mo_basis_mo_value_device(context);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return rc;

    ctx->mo_basis.mo_value = old_array;

    return QMCKL_SUCCESS_DEVICE;
}

//**********
// VARIOUS GETTERS/SETTERS
//**********

qmckl_exit_code_device
qmckl_get_mo_basis_mo_num_device(const qmckl_context_device context,
                                 int64_t *mo_num)
{
    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
                                     "qmckl_get_mo_basis_mo_num", NULL);
    }

    qmckl_context_struct_device *const ctx =
        (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    int32_t mask = 1;

    if ((ctx->mo_basis.uninitialized & mask) != 0)
    {
        return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
                                     "qmckl_get_mo_basis_mo_num", NULL);
    }

    assert(ctx->mo_basis.mo_num > (int64_t)0);
    *mo_num = ctx->mo_basis.mo_num;
    return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device
qmckl_set_mo_basis_mo_num_device(qmckl_context_device context, int64_t mo_num)
{

    int32_t mask = 1;

    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return QMCKL_NULL_CONTEXT_DEVICE;
    }

    qmckl_context_struct_device *ctx = (qmckl_context_struct_device *)context;

    if (mask != 0 && !(ctx->mo_basis.uninitialized & mask))
    {
        return qmckl_failwith_device(context, QMCKL_ALREADY_SET_DEVICE,
                                     "qmckl_set_mo_basis_mo_num_device", NULL);
    }

    if (mo_num <= 0)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
                                     "qmckl_set_mo_basis_mo_num_device",
                                     "mo_num <= 0");
    }

    ctx->mo_basis.mo_num = mo_num;

    ctx->mo_basis.uninitialized &= ~mask;
    ctx->mo_basis.provided = (ctx->mo_basis.uninitialized == 0);
    if (ctx->mo_basis.provided)
    {
        qmckl_exit_code_device rc_ = qmckl_finalize_mo_basis_device(context);
        if (rc_ != QMCKL_SUCCESS_DEVICE)
            return rc_;
    }

    return QMCKL_SUCCESS_DEVICE;
}

qmckl_exit_code_device
qmckl_set_mo_basis_coefficient_device(qmckl_context_device context,
                                      double *coefficient)
{

    int32_t mask = 1 << 1;

    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return QMCKL_NULL_CONTEXT_DEVICE;
    }

    qmckl_context_struct_device *ctx = (qmckl_context_struct_device *)context;

    if (mask != 0 && !(ctx->mo_basis.uninitialized & mask))
    {
        return qmckl_failwith_device(context, QMCKL_ALREADY_SET_DEVICE,
                                     "qmckl_set_mo_basis_coefficient_device",
                                     NULL);
    }

    if (ctx->mo_basis.coefficient != NULL)
    {
        qmckl_exit_code_device rc =
            qmckl_free_device(context, ctx->mo_basis.coefficient);
        if (rc != QMCKL_SUCCESS_DEVICE)
        {
            return qmckl_failwith_device(
                context, rc, "qmckl_set_mo_basis_coefficient_device", NULL);
        }
    }

    double *new_array = (double *)qmckl_malloc_device(
        context, ctx->ao_basis.ao_num * ctx->mo_basis.mo_num * sizeof(double));
    if (new_array == NULL)
    {
        return qmckl_failwith_device(context, QMCKL_ALLOCATION_FAILED_DEVICE,
                                     "qmckl_set_mo_basis_coefficient_device",
                                     NULL);
    }

    qmckl_memcpy_D2D(context, new_array, coefficient,
                     ctx->ao_basis.ao_num * ctx->mo_basis.mo_num *
                         sizeof(double));

    ctx->mo_basis.coefficient = new_array;

    ctx->mo_basis.uninitialized &= ~mask;
    ctx->mo_basis.provided = (ctx->mo_basis.uninitialized == 0);
    if (ctx->mo_basis.provided)
    {
        qmckl_exit_code_device rc_ = qmckl_finalize_mo_basis_device(context);
        if (rc_ != QMCKL_SUCCESS_DEVICE)
            return rc_;
    }

    return QMCKL_SUCCESS_DEVICE;
}
