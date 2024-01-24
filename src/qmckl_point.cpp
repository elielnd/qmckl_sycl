#include "../include/qmckl_point.hpp"

using namespace sycl;

qmckl_exit_code_device qmckl_get_point_device(const qmckl_context_device context, const char transp,
                                              double *const coord, const int64_t size_max)
{

    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return QMCKL_INVALID_CONTEXT_DEVICE;
    }

    if (coord == NULL)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE,
                                     "qmckl_get_point_coord",
                                     "coord is a null pointer");
    }

    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    int64_t point_num = ctx->point.num;
    if (point_num == 0)
        return QMCKL_NOT_PROVIDED_DEVICE;

    assert(ctx->point.coord.data != NULL);

    if (size_max < 3 * point_num)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE,
                                     "qmckl_get_point_coord_device",
                                     "size_max too small");
    }

    qmckl_exit_code_device rc;
    if (transp == 'N')
    {
        qmckl_matrix_device At = qmckl_matrix_alloc_device(context, 3, point_num);
        rc = qmckl_transpose_device(context, ctx->point.coord, At);
        if (rc != QMCKL_SUCCESS_DEVICE)
            return rc;

        // Copy content of At into coord
        // rc = qmckl_double_of_matrix_device(context, At, coord, size_max);
        qmckl_memcpy_D2D(context, coord, At.data,
                         At.size[0] * At.size[1] * sizeof(double));

        if (rc != QMCKL_SUCCESS_DEVICE)
            return rc;
        rc = qmckl_matrix_free_device(context, &At);
    }
    else
    {
        // Copy content of ctx->point.coord into coord
        // rc = qmckl_double_of_matrix_device(context, ctx->point.coord, coord,
        // size_max);
        qmckl_memcpy_D2D(context, coord, ctx->point.coord.data,
                         ctx->point.coord.size[0] * ctx->point.coord.size[1] *
                             sizeof(double));
    }
    if (rc != QMCKL_SUCCESS_DEVICE)
        return rc;

    return rc;
}

qmckl_exit_code_device qmckl_set_point_device(qmckl_context_device context, char transp, int64_t num, double *coord, int64_t size_max)
{

    size_t device_id = qmckl_get_device_id(context);
    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
    {
        return QMCKL_NULL_CONTEXT_DEVICE;
    }

    if (size_max < 3 * num)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_4_DEVICE, "qmckl_set_point_device", "Array too small");
    }

    if (transp != 'N' && transp != 'T')
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_2_DEVICE, "qmckl_set_point_device", "transp should be 'N' or 'T'");
    }

    if (coord == NULL)
    {
        return qmckl_failwith_device(context, QMCKL_INVALID_ARG_3_DEVICE, "qmckl_set_point_device", "coord is a NULL pointer");
    }

    qmckl_context_struct_device *ctx = (qmckl_context_struct_device *)context;
    assert(ctx != NULL);

    qmckl_exit_code_device rc;
    if (num != ctx->point.num)
    {
        if (ctx->point.coord.data != nullptr)
        {
            rc = qmckl_matrix_free_device(context, &(ctx->point.coord));
            assert(rc == QMCKL_SUCCESS_DEVICE);
        }

        ctx->point.coord = qmckl_matrix_alloc_device(context, num, 3);
        if (ctx->point.coord.data == nullptr)
        {
            return qmckl_failwith_device(context, QMCKL_ALLOCATION_FAILED_DEVICE, "qmckl_set_point", nullptr);
        }
    }

    ctx->point.num = num;

    double *a = ctx->point.coord.data;
    int size_0 = ctx->point.coord.size[0];

    queue q;

    if (transp == 'T')
    {
        q.parallel_for(sycl::range<1>(3 * num), [=](id<1> i)
                       { a[i] = coord[i]; })
            .wait();
    }
    else
    {
        q.parallel_for(sycl::range<1>(num), [=](sycl::id<1> i)
                       {
            a[i] = coord[3 * i];
            a[i + size_0] = coord[3 * i + 1];
            a[i + 2 * size_0] = coord[3 * i + 2]; })
            .wait();
    }
    rc = qmckl_context_touch_device(context);
    assert(rc == QMCKL_SUCCESS_DEVICE);

    return QMCKL_SUCCESS_DEVICE;
}
