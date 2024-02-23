#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>
#include <iostream>
#include "chbrclf.hpp"
#include "../include/qmckl_gpu.hpp"

#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#define AO_VALUE_ID(x, y) ao_num *x + y
#define AO_VGL_ID(x, y, z) 5 * ao_num *x + ao_num *y + z

using namespace sycl;

int main()
{
    queue q;
    try
    {
        // Try to initialize the queue with the GPU selector
        q = queue(gpu_selector_v);
        std::cout << "Running on GPU: " << q.get_device().get_info<info::device::name>() << "\n";
    }
    catch (const sycl::exception &accelerator_exception)
    {
        try
        {
            // If initialization with the GPU selector fails as well, use the accelerator selector
            q = queue(accelerator_selector_v);
            std::cout << "Running on ACCELERATOR: " << q.get_device().get_info<info::device::name>() << "\n";
        }
        catch (const sycl::exception &gpu_exception)
        {
            // If initialization with the accelerator selector fails as well, use the host selector
            q = queue();
            std::cerr << "Warning: No device found";
            std::cout << "Running on Host: " << q.get_device().get_info<info::device::name>() << "\n";
        }
    };

    qmckl_context_device context;

    context = qmckl_context_create_device(q);

    int64_t nucl_num = chbrclf_nucl_num;

    // Put nucleus stuff in CPU arrays
    double *nucl_charge = chbrclf_charge;
    double *nucl_coord = &(chbrclf_nucl_coord[0][0]);

    // Put nucleus stuff in GPU arrays
    double *nucl_charge_d = (double *)qmckl_malloc_device(context, nucl_num * sizeof(double));
    double *nucl_coord_d = (double *)qmckl_malloc_device(context, 3 * nucl_num * sizeof(double));

    qmckl_memcpy_H2D(context, nucl_charge_d, nucl_charge, nucl_num * sizeof(double));
    qmckl_memcpy_H2D(context, nucl_coord_d, nucl_coord, 3 * nucl_num * sizeof(double));

    // Set nucleus stuff in context
    qmckl_exit_code_device rc;
    rc = qmckl_set_nucleus_num_device(context, nucl_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_nucleus_coord_device(context, 'T', nucl_coord_d, 3 * nucl_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_nucleus_charge_device(context, nucl_charge_d, nucl_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    if (!qmckl_nucleus_provided_device(context))
        return 1;

    int64_t shell_num = chbrclf_shell_num;
    int64_t prim_num = chbrclf_prim_num;
    int64_t ao_num = chbrclf_ao_num;

    // Put other stuff in CPU arrays
    int64_t *nucleus_index = &(chbrclf_basis_nucleus_index[0]);
    int64_t *nucleus_shell_num = &(chbrclf_basis_nucleus_shell_num[0]);
    int32_t *shell_ang_mom = &(chbrclf_basis_shell_ang_mom[0]);
    int64_t *shell_prim_num = &(chbrclf_basis_shell_prim_num[0]);
    int64_t *shell_prim_index = &(chbrclf_basis_shell_prim_index[0]);
    double *shell_factor = &(chbrclf_basis_shell_factor[0]);
    double *exponent = &(chbrclf_basis_exponent[0]);
    double *coefficient = &(chbrclf_basis_coefficient[0]);
    double *prim_factor = &(chbrclf_basis_prim_factor[0]);
    double *ao_factor = &(chbrclf_basis_ao_factor[0]);

    // Put other stuff in GPU arrays
    int64_t *nucleus_index_d = (int64_t *)qmckl_malloc_device(context, nucl_num * sizeof(int64_t));
    int64_t *nucleus_shell_num_d = reinterpret_cast<int64_t *>(qmckl_malloc_device(context, nucl_num * sizeof(int64_t)));
    int32_t *shell_ang_mom_d = reinterpret_cast<int32_t *>(qmckl_malloc_device(context, shell_num * sizeof(int32_t)));
    int64_t *shell_prim_num_d = reinterpret_cast<int64_t *>(qmckl_malloc_device(context, shell_num * sizeof(int64_t)));
    int64_t *shell_prim_index_d = reinterpret_cast<int64_t *>(qmckl_malloc_device(context, shell_num * sizeof(int64_t)));
    double *shell_factor_d = reinterpret_cast<double *>(qmckl_malloc_device(context, shell_num * sizeof(double)));
    double *exponent_d = reinterpret_cast<double *>(qmckl_malloc_device(context, prim_num * sizeof(double)));
    double *coefficient_d = reinterpret_cast<double *>(qmckl_malloc_device(context, prim_num * sizeof(double)));
    double *prim_factor_d = reinterpret_cast<double *>(qmckl_malloc_device(context, prim_num * sizeof(double)));
    double *ao_factor_d = reinterpret_cast<double *>(qmckl_malloc_device(context, ao_num * sizeof(double)));

    qmckl_memcpy_H2D(context, nucleus_index_d, nucleus_index, nucl_num * sizeof(int64_t));
    qmckl_memcpy_H2D(context, nucleus_shell_num_d, nucleus_shell_num, nucl_num * sizeof(int64_t));
    qmckl_memcpy_H2D(context, shell_ang_mom_d, shell_ang_mom, shell_num * sizeof(int32_t));
    qmckl_memcpy_H2D(context, shell_prim_num_d, shell_prim_num, shell_num * sizeof(int64_t));
    qmckl_memcpy_H2D(context, shell_prim_index_d, shell_prim_index, shell_num * sizeof(int64_t));
    qmckl_memcpy_H2D(context, shell_factor_d, shell_factor, shell_num * sizeof(double));
    qmckl_memcpy_H2D(context, exponent_d, exponent, prim_num * sizeof(double));
    qmckl_memcpy_H2D(context, coefficient_d, coefficient, prim_num * sizeof(double));
    qmckl_memcpy_H2D(context, prim_factor_d, prim_factor, prim_num * sizeof(double));
    qmckl_memcpy_H2D(context, ao_factor_d, ao_factor, ao_num * sizeof(double));

    char typ = 'G';

    rc = qmckl_set_ao_basis_type_device(context, typ);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_shell_num_device(context, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_prim_num_device(context, prim_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_nucleus_index_device(context, nucleus_index_d, nucl_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_nucleus_shell_num_device(context, nucleus_shell_num_d, nucl_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_shell_ang_mom_device(context, shell_ang_mom_d, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_shell_factor_device(context, shell_factor_d, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_shell_prim_num_device(context, shell_prim_num_d, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_shell_prim_index_device(context, shell_prim_index_d, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_exponent_device(context, exponent_d, prim_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_coefficient_device(context, coefficient_d, prim_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_prim_factor_device(context, prim_factor_d, prim_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_ao_num_device(context, ao_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    rc = qmckl_set_ao_basis_ao_factor_device(context, ao_factor_d, ao_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    if (!qmckl_ao_basis_provided_device(context))
        return 1;

    // Checking arrays after context set and get

    int64_t shell_num_test;
    int64_t prim_num_test;
    int64_t ao_num_test;
    int64_t *nucleus_index_test;
    int64_t *nucleus_shell_num_test;
    int32_t *shell_ang_mom_test;
    int64_t *shell_prim_num_test;
    int64_t *shell_prim_index_test;
    double *shell_factor_test;
    double *exponent_test;
    double *coefficient_test;
    double *prim_factor_test;
    double *ao_factor_test;
    char typ_test;

    rc = qmckl_get_ao_basis_type_device(context, &typ_test);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;
    if (typ != typ_test)
        return 1;

    rc = qmckl_get_ao_basis_shell_num_device(context, &shell_num_test);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;
    if (shell_num != shell_num_test)
        return 1;

    rc = qmckl_get_ao_basis_prim_num_device(context, &prim_num_test);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;
    if (prim_num != prim_num_test)
        return 1;

    nucleus_index_test = (int64_t *)qmckl_malloc_device(context, nucl_num * sizeof(int64_t));
    rc = qmckl_get_ao_basis_nucleus_index_device(context, nucleus_index_test, nucl_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    int wrong_val = 0;
    buffer<int, 1> buff_wrong_val(&wrong_val, range<1>(1));

    q.submit([&](handler &h)
             {
            auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);

            h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                           {
                                if ( nucleus_index_test[i]!= nucleus_index_d[i])
                                {
                                    acc_wrong_val[0] = 1;
                                } }); });
    q.wait();
    buff_wrong_val.get_host_access();

    qmckl_free_device(context, nucleus_index_test);
    if (wrong_val)
        return 1;

    nucleus_shell_num_test = (int64_t *)qmckl_malloc_device(context, nucl_num * sizeof(int64_t));
    rc = qmckl_get_ao_basis_nucleus_shell_num_device(context, nucleus_shell_num_test, nucl_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    q.submit([&](handler &h)
             {
                auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);
                h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                            {
                    if (nucleus_shell_num_test[i] != nucleus_shell_num_d[i])
                        acc_wrong_val[0] = 1;
                            }); });
    q.wait();
    buff_wrong_val.get_host_access();

    qmckl_free_device(context, nucleus_shell_num_test);
    if (wrong_val)
        return 1;

    shell_ang_mom_test = (int32_t *)qmckl_malloc_device(context, shell_num * sizeof(int32_t));
    rc = qmckl_get_ao_basis_shell_ang_mom_device(context, shell_ang_mom_test, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    q.submit([&](handler &h)
             {
                    auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);
                    h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                        {
                            if (shell_ang_mom_test[i] != shell_ang_mom_d[i])
                                acc_wrong_val[i] = 1;
                        }); });
    q.wait();
    buff_wrong_val.get_host_access();

    qmckl_free_device(context, shell_ang_mom_test);
    if (wrong_val)
        return 1;

    shell_factor_test = (double *)qmckl_malloc_device(context, shell_num * sizeof(double));
    rc = qmckl_get_ao_basis_shell_factor_device(context, shell_factor_test, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    q.submit([&](handler &h)
             {
        auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);
        h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                        {
                if (shell_factor_test[i] != shell_factor_d[i])
                    acc_wrong_val[0] = 1; }); });
    q.wait();
    buff_wrong_val.get_host_access();

    qmckl_free_device(context, shell_factor_test);
    if (wrong_val)
        return 1;

    shell_prim_num_test = (int64_t *)qmckl_malloc_device(context, shell_num * sizeof(int64_t));
    rc = qmckl_get_ao_basis_shell_prim_num_device(context, shell_prim_num_test, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    shell_prim_index_test = (int64_t *)qmckl_malloc_device(context, shell_num * sizeof(int64_t));
    rc = qmckl_get_ao_basis_shell_prim_index_device(context, shell_prim_index_test, shell_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;
    q.submit([&](handler &h)
             {
        auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);
        h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                        {
                if (shell_prim_index_test[i] != shell_prim_index_d[i])
                    acc_wrong_val[0] = 1; }); });
    q.wait();
    buff_wrong_val.get_host_access();

    qmckl_free_device(context, shell_prim_index_test);
    if (wrong_val)
        return 1;

    exponent_test = (double *)qmckl_malloc_device(context, prim_num * sizeof(double));
    rc = qmckl_get_ao_basis_exponent_device(context, exponent_test, prim_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    q.submit([&](handler &h)
             {
        auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);
        h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                        {
                if (exponent_test[i] != exponent_d[i])
                    acc_wrong_val[0] = 1; }); });
    q.wait();

    qmckl_free_device(context, exponent_test);
    if (wrong_val)
        return 1;

    coefficient_test = (double *)qmckl_malloc_device(context, prim_num * sizeof(double));
    rc = qmckl_get_ao_basis_coefficient_device(context, coefficient_test, prim_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    q.submit([&](handler &h)
             {
        auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);
        h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                        {
            if (coefficient_test[i] != coefficient_d[i])
                acc_wrong_val[0] = 1; }); });
    q.wait();
    buff_wrong_val.get_host_access();

    qmckl_free_device(context, coefficient_test);
    if (wrong_val)
        return 1;

    prim_factor_test = (double *)qmckl_malloc_device(context, prim_num * sizeof(double));
    rc = qmckl_get_ao_basis_prim_factor_device(context, prim_factor_test,
                                               prim_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    q.submit([&](handler &h)
             {
        auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);
        h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                        {
            if (prim_factor_test[i] != prim_factor_d[i])
                acc_wrong_val[0] = 1; }); });
    q.wait();
    buff_wrong_val.get_host_access();

    qmckl_free_device(context, prim_factor_test);
    if (wrong_val)
        return 1;

    rc = qmckl_get_ao_basis_ao_num_device(context, &ao_num_test);
    if (ao_num != ao_num_test)
        return 1;

    ao_factor_test = (double *)qmckl_malloc_device(context, ao_num * sizeof(double));
    rc = qmckl_get_ao_basis_ao_factor_device(context, ao_factor_test, ao_num);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    q.submit([&](handler &h)
             {
            auto acc_wrong_val = buff_wrong_val.get_access<access::mode::write>(h);
            h.parallel_for(range<1>(nucl_num), [=](id<1> i)
                           {
                if (ao_factor_test[i] != ao_factor_d[i])
                    acc_wrong_val[0] = 1; }); });
    q.wait();
    buff_wrong_val.get_host_access();

    qmckl_free_device(context, ao_factor_test);
    if (wrong_val)
        return 1;

#define shell_num chbrclf_shell_num
#define ao_num chbrclf_ao_num
#define elec_num chbrclf_elec_num

    double *elec_coord = &(chbrclf_elec_coord[0][0][0]);
    int64_t point_num = elec_num;
    double *elec_coord_d = (double *)qmckl_malloc_device(context, 3 * point_num * sizeof(double));

    qmckl_memcpy_H2D(context, elec_coord_d, elec_coord, 3 * point_num * sizeof(double));

    // TODO Fix this
    // if (!qmckl_electron_provided(context))
    //	return 1;

    rc = qmckl_set_point_device(context, 'N', point_num, elec_coord_d, point_num * 3);
    if (rc != QMCKL_SUCCESS_DEVICE)
        return 1;

    // Get & test ao_value values
    double *ao_value_d = (double *)qmckl_malloc_device(context, point_num * ao_num * sizeof(double));
    double *ao_value = (double *)malloc(point_num * ao_num * sizeof(double));
    qmckl_context_struct_device *const ctx = (qmckl_context_struct_device *)context;

    rc = qmckl_get_ao_basis_ao_value_device(context, ctx->ao_basis.ao_value, (int64_t)(point_num * ao_num));

    q.memcpy(ao_value, ctx->ao_basis.ao_value, point_num * ao_num * sizeof(double)).wait();

    // qmckl_memcpy_D2H(context, ao_value, ao_value_d, point_num * ao_num * sizeof(double));

    // Print the values [26][219] [26][220] [26][221] [26][222

    printf(" ao_value ao_value[26][219] %25.15e\n",
           ao_value[AO_VALUE_ID(26, 219)]);
    printf(" ao_value ao_value[26][220] %25.15e\n",
           ao_value[AO_VALUE_ID(26, 220)]);
    printf(" ao_value ao_value[26][221] %25.15e\n",
           ao_value[AO_VALUE_ID(26, 221)]);
    printf(" ao_value ao_value[26][222] %25.15e\n",
           ao_value[AO_VALUE_ID(26, 222)]);
    printf("\n");

    // if (fabs(ao_value[AO_VALUE_ID(26, 219)] - (1.020298798341620e-08)) > 1.e-14)
    //     return 1;
    // if (fabs(ao_value[AO_VALUE_ID(26, 220)] - (1.516643537739178e-08)) > 1.e-14)
    //     return 1;
    // if (fabs(ao_value[AO_VALUE_ID(26, 221)] - (-4.686370882518819e-09)) >
    //     1.e-14)
    //     return 1;
    // if (fabs(ao_value[AO_VALUE_ID(26, 222)] - (7.514816980753531e-09)) > 1.e-14)
    //     return 1;

    //     // Get & test ao_vgl values
    //     double *ao_vgl_d = (double *)qmckl_malloc_device(context, point_num * 5 * ao_num * sizeof(double));
    //     double *ao_vgl = (double *)malloc(point_num * 5 * ao_num * sizeof(double));

    //     printf("About to get vgl\n");
    //     rc = qmckl_get_ao_basis_ao_vgl_device(context, ao_vgl_d,
    //                                           (int64_t)5 * point_num * ao_num);
    //     qmckl_memcpy_D2H(context, ao_vgl, ao_vgl_d,
    //                      point_num * 5 * ao_num * sizeof(double));
    //     if (rc != QMCKL_SUCCESS_DEVICE)
    //         return 1;

    //     printf("\n");
    //     printf(" ao_vgl ao_vgl[26][0][219] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 0, 219)]);
    //     printf(" ao_vgl ao_vgl[26][1][219] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 1, 219)]);
    //     printf(" ao_vgl ao_vgl[26][2][219] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 2, 219)]);
    //     printf(" ao_vgl ao_vgl[26][3][219] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 3, 219)]);
    //     printf(" ao_vgl ao_vgl[26][4][219] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 4, 219)]);
    //     printf(" ao_vgl ao_vgl[26][0][220] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 0, 220)]);
    //     printf(" ao_vgl ao_vgl[26][1][220] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 1, 220)]);
    //     printf(" ao_vgl ao_vgl[26][2][220] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 2, 220)]);
    //     printf(" ao_vgl ao_vgl[26][3][220] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 3, 220)]);
    //     printf(" ao_vgl ao_vgl[26][4][220] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 4, 220)]);
    //     printf(" ao_vgl ao_vgl[26][0][221] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 0, 221)]);
    //     printf(" ao_vgl ao_vgl[26][1][221] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 1, 221)]);
    //     printf(" ao_vgl ao_vgl[26][2][221] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 2, 221)]);
    //     printf(" ao_vgl ao_vgl[26][3][221] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 3, 221)]);
    //     printf(" ao_vgl ao_vgl[26][4][221] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 4, 221)]);
    //     printf(" ao_vgl ao_vgl[26][0][222] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 0, 222)]);
    //     printf(" ao_vgl ao_vgl[26][1][222] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 1, 222)]);
    //     printf(" ao_vgl ao_vgl[26][2][222] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 2, 222)]);
    //     printf(" ao_vgl ao_vgl[26][3][222] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 3, 222)]);
    //     printf(" ao_vgl ao_vgl[26][4][222] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 4, 222)]);
    //     printf(" ao_vgl ao_vgl[26][0][223] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 0, 223)]);
    //     printf(" ao_vgl ao_vgl[26][1][223] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 1, 223)]);
    //     printf(" ao_vgl ao_vgl[26][2][223] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 2, 223)]);
    //     printf(" ao_vgl ao_vgl[26][3][223] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 3, 223)]);
    //     printf(" ao_vgl ao_vgl[26][4][223] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 4, 223)]);
    //     printf(" ao_vgl ao_vgl[26][0][224] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 0, 224)]);
    //     printf(" ao_vgl ao_vgl[26][1][224] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 1, 224)]);
    //     printf(" ao_vgl ao_vgl[26][2][224] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 2, 224)]);
    //     printf(" ao_vgl ao_vgl[26][3][224] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 3, 224)]);
    //     printf(" ao_vgl ao_vgl[26][4][224] %25.15e\n",
    //            ao_vgl[AO_VGL_ID(26, 4, 224)]);
    //     printf("\n");

    //     // Read the ao_vgl ref

    //     // We will try to open ao_reference.txt "from" qmckl_gpu/ and
    //     // qmckl_gpu/tests/
    //     FILE *fp = fopen("tests/ao_reference.txt", "r");
    //     if (fp == NULL)
    //     {
    //         fp = fopen("ao_reference.txt", "r");
    //     }
    //     if (fp == NULL)
    //     {
    //         printf("Error : ao_reference.txt not found, leaving\n");
    //         exit(1);
    //     }

    //     double ref;
    //     printf("%ld %d\n", point_num, ao_num);
    //     for (int i = 0; i < point_num; i++)
    //     {
    //         for (int j = 0; j < 5; j++)
    //         {
    //             for (int k = 0; k < ao_num; k++)
    //             {
    //                 fscanf(fp, "%lf", &ref);
    //                 if (fabs(ao_vgl[AO_VGL_ID(i, j, k)] - ref) > 1.e-14)
    //                     return 1;
    //             }
    //         }
    //     }

    //     rc = qmckl_context_destroy_device(context);
    //     if (rc != QMCKL_SUCCESS_DEVICE)
    //         return 1;

    return 0;
}

// int point = 0;
// int *ptr_point = &point;

// q.parallel_for(range<1>(N), [=](id<1> i)
//                { *ptr_point += 1; });
// q.wait();

// std::cout << "Point: " << point << std::endl;

// //**********************
// // Test Memory
// //**********************

// try
// {
//     double chbrclf_nucl_coord[3][chbrclf_nucl_num] =
//         {{1.096243353458458e+00, 1.168459237342663e+00, 1.487097297712132e+00, 3.497663849983889e+00, -2.302574592081335e+00},
//          {8.907054016973815e-01, 1.125660720053393e+00, 3.119652484478797e+00, -1.302920810073182e+00, -3.542027060505035e-01},
//          {7.777092280258892e-01, 2.833370314829343e+00, -3.855438138411500e-01, -1.272220319439064e-01, -5.334129934317614e-02}};
//     double chbrclf_charge[chbrclf_nucl_num] = {6., 1., 9., 17., 35.};

//     qmckl_context_device context;
//     context = qmckl_context_create_device(q);

//     int64_t nucl_num = chbrclf_nucl_num;
//     qmckl_exit_code_device exit_code;

//     // Put nucleus stuff in CPU arrays
//     double *nucl_charge = chbrclf_charge;
//     double *nucl_coord = &(chbrclf_nucl_coord[0][0]);

//     // Put nucleus stuff in GPU arrays
//     double *nucl_charge_d = (double *)qmckl_malloc_device(context, nucl_num * sizeof(double));
//     double *nucl_coord_d = (double *)qmckl_malloc_device(context, 3 * nucl_num * sizeof(double));

//     assert(nucl_charge_d != nullptr);
//     assert(nucl_coord_d != nullptr);

//     exit_code = qmckl_memcpy_H2D(context, nucl_charge_d, nucl_charge, nucl_num * sizeof(double));
//     assert(exit_code != QMCKL_FAILURE_DEVICE);

//     exit_code = qmckl_memcpy_H2D(context, nucl_coord_d, nucl_coord, 3 * nucl_num * sizeof(double));
//     assert(exit_code != QMCKL_FAILURE_DEVICE);

//     // Set nucleus stuff in context
//     qmckl_exit_code_device rc;
//     rc = qmckl_set_nucleus_num_device(context, nucl_num);
//     if (rc != QMCKL_SUCCESS_DEVICE)
//         return 1;

//     rc = qmckl_set_nucleus_coord_device(context, 'T', nucl_coord_d,
//                                         3 * nucl_num);
//     if (rc != QMCKL_SUCCESS_DEVICE)
//         return 1;

//     rc = qmckl_set_nucleus_charge_device(context, nucl_charge_d, nucl_num);
//     if (rc != QMCKL_SUCCESS_DEVICE)
//         return 1;

//     if (!qmckl_nucleus_provided_device(context))
//         return 1;

//     qmckl_context_struct_device *ctx = (qmckl_context_struct_device *)context;

//     std::cout << "Device end:  " << ctx->q.get_device().get_info<info::device::name>() << "\n";
//     std::cout << "Size : " << ctx->memory.array_size << "\n";
// }
// catch (const sycl::exception &e)
// {
//     std::cerr << "3: " << e.what() << '\n';
// }

// return 0;
