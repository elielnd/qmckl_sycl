#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include "chbrclf.hpp"
#include "../include/qmckl_gpu.hpp"


using namespace sycl;
static const int N = 4;

int main()
{
    queue q;

    try
    {
        // define queue with accelerator selector
        q = queue(accelerator_selector_v);
    }
    catch (const sycl::exception &e)
    {
        q = queue();
        std::cerr << "Could not create GPU queue. Using default queue.\n";
    };

    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    int point = 0;
    int *ptr_point = &point;

    q.parallel_for(range<1>(N), [=](id<1> i)
                   { *ptr_point += 1; });
    q.wait();

    std::cout << "Point: " << point << std::endl;

    //**********************
    // Test Memory
    //**********************

    try
    {
        double chbrclf_nucl_coord[3][chbrclf_nucl_num] =
            {{1.096243353458458e+00, 1.168459237342663e+00, 1.487097297712132e+00, 3.497663849983889e+00, -2.302574592081335e+00},
             {8.907054016973815e-01, 1.125660720053393e+00, 3.119652484478797e+00, -1.302920810073182e+00, -3.542027060505035e-01},
             {7.777092280258892e-01, 2.833370314829343e+00, -3.855438138411500e-01, -1.272220319439064e-01, -5.334129934317614e-02}};
        double chbrclf_charge[chbrclf_nucl_num] = {6., 1., 9., 17., 35.};

        qmckl_context_device context;
        context = qmckl_context_create_device(q);

        int64_t nucl_num = chbrclf_nucl_num;
        qmckl_exit_code_device exit_code;

        // Put nucleus stuff in CPU arrays
        double *nucl_charge = chbrclf_charge;
        double *nucl_coord = &(chbrclf_nucl_coord[0][0]);

        // Put nucleus stuff in GPU arrays
        double *nucl_charge_d = (double *)qmckl_malloc_device(context, nucl_num * sizeof(double));
        double *nucl_coord_d = (double *)qmckl_malloc_device(context, 3 * nucl_num * sizeof(double));

        assert(nucl_charge_d != nullptr);
        assert(nucl_coord_d != nullptr);

        exit_code = qmckl_memcpy_H2D(context, nucl_charge_d, nucl_charge, nucl_num * sizeof(double));
        assert(exit_code != QMCKL_FAILURE_DEVICE);

        exit_code = qmckl_memcpy_H2D(context, nucl_coord_d, nucl_coord, 3 * nucl_num * sizeof(double));
        assert(exit_code != QMCKL_FAILURE_DEVICE);

        // Set nucleus stuff in context
        qmckl_exit_code_device rc;
        rc = qmckl_set_nucleus_num_device(context, nucl_num);
        if (rc != QMCKL_SUCCESS_DEVICE)
            return 1;

        rc = qmckl_set_nucleus_coord_device(context, 'T', nucl_coord_d,
                                            3 * nucl_num);
        if (rc != QMCKL_SUCCESS_DEVICE)
            return 1;

        rc = qmckl_set_nucleus_charge_device(context, nucl_charge_d, nucl_num);
        if (rc != QMCKL_SUCCESS_DEVICE)
            return 1;

        if (!qmckl_nucleus_provided_device(context))
            return 1;

        qmckl_context_struct_device *ctx = (qmckl_context_struct_device *)context;

        std::cout << "Device end:  " << ctx->q.get_device().get_info<info::device::name>() << "\n";
        std::cout << "Size : " << ctx->memory.array_size << "\n";
    }
    catch (const sycl::exception &e)
    {
        std::cerr << "3: " << e.what() << '\n';
    }

    return 0;
}
