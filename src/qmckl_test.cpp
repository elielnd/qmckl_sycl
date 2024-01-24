#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include "../include/qmckl_memory.hpp"

using namespace sycl;
static const int N = 4;

int main()
{
    queue q;

    try
    {
        // define queue with accelerator selector
        q = queue(cl::sycl::accelerator_selector_v);
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
                   { 
                *ptr_point += 1; 
    });
    q.wait();

    std::cout << "Point: " << point << std::endl;

    return 0;
}
