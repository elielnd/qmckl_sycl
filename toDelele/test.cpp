#include <CL/sycl.hpp>

int main()
{
    // Print all available devices
    for (const auto &device : cl::sycl::device::get_devices())
    {
        std::cout << "Device: " << device.get_info<cl::sycl::info::device::name>() << std::endl;
    }
    //print all available platforms
    for (const auto &platform : cl::sycl::platform::get_platforms())
    {
        std::cout << "Platform: " << platform.get_info<cl::sycl::info::platform::name>() << std::endl;
    }

    constexpr size_t N = 10;
    int array[N];

    // Initialize array on the host
    for (size_t i = 0; i < N; i++)
    {
        array[i] = i;
    }

    try
    {
        // Create a SYCL queue
        cl::sycl::queue queue(sycl::gpu_selector_v);

        // Print the name of the device that the queue is using
        std::cout << "Running on " << queue.get_device().get_info<cl::sycl::info::device::name>() << std::endl;

        // Create a SYCL buffer to hold the array on the device
        cl::sycl::buffer<int, 1> buffer(array, cl::sycl::range<1>(N));

        // Submit a command group to the queue
        queue.submit([&](cl::sycl::handler &cgh)
                     {
            // Get write access to the buffer on the device
            auto accessor = buffer.get_access<cl::sycl::access::mode::write>(cgh);

            // Change the values of the array on the device
            cgh.parallel_for<class change_values>(cl::sycl::range<1>(N), [=](cl::sycl::id<1> idx) {
                accessor[idx] *= 2;
            }); });

        // Wait for the command group to finish
        queue.wait_and_throw();
    }
    catch (cl::sycl::exception e)
    {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }

    // Print the updated array
    for (size_t i = 0; i < N; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
