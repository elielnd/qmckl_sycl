/*#include <CL/sycl.hpp>

using namespace sycl;


int main() {

  // No device of requested type
  // queue q(gpu_selector_v);
  
  // Device: Intel(R) FPGA Emulation Device
  //queue q(accelerator_selector_v);
  
  // Device: AMD Ryzen 5 5500
  queue q;
  //queue q(cpu_selector_v);

  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  return 0;
}*/