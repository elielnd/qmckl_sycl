#include <sycl/sycl.hpp>

#include "../include/qmckl_memory.hpp"

using namespace sycl;


int main() {

  int rows = 2;
  int cols = 2;

  // No device of requested type
  // queue q(gpu_selector_v);
  
  // Device: Intel(R) FPGA Emulation Device
  //queue q(accelerator_selector_v);
  
  // Device: AMD Ryzen 5 5500
  queue q(default_selector_v);
  //queue q(cpu_selector_v);



  if (q.get_device().get_info<info::device::name>() == "") 
  {
    std::cout << "DeviceDeviceDeviceDeviceDeviceDeviceDevice:\n\n\n\n\n";
  }
  else
  {
      std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
  }

  return 0;
}