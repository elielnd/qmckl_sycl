#include <sycl/sycl.hpp>

using namespace sycl;


int main() {

  /*
    int A[2][2] = {{1, 2}, {3, 4}};
  int B[2][2] = {{5, 6}, {7, 8}};
  int C[2][2] = {{0, 0}, {0, 0}};

  auto R = range<1>(3);

  buffer A_buf(A, R);
  buffer B_buf(B, R);
  buffer C_buf(C, R);

  queue Q;

  Q.submit([&](handler& h) {

    accessor A_acc(A_buf, h, read_only);
    accessor B_acc(B_buf, h, read_only);
    accessor C_acc(C_buf, h, write_only);

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        int sum = 0;
        for (int k = 0; k < 2; k++) {
          sum += A_acc[i][k] * B_acc[k][j];
        }
        C_acc[i][j] = sum;
      }
    }
  });

  Q.wait();

  host_accessor C_host(C_buf, read_only);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      std::cout << C_host[i][j] << " ";
    }
    std::cout << std::endl;
  }
  */
  queue q(default_selector_v);
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