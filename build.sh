#bin/bash
/cluster/intel/oneapi/2024.0.0/compiler/2024.0/bin/compiler/clang++ -fsycl -g -I./home/enditar/qmckl_sycl/_INSTALL/include -c try.cpp -o try.o 
/cluster/intel/oneapi/2024.0.0/compiler/2024.0/bin/compiler/clang++ -fsycl -g -L/home/enditar/qmckl_sycl/_INSTALL/lib -o output try.o -lqmckl_gpu
#./configure CXX="/cluster/intel/oneapi/2024.0.0/compiler/2024.0/bin/compiler/clang++" --prefix=~/qmckl_sycl/_INSTALL/