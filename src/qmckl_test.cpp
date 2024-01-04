#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include "../include/qmckl_memory.hpp"

using namespace sycl;

void displayMatrix(int* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    qmckl_context_device context;
    int size_type = 3;
    sycl::queue q;
    int *matrix_a = static_cast<int*>(qmckl_malloc_device(q, context, sizeof(int) * size_type * size_type));
    int *matrix_b = static_cast<int*>(qmckl_malloc_device(q, context, sizeof(int) * size_type * size_type));
    int *matrix_c = static_cast<int*>(qmckl_malloc_device(q, context, sizeof(int) * size_type * size_type));

    for (int i = 0; i < size_type; i++) {
        for (int j = 0; j < size_type; j++) {
            matrix_a[i * size_type + j] = 2;
            matrix_b[i * size_type + j] = 2;
        }    
    }

    for (int i = 0; i < size_type; ++i) {
        for (int j = 0; j < size_type; ++j) {
            matrix_c[i * size_type + j] = 0;
            for (int k = 0; k < size_type; ++k) {
                matrix_c[i * size_type + j] += matrix_a[i * size_type + k] * matrix_b[k * size_type + j];
            }
        }
    }

    // Display the result
    std::cout << "Result of matrix multiplication:" << std::endl;
    displayMatrix(matrix_c, size_type, size_type);

    qmckl_free_device(q, context, matrix_a);
    qmckl_free_device(q, context, matrix_b);
    qmckl_free_device(q, context, matrix_c);

    return 0;
}
