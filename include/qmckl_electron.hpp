#include <CL/sycl.hpp>

#include "qmckl_distance.hpp"
#include "qmckl_types.hpp"
#include "qmckl_basic_functions.hpp"
#include "qmckl_context.hpp"
#include "qmckl_memory.hpp"
#include "qmckl_blas.hpp"
#include "qmckl_point.hpp"

//**********
// SETTERS
//**********
qmckl_exit_code_device qmckl_set_electron_num_device(qmckl_context_device context, int64_t up_num, int64_t down_num);

qmckl_exit_code_device qmckl_set_electron_coord_device(qmckl_context_device context, char transp,
                                                       int64_t walk_num, double *coord, int64_t size_max);

//**********
// GETTERS
//**********
qmckl_exit_code_device qmckl_get_electron_coord_device(const qmckl_context_device context, const char transp,
                                                       double *const coord, const int64_t size_max);

//**********
// COMPUTE
//**********
qmckl_exit_code_device qmckl_compute_en_distance_device(const qmckl_context_device context, const int64_t point_num,
                                                        const int64_t nucl_num, const double *elec_coord,
                                                        const double *nucl_coord, double *const en_distance);

qmckl_exit_code_device qmckl_compute_ee_distance_device(const qmckl_context_device context, const int64_t elec_num,
                                                        const int64_t walk_num, const double *coord, double *const ee_distance);

qmckl_exit_code_device qmckl_distance_device(const qmckl_context_device context, const char transa,
                                            const char transb, const int64_t m, const int64_t n,
                                            const double *A, const int64_t lda, const double *B,
                                            const int64_t ldb, double *const C, const int64_t ldc);

//**********
// PROVIDES
//**********
bool qmckl_electron_provided_device(qmckl_context_device context);