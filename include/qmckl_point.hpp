#include "qmckl_types.hpp"
#include "qmckl_basic_functions.hpp"
#include "qmckl_context.hpp"
#include "qmckl_memory.hpp"
#include "qmckl_blas.hpp"

qmckl_exit_code_device qmckl_set_point_device(qmckl_context_device context,
											  char transp, int64_t num,
											  double *coord, int64_t size_max);

qmckl_exit_code_device qmckl_get_point_device(const qmckl_context_device context, const char transp,
					   double *const coord, const int64_t size_max);
