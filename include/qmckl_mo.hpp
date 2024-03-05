#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>

#include "../include/qmckl_basic_functions.hpp"
#include "../include/qmckl_context.hpp"
#include "../include/qmckl_memory.hpp"

bool qmckl_mo_basis_provided_device(qmckl_context_device context);

qmckl_exit_code_device
qmckl_get_mo_basis_mo_vgl_device(qmckl_context_device context,
                                 double *const mo_vgl, const int64_t size_max);

qmckl_exit_code_device
qmckl_get_mo_basis_mo_value_device(qmckl_context_device context,
                                   double *mo_value, int64_t size_max);

qmckl_exit_code_device
qmckl_get_mo_basis_mo_value_inplace_device(qmckl_context_device context,
                                           double *mo_value, int64_t size_max);

qmckl_exit_code_device
qmckl_provide_mo_basis_mo_value_device(qmckl_context_device context);

qmckl_exit_code_device
qmckl_provide_mo_basis_mo_vgl_device(qmckl_context_device context);

qmckl_exit_code_device qmckl_compute_mo_basis_mo_value_device(
    qmckl_context_device context, int64_t ao_num, int64_t mo_num,
    int64_t point_num, double *__restrict__ coefficient_t,
    double *__restrict__ ao_value, double *__restrict__ mo_value);

qmckl_exit_code_device qmckl_compute_mo_basis_mo_vgl_device(
    qmckl_context_device context, int64_t ao_num, int64_t mo_num,
    int64_t point_num, double *__restrict__ coefficient_t, double *__restrict__ ao_vgl,
    double *__restrict__ mo_vgl);

qmckl_exit_code_device
qmckl_finalize_mo_basis_device(qmckl_context_device context);
qmckl_exit_code_device
qmckl_set_mo_basis_mo_num_device(qmckl_context_device context, int64_t mo_num);
qmckl_exit_code_device
qmckl_set_mo_basis_coefficient_device(qmckl_context_device context,
                                      double *coefficient);