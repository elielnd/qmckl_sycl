#pragma once

// This file contains functions prototypes for functions initializing the
// context from a TREXIO file

#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>
#include "trexio.h"

#include "../include/qmckl_types.hpp"
#include "../include/qmckl_basic_functions.hpp"
#include "../include/qmckl_nucleus.hpp"
#include "../include/qmckl_electron.hpp"
#include "../include/qmckl_context.hpp"
#include "../include/qmckl_ao.hpp"
#include "../include/qmckl_mo.hpp"

// Prototype for standard QMCkl function
trexio_t *qmckl_trexio_open_X_device(char *file_name, qmckl_exit_code_device *rc);

qmckl_exit_code_device qmckl_trexio_read_device(qmckl_context_device context, char *file_name,  int64_t size_max);

//**********
// CONTEXT FILL
//**********

qmckl_exit_code_device qmckl_trexio_read_nucleus_X_device(qmckl_context_device context, trexio_t *file);
qmckl_exit_code_device qmckl_trexio_read_ao_X_device(qmckl_context_device context, trexio_t *file);
qmckl_exit_code_device qmckl_trexio_read_mo_X_device(qmckl_context_device context, trexio_t *file);
qmckl_exit_code_device qmckl_trexio_read_device(qmckl_context_device context, char *file_name, int64_t size_max);