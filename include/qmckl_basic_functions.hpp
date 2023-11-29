#pragma once

#include <cstdint>
#include <cstdlib>

#include "qmckl_types.hpp"
#include "qmckl_context.hpp"


/* Error */
qmckl_exit_code_device qmckl_failwith_device(qmckl_context_device context, const qmckl_exit_code_device exit_code, const char *function, const char *message);