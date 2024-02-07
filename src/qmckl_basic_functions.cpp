#include "../include/qmckl_basic_functions.hpp"

#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <cassert>
#include <thread>
#include <system_error>
#include <cmath>

/* Error */
qmckl_exit_code_device qmckl_set_error_device(qmckl_context_device context,
                                              const qmckl_exit_code_device exit_code,
                                              const char *function_name, const char *message)
{
    /* Passing a function name and a message is mandatory. */
    assert(function_name != nullptr);
    assert(message != nullptr);

    /* Exit codes are assumed valid. */
    assert(exit_code >= 0);
    assert(exit_code != QMCKL_SUCCESS_DEVICE);
    assert(exit_code < QMCKL_INVALID_EXIT_CODE_DEVICE);

    /* The context is assumed to exist. */
    assert(qmckl_context_check_device(context) != QMCKL_NULL_CONTEXT_DEVICE);

    qmckl_lock_device(context);
    {
        qmckl_context_struct_device *const ctx = (qmckl_context_struct_device *)context;
        assert(ctx != nullptr); /* Impossible because the context is valid. */

        ctx->error.exit_code = exit_code;
        strncpy(ctx->error.function, function_name, QMCKL_MAX_FUN_LEN_DEVICE - 1);
        strncpy(ctx->error.message, message, QMCKL_MAX_MSG_LEN_DEVICE - 1);
    }
    qmckl_unlock_device(context);

    return QMCKL_SUCCESS_DEVICE;
}

const char *qmckl_string_of_error_device(const qmckl_exit_code_device error)
{
    switch (error)
    {
    }
    return "Unknown error";
}

qmckl_exit_code_device
qmckl_failwith_device(qmckl_context_device context,
                      const qmckl_exit_code_device exit_code,
                      const char *function, const char *message)
{
    assert(exit_code > 0);
    assert(exit_code < QMCKL_INVALID_EXIT_CODE_DEVICE);
    assert(function != nullptr);
    assert(strlen(function) < QMCKL_MAX_FUN_LEN_DEVICE);
    if (message != nullptr)
    {
        assert(strlen(message) < QMCKL_MAX_MSG_LEN_DEVICE);
    }

    if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE)
        return QMCKL_INVALID_CONTEXT_DEVICE;

    if (message == nullptr)
    {
        qmckl_exit_code_device rc = qmckl_set_error_device(context, exit_code, function, qmckl_string_of_error_device(exit_code));
        assert(rc == QMCKL_SUCCESS_DEVICE);
    }
    else
    {
        qmckl_exit_code_device rc = qmckl_set_error_device(context, exit_code, function, message);
        assert(rc == QMCKL_SUCCESS_DEVICE);
    }

    return exit_code;
}