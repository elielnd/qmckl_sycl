#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([qmckl_gpu], [0.0.1], [BUG-REPORT-ADDRESS])
AM_INIT_AUTOMAKE([-Wall subdir-objects])

AC_PROG_CXX
AC_CONFIG_FILES([Makefile])
AC_OUTPUT

# Initialize Makefile
LT_INIT
AC_CONFIG_MACRO_DIRS([m4])
AC_ENABLE_SHARED
AC_ENABLE_STATIC

# Add mandatory CFLAGS/FCFLAGS
CFLAGS="-Iinclude ${CFLAGS}"
FCFLAGS="-Iinclude ${FCFLAGS}"


###
# TREXIO
###

# Get TREXIO path if any
AC_ARG_ENABLE([trexio],
            [AS_HELP_STRING([--[enable|disable]-trexio],
                            [enable or disable support for TREXIO (configure will try to enable it by default)])],
            [have_trexio=$enableval],
            [have_trexio=yes])

AS_CASE([$have_trexio],

  # Explicitely disabled
  [no], [
    AC_OUTPUT
    echo \
    "Note: TREXIO explicitely disabled, skipping it."
  ],

  # Implicitely enabled (not mandatory)
  [yes],
  [AC_CHECK_LIB([trexio], [trexio_open], [],
                [PKG_CHECK_MODULES([TREXIO], [trexio]) ])
  CFLAGS="$CFLAGS $TREXIO_CFLAGS"
  LIBS="$LIBS $TREXIO_LIBS"
  AC_CHECK_HEADERS([trexio.h], [
      ARGS="${ARGS} trexio"
      have_trexio=yes
  ], [
    AC_OUTPUT
    echo \
    "Warning: TREXIO not found. Library will be built without it."
    have_trexio=no
  ])
  ARGS="${ARGS} trexio"],

  # Default case : explicitely enabled with specified path (mandatory)
  [AS_IF([test ! -d "$enableval"], [

    AC_MSG_ERROR([--enable-trexio=$enableval: argument is not a directory])])
    AC_CHECK_HEADERS([trexio.h], [
      TREXIO_LIBS="-L$have_trexio/lib -ltrexio"
      TREXIO_CFLAGS="-I$have_trexio/include"
      CFLAGS="$CFLAGS $TREXIO_CFLAGS"
      LIBS="$LIBS $TREXIO_LIBS"
      ARGS="${ARGS} trexio"
      have_trexio=yes
    ],
    [
      AC_MSG_ERROR("trexio.h not found")
    ])
])




