#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([qmckl_gpu], [0.0.1], [BUG-REPORT-ADDRESS])
AM_INIT_AUTOMAKE([foreign -Wall subdir-objects])
AM_PROG_AR

AC_PROG_CXX([icpx clang++])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT

# Initialize Makefile
LT_INIT
AC_CONFIG_MACRO_DIRS([m4])
AC_ENABLE_SHARED
AC_ENABLE_STATIC

# Add mandatory CFLAGS/FCFLAGS
CXXFLAGS="-Iinclude ${CXXFLAGS}"


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
  CXXFLAGS="$CXXFLAGS $TREXIO_CFLAGS"
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
      CXXFLAGS="$CXXFLAGS $TREXIO_CFLAGS"
      LIBS="$LIBS $TREXIO_LIBS"
      ARGS="${ARGS} trexio"
      have_trexio=yes
    ],
    [
      AC_MSG_ERROR("trexio.h not found")
    ])
])

# Auto offloading flags
AC_ARG_ENABLE(autoflags, [AS_HELP_STRING([--disable-autoflags],[By default, configure will try to infer and add offloading compiler flags by itself. Disable this option and set your flags manually if you encounter issues.])], have_autoflags=no, have_autoflags=yes)
AS_IF([test "x$have_autoflags" = "xyes"], [
	echo "Autoflags have been kept enabled"
], [
	echo "Autoflags have been disabled"
])

# If TREXIO define Automake & preprocessors variables
AS_IF([test "x$have_trexio" = "xyes"], [
	AC_DEFINE([HAVE_TREXIO], [1], [Enable compilation of TREXIO functions])
])


####
## Define Makefile variables from configure flags
####

AM_CONDITIONAL([HAVE_TREXIO], [test "x$have_trexio"  = "xyes"])


####
# Print configure status
####

AC_OUTPUT
echo \
"-------------------------------------------------
${PACKAGE_NAME} Version ${PACKAGE_VERSION}

Prefix: '${prefix}'

CXX..................: ${CXX}
CXXFLAGS.............: ${CXXFLAGS}
LDFLAGS:.............: ${LDFLAGS}

Autoflags ...........: ${have_autoflags}

TREXIO ..............: ${have_trexio}


Now type 'make @<:@<target>@:>@'
where the optional <target> is:
  all           - build ${PACKAGE_NAME}
  check         - run tests
  install       - install
--------------------------------------------------"