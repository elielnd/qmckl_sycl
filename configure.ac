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

# Add mandatory CXXFLAGS
CXXFLAGS="-Iinclude ${CXXFLAGS}" 


###
# TREXIO
###



# Auto offloading flags
AC_ARG_ENABLE(autoflags, [AS_HELP_STRING([--disable-autoflags],[By default, configure will try to infer and add offloading compiler flags by itself. Disable this option and set your flags manually if you encounter issues.])], have_autoflags=no, have_autoflags=yes)
AS_IF([test "x$have_autoflags" = "xyes"], [
	echo "Autoflags have been kept enabled"
], [
	echo "Autoflags have been disabled"
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
LIBS:................: ${LIBS}
ARGS:................: ${ARGS}

Autoflags ...........: ${have_autoflags}

TREXIO ..............: ${have_trexio}


Now type 'make @<:@<target>@:>@'
where the optional <target> is:
  all           - build ${PACKAGE_NAME}
  check         - run tests
  install       - install
--------------------------------------------------"