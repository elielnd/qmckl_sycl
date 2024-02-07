###
# Build sources
###


ACLOCAL_AMFLAGS = -I m4 -I include 

AM_CPPFLAGS  = -fsycl -I$(top_builddir)/src -I$(top_builddir)/include 

include_HEADERS = include/qmckl_gpu.hpp

lib_LTLIBRARIES = libqmckl_gpu.la

libqmckl_gpu_la_SOURCES = ""
libqmckl_gpu_la_SOURCES += src/qmckl_context.cpp src/qmckl_basic_functions.cpp src/qmckl_point.cpp \
src/qmckl_memory.cpp src/qmckl_blas.cpp src/qmckl_electron.cpp src/qmckl_nucleus.cpp \
src/qmckl_ao.cpp src/qmckl_mo.cpp src/qmckl_jastrow.cpp

# Enable TREXIO interfacing
if HAVE_TREXIO
  libqmckl_gpu_la_SOURCES += src/qmckl_trexio.cpp
endif

###
# Build test programs
###

check_PROGRAMS = tests/test_qmckl_ao tests/test_qmckl_mo tests/test_qmckl_jastrow

tests_test_qmckl_ao_SOURCES = tests/test_qmckl_ao_omp.c chbrclf.h
tests_test_qmckl_ao_LDFLAGS = -pthread

tests_test_qmckl_mo_SOURCES = tests/test_qmckl_mo_omp.c chbrclf.h
tests_test_qmckl_mo_LDFLAGS = -pthread 

tests_test_qmckl_jastrow_SOURCES = tests/test_qmckl_jastrow_omp.c n2.h
tests_test_qmckl_jastrow_LDFLAGS = --pthread 

# This shoud be added whether we use OpenMP or ACC
tests_test_qmckl_ao_LDADD = libqmckl_gpu.la
tests_test_qmckl_mo_LDADD = libqmckl_gpu.la
tests_test_qmckl_jastrow_LDADD = libqmckl_gpu.la

###
# Run test programs
###

TESTS = $(check_PROGRAMS)
