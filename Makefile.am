###
# Build sources
###


ACLOCAL_AMFLAGS = -I m4 -I include 

AM_CXXFLAGS  = -fsycl -g   -I$(top_builddir)/src -I$(top_builddir)/include 

include_HEADERS = include/qmckl_gpu.hpp

lib_LIBRARIES = libqmckl_gpu.a

libqmckl_gpu_a_SOURCES = src/qmckl_context.cpp src/qmckl_basic_functions.cpp src/qmckl_point.cpp \
src/qmckl_memory.cpp src/qmckl_blas.cpp src/qmckl_electron.cpp src/qmckl_nucleus.cpp \
src/qmckl_ao.cpp src/qmckl_ao_sycl.cpp  src/qmckl_mo.cpp src/qmckl_mo_sycl.cpp  src/qmckl_jastrow.cpp
libqmckl_qpu_a_CXXFLAGS = $(AM_CXXFLAGS)




# Enable TREXIO interfacing
if HAVE_TREXIO
  libqmckl_gpu_a_SOURCES += src/qmckl_trexio.cpp
endif


###
# Build test programs
###

check_PROGRAMS = tests/test_qmckl_ao tests/test_qmckl_mo tests/test_qmckl_jastrow

tests_test_qmckl_ao_SOURCES = tests/test_qmckl_ao.cpp chbrclf.hpp
tests_test_qmckl_ao_LDFLAGS = -pthread

tests_test_qmckl_mo_SOURCES = tests/test_qmckl_mo.cpp chbrclf.hpp
tests_test_qmckl_mo_LDFLAGS = -pthread 

tests_test_qmckl_jastrow_SOURCES = tests/test_qmckl_jastrow.cpp n2.hpp
tests_test_qmckl_jastrow_LDFLAGS = -pthread 

# This shoud be added whether we use OpenMP or ACC
tests_test_qmckl_ao_LDADD = libqmckl_gpu.a 
tests_test_qmckl_mo_LDADD = libqmckl_gpu.a
tests_test_qmckl_jastrow_LDADD = libqmckl_gpu.a

###
# Run test programs
###

TESTS = $(check_PROGRAMS)

clean-local: 
	-rm -rf autom4te.cache/ m4/
	-rm -f aclocal.m4 ar-lib compile
	-rm -f config.* configure
	-rm -f depcomp install-sh libtool ltmain.sh missing
	-rm -f Makefile  Makefile.in
	-rm -f test-driver
