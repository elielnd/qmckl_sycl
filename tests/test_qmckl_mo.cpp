#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>
#include <iostream>
#include "chbrclf.hpp"
#include "../include/qmckl_gpu.hpp"

using namespace sycl;

#define AO_VALUE_ID(x, y) ao_num *x + y
#define AO_VGL_ID(x, y, z) 5 * ao_num *x + ao_num *y + z

int main()
{
    //TO DO
} 
