#include <CL/sycl.hpp>

#include "include/qmckl_ao.h"
#define MAX_MEMORY_SIZE 16.0 * 1024 * 1024 * 1024

//**********
// COMPUTE
//**********

/* shell_vgl */

qmckl_exit_code_device qmckl_compute_ao_basis_shell_gaussian_vgl_device(
	qmckl_context_device context, int prim_num, int shell_num, int point_num,
	int nucl_num, int64_t *nucleus_shell_num, int64_t *nucleus_index,
	double *nucleus_range, int64_t *shell_prim_index, int64_t *shell_prim_num,
	double *coord, double *nucl_coord, double *expo, double *coef_normalized,
	double *shell_vgl) {

    // Don't compute exponentials when the result will be almost zero.
	// TODO : Use numerical precision here
	double cutoff = 27.631021115928547; //-dlog(1.d-12)

	// Use SYCL buffer to allocate memory on the device
    sycl::buffer<int, 1> prim_num_buffer(prim_num, sycl::range<1>(1));		// buffer for prim_num
    sycl::buffer<int, 1> shell_num_buffer(shell_num, sycl::range<1>(1));		// buffer for shell_num
    sycl::buffer<int, 1> point_num_buffer(point_num, sycl::range<1>(1));		// buffer for point_num
    sycl::buffer<int, 1> nucl_num_buffer(nucl_num, sycl::range<1>(1));		// buffer for nucl_num
    sycl::buffer<int64_t, 1> nucleus_shell_num_buffer(nucleus_shell_num, sycl::range<1>(nucl_num));		// buffer for nucleus_shell_num
    sycl::buffer<int64_t, 1> nucleus_index_buffer(nucleus_index, sycl::range<1>(nucl_num));		// buffer for nucleus_index
    sycl::buffer<double, 1> nucleus_range_buffer(nucleus_range, sycl::range<1>(shell_num));		// buffer for nucleus_range
    sycl::buffer<int64_t, 1> shell_prim_index_buffer(shell_prim_index, sycl::range<1>(shell_num));		// buffer for shell_prim_index
    sycl::buffer<int64_t, 1> shell_prim_num_buffer(shell_prim_num, sycl::range<1>(shell_num));		// buffer for shell_prim_num
    sycl::buffer<double, 1> coord_buffer(coord, sycl::range<1>(point_num));		// buffer for coord
    sycl::buffer<double, 1> nucl_coord_buffer(nucl_coord, sycl::range<1>(point_num));		// buffer for nucl_coord

    sycl::queue queue;

	int *shell_to_nucl = qmckl_malloc_device(queue, context, sizeof(int) * shell_num);

    queue.submit([&](handler &h) {
		h.parallel_for(range<1>(nucl_num), [=](id<1> nucl_num_item) {
			int shell_num_item_start = nucleus_index[nucl_num_item];
			int shell_num_item_end =
				nucleus_index[nucl_num_item] + nucleus_shell_num[nucl_num_item] - 1;
			h.parallel_for(range<1>(shell_num_item_end), [=](id<1> item2) {
				shell_to_nucl[item2 + shell_num_item_start] = nucl_num_item;
			});
		});
	}).wait();

	queue.submit([&](handler &h) {
		h.parallel_for(range<1>(point_num), [=](id<1> point_num_item) {

			h.parallel_for(range<1>(shell_num), [=](id<1> shell_num_item) {
				int inucl = shell_to_nucl[shell_num_item];

				double x = coord[point_num_item] - nucl_coord[inucl];
				double y =
					coord[point_num_item + point_num] - nucl_coord[inucl + nucl_num];
				double z = coord[point_num_item + 2 * point_num] -
						   nucl_coord[inucl + 2 * nucl_num];

				double r2 = x * x + y * y + z * z;

				if (r2 > cutoff * nucleus_range[inucl]) {
					continue;
				}

				double t0 = 0;
				double t1 = 0;
				double t2 = 0;
				double t3 = 0;
				double t4 = 0;

				int iprim_start = shell_prim_index[shell_num_item];
				int iprim_end =
					shell_prim_index[shell_num_item] + shell_prim_num[shell_num_item] - 1;
				
				h.parallel_for(range<1>(iprim_end), [=](id<1> iprim) {
					double ar2 = expo[iprim] * r2;
					if (ar2 > cutoff) {
						continue;
					}

					double v = coef_normalized[iprim] * exp(-ar2);
					double two_a = -2 * expo[iprim] * v;

					t0 += v;
					t1 += two_a * x;
					t2 += two_a * y;
					t3 += two_a * z;
					t4 += two_a * (3 - 2 * ar2);
				});

				shell_vgl[shell_num_item + 0 * shell_num + point_num_item * shell_num * 5] = t0;

				shell_vgl[shell_num_item + 1 * shell_num + point_num_item * shell_num * 5] = t1;

				shell_vgl[shell_num_item + 2 * shell_num + point_num_item * shell_num * 5] = t2;

				shell_vgl[shell_num_item + 3 * shell_num + point_num_item * shell_num * 5] = t3;

				shell_vgl[shell_num_item + 4 * shell_num + point_num_item * shell_num * 5] = t4;
			});
		});
	}).wait();

    qmckl_free_device(queue, context, shell_to_nucl);

	qmckl_exit_code_device info = QMCKL_SUCCESS_DEVICE;
	return info;
}


/* ao_vgl */

qmckl_exit_code_device qmckl_compute_ao_vgl_gaussian_device(
	const qmckl_context_device context, const int64_t ao_num,
	const int64_t shell_num, const int64_t point_num, const int64_t nucl_num,
	const double *restrict coord, const double *restrict nucl_coord,
	const int64_t *restrict nucleus_index,
	const int64_t *restrict nucleus_shell_num, const double *nucleus_range,
	const int32_t *restrict nucleus_max_ang_mom,
	const int32_t *restrict shell_ang_mom, const double *restrict ao_factor,
	double *shell_vgl, double *restrict const ao_vgl) {

	double cutoff = 27.631021115928547;

	// TG: MAX_MEMORY_SIZE should be roughly the GPU RAM, to be set at configure
	// time? Not to exceed GPU memory when allocating poly_vgl
	int64_t target_chunk =
		(MAX_MEMORY_SIZE / 4.) / (sizeof(double) * 5 * ao_num);
	// A good guess, can result in large memory occupation though
	// int64_t target_chunk = 128*1024;
	size_t max_chunk_size = ((target_chunk) / nucl_num) * nucl_num;
	int64_t num_iters = point_num * nucl_num;
	int64_t chunk_size =
		(num_iters < max_chunk_size) ? num_iters : max_chunk_size;
	int64_t num_sub_iters = (num_iters + chunk_size - 1) / chunk_size;
	int64_t poly_dim = 5 * ao_num * chunk_size;

	sycl::queue queue;

	double *poly_vgl_shared =
		qmckl_malloc_device(queue, context, sizeof(double) * poly_dim);
	int64_t *ao_index =
		qmckl_malloc_device(queue, context, sizeof(int64_t) * shell_num);
	int64_t *lstart = qmckl_malloc_device(queue, context, sizeof(int64_t) * 21);

	// Specific calling function
	int lmax = -1;
#pragma omp target is_device_ptr(nucleus_max_ang_mom) map(tofrom : lmax)
	{
		for (int i = 0; i < nucl_num; i++) {
			if (lmax < nucleus_max_ang_mom[i]) {
				lmax = nucleus_max_ang_mom[i];
			}
		}
	}
	size_t pows_dim = (lmax + 3) * 3 * chunk_size;
	double *pows_shared =
		qmckl_malloc_device(context, sizeof(double) * pows_dim);

#pragma omp target is_device_ptr(lstart)
	{
		for (int l = 0; l < 21; l++) {
			lstart[l] = l * (l + 1) * (l + 2) / 6 + 1;
		}
	}

	int k = 1;
	int *shell_to_nucl = qmckl_malloc_device(context, sizeof(int) * shell_num);
#pragma omp target is_device_ptr(nucleus_index, nucleus_shell_num,             \
								 shell_ang_mom, ao_index, lstart,              \
								 shell_to_nucl) map(tofrom                     \
													: k)
	{
		for (int inucl = 0; inucl < nucl_num; inucl++) {
			int ishell_start = nucleus_index[inucl];
			int ishell_end =
				nucleus_index[inucl] + nucleus_shell_num[inucl] - 1;
			for (int ishell = ishell_start; ishell <= ishell_end; ishell++) {
				int l = shell_ang_mom[ishell];
				ao_index[ishell] = k;
				k = k + lstart[l + 1] - lstart[l];
				shell_to_nucl[ishell] = inucl;
			}
		}
	}

	double(*poly_vgl)[chunk_size] = (double(*)[chunk_size])poly_vgl_shared;
	double(*pows)[chunk_size] = (double(*)[chunk_size])pows_shared;

	// End of target data region
	qmckl_free_device(context, ao_index);
	qmckl_free_device(context, poly_vgl_shared);
	qmckl_free_device(context, pows_shared);
	qmckl_free_device(context, shell_to_nucl);
	qmckl_free_device(context, lstart);

	return QMCKL_SUCCESS_DEVICE;
}
