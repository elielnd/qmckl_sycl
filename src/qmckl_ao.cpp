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

	

    sycl::queue queue;

	int *shell_to_nucl = qmckl_malloc_device(queue, context, sizeof(int) * shell_num);

    queue.submit([&](handler &h) {
		h.parallel_for(range<1>(nucl_num), [=](id<1> inucl) {
			int ishell_start = nucleus_index[inucl];
			int ishell_end =
				nucleus_index[inucl] + nucleus_shell_num[inucl] - 1;
			for (int ishell = ishell_start; ishell <= ishell_end; ishell++) {
				shell_to_nucl[ishell] = inucl;
			}
		});
	}).wait();

	queue.submit([&](handler &h) {
		h.parallel_for(range<1>(point_num), [=](id<1> ipoint) {
			for (int ishell = 0; ishell < shell_num; ishell++) {

				int inucl = shell_to_nucl[ishell];

				double x = coord[ipoint] - nucl_coord[inucl];
				double y =
					coord[ipoint + point_num] - nucl_coord[inucl + nucl_num];
				double z = coord[ipoint + 2 * point_num] -
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

				int iprim_start = shell_prim_index[ishell];
				int iprim_end =
					shell_prim_index[ishell] + shell_prim_num[ishell] - 1;

				// BEWARE: set noautopar with nvidia compilers,
				// this loop must be executed serially not to degrade
				// performances. More safely, one can substitute the omp
				// collapse above with the commented line losing about 10% perf.
				for (int iprim = iprim_start; iprim <= iprim_end; iprim++) {

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
				}

				shell_vgl[ishell + 0 * shell_num + ipoint * shell_num * 5] = t0;

				shell_vgl[ishell + 1 * shell_num + ipoint * shell_num * 5] = t1;

				shell_vgl[ishell + 2 * shell_num + ipoint * shell_num * 5] = t2;

				shell_vgl[ishell + 3 * shell_num + ipoint * shell_num * 5] = t3;

				shell_vgl[ishell + 4 * shell_num + ipoint * shell_num * 5] = t4;
			}
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

	int lmax = -1;
	queue.submit([&](handler &h) {
		for (int i = 0; i < nucl_num; i++) {
			if (lmax < nucleus_max_ang_mom[i]) {
				lmax = nucleus_max_ang_mom[i];
			}
		}
	}).wait();

	size_t pows_dim = (lmax + 3) * 3 * chunk_size;
	double *pows_shared =
		qmckl_malloc_device(queue, context, sizeof(double) * pows_dim);
	
	queue.submit([&](handler &h) {
		for (int l = 0; l < 21; l++) {
			lstart[l] = l * (l + 1) * (l + 2) / 6 + 1;
		}
	}).wait();

	int k = 1;
	int *shell_to_nucl = qmckl_malloc_device(queue, context, sizeof(int) * shell_num);

	queue.submit([&](handler &h) {
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
	}).wait();

	double(*poly_vgl)[chunk_size] = (double(*)[chunk_size])poly_vgl_shared;
	double(*pows)[chunk_size] = (double(*)[chunk_size])pows_shared;

	for (int sub_iter = 0; sub_iter < num_sub_iters; sub_iter++) {
		queue.submit([&](handler &h) {
			h.parallel_for(range<1>(chunk_size), [=](id<1> iter) {
				int step = iter + sub_iter * chunk_size;
				if (step >= num_iters)
					continue;

				int ipoint = step / nucl_num;
				int inucl = step % nucl_num;

				double e_coord_0 = coord[0 * point_num + ipoint];
				double e_coord_1 = coord[1 * point_num + ipoint];
				double e_coord_2 = coord[2 * point_num + ipoint];

				double n_coord_0 = nucl_coord[0 * nucl_num + inucl];
				double n_coord_1 = nucl_coord[1 * nucl_num + inucl];
				double n_coord_2 = nucl_coord[2 * nucl_num + inucl];

				double x = e_coord_0 - n_coord_0;
				double y = e_coord_1 - n_coord_1;
				double z = e_coord_2 - n_coord_2;

				double r2 = x * x + y * y + z * z;

				if (r2 > cutoff * nucleus_range[inucl]) {
					continue;
				}

				// Beginning of ao_polynomial computation (now inlined)
				int n;

				// Already computed outsite of the ao_polynomial part
				double Y1 = x;
				double Y2 = y;
				double Y3 = z;

				int llmax = nucleus_max_ang_mom[inucl];
				if (llmax == 0) {
					poly_vgl[0][iter] = 1.;
					poly_vgl[1][iter] = 0.;
					poly_vgl[2][iter] = 0.;
					poly_vgl[3][iter] = 0.;
					poly_vgl[4][iter] = 0.;

					n = 0;
				} else if (llmax > 0) {
					// Reset pows to 0 for safety. Then we will write over
					// the top left submatrix of size (llmax+3)x3. We will
					// compute indices with llmax and not lmax, so we will
					// use the (llmax+3)*3 first elements of the array
					for (int i = 0; i < 3 * (lmax + 3); i++) {
						pows[i][iter] = 0.;
					}

					for (int i = 0; i < 3; i++) {
						for (int j = 0; j < 3; j++) {
							pows[i + (llmax + 3) * j][iter] = 1.;
						}
					}

					for (int i = 3; i < llmax + 3; i++) {
						pows[i][iter] = pows[(i - 1)][iter] * Y1;
						pows[i + (llmax + 3)][iter] =
							pows[(i - 1) + (llmax + 3)][iter] * Y2;
						pows[i + 2 * (llmax + 3)][iter] =
							pows[(i - 1) + 2 * (llmax + 3)][iter] * Y3;
					}

					for (int i = 0; i < 5; i++) {
						for (int j = 0; j < 4; j++) {
							poly_vgl[i + 5 * j][iter] = 0.;
						}
					}

					poly_vgl[0][iter] = 1.;

					poly_vgl[5][iter] = pows[3][iter];
					poly_vgl[6][iter] = 1.;

					poly_vgl[10][iter] = pows[3 + (llmax + 3)][iter];
					poly_vgl[12][iter] = 1.;

					poly_vgl[15][iter] = pows[3 + 2 * (llmax + 3)][iter];
					poly_vgl[18][iter] = 1.;

					n = 3;
				}

				double xy, yz, xz;
				double da, db, dc, dd;

				// l>=2
				dd = 2.;
				for (int d = 2; d <= llmax; d++) {

					da = dd;
					for (int a = d; a >= 0; a--) {

						db = dd - da;

						for (int b = d - a; b >= 0; b--) {

							int c = d - a - b;
							dc = dd - da - db;
							n = n + 1;

							xy = pows[(a + 2)][iter] *
								 pows[(b + 2) + (llmax + 3)][iter];
							yz = pows[(b + 2) + (llmax + 3)][iter] *
								 pows[(c + 2) + 2 * (llmax + 3)][iter];
							xz = pows[(a + 2)][iter] *
								 pows[(c + 2) + 2 * (llmax + 3)][iter];

							poly_vgl[5 * (n)][iter] =
								xy * pows[c + 2 + 2 * (llmax + 3)][iter];

							xy = dc * xy;
							xz = db * xz;
							yz = da * yz;

							poly_vgl[1 + 5 * n][iter] = pows[a + 1][iter] * yz;
							poly_vgl[2 + 5 * n][iter] =
								pows[b + 1 + (llmax + 3)][iter] * xz;
							poly_vgl[3 + 5 * n][iter] =
								pows[c + 1 + 2 * (llmax + 3)][iter] * xy;

							poly_vgl[4 + 5 * n][iter] =
								(da - 1.) * pows[a][iter] * yz +
								(db - 1.) * pows[b + (llmax + 3)][iter] * xz +
								(dc - 1.) * pows[c + 2 * (llmax + 3)][iter] *
									xy;

							db -= 1.;
						}
						da -= 1.;
					}
					dd += 1.;
				}
				// End of ao_polynomial computation (now inlined)
				// poly_vgl is now set from here
			});
		}).wait();

		queue.submit([&](handler &h) {
			h.parallel_for(range<1>(chunk_size / nucl_num), [=](id<1> iter_new) {
				for (int ishell = 0; ishell < shell_num; ishell++) {

					int ipoint = iter_new + chunk_size / nucl_num * sub_iter;
					int inucl = shell_to_nucl[ishell];
					int iter = iter_new * nucl_num + inucl;

					if (ipoint >= point_num)
						continue;

					double e_coord_0 = coord[0 * point_num + ipoint];
					double e_coord_1 = coord[1 * point_num + ipoint];
					double e_coord_2 = coord[2 * point_num + ipoint];

					double n_coord_0 = nucl_coord[0 * nucl_num + inucl];
					double n_coord_1 = nucl_coord[1 * nucl_num + inucl];
					double n_coord_2 = nucl_coord[2 * nucl_num + inucl];

					double x = e_coord_0 - n_coord_0;
					double y = e_coord_1 - n_coord_1;
					double z = e_coord_2 - n_coord_2;

					double r2 = x * x + y * y + z * z;

					if (r2 > cutoff * nucleus_range[inucl]) {
						continue;
					}

					int k = ao_index[ishell] - 1;
					int l = shell_ang_mom[ishell];

					for (int il = lstart[l] - 1; il <= lstart[l + 1] - 2;
						 il++) {

						// value
						ao_vgl[k + 0 * ao_num + ipoint * 5 * ao_num] =
							poly_vgl[il * 5 + 0][iter] *
							shell_vgl[ishell + 0 * shell_num +
									  ipoint * shell_num * 5] *
							ao_factor[k];

						// Grad x
						ao_vgl[k + 1 * ao_num + ipoint * 5 * ao_num] =
							(poly_vgl[il * 5 + 1][iter] *
								 shell_vgl[ishell + 0 * shell_num +
										   ipoint * shell_num * 5] +
							 poly_vgl[il * 5 + 0][iter] *
								 shell_vgl[ishell + 1 * shell_num +
										   ipoint * shell_num * 5]) *
							ao_factor[k];

						// grad y
						ao_vgl[k + 2 * ao_num + ipoint * 5 * ao_num] =
							(poly_vgl[il * 5 + 2][iter] *
								 shell_vgl[ishell + 0 * shell_num +
										   ipoint * shell_num * 5] +
							 poly_vgl[il * 5 + 0][iter] *
								 shell_vgl[ishell + 2 * shell_num +
										   ipoint * shell_num * 5]) *
							ao_factor[k];

						// grad z
						ao_vgl[k + 3 * ao_num + ipoint * 5 * ao_num] =
							(poly_vgl[il * 5 + 3][iter] *
								 shell_vgl[ishell + 0 * shell_num +
										   ipoint * shell_num * 5] +
							 poly_vgl[il * 5 + 0][iter] *
								 shell_vgl[ishell + 3 * shell_num +
										   ipoint * shell_num * 5]) *
							ao_factor[k];

						// Lapl_z
						ao_vgl[k + 4 * ao_num + ipoint * 5 * ao_num] =
							(poly_vgl[il * 5 + 4][iter] *
								 shell_vgl[ishell + 0 * shell_num +
										   ipoint * shell_num * 5] +
							 poly_vgl[il * 5 + 0][iter] *
								 shell_vgl[ishell + 4 * shell_num +
										   ipoint * shell_num * 5] +
							 2.0 * (poly_vgl[il * 5 + 1][iter] *
										shell_vgl[ishell + 1 * shell_num +
												  ipoint * shell_num * 5] +
									poly_vgl[il * 5 + 2][iter] *
										shell_vgl[ishell + 2 * shell_num +
												  ipoint * shell_num * 5] +
									poly_vgl[il * 5 + 3][iter] *
										shell_vgl[ishell + 3 * shell_num +
												  ipoint * shell_num * 5])) *
							ao_factor[k];
						k = k + 1;
					}
				}
			});
			// End of outer compute loop
		}).wait();

	}
	// End of target data region
	qmckl_free_device(queue, context, ao_index);
	qmckl_free_device(queue, context, poly_vgl_shared);
	qmckl_free_device(queue, context, pows_shared);
	qmckl_free_device(queue, context, shell_to_nucl);
	qmckl_free_device(queue, context, lstart);

	return QMCKL_SUCCESS_DEVICE;
}

/* ao_value */

qmckl_exit_code_device qmckl_compute_ao_value_gaussian_device(
	const qmckl_context_device context, const int64_t ao_num,
	const int64_t shell_num, const int64_t point_num, const int64_t nucl_num,
	const double *restrict coord, const double *restrict nucl_coord,
	const int64_t *restrict nucleus_index,
	const int64_t *restrict nucleus_shell_num, const double *nucleus_range,
	const int32_t *restrict nucleus_max_ang_mom,
	const int32_t *restrict shell_ang_mom, const double *restrict ao_factor,
	double *shell_vgl, double *restrict const ao_value) {

	double cutoff = 27.631021115928547;

	// int64_t target_chunk = 128*1024;
	int64_t target_chunk = (MAX_MEMORY_SIZE / 4.) / (sizeof(double) * ao_num);
	size_t max_chunk_size = ((target_chunk) / nucl_num) * nucl_num;
	int64_t num_iters = point_num * nucl_num;
	int64_t chunk_size =
		(num_iters < max_chunk_size) ? num_iters : max_chunk_size;
	int64_t num_sub_iters = (num_iters + chunk_size - 1) / chunk_size;
	int64_t poly_dim = ao_num * chunk_size;

	sycl::queue queue;

	double *poly_vgl_shared =
		qmckl_malloc_device(queue, context, sizeof(double) * poly_dim);
	int64_t *ao_index =
		qmckl_malloc_device(queue, context, sizeof(int64_t) * shell_num);
	int64_t *lstart = qmckl_malloc_device(queue, context, sizeof(int64_t) * 21);


	// Specific calling function
	int lmax = -1;
	queue.submit([&](handler &h) {
		for (int i = 0; i < nucl_num; i++) {
			if (lmax < nucleus_max_ang_mom[i]) {
				lmax = nucleus_max_ang_mom[i];
			}
		}
	}).wait();
	size_t pows_dim = (lmax + 3) * 3 * chunk_size;
	double *pows_shared =
		qmckl_malloc_device(queue, context, sizeof(double) * pows_dim);

	queue.submit([&](handler &h) {
		for (int l = 0; l < 21; l++) {
			lstart[l] = l * (l + 1) * (l + 2) / 6 + 1;
		}
	}).wait();

	int k = 1;
	int *shell_to_nucl = qmckl_malloc_device(queue, context, sizeof(int) * shell_num);

	queue.submit([&](handler &h) {
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
	}).wait();

	double(*poly_vgl)[chunk_size] = (double(*)[chunk_size])poly_vgl_shared;
	double(*pows)[chunk_size] = (double(*)[chunk_size])pows_shared;

	for (int sub_iter = 0; sub_iter < num_sub_iters; sub_iter++) {
		queue.submit([&](handler &h) {
			h.parallel_for(range<1>(chunk_size), [=](id<1> iter) {
				int step = iter + sub_iter * chunk_size;
				if (step >= num_iters)
					continue;

				int ipoint = step / nucl_num;
				int inucl = step % nucl_num;

				double e_coord_0 = coord[0 * point_num + ipoint];
				double e_coord_1 = coord[1 * point_num + ipoint];
				double e_coord_2 = coord[2 * point_num + ipoint];

				double n_coord_0 = nucl_coord[0 * nucl_num + inucl];
				double n_coord_1 = nucl_coord[1 * nucl_num + inucl];
				double n_coord_2 = nucl_coord[2 * nucl_num + inucl];

				double x = e_coord_0 - n_coord_0;
				double y = e_coord_1 - n_coord_1;
				double z = e_coord_2 - n_coord_2;

				double r2 = x * x + y * y + z * z;

				if (r2 > cutoff * nucleus_range[inucl]) {
					continue;
				}

				// Beginning of ao_polynomial computation (now inlined)
				int n;

				// Already computed outsite of the ao_polynomial part
				double Y1 = x;
				double Y2 = y;
				double Y3 = z;

				int llmax = nucleus_max_ang_mom[inucl];
				if (llmax == 0) {
					poly_vgl[0][iter] = 1.;
					poly_vgl[1][iter] = 0.;
					poly_vgl[2][iter] = 0.;
					poly_vgl[3][iter] = 0.;
					poly_vgl[4][iter] = 0.;

					n = 0;
				} else if (llmax > 0) {
					// Reset pows to 0 for safety. Then we will write over
					// the top left submatrix of size (llmax+3)x3. We will
					// compute indices with llmax and not lmax, so we will
					// use the (llmax+3)*3 first elements of the array
					for (int i = 0; i < 3 * (lmax + 3); i++) {
						pows[i][iter] = 0.;
					}

					for (int i = 0; i < 3; i++) {
						for (int j = 0; j < 3; j++) {
							pows[i + (llmax + 3) * j][iter] = 1.;
						}
					}

					for (int i = 3; i < llmax + 3; i++) {
						pows[i][iter] = pows[(i - 1)][iter] * Y1;
						pows[i + (llmax + 3)][iter] =
							pows[(i - 1) + (llmax + 3)][iter] * Y2;
						pows[i + 2 * (llmax + 3)][iter] =
							pows[(i - 1) + 2 * (llmax + 3)][iter] * Y3;
					}

					for (int i = 0; i < 5; i++) {
						for (int j = 0; j < 4; j++) {
							poly_vgl[i + 5 * j][iter] = 0.;
						}
					}

					poly_vgl[0][iter] = 1.;

					poly_vgl[5][iter] = pows[3][iter];
					poly_vgl[6][iter] = 1.;

					poly_vgl[10][iter] = pows[3 + (llmax + 3)][iter];
					poly_vgl[12][iter] = 1.;

					poly_vgl[15][iter] = pows[3 + 2 * (llmax + 3)][iter];
					poly_vgl[18][iter] = 1.;

					n = 3;
				}

				double xy, yz, xz;
				double da, db, dc, dd;

				// l>=2
				dd = 2.;
				for (int d = 2; d <= llmax; d++) {

					da = dd;
					for (int a = d; a >= 0; a--) {

						db = dd - da;
						for (int b = d - a; b >= 0; b--) {

							int c = d - a - b;
							dc = dd - da - db;
							n = n + 1;

							xy = pows[(a + 2)][iter] *
								 pows[(b + 2) + (llmax + 3)][iter];
							yz = pows[(b + 2) + (llmax + 3)][iter] *
								 pows[(c + 2) + 2 * (llmax + 3)][iter];
							xz = pows[(a + 2)][iter] *
								 pows[(c + 2) + 2 * (llmax + 3)][iter];

							poly_vgl[5 * (n)][iter] =
								xy * pows[c + 2 + 2 * (llmax + 3)][iter];

							xy = dc * xy;
							xz = db * xz;
							yz = da * yz;

							poly_vgl[1 + 5 * n][iter] = pows[a + 1][iter] * yz;
							poly_vgl[2 + 5 * n][iter] =
								pows[b + 1 + (llmax + 3)][iter] * xz;
							poly_vgl[3 + 5 * n][iter] =
								pows[c + 1 + 2 * (llmax + 3)][iter] * xy;

							poly_vgl[4 + 5 * n][iter] =
								(da - 1.) * pows[a][iter] * yz +
								(db - 1.) * pows[b + (llmax + 3)][iter] * xz +
								(dc - 1.) * pows[c + 2 * (llmax + 3)][iter] *
									xy;

							db -= 1.;
						}
						da -= 1.;
					}
					dd += 1.;
				}
			});
			// End of ao_polynomial computation (now inlined)
			// poly_vgl is now set from here
		}).wait();

		queue.submit([&](handler &h) {
			h.parallel_for(range<1>(chunk_size / nucl_num), [=](id<1> iter_new) {
				for (int ishell = 0; ishell < shell_num; ishell++) {

					int ipoint = iter_new + chunk_size / nucl_num * sub_iter;
					int inucl = shell_to_nucl[ishell];
					int iter = iter_new * nucl_num + inucl;

					if (ipoint >= point_num)
						continue;

					double e_coord_0 = coord[0 * point_num + ipoint];
					double e_coord_1 = coord[1 * point_num + ipoint];
					double e_coord_2 = coord[2 * point_num + ipoint];

					double n_coord_0 = nucl_coord[0 * nucl_num + inucl];
					double n_coord_1 = nucl_coord[1 * nucl_num + inucl];
					double n_coord_2 = nucl_coord[2 * nucl_num + inucl];

					double x = e_coord_0 - n_coord_0;
					double y = e_coord_1 - n_coord_1;
					double z = e_coord_2 - n_coord_2;

					double r2 = x * x + y * y + z * z;

					if (r2 > cutoff * nucleus_range[inucl]) {
						continue;
					}

					int k = ao_index[ishell] - 1;
					int l = shell_ang_mom[ishell];

					for (int il = lstart[l] - 1; il <= lstart[l + 1] - 2;
						 il++) {

						ao_value[k + ipoint * ao_num] =
							poly_vgl[il * 5][iter] *
							shell_vgl[ishell + ipoint * shell_num * 5] *
							ao_factor[k];
						k = k + 1;
					}
				}
			});
		}).wait();
		// End of outer compute loop
	}

	// End of target data region
	qmckl_free_device(queue, context, ao_index);
	qmckl_free_device(queue, context, poly_vgl_shared);
	qmckl_free_device(queue, context, pows_shared);
	qmckl_free_device(queue, context, shell_to_nucl);
	qmckl_free_device(queue, context, lstart);

	return QMCKL_SUCCESS_DEVICE;
}

//**********
// PROVIDE
//**********

/* ao_value */

qmckl_exit_code_device
qmckl_provide_ao_basis_ao_value_device(qmckl_context_device context) {

	qmckl_exit_code_device rc = QMCKL_SUCCESS_DEVICE;

	if (qmckl_context_check_device(context) == QMCKL_NULL_CONTEXT_DEVICE) {
		return qmckl_failwith_device(context, QMCKL_INVALID_CONTEXT_DEVICE,
									 "qmckl_provide_ao_basis_ao_value", NULL);
	}

	qmckl_context_struct_device *const ctx =
		(qmckl_context_struct_device *)context;
	assert(ctx != NULL);

	sycl::queue queue;

	if (!ctx->ao_basis.provided) {
		return qmckl_failwith_device(context, QMCKL_NOT_PROVIDED_DEVICE,
									 "qmckl_provide_ao_basis_ao_value", NULL);
	}

	/* Compute if necessary */
	if (ctx->point.date > ctx->ao_basis.ao_value_date) {

		/* Allocate array */
		if (ctx->ao_basis.ao_value == NULL) {

			double *ao_value = (double *)qmckl_malloc_device(
				context,
				ctx->ao_basis.ao_num * ctx->point.num * sizeof(double));

			if (ao_value == NULL) {
				return qmckl_failwith_device(
					context, QMCKL_ALLOCATION_FAILED_DEVICE,
					"qmckl_provide_ao_basis_ao_value", NULL);
			}
			ctx->ao_basis.ao_value = ao_value;
		}

		if (ctx->point.date <= ctx->ao_basis.ao_vgl_date &&
			ctx->ao_basis.ao_vgl != NULL) {
			// ao_vgl is already computed and recent enough, we just need to
			// copy the required data to ao_value

			double *v = ctx->ao_basis.ao_value;
			double *vgl = ctx->ao_basis.ao_vgl;
			int point_num = ctx->point.num;
			int ao_num = ctx->ao_basis.ao_num;

			queue.submit([&](handler &h) {
				for (int i = 0; i < point_num; ++i) {
					for (int k = 0; k < ao_num; ++k) {
						v[i * ao_num + k] = vgl[i * ao_num * 5 + k];
					}
				}
			}).wait();
		
		} else {
			// We don't have ao_vgl, so we will compute the values only

			/* Checking for shell_vgl */
			if (ctx->ao_basis.shell_vgl == NULL ||
				ctx->point.date > ctx->ao_basis.shell_vgl_date) {
				qmckl_provide_ao_basis_shell_vgl_device(context);
			}

			if (ctx->ao_basis.type == 'G') {
				rc = qmckl_compute_ao_value_gaussian_device(
					context, ctx->ao_basis.ao_num, ctx->ao_basis.shell_num,
					ctx->point.num, ctx->nucleus.num, ctx->point.coord.data,
					ctx->nucleus.coord.data, ctx->ao_basis.nucleus_index,
					ctx->ao_basis.nucleus_shell_num,
					ctx->ao_basis.nucleus_range,
					ctx->ao_basis.nucleus_max_ang_mom,
					ctx->ao_basis.shell_ang_mom, ctx->ao_basis.ao_factor,
					ctx->ao_basis.shell_vgl, ctx->ao_basis.ao_value);
			} else {
				return qmckl_failwith_device(context, QMCKL_ERRNO_DEVICE,
											 "qmckl_ao_basis_ao_value", NULL);
			}
		}



	}
}