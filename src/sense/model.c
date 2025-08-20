/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <uecker@eecs.berkeley.edu>
 *
 *
 * Ra JB, Rim CY. Fast imaging using subencoding data sets from multiple detectors. 
 * Magn Reson Med 1993; 30:142-145.
 *
 * Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast
 * MRI. Magn Reson Med 1999; 42:952-962.
 *
 * Pruessmann KP, Weiger M, Boernert P, Boesiger P. Advances in sensitivity
 * encoding with arbitrary k-space trajectories. 
 * Magn Reson Med 2001; 46:638-651.
 *
 * Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M.
 * ESPIRiT - An Eigenvalue Approach to  Autocalibrating Parallel MRI: Where SENSE 
 * meets GRAPPA. Magn Reson Med 2014; 71:990-1001.
 *
 */

#include <string.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"


#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "misc/mri.h"

#include "model.h"


struct linop_s* linop_sampling_create(const long dims[DIMS], const long pat_dims[DIMS], const complex float* pattern)
{
	assert(md_check_compat(DIMS, ~0UL, dims, pat_dims));

	auto ret = linop_cdiag_create(DIMS, dims, md_nontriv_dims(DIMS, pat_dims), NULL);
	linop_gdiag_set_diag_ref(ret, DIMS, pat_dims, pattern);

	return ret;
}


/**
 * Create maps operator, m = S x
 *
 * @param shared_img_flags select dimensions not present in image
 * @param max_dims maximal dimensions across all data structures
 * @param sens_flags active map dimensions
 * @param sens sensitivities
 */
struct linop_s* maps_create(unsigned long shared_img_flags, const long max_dims[DIMS], 
			unsigned long sens_flags, const complex float* sens)
{
	long mps_dims[DIMS];
	md_select_dims(DIMS, sens_flags, mps_dims, max_dims);
	complex float* nsens = md_alloc_sameplace(DIMS, mps_dims, CFL_SIZE, sens);
	fftscale(DIMS, mps_dims, FFT_FLAGS, nsens, sens);

	long cim_dims[DIMS];
	long img_dims[DIMS];

	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, max_dims);
	md_select_dims(DIMS, ~COIL_FLAG & ~shared_img_flags, img_dims, max_dims);

	auto ret = (struct linop_s*)linop_fmac_dims_create(DIMS, cim_dims, img_dims, mps_dims, NULL);
	linop_fmac_set_tensor_F(ret, DIMS, mps_dims, nsens);

	return ret;
}



struct linop_s* maps2_create(const long coilim_dims[DIMS], const long maps_dims[DIMS], const long img_dims[DIMS], const complex float* maps)
{
	unsigned long sens_flags = 0;

	for (int i = 0; i < DIMS; i++)
		if (1 != maps_dims[i])
			sens_flags = MD_SET(sens_flags, i);

	assert(1 == coilim_dims[MAPS_DIM]);
	assert(1 == img_dims[COIL_DIM]);
	assert(maps_dims[COIL_DIM] == coilim_dims[COIL_DIM]);
	assert(maps_dims[MAPS_DIM] == img_dims[MAPS_DIM]);

	auto ret = (struct linop_s*)linop_fmac_dims_create(DIMS, coilim_dims, img_dims, maps_dims, NULL);
	linop_fmac_set_tensor_ref(ret, DIMS, maps_dims, maps);

	return ret;
}




/**
 * Create sense operator, y = F S x,
 * where F is the Fourier transform and S is the sensitivity maps
 *
 * @param shared_img_flags select dimensions not present in image
 * @param max_dims maximal dimensions across all data structures
 * @param sens_flags active map dimensions
 * @param sens sensitivities
 */
struct linop_s* sense_init(unsigned long shared_img_flags, const long max_dims[DIMS], 
			unsigned long sens_flags, const complex float* sens)
{
	long ksp_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, ksp_dims, max_dims);

	struct linop_s* fft = linop_fft_create(DIMS, ksp_dims, FFT_FLAGS);
	struct linop_s* maps = maps_create(shared_img_flags, max_dims, sens_flags, sens);

	struct linop_s* sense_op = linop_chain(maps, fft);

	linop_free(fft);
	linop_free(maps);

	return sense_op;
}


// Shreya start - adding for polynomial preconditioned forward op
/** 
* linop_s is struct that is collection of operator_s, one for each of forward/adjoint/normal/norm_inv. (see linop.h)
* We want new linop_polyprecond to replace the sense_op, and are just replacing the adjoint and normal as:
* AHA --> P(AHA) * AHA, and AH --> P(AHA) * AH
*/

struct linop_s* polyprecond_sense_normal(const struct linop_s* sense_linop_normal, 
												const float* coeffs, 
												const int D, const int N, 
												const long *img_dims) {

	// Linop that scales by first coefficient passed in
	struct linop_s* out_linop = linop_scale_create(N, img_dims, coeffs[0]);

	if (D == 1) {
		return out_linop;
	}
	else {
		// Increment pointer, update length, and run recursively
		// return c[0]*I + AHA * phelper
		return linop_plus(out_linop, 
				linop_chain( 
					polyprecond_sense_normal(sense_linop_normal, coeffs + 1, D - 1, N, img_dims), 
					sense_linop_normal)
			);
	}
	
}

/**
* Linop for Polynomial Preconditioning
* 
* @param sense_op	normal sense forward linop to replace with preconditioned version
* @param coeffs		polynomial coefficients c[0] * I + c[1] * AHA
*/
struct linop_s* linop_polyprecond_create(const struct linop_s* sense_linop, 
										const float* coeffs, int D, 
										const long img_dims[DIMS]) {

	PTR_ALLOC(struct linop_s, p);

	p->forward = operator_ref(sense_linop->forward);

	// Get normal of SENSE as linop
	const struct linop_s* sense_normal_linop = linop_get_normal(sense_linop);

	// Get preconditioner operator
	const struct linop_s* precond_linop = polyprecond_sense_normal(sense_normal_linop, coeffs, D, DIMS, img_dims);

	// Replace adjoint as P * AH
	const struct linop_s* adjoint_precond_linop = linop_chain(linop_get_adjoint(sense_linop), precond_linop);
	p->adjoint = operator_ref(adjoint_precond_linop->forward);

	// Replace normal as P * AHA
	const struct linop_s* normal_precond_linop = linop_chain_FF(sense_normal_linop, precond_linop);
	p->normal = operator_ref(normal_precond_linop->forward);
	p->norm_inv = NULL;

	// Clean up
	linop_free(normal_precond_linop);
	linop_free(adjoint_precond_linop);
	linop_free(sense_linop);

	return PTR_PASS(p);

}

/* Returns coeffs array with polynomial coefficients from l_2_opt calculations
*  Important - these coefficients only work when max eigenvalue of A is 1
*  So A must be normalized before preconditioner applied
*/
float* get_polyprecond_coeffs(const int polyprecond_deg) {
	// TODO - replace with function that calculates coeffs given degree and max_eig

	// First coefficient is just scaling, so degree 4 means polynomial is:
	// c[0] + c[1]*x + c[2]*x^2 + c[3]*x^3 (highest power is 3)
	if (polyprecond_deg == 4) {
		static float coeffs[4] = {12.0, -42.0, 56.0, -25.2000007629395};
		return coeffs;
	}
	else if (polyprecond_deg == 6) {
		static float coeffs[6] = {24.0, -180.0, 600.0, -990.0, 792.0, -245.142852783203};
		return coeffs;
	}
	else if (polyprecond_deg == 8) {
		static float coeffs[8] = {39.9999809265137, -513.332824707031, 3079.99609375, 
								-10009.984375, 18685.30078125, -20019.9609375, 
								11439.9755859375, -2701.10498046875};
		return coeffs;
	}
	else {
		return NULL; // unsupported degree
	}
}
// Shreya end
