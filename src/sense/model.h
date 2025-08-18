
#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"

struct linop_s;

extern struct linop_s* linop_sampling_create(const long dims[DIMS], const long pat_dims[DIMS], const _Complex float* pattern);

extern struct linop_s* sense_init(unsigned long shared_img_flags, const long max_dims[DIMS], unsigned long sens_flags, const _Complex float* sens);
extern struct linop_s* maps_create(unsigned long shared_img_flags, const long max_dims[DIMS], 
			unsigned long sens_flags, const _Complex float* sens);
extern struct linop_s* maps2_create(const long coilim_dims[DIMS], const long maps_dims[DIMS], const long img_dims[DIMS], const _Complex float* maps);

extern struct linop_s* polyprecond_sense_normal(const struct linop_s* sense_op_normal, 
												const float* coeffs, 
												const int D, 
												const int N, 
												const long *img_dims);
extern struct linop_s* linop_polyprecond_create(const struct linop_s* sense_op, 
												const float* coeffs, int D,
												const long img_dims[DIMS]); 
float* get_polyprecond_coeffs(const int polyprecond_deg);

#ifdef __cplusplus
}
#endif


