
#ifndef SURV_H_INCLUDED
#define SURV_H_INCLUDED


# if defined(_MSC_VER)
#  define isn(x) (_isnan(x))
#  define isf(x) (_finite(x))
# else
#  define isn(x) (isnan(x))
#  define isf(x) (isfinite(x))
#endif

#include "corpus.h"
#include "ctm.h"
#include "params.h"
#include "inference.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

extern llna_params PARAMS;

int cox_reg(llna_model* model,	corpus* c, double* f);

/*
double cox_reg_cross_val(
	int group, 
	double * newbeta, 
	double ** var,
	int nvar, 
	double lambda, 
	const suffstats * ss, 
	const settings* setting, 
	int base
	);

int cox_reg_sparse(
	double * beta,
	double * zbeta,
	const corpus *c,
	int nvar,
	double lambda,
	const suffstats * ss,
	double * f,
	const settings* setting);
*/

#endif // OPT_H_INCLUDED

