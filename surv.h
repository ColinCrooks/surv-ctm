
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

int cox_reg(llna_model* model, corpus* c, double* f);
int cox_reg_dist(llna_model* model, corpus* c, double* f);
void cox_reg_accumulation(llna_model* model, corpus* c, int size, int rank, int bn, int lastvar, double dif,
	gsl_vector* beta, gsl_vector* xb, gsl_vector* cumulrisk, gsl_vector* cumulgdiag, gsl_vector* cumulhdiag,
	gsl_vector* cumul2risk, gsl_vector* cumulg2diag, gsl_vector* cumulh2diag);
void cox_reg_distr_init(llna_model* model, gsl_matrix* sum_zbar_events_cumulative, corpus* c, int size, int rank);
int cox_reg_hes(llna_model* model, corpus* c, double* f);
void cox_reg_accumul_hes(llna_model* model, corpus* c, int size, int rank,
	gsl_vector* beta, gsl_vector* cumulxb, gsl_vector* cumulrisk, gsl_matrix* cumulgdiag, gsl_matrix** cumulhdiag,
	gsl_vector* cumul2risk, gsl_matrix* cumulg2diag, gsl_matrix** cumulh2diag);
void cox_reg_hes_init(llna_model* model, gsl_matrix* sum_zbar_events_cumulative, corpus* c, int size, int rank);


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

