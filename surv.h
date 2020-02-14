
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
	gsl_vector* beta, gsl_vector* cumulxb,
	gsl_vector* cumulrisk, gsl_matrix* cumulgdiag, gsl_matrix** cumulhdiag, gsl_vector* atemp, gsl_matrix* ctemp);
void cox_reg_accumul_hessian_atomic(llna_model* model, corpus* c, int size, int rank,
	gsl_vector* beta, gsl_vector* cumulxb,
	gsl_vector* cumulrisk, gsl_matrix* cumulgdiag, gsl_matrix** cumulhdiag);
int cox_reg_fullefron(llna_model* model, corpus* c, double* f);

void cox_reg_accumul_fullefron(llna_model* model, corpus* c, int size, int rank,
	gsl_vector* beta, 	gsl_vector* cumulrisk, gsl_matrix* cumulgdiag, gsl_matrix** cumulhdiag,
	gsl_vector* cumul2risk, gsl_matrix* cumul2gdiag, gsl_matrix** cumul2hdiag,
	gsl_vector* atemp, gsl_matrix* ctemp);

int cox_reg_fullefron(llna_model* model, corpus* c, double* f);

void cox_reg_hes_init_accumul(llna_model* model, gsl_matrix* sum_zbar_events_cumulative, corpus* c, int size, int rank);
//void cox_reg_hes_intitialise(int nvar, int ntimes, llna_model* model, gsl_matrix* sum_zbar, gsl_vector* gdiag, gsl_matrix* hdiag, gsl_vector* mean_z,
//	gsl_vector* scale_z, gsl_vector* beta, gsl_vector* newbeta, gsl_vector* cumulxb, /*gsl_vector* cumul2risk, gsl_matrix* cumulg2diag,
//	gsl_matrix** cumulh2diag,*/ gsl_vector* cumulrisk_start, gsl_matrix* cumulgdiag_start, gsl_matrix** cumulhdiag_start, gsl_vector* cumulrisk_end,
//	gsl_matrix* cumulgdiag_end, gsl_matrix** cumulhdiag_end, gsl_vector* running_gdiag, gsl_matrix* running_hdiag, gsl_vector* htemp, gsl_vector* atemp,
//	gsl_vector** cumulxb_private, /* gsl_vector** cumul2risk_private, gsl_matrix*** cumulh2diag_private, gsl_matrix** cumulg2diag_private,*/ gsl_vector** cumulrisk_start_private,
//	gsl_matrix** cumulgdiag_start_private, gsl_matrix*** cumulhdiag_start_private, gsl_vector** cumulrisk_end_private, gsl_matrix** cumulgdiag_end_private, gsl_matrix*** cumulhdiag_end_private);
//void cox_reg_hes_free(int nvar, int ntimes, llna_model* model, gsl_matrix* sum_zbar, gsl_vector* gdiag, gsl_matrix* hdiag, gsl_vector* mean_z,
//	gsl_vector* scale_z, gsl_vector* beta, gsl_vector* newbeta, gsl_vector* cumulxb/*, gsl_vector* cumul2risk, gsl_matrix* cumulg2diag,
//	gsl_matrix** cumulh2diag*/, gsl_vector* cumulrisk_start, gsl_matrix* cumulgdiag_start, gsl_matrix** cumulhdiag_start, gsl_vector* cumulrisk_end,
//	gsl_matrix* cumulgdiag_end, gsl_matrix** cumulhdiag_end, gsl_vector* running_gdiag, gsl_matrix* running_hdiag, gsl_vector* htemp, gsl_vector* atemp,
//	gsl_vector** cumulxb_private,/* gsl_vector** cumul2risk_private, gsl_matrix*** cumulh2diag_private, gsl_matrix** cumulg2diag_private,*/ gsl_vector** cumulrisk_start_private,
//	gsl_matrix** cumulgdiag_start_private, gsl_matrix*** cumulhdiag_start_private, gsl_vector** cumulrisk_end_private, gsl_matrix** cumulgdiag_end_private, gsl_matrix*** cumulhdiag_end_private);

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

