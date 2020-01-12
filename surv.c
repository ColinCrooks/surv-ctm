
#include "surv.h"
#include <gsl/gsl_blas.h>
#include <float.h>
#include <omp.h>
#include <assert.h>
#include <math.h>
#include <limits.h>


int cox_reg(llna_model* model, corpus* c, double* f, int base_index)
{
	//Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
	int i, k, person, iter;
	int nvar = model->k;
	int nused = c->ndocs, lastvar = 0;
	int ntimes = model->range_t;
	gsl_vector* denom = gsl_vector_calloc(ntimes);
	gsl_vector* a = gsl_vector_calloc(ntimes);
	gsl_vector* cdiag = gsl_vector_calloc(ntimes);
	double  risk = 0.0, temp = 0.0, temp2 = 0.0, loglik = 0.0, newlk = 0.0, d2 = 0.0, efron_wt = 0.0;
	double dif = 0.0, a2 = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	gsl_vector* newbeta = gsl_vector_calloc(nvar);
	gsl_vector* step = gsl_vector_calloc(nvar);
	gsl_vector* zbeta = gsl_vector_calloc(nused);
	gsl_vector_set_zero(zbeta);
	// Find baseline topic with most allocations

	for (i = 0; i < nvar; i++)
	{
		vset(newbeta, i, i == base_index ? 0.0 : vget(model->topic_beta, i));
		vset(step, i, 1.0);
	}

	gsl_blas_dgemv(CblasNoTrans, 1, c->zbar, model->topic_beta, 0, zbeta);


	for (person = nused - 1; person >= 0; person--)
	{
		vset(zbeta, person, vget(zbeta, person) > 22 ? 22 : vget(zbeta, person));
		vset(zbeta, person, vget(zbeta, person) < -200 ? -200 : vget(zbeta, person));
	}
	for (iter = 1; iter <= PARAMS.surv_max_iter; iter++)
	{
		newlk = -(log(sqrt(PARAMS.surv_penalty)) * ((double)nvar - 1));
		for (i = 0; i < nvar; i++)
		{
			if (i == base_index) continue;
			/*
			** The data is sorted from smallest time to largest
			** Start at the largest time, accumulating the risk set 1 by 1
			*/
			gsl_vector_set_zero(denom);
			gsl_vector_set_zero(a);
			gsl_vector_set_zero(cdiag);
			efron_wt = 0.0;

			gdiag = 0.0;
			hdiag = 0.0;
			a2 = 0.0;
			cdiag2 = 0.0;

			for (person = nused - 1; person >= 0; person--)
			{
				int t_enter = c->docs[person].t_enter;
				int t_exit = c->docs[person].t_exit;
				int label = c->docs[person].label;
				int mark = (int) vget(c->cmark, person);
				double xb = vget(zbeta,person) - (dif * mget(c->zbar, person, lastvar));
				xb = xb > 22 ? 22 : xb;
				xb = xb < -200 ? -200 : xb;
				vset(zbeta, person, xb);		
				risk = exp(xb);
				double z = mget(c->zbar, person, i);
				//cumumlative sums for all patients
/*#pragma omp parallel for 
				for(r = c->docs[person].t_enter; r < c->docs[person].t_exit; r++)*/
#pragma omp parallel default(none) shared(c, risk, z, denom, a , cdiag, person, t_enter, t_exit) 
				{
					int size = omp_get_num_threads(); // get total number of processes
					int rank = omp_get_thread_num(); // get rank of current
					int t = t_exit - t_enter;
					for (int r = t_enter + (rank * t / size); r < t_enter + ((rank + 1) * t / size); r++)
					{
						vinc(denom, r, risk);
						vinc(a, r, risk * z);
						vinc(cdiag, r, risk * z * z);
					}
				}
				if (label > 0)
				{
					//cumumlative sums for event patients
					newlk += xb;
					gdiag -= z;

					efron_wt += risk; /* sum(denom) for tied deaths*/
					a2 += risk * z;
					cdiag2 += risk * z * z;
				}
				
				/* once per unique death time */
				for (k = 0; k < mark; k++)
				{
					temp = (double) k
						/ (double) mark;
					d2 = vget(denom, t_exit) - (temp * efron_wt); /* sum(denom) adjusted for tied deaths*/
					newlk -= log(d2);
					temp2 = (vget(a, t_exit) - (temp * a2)) / d2;
					gdiag += temp2;
					hdiag += ((vget(cdiag, t_exit) - (temp * cdiag2)) / d2) -
						(temp2 * temp2);
				}
				efron_wt = 0.0;
				a2 = 0.0;
				cdiag2 = 0.0;
				
			}   /* end  of accumulation loop  */

			dif = (gdiag + (vget(newbeta, i) / PARAMS.surv_penalty)) / (hdiag + (1.0 / PARAMS.surv_penalty));
			if (fabs(dif) > vget(step, i))
				dif = (dif > 0.0) ? vget(step, i) : -vget(step, i);

			vset(step, i, ((2.0 * fabs(dif)) > (vget(step, i) / 2.0)) ? 2.0 * fabs(dif) : (vget(step, i) / 2.0)); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			vinc(newbeta, i, -dif);
			lastvar = i;
		}

		for (i = 0; i < nvar; i++)
		{
			if (i == base_index) continue;
			newlk -= (vget(newbeta, i) * vget(newbeta, i)) / (2.0 * PARAMS.surv_penalty);
		}
		if (fabs(1.0 - (newlk / loglik)) <= model->surv_convergence) break;
		loglik = newlk;
	}   /* return for another iteration */

	*f = loglik;
	for (i = 0; i < nvar; i++)
		vset(model->topic_beta, i, vget(newbeta, i) / nused);


	gsl_vector_free(newbeta);
	gsl_vector_free(step);
	gsl_vector_free(denom);
	gsl_vector_free(a);
	gsl_vector_free(cdiag);
	gsl_vector_free(zbeta);
	return iter;
}

//zbar cumulative sum only needed once as doesn't  change:

void cox_reg_distr_init(llna_model* model, gsl_matrix* sum_zbar_events_cumulative, corpus* c, int size, int rank)
{
	int nused = c->ndocs;
	int nvar = model->k;
	int ntimes = model->range_t;
	gsl_matrix_set_zero(sum_zbar_events_cumulative);

	for (int person = rank * c->ndocs / size; person < (rank + 1) * c->ndocs / size; person++)
	{
		int t_enter = c->docs[person].t_enter;
		int t_exit = c->docs[person].t_exit;
		int label = c->docs[person].label;
		if (label > 0)
		{
			gsl_vector_view personzbar = gsl_matrix_row(c->zbar, person);
			//for (r = t_enter; r <= t_exit; r++)
			//{
				gsl_vector_view timeszbar = gsl_matrix_row(sum_zbar_events_cumulative, t_exit);
				gsl_vector_add(&timeszbar.vector, &personzbar.vector);
			//}
		}
	}

	return;
}

void cox_reg_accumulation(llna_model* model, corpus* c, int size, int rank, int bn, 
	gsl_vector* beta, gsl_vector* cumulrisk, gsl_vector* cumulgdiag, gsl_vector* cumulhdiag, 
	gsl_vector* cumul2risk, gsl_vector* cumulg2diag, gsl_vector* cumulh2diag)
{
	int nvar = model->k;
	int nused = c->ndocs;
	int ntimes = model->range_t;
	int start = rank * c->ndocs / size;
	int nrows = (c->ndocs / size) - 1;
	gsl_matrix_view personrisk = gsl_matrix_submatrix(c->zbar, start, 0, nrows, nvar);
	gsl_vector_view zb = gsl_vector_subvector(c->zbeta, start, nrows);
	gsl_blas_dgemv(CblasNoTrans, 1.0, &personrisk.matrix, beta, 0.0, &zb.vector);

	//gsl_vector* z = gsl_vector_calloc(nvar);
	for (int person = rank * c->ndocs / size; person < (rank + 1) * c->ndocs / size; person++)
	{
		int t_enter = c->docs[person].t_enter;
		int t_exit = c->docs[person].t_exit;

		gsl_vector_view zbar = gsl_matrix_row(c->zbar, person);
		double expxb = exp(vget(c->zbeta, person));
		double z = expxb * vget(&zbar.vector, bn);
		double zz = z *  vget(&zbar.vector, bn);		
		//printf("xb %f expxb %f z %f zz %f\n", xb, expxb, z, zz);
		//vprint(beta);
		//printf("\n");
		//assert(!isnan(xb));
		gsl_vector_view timeupdate = gsl_vector_subvector(cumulrisk, t_enter, t_exit - t_enter + 1);
		gsl_vector_view timeupdate2 = gsl_vector_subvector(cumulgdiag, t_enter, t_exit - t_enter + 1);
		gsl_vector_view timeupdate3 = gsl_vector_subvector(cumulhdiag, t_enter, t_exit - t_enter + 1);
		gsl_vector_add_constant(&timeupdate.vector, expxb);
		gsl_vector_add_constant(&timeupdate2.vector, z);
		gsl_vector_add_constant(&timeupdate3.vector, zz);

		if (c->docs[person].label > 0)
		{
			//cumumlative sums for event patients
			vinc(cumul2risk, t_exit, expxb);
			vinc(cumulg2diag, t_exit, z);
			vinc(cumulh2diag, t_exit, zz);
		}
	}
}


// using the suggested accumulations from 	//Lu CL, Wang S, Ji Z, et al. WebDISCO: a web service for distributed cox model learning without patient-level data sharing. J Am Med Inform Assoc. 2015;22(6):1212–1219. doi:10.1093/jamia/ocv083v
// and the penalised model with cyclical descent algorithm derived from Mittal, S. Biostatistics 15, 207–221 (2014). 

int cox_reg_dist(llna_model* model, corpus* c, double* f, int base_index)
{
	int iter, nvar = model->k;
	int nused = c->ndocs;
	int ntimes = model->range_t;
	gsl_matrix* sum_zbar = gsl_matrix_calloc(ntimes, nvar);
	gsl_matrix_set_zero(sum_zbar);
#pragma omp parallel default(none) shared(c, model, sum_zbar, ntimes, nvar) /* for (i = 0; i < corpus->ndocs; i++) */
	{
		int size = omp_get_num_threads(); // get total number of processes
		int rank = omp_get_thread_num(); // get rank of current
		gsl_matrix* sum_zbar_private = gsl_matrix_calloc(ntimes, nvar);
		cox_reg_distr_init(model, sum_zbar_private, c, size, rank);
#pragma omp critical
		gsl_matrix_add(sum_zbar, sum_zbar_private);
		gsl_matrix_free(sum_zbar_private);
	}
	gsl_vector* beta = gsl_vector_calloc(nvar);
	gsl_matrix* cumulrisk = gsl_vector_calloc(ntimes);
	gsl_vector* cumulgdiag = gsl_vector_calloc(ntimes);
	gsl_vector* cumulhdiag = gsl_vector_calloc(ntimes);
	gsl_matrix* cumul2risk = gsl_vector_calloc(ntimes);
	gsl_vector* cumulg2diag = gsl_vector_calloc(ntimes);
	gsl_vector* cumulh2diag = gsl_vector_calloc(ntimes);
	gsl_vector* step = gsl_vector_calloc(nvar);
	gsl_vector_set_all(step, 1.0);

	gsl_vector_memcpy(beta, model->topic_beta);
	double loglik = 0.0;
	for (iter = 1; iter <= PARAMS.surv_max_iter; iter++)
	{
		double newlk = 0.0;
		//gsl_vector_set_zero(dl);
		//gsl_vector_set_zero(dll);
		for (int bn = 0; bn < nvar; bn++)
		{	
			//if (bn == base_index) continue;
			double dl = 0.0;
			double dll = 0.0;
			gsl_vector_set_zero(cumulrisk);
			gsl_vector_set_zero(cumulgdiag);
			gsl_vector_set_zero(cumulhdiag);
			gsl_vector_set_zero(cumul2risk);
			gsl_vector_set_zero(cumulg2diag);
			gsl_vector_set_zero(cumulh2diag);
#pragma omp parallel reduction(+:newlk) default(none) shared(c, model, ntimes, nvar, cumulrisk, cumulgdiag, cumulhdiag, cumul2risk, cumulg2diag, cumulh2diag, sum_zbar, dl , dll, bn) firstprivate(beta) /* for (i = 0; i < corpus->ndocs; i++) */
			{
				int size = omp_get_num_threads(); // get total number of processes
				int rank = omp_get_thread_num(); // get rank of current
				gsl_vector* cumulrisk_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulgdiag_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulhdiag_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumul2risk_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulg2diag_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulh2diag_private = gsl_vector_calloc(ntimes);
				gsl_vector_set_zero(cumulrisk_private);
				gsl_vector_set_zero(cumulgdiag_private);
				gsl_vector_set_zero(cumulhdiag_private);
				gsl_vector_set_zero(cumul2risk_private);
				gsl_vector_set_zero(cumulg2diag_private);
				gsl_vector_set_zero(cumulh2diag_private);

				cox_reg_accumulation(model, c, size, rank, bn, beta, cumulrisk_private, cumulgdiag_private, cumulhdiag_private, 
					cumul2risk_private, cumulg2diag_private, cumulh2diag_private);

#pragma omp critical
				{
					gsl_vector_add(cumulrisk, cumulrisk_private);
					gsl_vector_add(cumulgdiag, cumulgdiag_private);
					gsl_vector_add(cumulhdiag, cumulhdiag_private);
					gsl_vector_add(cumul2risk, cumul2risk_private);
					gsl_vector_add(cumulg2diag, cumulg2diag_private);
					gsl_vector_add(cumulh2diag, cumulh2diag_private);
				}
				gsl_vector_free(cumulrisk_private);
				gsl_vector_free(cumulgdiag_private);
				gsl_vector_free(cumulhdiag_private);
				gsl_vector_free(cumul2risk_private);
				gsl_vector_free(cumulg2diag_private);
				gsl_vector_free(cumulh2diag_private);
			}

			int range_t = model->range_t;

#pragma omp parallel reduction(+:dl,dll, newlk) default(none) shared(c, beta, sum_zbar, range_t, bn, cumulrisk, cumulgdiag, cumulhdiag, cumul2risk, cumulg2diag, cumulh2diag, ntimes)
			{
				int size = omp_get_num_threads(); // get total number of processes
				int rank = omp_get_thread_num(); // get rank of current
				for (int r = rank * range_t / size; r < (rank + 1) * range_t / size; r++)
				{
					dl -= mget(sum_zbar, r, bn);
					newlk += safe_log(vget(cumul2risk, r));
					for (int k = 0; k < vget(c->cmark, r); k++)
					{
						double temp = (double)k
							/ (double)vget(c->cmark, r);
						double denom = vget(cumulrisk, r) - (temp * vget(cumul2risk, r)); /* sum(denom) adjusted for tied deaths*/
						newlk -= log(denom);
						double temp2 = (vget(cumulgdiag, r) - (temp * vget(cumulg2diag, r))) / denom;
						dl += temp2;
						dll += (vget(cumulhdiag, r) - (temp * vget(cumulh2diag, r)) / denom) -
							(temp2 * temp2);
					}
				}
			}
			//printf("dl %f\t",dl);
			//printf("dll %f\n",dll);

			double dif = (dl + (vget(beta, bn)/PARAMS.surv_penalty)) / (dll + (1.0 / PARAMS.surv_penalty));
			if (fabs(dif) > vget(step, bn))
				dif = dif > 0.0 ? vget(step, bn) : -vget(step, bn);
			vset(step, bn, ((2.0 * fabs(dif)) > (vget(step, bn) / 2.0)) ? 2.0 * fabs(dif) : vget(step, bn) / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			vinc(beta, bn, -dif);		
			if (isn(vget(beta,bn)))
				return INT_MIN;

#pragma omp parallel default(none) shared(c, bn, nused, dif)  /* for (i = 0; i < corpus->ndocs; i++) */
			{
				int size = omp_get_num_threads(); // get total number of processes
				int rank = omp_get_thread_num(); // get rank of current
				for (int person = rank * c->ndocs / size; person < (rank + 1) * c->ndocs / size; person++)
				{
					vinc(c->zbeta, person, -dif * mget(c->zbar, person, bn));
					double xb = vget(c->zbeta, person);
					xb = xb > 22 ? 22 : xb;
					xb = xb < -200 ? -200 : xb;
					vset(c->zbeta, person, xb);
				}
			}

		}
		newlk = -(log(sqrt(PARAMS.surv_penalty)) * ((double) nvar - 1.0));
		for (int bn = 0; bn < nvar; bn++)
		{
			if (bn == base_index) continue;
			newlk -= (vget(beta, bn) * vget(beta, bn)) / (2.0 * PARAMS.surv_penalty);
		}	
		if (isn(newlk))
			return INT_MIN;
		if (iter > 0 && fabs(1.0 - (newlk / loglik)) <= model->surv_convergence) break;
		loglik = newlk;
		//vprint(beta);
	}



	*f = loglik;

	gsl_vector_memcpy(model->topic_beta, beta);
	gsl_matrix_free(sum_zbar);
	gsl_vector_free(cumulrisk);
	gsl_vector_free(cumulgdiag);
	gsl_vector_free(cumulhdiag);
	gsl_vector_free(cumul2risk);
	gsl_vector_free(cumulg2diag);
	gsl_vector_free(cumulh2diag);
	gsl_vector_free(beta);
	return(iter);
}


int cox_reg_dac(llna_model* model, corpus* c, double* f, int group, int base_index, gsl_vector* beta)
{
	//Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
	int i,k, groupperson, person, iter, r, z;
	int nvar = model->k;
	int nused = (int) c->group->size1, lastvar = 0;
	int ntimes = model->range_t;
	double  risk = 0.0, temp = 0.0, temp2 = 0.0, loglik = 0.0, newlk = 0.0, d2 = 0.0, efron_wt = 0.0;
	double dif = 0.0, a2 = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;

	gsl_vector* denom = gsl_vector_calloc(ntimes);
	gsl_vector* a = gsl_vector_calloc(ntimes);
	gsl_vector* cdiag = gsl_vector_calloc(ntimes);
	gsl_vector* step = gsl_vector_calloc(nvar);
	gsl_vector* zbeta = gsl_vector_calloc(nused);
	gsl_vector_set_zero(zbeta);
	gsl_vector_set_all(step, 1.0);
	// Find baseline topic with most allocations

	efron_wt = 0.0;

	gdiag = 0.0;
	hdiag = 0.0;
	a2 = 0.0;
	cdiag2 = 0.0;

	for (person=nused-1; person >=0; person--)
	{
		
		groupperson = (int) mget(c->group, person, group);
		if (groupperson == -1)
		{
			nused--;
			continue;
		}
		int t_enter = c->docs[person].t_enter;
		int t_exit = c->docs[person].t_exit;
		int label = c->docs[person].label;
		int mark = (int) mget(c->mark, person, group);

		z = 0;
		for (i = 0; i < nvar; i++)
		{
			if (i == base_index) continue;
			z+= mget(c->zbar, groupperson, i) * vget(beta,i);
		}
		z = z > 22 ? 22 : z;
		z = z < -200 ? -200 : z;

		vset(zbeta, person, z);
	}
	for (iter=1; iter<=PARAMS.surv_max_iter;  iter++)
	{
		newlk = -(log(sqrt(PARAMS.surv_penalty)) * ((double) nvar-1));
		for (i = 0; i < nvar; i++)
		{
			if (i == base_index) continue;
			/*
			** The data is sorted from smallest time to largest
			** Start at the largest time, accumulating the risk set 1 by 1
			*/
			gsl_vector_set_zero(denom);
			gsl_vector_set_zero(a);
			gsl_vector_set_zero(cdiag);

			efron_wt = 0.0;
			
			gdiag = 0.0;
			hdiag = 0.0;
			a2 = 0.0;
			cdiag2 = 0.0;

			for (person = nused - 1; person >= 0; person--)
			{
				groupperson = (int) mget(c->group, person, group);
				if (groupperson == -1) continue;

				// calculate or access things once per loop
				int t_enter = c->docs[person].t_enter;
				int t_exit = c->docs[person].t_exit;
				int label = c->docs[person].label;
				int mark = (int) mget(c->mark, person, group);
				double xb = vget(zbeta, person) - (dif * mget(c->zbar, groupperson, lastvar));
				xb = xb > 22 ? 22 : xb;
				xb = xb < -200 ? -200 : xb;
				vset(zbeta, person, xb);
				risk = exp(xb);
				double z = mget(c->zbar, groupperson, i);
				int t = t_exit - t_enter;

				//cumumlative sums for all patients
				for (r = t_enter; r <= t_exit; r++)
				{
					vinc(denom, r, risk);
					vinc(a, r, risk * z);
					vinc(cdiag, r,  risk * z * z);
				}
				if (label > 0)
				{
					//cumumlative sums for event patients
					newlk += z;
					gdiag -= xb;

					efron_wt += risk; /* sum(denom) for tied deaths*/
					a2 += risk * z;
					cdiag2 += risk * z * z;
				}
				/* once per unique death time */
				for (k = 0; k < mark; k++)
				{
					temp = (double) k
						/ (double) mark;
					d2 = vget(denom, t_exit) - (temp * efron_wt); /* sum(denom) adjusted for tied deaths*/
					newlk -= log(d2);
					temp2 = (vget(a, t_exit) - (temp * a2)) / d2;
					gdiag += temp2;
					hdiag += ((vget(cdiag, t_exit) - (temp * cdiag2)) / d2) -
						(temp2 * temp2);
				}
				efron_wt = 0.0;
				a2 = 0.0;
				cdiag2 = 0.0;
				
			}   /* end  of accumulation loop  */

			dif = (gdiag + (vget(beta, i) / PARAMS.surv_penalty)) / (hdiag + (1.0 / PARAMS.surv_penalty));
			if (fabs(dif) > vget(step, i))	
				dif = (dif > 0.0) ? vget(step, i) : -vget(step, i);

			vset(step, i, ((2.0 * fabs(dif)) > (vget(step, i) / 2.0)) ? 2.0 * fabs(dif) : (vget(step, i) / 2.0)); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			vinc(beta, i, -dif);
			lastvar = i;
		}
		
		for (i = 0; i < nvar; i++)
		{
			if (i == base_index) continue;
			newlk -= (vget(beta, i) * vget(beta, i)) / (2.0 * PARAMS.surv_penalty);
		}
		if (fabs(1.0 - (newlk / loglik)) <= model->surv_convergence) break;
		loglik = newlk;
	}   /* return for another iteration */

	* f = loglik;
	gsl_vector_free(step);
	gsl_vector_free(denom);
	gsl_vector_free(a);
	gsl_vector_free(cdiag);
	gsl_vector_free(zbeta);
	return iter;
}

