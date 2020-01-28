
#include "surv.h"
#include <gsl/gsl_blas.h>
#include <float.h>
#include <omp.h>
#include <assert.h>
#include <mathimf.h>
#include <limits.h>


//Sequential Cox model 
int cox_reg(llna_model* model, corpus* c, double* f)
{
	//Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
	int i, k, person, iter;
	int nvar = model->k - 1;
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
		//vset(newbeta, i, i == base_index ? 0.0 : vget(model->topic_beta, i));
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
		newlk = 0.0;
		for (i = 0; i < nvar; i++)
		{
			//if (i == base_index) continue;
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
			double sumz = 0.0;
			double sumxb = 0.0;
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
/*#pragma omp parallel for defaulte(none) shared(t_enter, t_exit, denom, a, cdiag, z, risk) */ 
				for(int r = t_enter; r <= t_exit; r++)
				{
						vinc(denom, r, risk);
						vinc(a, r, risk * z);
						vinc(cdiag, r, risk * z * z);
				}
				if (label > 0)
				{
					//cumumlative sums for event patients
					newlk += z * vget(newbeta, i);
					gdiag -= z;

				//	sumz += z;
				//	sumxb += z * vget(newbeta, i);
					efron_wt += risk; /* sum(denom) for tied deaths*/
					a2 += risk * z;
					cdiag2 += risk * z * z;
			//		printf(" Seq: %f ", sumz);
				}
			//	printf("\n person %d\t label %d\t", person, label);
				/* once per unique death time */
				if (mark > 0)
				{
					for (k = 0; k < mark; k++)
					{
						temp = (double)k
							/ (double)mark;
						d2 = vget(denom, t_exit) - (temp * efron_wt); /* sum(denom) adjusted for tied deaths*/
						newlk -= log(d2);
						temp2 = (vget(a, t_exit) - (temp * a2)) / d2;
						gdiag += temp2;
						hdiag += ((vget(cdiag, t_exit) - (temp * cdiag2)) / d2) -
							(temp2 * temp2);
					}

		//			if(person==0 && iter==1) printf("Seq beta %d (%f)\t time %d\t sumzbar %f \t cumulxb %f\t cumulrisk %f\t cumulgdiag %f \t cumulhdiag %f \t cumul2risk %f\t cumul2gdiag %f \t cumul2hdiag %f\n ", i, vget(newbeta, i), ntimes, sumz, sumxb,vget(denom,ntimes-1), vget(a,ntimes-1), vget(cdiag,ntimes-1), efron_wt, a2,cdiag2);
					efron_wt = 0.0;
					a2 = 0.0;
					cdiag2 = 0.0;
					sumz = 0;
					sumxb = 0;
				}
			}   /* end  of accumulation loop  */
			newlk = -(log(sqrt(PARAMS.surv_penalty)) * ((double)nvar - 1));

			//printf("Sequential dl %f\t", gdiag);
			//printf("dll %f\n", hdiag);

			dif = (gdiag + (vget(newbeta, i) / PARAMS.surv_penalty)) / (hdiag + (1.0 / PARAMS.surv_penalty));
			if (fabs(dif) > vget(step, i))
				dif = (dif > 0.0) ? vget(step, i) : -vget(step, i);

			vset(step, i, ((2.0 * fabs(dif)) > (vget(step, i) / 2.0)) ? 2.0 * fabs(dif) : (vget(step, i) / 2.0)); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			vinc(newbeta, i, -dif);
			lastvar = i;
		}


		for (i = 0; i < nvar; i++)
		{
			//if (i == base_index) continue;
			newlk -= (vget(newbeta, i) * vget(newbeta, i)) / (2.0 * PARAMS.surv_penalty);
		}
		if (fabs(1.0 - (newlk / loglik)) <= model->surv_convergence) break;
		loglik = newlk;
	}   /* return for another iteration */

	*f = loglik;
	for (i = 0; i < nvar; i++)
		vset(model->topic_beta, i, vget(newbeta, i));


	gsl_vector_free(newbeta);
	gsl_vector_free(step);
	gsl_vector_free(denom);
	gsl_vector_free(a);
	gsl_vector_free(cdiag);
	gsl_vector_free(zbeta);
	return iter;
}

//Parallel Cox model accumulation

//zbar cumulative sum only needed once as doesn't  change:

void cox_reg_distr_init(llna_model* model, gsl_matrix* sum_zbar_events_cumulative, corpus* c, int size, int rank)
{
	unsigned int nused = c->ndocs;
	for (unsigned int person = rank * nused / size; person < (rank + 1) * nused / size; person++)
	{
		unsigned int t_exit = c->docs[person].t_exit;
		int label = c->docs[person].label;
		gsl_vector_view personrisk = gsl_matrix_row(c->zbar, person);
		double zb = 0.0;
		gsl_blas_ddot(&personrisk.vector, model->topic_beta, &zb);
		vset(c->zbeta, person, zb);
	//	printf("Person %d label %d mark %f exit %d\n", person, label, vget(c->cmark,person), t_exit);
		if (label > 0)
		{
		//	gsl_vector_view personzbar = gsl_matrix_subrow(c->zbar, person, 0, nvar);
			gsl_vector_view timeszbar = gsl_matrix_row(sum_zbar_events_cumulative, t_exit);
			gsl_blas_daxpy(1.0, &personrisk.vector, &timeszbar.vector);
		//	printf("par ");
		//	vprint(&timeszbar.vector);
		//	printf("\n");
		}
	}

	return;
}


void cox_reg_accumulation(llna_model* model, corpus* c, int size, int rank, int bn, int lastvar, double dif,
	gsl_vector* beta, gsl_vector* cumulxb, gsl_vector* cumulrisk, gsl_vector* cumulgdiag, gsl_vector* cumulhdiag,
	gsl_vector* cumul2risk, gsl_vector* cumulg2diag, gsl_vector* cumulh2diag)
{
	int nused = c->ndocs;
	int nvar = model->k - 1;
	double b = vget(beta, bn);
	
	for (int person = rank * nused / size ; person < (rank + 1) * nused / size; person++ )
	{
		unsigned int t_enter = (unsigned int) c->docs[person].t_enter;
		unsigned int t_exit = (unsigned int) c->docs[person].t_exit;

		double xb = vget(c->zbeta, person) - (dif * mget(c->zbar, person, lastvar));
		xb = xb > 22 ? 22 : xb;
		xb = xb < -200 ? -200 : xb;
		vset(c->zbeta, person, xb);

		double expxb = exp(xb);
		double zbar = mget(c->zbar, person, bn);
		double z = expxb * zbar;
		double zz = z * zbar;		
		//printf("xb %f expxb %f z %f zz %f\n", xb, expxb, z, zz);
		//vprint(beta);
		//printf("\n");
		//assert(!isnan(xb));
		gsl_vector_view timeupdate1 = gsl_vector_subvector(cumulrisk, t_enter, ((size_t)t_exit - (size_t)t_enter) +  1);
		gsl_vector_view timeupdate2 = gsl_vector_subvector(cumulgdiag, t_enter, ((size_t)t_exit - (size_t)t_enter) +  1);
		gsl_vector_view timeupdate3 = gsl_vector_subvector(cumulhdiag, t_enter, ((size_t)t_exit - (size_t)t_enter) +  1);
		gsl_vector_add_constant(&timeupdate1.vector, expxb);
		gsl_vector_add_constant(&timeupdate2.vector, z);
		gsl_vector_add_constant(&timeupdate3.vector, zz);

		if (c->docs[person].label > 0)
		{
			//cumumlative sums for event patients
			vinc(cumulxb, t_exit, zbar * b);
			vinc(cumul2risk, t_exit, expxb);
			vinc(cumulg2diag, t_exit, z);
			vinc(cumulh2diag, t_exit, zz);
		}
	
	}
}


// using the suggested accumulations from 	//Lu CL, Wang S, Ji Z, et al. WebDISCO: a web service for distributed cox model learning without patient-level data sharing. J Am Med Inform Assoc. 2015;22(6):1212–1219. doi:10.1093/jamia/ocv083v
// and the penalised model with cyclical descent algorithm derived from Mittal, S. Biostatistics 15, 207–221 (2014). 

int cox_reg_dist(llna_model* model, corpus* c, double* f)
{
	const double PI = 3.141592653589793238463;
	int iter = 0, nvar = model->k - 1, ntimes = model->range_t;
	gsl_matrix* sum_zbar = gsl_matrix_calloc(ntimes, model->k);
#pragma omp parallel default(none) shared(c, model, sum_zbar, ntimes, nvar) /* for (i = 0; i < corpus->ndocs; i++) */
	{
		int size = omp_get_num_threads(); // get total number of processes
		int rank = omp_get_thread_num(); // get rank of current
		gsl_matrix* sum_zbar_private = gsl_matrix_calloc(ntimes, model->k);
		cox_reg_distr_init(model, sum_zbar_private, c, size, rank);
#pragma omp critical
		{
			gsl_matrix_add(sum_zbar, sum_zbar_private);
		}
		gsl_matrix_free(sum_zbar_private);
	}
	gsl_vector* beta = gsl_vector_calloc(nvar);
	gsl_vector* cumulxb = gsl_vector_calloc(ntimes);
	gsl_vector* cumulrisk = gsl_vector_calloc(ntimes);
	gsl_vector* cumulgdiag = gsl_vector_calloc(ntimes);
	gsl_vector* cumulhdiag = gsl_vector_calloc(ntimes);
	gsl_vector* cumul2risk = gsl_vector_calloc(ntimes);
	gsl_vector* cumulg2diag = gsl_vector_calloc(ntimes);
	gsl_vector* cumulh2diag = gsl_vector_calloc(ntimes);
	gsl_vector* step = gsl_vector_alloc(nvar);
	gsl_vector_set_all(step, 1.0);
	gsl_vector_view topic_beta = gsl_vector_subvector(model->topic_beta, 0, nvar);
	gsl_blas_dcopy(&topic_beta.vector, beta);
	double loglik = 0.0;
	double dif = 0.0;
	int lastvar = 0;
	for (iter = 1; iter <= PARAMS.surv_max_iter; iter++)
	{
		double newlk = 0.0;
		//gsl_vector_set_zero(dl);
		//gsl_vector_set_zero(dll);
		for (int bn = 0; bn < nvar; bn++)
		{	
			//if (bn == base_index) continue;

			gsl_vector_set_zero(cumulxb);
			gsl_vector_set_zero(cumulrisk);
			gsl_vector_set_zero(cumulgdiag);
			gsl_vector_set_zero(cumulhdiag);
			gsl_vector_set_zero(cumul2risk);
			gsl_vector_set_zero(cumulg2diag);
			gsl_vector_set_zero(cumulh2diag);
#pragma omp parallel  default(none) shared(c, model, ntimes, nvar, cumulxb,  cumulrisk, cumulgdiag, cumulhdiag, cumul2risk, cumulg2diag, cumulh2diag, bn, lastvar, dif, beta)  /* for (i = 0; i < corpus->ndocs; i++) */
			{
				int size = omp_get_num_threads(); // get total number of processes
				int rank = omp_get_thread_num(); // get rank of current
				gsl_vector* cumulxb_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulrisk_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulgdiag_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulhdiag_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumul2risk_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulg2diag_private = gsl_vector_calloc(ntimes);
				gsl_vector* cumulh2diag_private = gsl_vector_calloc(ntimes);
				cox_reg_accumulation(model, c, size, rank, bn, lastvar, dif, beta, cumulxb_private, cumulrisk_private, cumulgdiag_private, cumulhdiag_private,
					cumul2risk_private, cumulg2diag_private, cumulh2diag_private);

#pragma omp critical
				{
					gsl_blas_daxpy(1.0, cumulxb_private, cumulxb);
					gsl_blas_daxpy(1.0,  cumulrisk_private, cumulrisk);
					gsl_blas_daxpy(1.0, cumulgdiag_private, cumulgdiag);
					gsl_blas_daxpy(1.0,  cumulhdiag_private, cumulhdiag);
					gsl_blas_daxpy(1.0, cumul2risk_private, cumul2risk);
					gsl_blas_daxpy(1.0, cumulg2diag_private, cumulg2diag);
					gsl_blas_daxpy(1.0,  cumulh2diag_private, cumulh2diag);
				
				}
				gsl_vector_free(cumulxb_private);
				gsl_vector_free(cumulrisk_private);
				gsl_vector_free(cumulgdiag_private);
				gsl_vector_free(cumulhdiag_private);
				gsl_vector_free(cumul2risk_private);
				gsl_vector_free(cumulg2diag_private);
				gsl_vector_free(cumulh2diag_private);
			}

		//	if (iter==1) printf("Parrallel beta %d (%f) \t time %d\t sumzbar %f \t cumulxb %f\t cumulrisk %f\t cumulgdiag %f \t cumulhdiag %f \t cumul2risk %f\t cumul2gdiag %f \t cumul2hdiag %f \n",	bn, vget(beta,bn) , ntimes, mget(sum_zbar,1,bn), vget(cumulxb,1), vget(cumulrisk,ntimes-1), vget(cumulgdiag, ntimes-1), vget(cumulhdiag, ntimes-1), vget(cumul2risk, 1), vget(cumulg2diag, 1), vget(cumulh2diag,1));

			int range_t = model->range_t;
			double dl = 0.0;
			double dll = 0.0;
			double lk = 0.0;
#pragma omp parallel reduction(+:dl, dll, lk) default(none) shared(c, beta, sum_zbar, range_t, bn, cumulxb, cumulrisk, cumulgdiag, cumulhdiag, cumul2risk, cumulg2diag, cumulh2diag, ntimes)
			{
				int size = omp_get_num_threads(); // get total number of processes
				int rank = omp_get_thread_num(); // get rank of current
				for (int r = rank * range_t / size; r < (rank + 1) * range_t / size; r++)
				{
					double mark = vget(c->mark, r);
					dl -= mget(sum_zbar, r, bn);
					lk += vget(cumulxb, r);

				/*	lk -= (double)mark* vget(cumulrisk, r);
					double temp2 = (double)mark * vget(cumulgdiag, r) / vget(cumulrisk, r);
					dl += temp2;
					dll += ((double)mark * vget(cumulhdiag, r) / vget(cumulrisk, r)) - (temp2 * temp2);
				*/
					//Using Efron's weights for tied events produces a stabler convergence of the cox model than the above
					for (int k = 0; k < (int)mark; k++)
					{
						double temp = (double)k
							/ mark;
						double denom = vget(cumulrisk, r) - (temp * vget(cumul2risk, r));
						lk -= safe_log(denom);
						double temp2 = (vget(cumulgdiag, r) - (temp * vget(cumulg2diag, r))) / denom;
						dl += temp2;
						dll += (vget(cumulhdiag, r) - (temp * vget(cumulh2diag, r)) / denom) -
							(temp2 * temp2);
					}
				}
			}
		//	printf("Iteration = %d, Parrallel dl %f\t",iter, dl);
		//	printf("dll %f\n",dll);
			newlk += lk;
			dif = (dl + (vget(beta, bn)/PARAMS.surv_penalty)) / (dll + (1.0 / PARAMS.surv_penalty));
			if (fabs(dif) > vget(step, bn))
				dif = dif > 0.0 ? vget(step, bn) : -vget(step, bn);
			vset(step, bn, ((2.0 * fabs(dif)) > (vget(step, bn) / 2.0)) ? 2.0 * fabs(dif) : vget(step, bn) / 2.0); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			vset(beta, bn, (vget(beta, bn) - dif));		
			newlk -= (vget(beta, bn) * vget(beta, bn)) / (2.0 * PARAMS.surv_penalty);
			lastvar = bn;
		}
		newlk -= safe_log(sqrt(2 * PI * PARAMS.surv_penalty)) * ((double)nvar); //Constant for a particular lambda so unnecessary for convergence, however, when comparing models it is appropriate.(- 1.0 not needed as nvar already -1)
		if (iter > 0 && fabs(1.0 - (newlk / loglik)) <= model->surv_convergence) 
			break;
		loglik = newlk;
		//vprint(beta);
	}



	*f = loglik;

	gsl_vector_memcpy(&topic_beta.vector, beta);
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




