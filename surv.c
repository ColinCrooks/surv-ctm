
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

	gsl_blas_dgemv(CblasNoTrans, 1, c->zbar_scaled, model->topic_beta, 0, zbeta);


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
		gsl_vector_view personrisk = gsl_matrix_row(c->zbar_scaled, person);
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
//	int nvar = model->k - 1;
	double b = vget(beta, bn);
	
	for (int person = rank * nused / size ; person < (rank + 1) * nused / size; person++ )
	{
		unsigned int t_enter = (unsigned int) c->docs[person].t_enter;
		unsigned int t_exit = (unsigned int) c->docs[person].t_exit;

		double xb = vget(c->zbeta, person) - (dif * mget(c->zbar_scaled, person, lastvar));
		xb = xb > 22 ? 22 : xb;
		xb = xb < -200 ? -200 : xb;
		vset(c->zbeta, person, xb);

		double expxb = exp(xb);
		double zbar = mget(c->zbar_scaled, person, bn);
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
//	const double PI = 3.141592653589793238463;
	int iter = 0, nvar = model->k, ntimes = model->range_t, nused = c->ndocs;
	gsl_vector_view topic_beta = gsl_vector_subvector(model->topic_beta, 0, nvar);
	gsl_vector* scale = gsl_vector_calloc(nvar);

	gsl_vector* ones = gsl_vector_alloc(nused);
	gsl_vector_set_all(ones, 1.0);
	gsl_blas_dgemv(CblasTrans, 1.0, c->zbar, ones, 0.0, scale);
	gsl_blas_dscal(1.0 / nused, scale);

	gsl_vector* beta = gsl_vector_alloc(nvar);
	gsl_vector_memcpy(beta, &topic_beta.vector);
#pragma omp parallel for 
	for (int person = 0; person < nused; person++)
	{
		gsl_vector_view zbar = gsl_matrix_row(c->zbar, person);
		gsl_vector_view zbar_scaled = gsl_matrix_row(c->zbar_scaled, person);
		gsl_blas_dcopy(&zbar.vector, &zbar_scaled.vector);
		gsl_blas_daxpy((-1.0), scale, &zbar_scaled.vector);
		gsl_vector_div(&zbar.vector, scale); //zbar has to be positive as probability
		vset(&zbar.vector, nvar - 1, 1.0);
	}

	gsl_vector_div(beta, scale);

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

	gsl_vector* cumulxb = gsl_vector_calloc(ntimes);
	gsl_vector* cumulrisk = gsl_vector_calloc(ntimes);
	gsl_vector* cumulgdiag = gsl_vector_calloc(ntimes);
	gsl_vector* cumulhdiag = gsl_vector_calloc(ntimes);
	gsl_vector* cumul2risk = gsl_vector_calloc(ntimes);
	gsl_vector* cumulg2diag = gsl_vector_calloc(ntimes);
	gsl_vector* cumulh2diag = gsl_vector_calloc(ntimes);
	gsl_vector* step = gsl_vector_alloc(nvar);

	int threadn = omp_get_num_procs();
	//	cumul2risk_private = malloc(sizeof(gsl_vector*) * threadn);
	//	cumulh2diag_private = malloc(sizeof(gsl_matrix**) * threadn);
	//	cumulg2diag_private = malloc(sizeof(gsl_matrix*) * threadn);
	gsl_vector** cumulxb_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_vector** cumulrisk_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_vector** cumulgdiag_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_vector** cumulhdiag_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_vector** cumul2risk_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_vector** cumulg2diag_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_vector** cumulh2diag_private = malloc(sizeof(gsl_vector*) * threadn);
	for (int n = 0; n < threadn; n++)
	{
		cumulxb_private[n] = gsl_vector_calloc(ntimes);
		cumulrisk_private[n] = gsl_vector_calloc(ntimes);
		cumulgdiag_private[n] = gsl_vector_calloc(ntimes);
		cumulhdiag_private[n] = gsl_vector_calloc(ntimes);
		cumul2risk_private[n] = gsl_vector_calloc(ntimes);
		cumulg2diag_private[n] = gsl_vector_calloc(ntimes);
		cumulh2diag_private[n] = gsl_vector_calloc(ntimes);
	}


	gsl_vector_set_all(step, 1.0);

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
			gsl_vector_set_zero(cumulxb);
			gsl_vector_set_zero(cumulrisk);
			gsl_vector_set_zero(cumulgdiag);
			gsl_vector_set_zero(cumulhdiag);
			gsl_vector_set_zero(cumul2risk);
			gsl_vector_set_zero(cumulg2diag);
			gsl_vector_set_zero(cumulh2diag);
#pragma omp parallel  default(none) shared(c, model, ntimes, nvar, cumulxb_private,  cumulrisk_private, cumulgdiag_private, cumulhdiag_private, cumul2risk_private, cumulg2diag_private, cumulh2diag_private, cumulxb,  cumulrisk, cumulgdiag, cumulhdiag, cumul2risk, cumulg2diag, cumulh2diag, bn, lastvar, dif, beta)  /* for (i = 0; i < corpus->ndocs; i++) */
			{
				int size = omp_get_num_threads(); // get total number of processes
				int rank = omp_get_thread_num(); // get rank of current
				gsl_vector_set_zero(cumulxb_private[rank]);
				gsl_vector_set_zero(cumulrisk_private[rank]);
				gsl_vector_set_zero(cumulgdiag_private[rank]);
				gsl_vector_set_zero(cumulhdiag_private[rank]);
				gsl_vector_set_zero(cumul2risk_private[rank]);
				gsl_vector_set_zero(cumulg2diag_private[rank]);
				gsl_vector_set_zero(cumulh2diag_private[rank]);
				cox_reg_accumulation(model, c, size, rank, bn, lastvar, dif, beta, cumulxb_private[rank], cumulrisk_private[rank], cumulgdiag_private[rank], cumulhdiag_private[rank],
					cumul2risk_private[rank], cumulg2diag_private[rank], cumulh2diag_private[rank]);

//				for (int r = 0; r < ntimes; r++)
//				{
//#pragma omp atomic update
//					cumulxb->data[r] += cumulxb_private[rank]->data[r];
//#pragma omp atomic update
//					cumulrisk->data[r] += cumulrisk_private[rank]->data[r];
//#pragma omp atomic update
//					cumul2risk->data[r] += cumul2risk_private[rank]->data[r];
//
//#pragma omp atomic update
//					cumulgdiag->data[r] += cumulgdiag_private[rank]->data[r];
//#pragma omp atomic update
//					cumulhdiag->data[r] += cumulhdiag_private[rank]->data[r];
//#pragma omp atomic update
//					cumulg2diag->data[r] += cumulg2diag_private[rank]->data[r];
//#pragma omp atomic update
//					cumulh2diag->data[r] += cumulh2diag_private[rank]->data[r];
//
//				}
#pragma omp critical
				{
					gsl_blas_daxpy(1.0, cumulxb_private[rank], cumulxb);
					gsl_blas_daxpy(1.0,  cumulrisk_private[rank], cumulrisk);
					gsl_blas_daxpy(1.0, cumulgdiag_private[rank], cumulgdiag);
					gsl_blas_daxpy(1.0,  cumulhdiag_private[rank], cumulhdiag);
					gsl_blas_daxpy(1.0, cumul2risk_private[rank], cumul2risk);
					gsl_blas_daxpy(1.0, cumulg2diag_private[rank], cumulg2diag);
					gsl_blas_daxpy(1.0,  cumulh2diag_private[rank], cumulh2diag);
				
				}
				//gsl_vector_free(cumulxb_private);
				//gsl_vector_free(cumulrisk_private);
				//gsl_vector_free(cumulgdiag_private);
				//gsl_vector_free(cumulhdiag_private);
				//gsl_vector_free(cumul2risk_private);
				//gsl_vector_free(cumulg2diag_private);
				//gsl_vector_free(cumulh2diag_private);
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

				//	lk -= (double)mark * vget(cumulrisk, r);
				//	double temp2 = (double)mark * vget(cumulgdiag, r) / vget(cumulrisk, r);
				//	dl += temp2;
				//	dll += ((double)mark * vget(cumulhdiag, r) / vget(cumulrisk, r)) - (temp2 * temp2);
				//}
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
			newlk -= ((vget(beta, bn) * vget(beta, bn)) / (2.0 * PARAMS.surv_penalty)) + log(sqrt(PARAMS.surv_penalty)) + 0.91893853;
			lastvar = bn;
		}
		//newlk -= safe_log(sqrt(2 * PI * PARAMS.surv_penalty)) * ((double)nvar); //Constant for a particular lambda so unnecessary for convergence, however, when comparing models it is appropriate.(- 1.0 not needed as nvar already -1)
		if (iter > 0 && fabs(1.0 - (newlk / loglik)) <= model->surv_convergence) 
			break;
		loglik = newlk;
		//vprint(beta);
	}



	*f = loglik;
	gsl_vector_mul(beta, scale);
	model->intercept = vget(beta, nvar - 1);
	vset(beta, nvar - 1, 0.0);
	printf("Intercept %f\t", model->intercept);
	gsl_vector_memcpy(&topic_beta.vector, beta);
	gsl_matrix_free(sum_zbar);
	gsl_vector_free(cumulrisk);
	gsl_vector_free(cumulgdiag);
	gsl_vector_free(cumulhdiag);
	gsl_vector_free(cumul2risk);
	gsl_vector_free(cumulg2diag);
	gsl_vector_free(cumulh2diag);
	gsl_vector_free(beta);
	gsl_vector_free(scale);
	for (int n = 0; n < threadn; n++)
	{
		gsl_vector_free(cumulxb_private[n]);
		//		cumul2risk_private[n] = gsl_vector_calloc(ntimes);
		//		cumulh2diag_private[n] = malloc(sizeof(gsl_matrix*) * ntimes);
		//		cumulg2diag_private[n] = gsl_matrix_calloc(ntimes, nvar);

		gsl_vector_free(cumulrisk_private[n]);
		gsl_vector_free(cumulgdiag_private[n]);
		gsl_vector_free(cumulhdiag_private[n]);
		gsl_vector_free(cumul2risk_private[n]);
		gsl_vector_free(cumulg2diag_private[n]);
		gsl_vector_free(cumulh2diag_private[n]);
	}

	free(cumulxb_private);
	free(cumulrisk_private);
	free(cumulgdiag_private);
	free(cumulhdiag_private);
	free(cumul2risk_private);
	free(cumulg2diag_private);
	free(cumulh2diag_private);
	return(iter);
}

int cholesky2(gsl_matrix* matrix, int n, double toler)
{
	double temp;
	int  i, j, k;
	double eps, pivot;
	int rank;
	int nonneg;

	nonneg = 1;
	eps = 0;
	for (i = 0; i < n; i++) {
		if (mget(matrix,i,i) > eps)  eps = mget(matrix, i, i);
		for (j = (i + 1); j < n; j++)  mset(matrix,j,i, mget(matrix,i,j));
	}
	if (eps == 0) eps = toler; /* no positive diagonals! */
	else eps *= toler;

	rank = 0;
	for (i = 0; i < n; i++) {
		pivot = mget(matrix,i,i);
		if (isfinite(pivot) == 0 || pivot < eps) {
			mset(matrix,i,i,0);
			if (pivot < -8 * eps) nonneg = -1;
		}
		else {
			rank++;
			for (j = (i + 1); j < n; j++) {
				temp = mget(matrix,j,i) / pivot;
				mset(matrix,j,i,temp);
				minc(matrix,j,j, (-temp) * temp * pivot);
				for (k = (j + 1); k < n; k++) minc(matrix,k,j, (-temp) * mget(matrix,k,i));
			}
		}
	}
	return(rank * nonneg);
}

void chsolve2(gsl_matrix* matrix, int n, gsl_vector* y)
{
	int i, j;
	double temp;

	/*
	** solve Fb =y
	*/
	for (i = 0; i < n; i++) {
		temp = vget(y,i);
		for (j = 0; j < i; j++)
			temp -= vget(y,j) * mget(matrix,i,j);
		vset(y,i,temp);
	}
	/*
	** solve DF'z =b
	*/
	for (i = (n - 1); i >= 0; i--) {
		if (mget(matrix,i,i) == 0)  vset(y,i,0);
		else {
			temp = vget(y,i) / mget(matrix,i,i);
			for (j = i + 1; j < n; j++)
				temp -= vget(y,j) * mget(matrix,j,i);
			vset(y,i,temp);
		}
	}
}


void chinv2(gsl_matrix* matrix, int n)
{
	double temp;
	int i, j, k;

	/*
	** invert the cholesky in the lower triangle
	**   take full advantage of the cholesky's diagonal of 1's
	*/
	for (i = 0; i < n; i++) {
		if (mget(matrix,i,i) > 0) {
			mset(matrix,i,i, 1.0 / mget(matrix,i,i));   /*this line inverts D */
			for (j = (i + 1); j < n; j++) {
				minc(matrix,j,i, -mget(matrix,j,i));
				for (k = 0; k < i; k++)     /*sweep operator */
					minc(matrix,j,k, mget(matrix,j,i) * mget(matrix,i,k));
			}
		}
	}

	/*
	** lower triangle now contains inverse of cholesky
	** calculate F'DF (inverse of cholesky decomp process) to get inverse
	**   of original matrix
	*/
	for (i = 0; i < n; i++) {
		if (mget(matrix,i,i) == 0) {  /* singular row */
			for (j = 0; j < i; j++) mset(matrix,j,i,0);
			for (j = i; j < n; j++) mset(matrix,i,j,0);
		}
		else {
			for (j = (i + 1); j < n; j++) {
				temp = mget(matrix,j,i) * mget(matrix,j,j);
				if (j != i) mset(matrix,i,j,temp);
				for (k = i; k < j; k++)
					minc(matrix,i,k, temp * mget(matrix,j,k));
			}
		}
	}
}


//void cox_reg_hes_intitialise(int nvar, int ntimes, llna_model* model, gsl_matrix* sum_zbar, gsl_vector* gdiag , gsl_matrix* hdiag , gsl_vector* mean_z ,
//gsl_vector* scale_z , gsl_vector* beta , gsl_vector* newbeta , gsl_vector* cumulxb ,/* gsl_vector* cumul2risk , gsl_matrix* cumulg2diag ,
//gsl_matrix** cumulh2diag ,*/ gsl_vector* cumulrisk_start , gsl_matrix* cumulgdiag_start , gsl_matrix** cumulhdiag_start , gsl_vector* cumulrisk_end ,
//gsl_matrix* cumulgdiag_end , gsl_matrix** cumulhdiag_end , gsl_vector* running_gdiag , gsl_matrix* running_hdiag , gsl_vector* htemp , gsl_vector* atemp ,
//gsl_vector** cumulxb_private ,/* gsl_vector** cumul2risk_private , gsl_matrix*** cumulh2diag_private, gsl_matrix** cumulg2diag_private,*/  gsl_vector** cumulrisk_start_private ,
//gsl_matrix** cumulgdiag_start_private , gsl_matrix*** cumulhdiag_start_private , gsl_vector** cumulrisk_end_private , gsl_matrix** cumulgdiag_end_private , gsl_matrix*** cumulhdiag_end_private )
//{
//	
//	return;
//}
//
//void cox_reg_hes_free(int nvar, int ntimes, llna_model* model, gsl_matrix* sum_zbar, gsl_vector* gdiag, gsl_matrix* hdiag, gsl_vector* mean_z,
//	gsl_vector* scale_z, gsl_vector* beta, gsl_vector* newbeta, gsl_vector* cumulxb,/* gsl_vector* cumul2risk, gsl_matrix* cumulg2diag,
//	gsl_matrix** cumulh2diag,*/ gsl_vector* cumulrisk_start, gsl_matrix* cumulgdiag_start, gsl_matrix** cumulhdiag_start, gsl_vector* cumulrisk_end,
//	gsl_matrix* cumulgdiag_end, gsl_matrix** cumulhdiag_end, gsl_vector* running_gdiag, gsl_matrix* running_hdiag, gsl_vector* htemp, gsl_vector* atemp,
//	gsl_vector** cumulxb_private /*, gsl_vector** cumul2risk_private, gsl_matrix*** cumulh2diag_private, gsl_matrix** cumulg2diag_private*/, gsl_vector** cumulrisk_start_private,
//	gsl_matrix** cumulgdiag_start_private, gsl_matrix*** cumulhdiag_start_private, gsl_vector** cumulrisk_end_private, gsl_matrix** cumulgdiag_end_private, gsl_matrix*** cumulhdiag_end_private)
//{
//	int threadn = omp_get_num_procs();
//	
//	return;
//}
//
int cox_reg_hes(llna_model* model, corpus* c, double* f)
{
	//Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
	int i, iter;
	int halving = 0;
	int nvar = model->k-1;
	int nused = c->ndocs;
	int ntimes = model->range_t;
	double loglik = 0.0, newlk = 0.0;
	gsl_vector * beta, * newbeta, * cumulxb, * gdiag, * cumulrisk; 
	gsl_matrix * sum_zbar, * hdiag, *cumulgdiag; 
	gsl_matrix ** cumulhdiag; 

	sum_zbar = gsl_matrix_calloc(ntimes, model->k);

	//Memory for 1st g() and 2nd h() derivative function output. Initialised in loop.
	gdiag = gsl_vector_alloc(nvar);
	hdiag = gsl_matrix_alloc(nvar, nvar);

	//Memory to keep track of current beta and last iteration beta for halving process (vector view in case want to omit a baseline variable)
	beta = gsl_vector_alloc(nvar);
	newbeta = gsl_vector_alloc(nvar);


	// Memory for cumulative sums to be collected in parallel (initialised in loop)
	// single updates at event times:
	cumulxb = gsl_vector_alloc(ntimes);

		// multiple time updates in denominators
	cumulrisk = gsl_vector_alloc(ntimes);
	cumulgdiag = gsl_matrix_alloc(ntimes, nvar);
	cumulhdiag = malloc(sizeof(gsl_matrix*) * ntimes);

	for (int r = 0; r < ntimes; r++)
		cumulhdiag[r] = gsl_matrix_alloc(nvar, nvar);

	//working memory for threads 
	int threadn = omp_get_num_procs();
	gsl_vector** atemp_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_matrix** ctemp_private = malloc(sizeof(gsl_matrix*) * threadn);
	gsl_vector** gdiag_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_matrix** hdiag_private = malloc(sizeof(gsl_matrix*) * threadn);
	for (int n = 0; n < threadn; n++)
	{
		atemp_private[n] = gsl_vector_calloc(nvar);
		ctemp_private[n] = gsl_matrix_calloc(nvar,nvar);
		gdiag_private[n] = gsl_vector_calloc(nvar);
		hdiag_private[n] = gsl_matrix_calloc(nvar, nvar);
	}



	//Initial sum of zbar from patients with events at each time point (doesn't change with each iteration)
#pragma omp parallel default(none) shared(c, model, sum_zbar,  ntimes, nvar) 
	{
		int size = omp_get_num_threads(); // get total number of processes
		int rank = omp_get_thread_num(); // get rank of current
		gsl_matrix* sum_zbar_private = gsl_matrix_calloc(ntimes, model->k);
		cox_reg_hes_init_accumul(model, sum_zbar_private, c, size, rank);
#pragma omp critical
		{
			gsl_matrix_add(sum_zbar, sum_zbar_private);
		}
		gsl_matrix_free(sum_zbar_private);
	}

	gsl_vector_view topic_beta = gsl_vector_subvector(model->topic_beta, 0, nvar);
	gsl_blas_dcopy(&topic_beta.vector, newbeta);
	gsl_blas_dcopy(&topic_beta.vector, beta);

	//Main iteration loop
	for (iter = 1; iter <= PARAMS.surv_max_iter; iter++)
	{
		//Initialise everything for start of iteration
		newlk = 0.0;
		gsl_vector_set_zero(cumulxb);
		gsl_vector_set_zero(cumulrisk);
		gsl_matrix_set_zero(cumulgdiag);

		for (int r = 0; r < ntimes; r++)
			gsl_matrix_set_zero(cumulhdiag[r]);
		

		//Acuumulation loop across all patients in parallel
#pragma omp parallel  default(none) shared(c, model, ntimes, nvar, cumulxb,  cumulrisk, cumulgdiag, cumulhdiag, newbeta, atemp_private, ctemp_private)
		{
			int size = omp_get_num_threads(); // get total number of processes
			int rank = omp_get_thread_num(); // get rank of current
			cox_reg_accumul_hes(model, c, size, rank, newbeta, cumulxb, cumulrisk, cumulgdiag, cumulhdiag, atemp_private[rank], ctemp_private[rank]); 
		}

// Accumulate over event time points to calculate 1st and 2nd derivatives
		gsl_vector_set_zero(gdiag);
		gsl_matrix_set_zero(hdiag);
#pragma omp parallel reduction(+:newlk) default(none) shared(c, newbeta, sum_zbar, cumulxb, cumulrisk, cumulgdiag, cumulhdiag, ntimes, nvar, ctemp_private, atemp_private, gdiag, hdiag, gdiag_private, hdiag_private)
		{
			int size =  omp_get_num_threads(); // get total number of processes
			int rank =  omp_get_thread_num(); // get rank of current
			gsl_vector_set_zero(gdiag_private[rank]);
			gsl_matrix_set_zero(hdiag_private[rank]);
			gsl_vector_view htemp = gsl_matrix_row(ctemp_private[rank], 0); //Try to reuse memory 
			for (int r = rank * ntimes / size; r < (rank + 1) * ntimes / size; r++)
			{
				//update running sums for time 
				double mark = vget(c->mark, r);
				double denom = vget(cumulrisk, r); 


				if (mark > 0)
				{
					gsl_vector_set_zero(atemp_private[rank]);		
					
					newlk += vget(cumulxb, r);	
					newlk -= mark * safe_log(denom);

					gsl_vector_view zsum = gsl_matrix_subrow(sum_zbar, r, 0, nvar);
					gsl_blas_daxpy((1.0), &zsum.vector, gdiag_private[rank]);


					double scale = 1.0 / denom;
					gsl_vector_view grow_running = gsl_matrix_row(cumulgdiag, r);
					gsl_blas_daxpy(scale, &grow_running.vector, atemp_private[rank]);
					gsl_blas_daxpy((-1.0 * mark), atemp_private[rank], gdiag_private[rank]);

					gsl_matrix_scale(cumulhdiag[r], scale);
					gsl_blas_dsyr(CblasUpper, -1.0, atemp_private[rank], cumulhdiag[r]);
					for (int i = 0; i < nvar; i++)
					{
						gsl_vector_view cumulhrow = gsl_matrix_row(cumulhdiag[r], i);
						gsl_vector_view hrow = gsl_matrix_row(hdiag_private[rank], i);
						gsl_blas_daxpy(mark, &cumulhrow.vector, &hrow.vector);
					}
				}
			}
#pragma omp critical
			{
				gsl_blas_daxpy(1.0, gdiag_private[rank], gdiag);
				gsl_matrix_add(hdiag, hdiag_private[rank]);
				//#pragma omp simd
//				for (int i = 0; i < nvar; i++)
//				{
//					gdiag->data[i] += gdiag_private[rank]->data[i];
//#pragma omp simd
//					for (int j = i; j < nvar; j++)
//						hdiag->data[(i * hdiag->tda) + j] += hdiag_private[rank]->data[(i * hdiag->tda) + j];
//				}
			}
		}
		double b2 = 0.0;
		gsl_blas_ddot(newbeta, newbeta, &b2);
		newlk -= b2 * PARAMS.surv_penalty;
		gsl_blas_daxpy((-1.0) * PARAMS.surv_penalty, newbeta, gdiag );
		gsl_matrix_add_constant(hdiag,  PARAMS.surv_penalty);

		double flag;
		flag = cholesky2(hdiag, nvar, PARAMS.surv_convergence);

		double notfinite = 0;
		for (i = 0; i < nvar; i++) {
			if (isfinite(vget(gdiag, i)) == 0) notfinite = 2;
			for (int j = 0; j < nvar; j++) {
				if (isfinite(mget(hdiag, i, j)) == 0) notfinite = 3;
			}
		}
		if (isfinite(newlk) == 0) notfinite = 4;

		if (notfinite == 0 && (fabs(1 - (loglik / newlk)) <= PARAMS.surv_convergence)) {
			loglik = newlk;
			//chinv2(hdiag, nvar);   //Will be useful if do more with covariance matrix in future. Would also need to be re scaled
			if (halving) flag = -2;
			break;
		}

		if ((notfinite > 0 || newlk < loglik) && iter > 1) {
			halving++;
			gsl_blas_daxpy((double)halving, beta, newbeta);
			gsl_blas_dscal(1.0 / ((double)halving + 1.0), newbeta);
			printf("Backing up\t likelihood %f\n", newlk);
		}
		else {
			halving = 0;
			loglik = newlk;
			chsolve2(hdiag, nvar, gdiag);
			gsl_blas_dcopy(newbeta, beta); //Keep copy of old beta incase need to do halving
			gsl_blas_daxpy(1.0, gdiag, newbeta);
		//	vprint(newbeta);
			printf("Cox iter %d \t  likelihood %f\n", iter, newlk);
		}
	}

	*f = loglik;
	gsl_blas_dcopy(newbeta, &topic_beta.vector);
	gsl_vector_free(beta);

	gsl_vector_free(newbeta);
	gsl_matrix_free(sum_zbar);
	gsl_vector_free(cumulxb);
	gsl_vector_free(cumulrisk);
	gsl_matrix_free(cumulgdiag);

	for (int r = 0; r < ntimes; r++)
		gsl_matrix_free(cumulhdiag[r]);
	free(cumulhdiag); 
	gsl_vector_free(gdiag);
	gsl_matrix_free(hdiag);

	//working memory for threads - takes too much memory

	for (int n = 0; n < threadn; n++)
	{
		gsl_vector_free(atemp_private[n]);
		gsl_matrix_free(ctemp_private[n]);
	}
	free(atemp_private);
	free(ctemp_private);

	return iter;
}



void cox_reg_accumul_hes(llna_model* model, corpus* c, int size, int rank,
	gsl_vector* beta, gsl_vector* cumulxb,
	gsl_vector* cumulrisk, gsl_matrix* cumulgdiag, gsl_matrix** cumulhdiag, gsl_vector* atemp, gsl_matrix* ctemp)
{
	
	unsigned int nused = c->ndocs;
	unsigned int nvar = model->k - 1;
	double risk = 0;
	double step = (double)nused / (double)size;

	unsigned int startperson = mget(c->group, rank, 0);
	unsigned int endperson = (rank + 1 < size) ? (unsigned int)mget(c->group, rank + 1, 0) : nused - 1;

	unsigned int start_cumultime = (unsigned int) mget(c->group, rank, 1);
	unsigned int end_cumultime = (unsigned int) mget(c->group, rank, 2);

	unsigned int person = startperson;
	while ((unsigned int)c->docs[person].t_enter < end_cumultime &&  person < nused)
	{
		//These were already centred on t_min in corpus.c so can directly map into vectors
		unsigned int t_enter = ((unsigned int)c->docs[person].t_enter < start_cumultime) ? start_cumultime : (unsigned int)c->docs[person].t_enter;
		unsigned int t_exit = ((unsigned int)c->docs[person].t_exit >= end_cumultime) ? end_cumultime - 1 :  (unsigned int)c->docs[person].t_exit;

		//Patient level linear predictor and risk
		gsl_vector_view z = gsl_matrix_subrow(c->zbar_scaled, person, 0, nvar);
		double xb = 0.0;
		gsl_blas_ddot(beta, &z.vector, &xb);
		xb = xb > 22 ? 22 : xb;
		xb = xb < -200 ? -200 : xb;
		risk = exp(xb);
		gsl_vector_set_zero(atemp);
		gsl_matrix_set_zero(ctemp);
		// X*exp(xb)
		gsl_blas_daxpy(risk, &z.vector, atemp);
		gsl_blas_dsyr(CblasUpper, risk, &z.vector, ctemp);
#pragma omp simd
		for (int r = t_enter; r <= t_exit; r++)
		{
			vinc(cumulrisk, r, risk);
			gsl_vector_view arow = gsl_matrix_row(cumulgdiag, r);
			gsl_blas_daxpy(1.0, atemp, &arow.vector);
#pragma omp simd
			for (unsigned int rowN = 0; rowN < nvar; rowN++)
				for (unsigned int colN = rowN; colN < nvar; colN++)
					cumulhdiag[r]->data[(rowN * cumulhdiag[r]->tda) + colN] += ctemp->data[(rowN * cumulhdiag[r]->tda) + colN];
		}
		if (c->docs[person].label > 0 && c->docs[person].t_exit< end_cumultime)	vinc(cumulxb, t_exit, xb);
		person++;
	}
}

void cox_reg_accumul_fullefron(llna_model* model, corpus* c, int size, int rank,
	gsl_vector* beta, 
	gsl_vector* cumulrisk, gsl_matrix* cumulgdiag, gsl_matrix** cumulhdiag,
	gsl_vector* cumul2risk, gsl_matrix* cumul2gdiag, gsl_matrix** cumul2hdiag,
	gsl_vector* atemp, gsl_matrix* ctemp)
{

	unsigned int nused = c->ndocs;
	unsigned int nvar = model->k;
	double risk = 0;
	double step = (double)nused / (double)size;

	unsigned int startperson = mget(c->group, rank, 0);
	unsigned int endperson = (rank + 1 < size) ? (unsigned int)mget(c->group, rank + 1, 0) : nused - 1;

	unsigned int start_cumultime = (unsigned int)mget(c->group, rank, 1);
	unsigned int end_cumultime = (unsigned int)mget(c->group, rank, 2);

	unsigned int person = startperson;
	while ((unsigned int)c->docs[person].t_enter < end_cumultime && person < nused)
	{
		//These were already centred on t_min in corpus.c so can directly map into vectors
		unsigned int t_enter = ((unsigned int)c->docs[person].t_enter < start_cumultime) ? start_cumultime : (unsigned int)c->docs[person].t_enter;
		unsigned int t_exit = ((unsigned int)c->docs[person].t_exit >= end_cumultime) ? end_cumultime - 1 : (unsigned int)c->docs[person].t_exit;

		//Patient level linear predictor and risk
		gsl_vector_view z = gsl_matrix_subrow(c->zbar_scaled, person, 0, nvar);
		double xb = 0.0;
		gsl_blas_ddot(beta, &z.vector, &xb);
		xb = xb > 22 ? 22 : xb;
		xb = xb < -200 ? -200 : xb;
		risk = exp(xb);
		gsl_vector_set_zero(atemp);
		gsl_matrix_set_zero(ctemp);
		// X*exp(xb)
		gsl_blas_daxpy(risk, &z.vector, atemp);
		gsl_blas_dsyr(CblasUpper, risk, &z.vector, ctemp);
		for (int r = t_enter; r <= t_exit; r++)
		{
			vinc(cumulrisk, r, risk);
			gsl_vector_view arow = gsl_matrix_row(cumulgdiag, r);
			gsl_blas_daxpy(1.0, atemp, &arow.vector);
#pragma omp simd
			for (unsigned int rowN = 0; rowN < nvar; rowN++)
			{
#pragma omp simd
				for (unsigned int colN = rowN; colN < nvar; colN++)
					cumulhdiag[r]->data[(rowN * cumulhdiag[r]->tda) + colN] += ctemp->data[(rowN * ctemp->tda) + colN];
			}
		}
		
		if (c->docs[person].label > 0 && c->docs[person].t_exit < end_cumultime)
		{
//			vinc(cumulxb, t_exit, xb);
			vinc(cumul2risk, t_exit, risk);
			gsl_vector_view arow = gsl_matrix_row(cumul2gdiag, t_exit);
			gsl_blas_daxpy(1.0, atemp, &arow.vector);
#pragma omp simd
			for (unsigned int rowN = 0; rowN < nvar; rowN++)
			{
#pragma omp simd
				for (unsigned int colN = rowN; colN < nvar; colN++)
					cumul2hdiag[t_exit]->data[(rowN * cumul2hdiag[t_exit]->tda) + colN] += ctemp->data[(rowN * ctemp->tda) + colN];
			}
		}
		person++;
	}
}






void cox_reg_hes_init_accumul(llna_model* model, gsl_matrix* sum_zbar_events_cumulative, corpus* c, int size, int rank)
{
	unsigned int nused = c->ndocs;
	for (unsigned int person = rank * nused / size; person < (rank + 1) * nused / size; person++)
	{
		unsigned int t_exit = c->docs[person].t_exit;
		int label = c->docs[person].label;
		gsl_vector_view personrisk = gsl_matrix_row(c->zbar_scaled, person);
	
		if (label > 0)
		{
			gsl_vector_view timeszbar = gsl_matrix_row(sum_zbar_events_cumulative, t_exit);
			gsl_blas_daxpy(1.0, &personrisk.vector, &timeszbar.vector);

		}
	}

	return;
}



void cox_reg_accumul_hessian_atomic(llna_model* model, corpus* c, int size, int rank,
	gsl_vector* newbeta, gsl_vector* cumulxb,
	gsl_vector* cumulrisk, gsl_matrix* cumulgdiag, gsl_matrix** cumulhdiag)
{

	int nused = c->ndocs;
	int nvar = model->k - 1;
	double risk;
	for (int person = rank * nused / size; person < (rank + 1) * nused / size; person++)
	{
		//These were already centred on t_min in corpus.c so can directly map into vectors
		unsigned int t_enter = (unsigned int)c->docs[person].t_enter;
		unsigned int t_exit = (unsigned int)c->docs[person].t_exit;

		//Patient level linear predictor and risk
		gsl_vector_view z = gsl_matrix_subrow(c->zbar_scaled, person, 0, nvar);
		double xb;

		gsl_blas_ddot(newbeta, &z.vector, &xb);
		xb = xb > 22 ? 22 : xb;
		xb = xb < -200 ? -200 : xb;
		risk = exp(xb);
#pragma omp simd
				for (int r = t_enter; r <= t_exit; r++)
				{
#pragma omp atomic update
					cumulrisk->data[r] += risk;

#pragma omp simd
					for (int rowN = 0; rowN < nvar; rowN++)
					{

						double rowz = c->zbar_scaled->data[(person * c->zbar_scaled->tda) + rowN];
						double zrisk = rowz * risk;
#pragma omp atomic update
						cumulgdiag->data[(r * cumulgdiag->tda) + rowN] += zrisk;
						//printf("gdiag %f\n", cumulgdiag->data[(rowN)]);
#pragma omp simd
						for (int colN = rowN; colN < nvar; colN++)
						{
							double colz = c->zbar_scaled->data[(person * c->zbar_scaled->tda) + colN];
#pragma omp atomic update
							cumulhdiag[r]->data[(rowN * cumulhdiag[r]->tda) + colN] += zrisk * colz; //ctemp->data[(rowN * nvar) + colN];
						//	printf("hdiag %f\n", cumulhdiag[r]->data[(rowN * nvar) + colN] );
						}
					}
				}
//			}
	
			if (c->docs[person].label > 0)
			{
#pragma omp atomic update
				cumulxb->data[t_exit] += xb;
			}
//			printf("cumulrisk\n");
//			printf("%f\n",cumulrisk->data[t_exit]);


//		}
	}
//	gsl_vector_free(atemp);
//	gsl_matrix_free(ctemp);
}


int cox_reg_fullefron(llna_model* model, corpus* c, double* f)
{
	//Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
	int i, iter;
	int halving = 0;
	int nvar = model->k;
	int nused = c->ndocs;
	int ntimes = model->range_t;
	double loglik = 0.0, newlk = 0.0;
	gsl_vector* beta, * newbeta,  * gdiag, * cumulrisk, *cumul2risk; //* cumulxb,
	gsl_matrix* sum_zbar, * hdiag, * cumulgdiag, * cumul2gdiag;
	gsl_matrix** cumulhdiag, **cumul2hdiag;
//	gsl_vector* step = gsl_vector_calloc(nvar);
	char backingup[12];
	sprintf(backingup, "           ");

	gsl_vector* scale = gsl_vector_calloc(model->k);

//	for (i = 0; i < nvar; i++)
//	{
//		//vset(newbeta, i, i == base_index ? 0.0 : vget(model->topic_beta, i));
//		vset(step, i, 1.0);
//	}

	sum_zbar = gsl_matrix_calloc(ntimes, model->k);

	//Memory for 1st g() and 2nd h() derivative function output. Initialised in loop.
	gdiag = gsl_vector_alloc(nvar);
	hdiag = gsl_matrix_alloc(nvar, nvar);

	//Memory to keep track of current beta and last iteration beta for halving process (vector view in case want to omit a baseline variable)
	beta = gsl_vector_alloc(nvar);
	newbeta = gsl_vector_alloc(nvar);


	// Memory for cumulative sums to be collected in parallel (initialised in loop)
	// single updates at event times:
//	cumulxb = gsl_vector_alloc(ntimes);

	// multiple time updates in denominators
	cumulrisk = gsl_vector_alloc(ntimes);
	cumul2risk = gsl_vector_alloc(ntimes);
	cumulgdiag = gsl_matrix_alloc(ntimes, nvar);
	cumul2gdiag = gsl_matrix_alloc(ntimes, nvar);
	cumulhdiag = malloc(sizeof(gsl_matrix*) * ntimes);
	cumul2hdiag = malloc(sizeof(gsl_matrix*) * ntimes);
	for (int r = 0; r < ntimes; r++)
	{
		cumulhdiag[r] = gsl_matrix_alloc(nvar, nvar);
		cumul2hdiag[r] = gsl_matrix_alloc(nvar, nvar);
	}
	//working memory for threads 
	int threadn = omp_get_num_procs();
	gsl_vector** atemp_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_matrix** ctemp_private = malloc(sizeof(gsl_matrix*) * threadn);
	gsl_vector** gdiag_private = malloc(sizeof(gsl_vector*) * threadn);
	gsl_matrix** hdiag_private = malloc(sizeof(gsl_matrix*) * threadn);
	for (int n = 0; n < threadn; n++)
	{
		atemp_private[n] = gsl_vector_calloc(nvar);
		ctemp_private[n] = gsl_matrix_calloc(nvar, nvar);
		gdiag_private[n] = gsl_vector_calloc(nvar);
		hdiag_private[n] = gsl_matrix_calloc(nvar, nvar);
	}

	//Scaling so centred on zero

	gsl_vector_view topic_beta = gsl_vector_subvector(model->topic_beta, 0, nvar);
	gsl_blas_dcopy(&topic_beta.vector, newbeta);
	int k = model->k;
	gsl_vector* ones = gsl_vector_alloc(nused);
	gsl_vector_set_all(ones, 1.0);
	gsl_blas_dgemv(CblasTrans, 1.0,  c->zbar, ones, 0.0, scale);
	gsl_blas_dscal(1.0 / nused, scale);
//#pragma omp parallel for reduction(+:scale[:k])
//	for (int person = 0; person < nused; person++)
//	{
//		gsl_vector_view zbar = gsl_matrix_row(c->zbar, person);
//		gsl_blas_daxpy(1.0, &zbar.vector, scale);
//	}

#pragma omp parallel for 
	for (int person = 0; person < nused; person++)
	{
		gsl_vector_view zbar = gsl_matrix_row(c->zbar, person);
		gsl_vector_view zbar_scaled = gsl_matrix_row(c->zbar_scaled, person);
		gsl_blas_dcopy(&zbar.vector, &zbar_scaled.vector);
		gsl_blas_daxpy((-1.0), scale, &zbar_scaled.vector);
		gsl_vector_div(&zbar.vector, scale); //zbar has to be positive as probability
		vset(&zbar.vector, nvar - 1, 1.0);
	}
	
	gsl_vector_div(newbeta, scale);	
	gsl_blas_dcopy(newbeta, beta);

	//Initial sum of zbar from patients with events at each time point (doesn't change with each iteration)
#pragma omp parallel default(none) shared(c, model, sum_zbar,  ntimes, nvar) 
	{
		int size = omp_get_num_threads(); // get total number of processes
		int rank = omp_get_thread_num(); // get rank of current
		gsl_matrix* sum_zbar_private = gsl_matrix_calloc(ntimes, model->k);
		cox_reg_hes_init_accumul(model, sum_zbar_private, c, size, rank);
#pragma omp critical
		{
			gsl_matrix_add(sum_zbar, sum_zbar_private);
		}
		gsl_matrix_free(sum_zbar_private);
	}

	//Main iteration loop
	for (iter = 1; iter <= PARAMS.surv_max_iter; iter++)
	{
		//Initialise everything for start of iteration
		newlk = 0.0;
	//	gsl_vector_set_zero(cumulxb);
		gsl_vector_set_zero(cumulrisk);
		gsl_matrix_set_zero(cumulgdiag);
		gsl_vector_set_zero(cumul2risk);
		gsl_matrix_set_zero(cumul2gdiag);

		for (int r = 0; r < ntimes; r++)
		{
			gsl_matrix_set_zero(cumulhdiag[r]);
			gsl_matrix_set_zero(cumul2hdiag[r]);
		}

		//Acuumulation loop across all patients in parallel
#pragma omp parallel  default(none) shared(c, model, ntimes, nvar,   cumulrisk, cumulgdiag, cumulhdiag,  cumul2risk, cumul2gdiag, cumul2hdiag, newbeta, atemp_private, ctemp_private) //cumulxb,
		{
			int size = omp_get_num_threads(); // get total number of processes
			int rank = omp_get_thread_num(); // get rank of current
			cox_reg_accumul_fullefron(model, c, size, rank, newbeta, cumulrisk, cumulgdiag, cumulhdiag, cumul2risk, cumul2gdiag, cumul2hdiag, atemp_private[rank], ctemp_private[rank]); //cumulxb, 
		}

		// Accumulate over event time points to calculate 1st and 2nd derivatives
		newlk = 0.0;
		gsl_vector_set_zero(gdiag);
		gsl_matrix_set_zero(hdiag);
#pragma omp parallel reduction(+:newlk) default(none) shared(c, newbeta, sum_zbar, cumulrisk, cumulgdiag, cumulhdiag, cumul2risk, cumul2gdiag, cumul2hdiag, ntimes, nvar, atemp_private, gdiag, hdiag, gdiag_private, hdiag_private) //, cumulxb
		{
			int size = omp_get_num_threads(); // get total number of processes
			int rank = omp_get_thread_num(); // get rank of current
			gsl_vector_set_zero(gdiag_private[rank]);
			gsl_matrix_set_zero(hdiag_private[rank]);
			for (int r = rank * ntimes / size; r < (rank + 1) * ntimes / size; r++)
			{
				//update running sums for time 
				double mark = vget(c->mark, r);

				if (mark > 0)
				{
					double xb = 0;
					
					//newlk += vget(cumulxb, r);
					gsl_vector_view zsum = gsl_matrix_subrow(sum_zbar, r, 0, nvar);
					gsl_blas_ddot(newbeta, &zsum.vector, &xb);
					newlk += xb;
					gsl_blas_daxpy((1.0), &zsum.vector, gdiag_private[rank]);

					double denom = vget(cumulrisk, r);
					double denom_efron = vget(cumul2risk, r);
					double temp = 1.0 / (double)mark;
					gsl_vector_view grow_running = gsl_matrix_row(cumulgdiag, r);
					gsl_vector_view grow_running2 = gsl_matrix_row(cumul2gdiag, r);

					newlk -= safe_log(denom);

					double scale = 1.0 / denom;

					gsl_vector_set_zero(atemp_private[rank]);
					gsl_blas_daxpy(scale, &grow_running.vector, atemp_private[rank]);
					gsl_blas_daxpy((-1.0), atemp_private[rank], gdiag_private[rank]);


					gsl_blas_dsyr(CblasUpper, -1.0, atemp_private[rank], hdiag_private[rank]);
					for (int i = 0; i < nvar; i++)
					{
						gsl_vector_view cumulhrow = gsl_matrix_row(cumulhdiag[r], i);
						gsl_vector_view hrow = gsl_matrix_row(hdiag_private[rank], i);
						gsl_blas_daxpy(scale, &cumulhrow.vector, &hrow.vector);
					}

					for (int k = 1; k < mark; k++)
					{

						denom -= (temp * denom_efron);
						newlk -= safe_log(denom);

						double scale = 1.0 / denom;

						gsl_vector_set_zero(atemp_private[rank]);
						gsl_blas_daxpy((-1.0) * temp, &grow_running2.vector, &grow_running.vector);
						gsl_blas_daxpy(scale, &grow_running.vector, atemp_private[rank]);
						gsl_blas_daxpy((-1.0), atemp_private[rank], gdiag_private[rank]);


						gsl_blas_dsyr(CblasUpper, -1.0, atemp_private[rank], hdiag_private[rank]);
						for (int i = 0; i < nvar; i++)
						{

							gsl_vector_view cumulhrow = gsl_matrix_row(cumulhdiag[r], i);
							gsl_vector_view cumul2hrow = gsl_matrix_row(cumul2hdiag[r], i);
							gsl_blas_daxpy((-1.0) * temp, &cumul2hrow.vector, &cumulhrow.vector);
							gsl_vector_view hrow = gsl_matrix_row(hdiag_private[rank], i);
							gsl_blas_daxpy(scale, &cumulhrow.vector, &hrow.vector);
						}
					}
				}
			}
#pragma omp critical
			{
//				gsl_blas_daxpy(1.0, gdiag_private[rank], gdiag);
#pragma omp simd
				for (int i = 0; i < nvar; i++)
				{
					gdiag->data[i] += gdiag_private[rank]->data[i];
#pragma omp simd
					for (int j = i; j < nvar; j++)
						hdiag->data[(i * hdiag->tda) + j] += hdiag_private[rank]->data[(i * hdiag->tda) + j];
				}
			}
		}
		
//#Penalised ridge regression. Multiply penalty lambda = 1/r so that it is possible to remove penaltiy when lambda=0
		double b2 = 0.0;
		gsl_blas_ddot(newbeta, newbeta, &b2);
		newlk -= (b2 / (2 * PARAMS.surv_penalty)) + ((log(sqrt(PARAMS.surv_penalty)) + 0.91893853) * (nvar)) ;
		gsl_blas_daxpy((-1.0) / PARAMS.surv_penalty, newbeta, gdiag);
		gsl_matrix_add_constant(hdiag, 1.0 / PARAMS.surv_penalty);
		
		double flag;
		flag = cholesky2(hdiag, nvar, PARAMS.surv_convergence);

		double notfinite = 0;
		for (i = 0; i < nvar; i++) {
			if (isfinite(vget(gdiag, i)) == 0) notfinite = 2;
			for (int j = 0; j < nvar; j++) {
				if (isfinite(mget(hdiag, i, j)) == 0) notfinite = 3;
			}
		}
		if (isfinite(newlk) == 0) notfinite = 4;

		if (notfinite == 0 && (fabs(1 - (loglik / newlk)) <= PARAMS.surv_convergence)) {
			loglik = newlk;
			//chinv2(hdiag, nvar);   //Will be useful if do more with covariance matrix in future. Would also need to be re scaled
			if (halving) flag = -2;
			break;
		}

		if ((notfinite > 0 || newlk < loglik) && iter > 1) {
			halving++;
			gsl_blas_daxpy((double)halving, beta, newbeta);
			gsl_blas_dscal(1.0 / ((double)halving + 1.0), newbeta);
			sprintf(backingup, "(backed up)");
			//printf("Backing up\t likelihood %f\n", newlk);
			iter--;
		}
		else {
			halving = 0;
			loglik = newlk;
			chsolve2(hdiag, nvar, gdiag);
			//for (int b = 0; b < nvar; b++)
			//{
			//	if (fabs(vget(gdiag, b) > vget(step, b)))
			//		vset(gdiag, b, vget(gdiag, b) > 0.0 ? vget(step, b) : -vget(step, b));
			//	vset(step, b, ((2.0 * fabs(vget(gdiag, b))) > (vget(step, b) / 2.0)) ? 2.0 * fabs(vget(gdiag, b)) : (vget(step, b) / 2.0)); //Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-Scale Bayesian Logistic Regression for Text Categorization. Technometrics, 49(3), 291–304. doi:10.1198/004017007000000245
			//}

			gsl_blas_dcopy(newbeta, beta); //Keep copy of old beta incase need to do halving
			gsl_blas_daxpy(1.0, gdiag, newbeta);
			//	vprint(newbeta);
			printf("Cox iter %d \t  likelihood %f %s\n", iter, newlk, backingup);
			sprintf(backingup, "           ");
		}
	}

	*f = loglik; //-(log(sqrt(PARAMS.surv_penalty)) + 0.91893853) * ((double)nvar - 1);;
	gsl_vector_mul(newbeta, scale);
	model->intercept = vget(newbeta, nvar - 1);
	vset(newbeta, nvar - 1, 0.0);
	printf("Intercept %f\t", model->intercept);
	gsl_blas_dcopy(newbeta, &topic_beta.vector);
	gsl_vector_free(beta);

	gsl_vector_free(newbeta);
	gsl_matrix_free(sum_zbar);
//	gsl_vector_free(cumulxb);
	gsl_vector_free(cumulrisk);
	gsl_matrix_free(cumulgdiag);
	gsl_vector_free(cumul2risk);
	gsl_matrix_free(cumul2gdiag);

	for (int r = 0; r < ntimes; r++)
	{
		gsl_matrix_free(cumulhdiag[r]);
		gsl_matrix_free(cumul2hdiag[r]);
	}
	free(cumulhdiag);
	free(cumul2hdiag);
	gsl_vector_free(gdiag);
	gsl_matrix_free(hdiag);

	//working memory for threads - takes too much memory

	for (int n = 0; n < threadn; n++)
	{
		gsl_vector_free(atemp_private[n]);
		gsl_matrix_free(ctemp_private[n]);
	}
	free(atemp_private);
	free(ctemp_private);

	return iter;
}
