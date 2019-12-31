
#include "surv.h"
#include <gsl/gsl_blas.h>
#include <float.h>

int cox_reg(llna_model* model, corpus* c, double* f)
{
	//Mittal, S., Madigan, D., Burd, R. S., & Suchard, M. a. (2013). High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis. Biostatistics (Oxford, England), 1–15. doi:10.1093/biostatistics/kxt043
	int i,k, person, iter;
	int nvar = model->k;
	int nused = c->ndocs, lastvar = 0;
	int ntimes = model->range_t;
	gsl_vector* denom = gsl_vector_calloc(ntimes);
	gsl_vector* a = gsl_vector_calloc(ntimes);
	gsl_vector* cdiag = gsl_vector_calloc(ntimes);
	double  risk=0.0, temp=0.0, temp2=0.0,	loglik = 0.0,  newlk=0.0, d2 = 0.0, efron_wt = 0.0;
	double dif = 0.0, a2 = 0.0, cdiag2 = 0.0, gdiag = 0.0, hdiag = 0.0;
	gsl_vector* newbeta = gsl_vector_calloc(nvar);
	gsl_vector* step = gsl_vector_calloc(nvar);
	gsl_vector* zbeta = gsl_vector_calloc(nused);
	// Find baseline topic with most allocations
	double max_ss = 0.0;
	int base_index = 0;
	gsl_vector* zsum = gsl_vector_calloc(model->k);
	col_sum(c->zbar, zsum);
	for (i = 0; i < model->k; i++)
	{
		if (max_ss < vget(zsum, i))
		{
			base_index = i;
			max_ss = vget(zsum, i);
		}
	}
	for (i = 0; i < nvar; i++)
	{
		vset(newbeta, i, i==base_index ? 0.0 : vget(model->topic_beta,i)) ;
		vset(step, i, 1.0);
	}

	gsl_blas_dgemv(CblasNoTrans,1, c->zbar, newbeta, 0, zbeta);

	for (person=nused-1; person>=0; person--)
	{
	/*	vset(zbeta, person, 0.0);
		for (i = 0; i < nvar; i++)
		{
			if (i == base_index) continue;
			vinc(zbeta, person, vget(zbeta, person) * vget(newbeta,i));
		} */
		vset(zbeta, person, vget(zbeta, person) >22 ? 22 : vget(zbeta, person));
		vset(zbeta, person, vget(zbeta, person) < -200 ? -200 : vget(zbeta, person));
	}
	for (iter=1; iter<=PARAMS.surv_max_iter;  iter++)
	{
		newlk = -(log(sqrt(PARAMS.surv_penalty)) * (nvar-1));
		for (i = 0; i < nvar; i++)
		{
			if (i == base_index) continue;
			/*
			** The data is sorted from smallest time to largest
			** Start at the largest time, accumulating the risk set 1 by 1
			*/
			for (int r = ntimes - 1; r >= 0; r--)
			{
				vset(denom, r, 0.0);
				vset(a, r, 0.0);
				vset(cdiag, r, 0.0);
			}
			efron_wt = 0.0;
			
			gdiag = 0.0;
			hdiag = 0.0;
			a2 = 0.0;
			cdiag2 = 0.0;

			for (person = nused - 1; person >= 0; person--)
			{
				vinc(zbeta, person, -dif * mget(c->zbar, person, lastvar));
				vset(zbeta, person, vget(zbeta, person) > 22 ? 22 : vget(zbeta, person));
				vset(zbeta, person, vget(zbeta, person) < -200 ? -200 : vget(zbeta, person));
				/*if (isn(zbeta[person]))
				{
					zbeta[person] = 0;
					std::cout << "zbeta reset as dif = " << dif << std::endl;
					for (i = 0; i<nvar; i++)
						std::cout<< newbeta[i]<<",";
					break;
				}*/
				risk = exp(vget(zbeta,person));

				//cumumlative sums for all patients
				for (int r = c->docs[person].t_enter; r <= c->docs[person].t_exit; r++)
				{
					vinc(denom, r, risk);
					vinc(a, r, risk * mget(c->zbar, person, i));
					vinc(cdiag, r,  risk * mget(c->zbar, person, i) * mget(c->zbar, person, i));
				}
				if (c->docs[person].label > 0)
				{
					//cumumlative sums for event patients
					newlk += vget(zbeta, person);
					gdiag -= mget(c->zbar, person, i);

					efron_wt += risk; /* sum(denom) for tied deaths*/
					a2 += risk * mget(c->zbar, person, i);
					cdiag2 += risk * mget(c->zbar, person, i) * mget(c->zbar, person, i);
				}
				if (vget(c->mark, person) > 0)
				{  /* once per unique death time */
					for (k = 0; k < vget(c->mark, person); k++)
					{
						temp = k
							/ vget(c->mark, person);
						d2 = vget(denom, c->docs[person].t_exit) - (temp * efron_wt); /* sum(denom) adjusted for tied deaths*/
						newlk -= log(d2);
						temp2 = (vget(a, c->docs[person].t_exit) - (temp * a2)) / d2;
						gdiag += temp2;
						hdiag += ((vget(cdiag, c->docs[person].t_exit) - (temp * cdiag2)) / d2) -
							(temp2 * temp2);
					}
					efron_wt = 0.0;
					a2 = 0.0;
					cdiag2 = 0.0;
				}
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
		if (fabs(1.0 - (newlk / loglik)) <= PARAMS.surv_convergence) break;
		loglik = newlk;
	}   /* return for another iteration */
	
	for (person = nused - 1; person >= 0; person--)
	{
		vinc(zbeta, person, -dif * mget(c->zbar, person, lastvar));
		vset(zbeta, person, vget(zbeta, person) >22 ? 22 : vget(zbeta, person));
		vset(zbeta, person, vget(zbeta, person) < -200 ? -200 : vget(zbeta, person));
	}

	* f = loglik;
	for (i=0; i<nvar; i++)
		vset(model->topic_beta, i, vget(newbeta,i));	


	gsl_vector_free(newbeta);
	gsl_vector_free(step);
	gsl_vector_free(denom);
	gsl_vector_free(a);
	gsl_vector_free(cdiag);
	return iter;
}

