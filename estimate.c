// (C) Copyright 2007, David M. Blei and John D. Lafferty

// This file is part of CTM-C.

// CTM-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// CTM-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

/*************************************************************************
 *
 * llna.c
 *
 * estimation of an llna model by variational em
 *
 *************************************************************************/


#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_heapsort.h>
#include <assert.h>
#include <omp.h>


#include "corpus.h"
#include "ctm.h"
#include "inference.h"
#include "gsl-wrappers.h"
#include "params.h"
#include "surv.h"

extern llna_params PARAMS;

//MSVS debugging
# if defined(_MSC_VER)
#  define isn(x) (_isnan(x))
#  define isf(x) (_finite(x))
#  define mkd(x) (_mkdir(x))
# define  _NO_CRT_STDIO_INLINE
#  include <direct.h> //windows header for _mkdir
# else
#  define isn(x) (isnan(x))
#  define isf(x) (isfinite(x))
#  define mkd(x) (mkdir(x))
# include <sys/stat.h> // linux header for mkdir
#endif

#define HAVE_INLINE 1                   
#define GSL_RANGE_CHECK_OFF   1       // Turn off range checking for arrays for speed up (can be switched back on if need to re debug the code)


/*
 * e step
 *
 */

void expectation(corpus* corpus, llna_model* model, llna_ss* ss, 
                 double* avg_niter, double* total_lhood,
                 gsl_matrix* corpus_lambda, gsl_matrix* corpus_nu,
                 gsl_matrix* corpus_phi_sum,
                 short reset_var, double* converged_pct)
{
    double total= 0.0; 
    *avg_niter = 0.0;
    *converged_pct = 0;
    int threadn = omp_get_num_procs();
    llna_var_param** var = malloc(sizeof(llna_var_param*)*threadn);
    double avniter = 0.0;
    double convergedpct = 0;
    gsl_matrix_set_zero(corpus->zbar);

    for (int n = 0; n < threadn; n++)
        var[n] = new_llna_var_param(corpus->nterms, model->k);
    


#pragma omp parallel reduction(+:total, avniter, convergedpct) default(none) shared(corpus, model, ss, var, corpus_lambda,  corpus_nu, corpus_phi_sum, PARAMS, reset_var) /* for (i = 0; i < corpus->ndocs; i++) */
    {
        int i;
        int size = omp_get_num_threads(); // get total number of processes
        int rank = omp_get_thread_num(); // get rank of current
        double  lhood;
        // keep of track of corpus level values to reset var if needed
        gsl_vector lambda, nu;
 //       gsl_vector* phi_sum;
        doc doc;
      //  llna_var_param* var;

 //       phi_sum = gsl_vector_alloc(model->k);
        for (i = (rank * corpus->ndocs / size); i < (rank + 1) * corpus->ndocs / size; i++)
        {
           // printf("doc %5d   ", i);
            doc = corpus->docs[i];

          /*  if (var == NULL)
                return;*/
            if (reset_var)
                init_var_unif(var[rank], &doc, model);
            else
            {
                lambda = gsl_matrix_row(corpus_lambda, i).vector;
                nu = gsl_matrix_row(corpus_nu, i).vector;
                init_var(var[rank], &doc, model, &lambda, &nu);
            }
            lhood = var_inference(var[rank], &doc, model);
            update_expected_ss(var[rank], &doc, ss);
            total += lhood;
            // printf("lhood %5.5e   niter %5d\n", lhood, var->niter);
            avniter += var[rank]->niter;
            convergedpct += var[rank]->converged;

            // Allocated topics for survival supervision
            gsl_vector_view zbarow = gsl_matrix_row(corpus->zbar, i);

            for (int n = 0; n < doc.nterms; n++)
            {
                double scale = (double)doc.count[n] / (double)doc.total;
                gsl_vector_view phirow = gsl_matrix_row(var[rank]->phi, n);
                gsl_blas_daxpy(scale, &phirow.vector, &zbarow.vector);
             //   for (j = 0; j < model->k; j++)
            //    {
                    //minc(corpus->zbar, i, j, mget(var[i]->phi, n, j) * (double)doc.count[n] / (double)doc.total);
             //   }
            }
                //mset(corpus->zbar, i, j, vget(var->lambda, j));
//            

                    //
            //printf("zbar\t");
            //gsl_vector_view zbar = gsl_matrix_row(corpus->zbar, i);
            //vprint(&zbar.vector);

            gsl_matrix_set_row(corpus_lambda, i, var[rank]->lambda);
            gsl_matrix_set_row(corpus_nu, i, var[rank]->nu);
            gsl_matrix_set_row(corpus_phi_sum, i, var[rank]->sum_phi);
            
        }

    //    gsl_vector_free(phi_sum);
    }
    for (int n = 0; n < threadn; n++)
        free_llna_var_param(var[n]);
    
    free(var);
    *avg_niter = avniter / corpus->ndocs;
    *converged_pct = convergedpct / corpus->ndocs;
    *total_lhood = total;
}


/*
 * m step
 *
 */

void cov_shrinkage(gsl_matrix* mle, int n, gsl_matrix* result)
{
    int p = (int) mle->size1, i;
    double alpha = 0, tau = 0, log_lambda_s = 0;
    gsl_vector
        *lambda_star = gsl_vector_calloc(p),
        t, u,
        *eigen_vals = gsl_vector_calloc(p),
        *s_eigen_vals = gsl_vector_calloc(p);
    gsl_matrix
        *d = gsl_matrix_calloc(p,p),
        *eigen_vects = gsl_matrix_calloc(p,p),
        *s_eigen_vects = gsl_matrix_calloc(p,p),
        *result1 = gsl_matrix_calloc(p,p);

    // get eigen decomposition

    sym_eigen(mle, eigen_vals, eigen_vects);
    for (i = 0; i < p; i++)
    {
        // compute shrunken eigenvalues
        alpha = 1.0 / ( (double) n + (double) p + 1 - 2 * (double) i);
        vset(lambda_star, i, n * alpha * vget(eigen_vals, i));
    }

    // get diagonal mle and eigen decomposition

    t = gsl_matrix_diagonal(d).vector;
    u = gsl_matrix_diagonal(mle).vector;
    gsl_vector_memcpy(&t, &u);
    sym_eigen(d, s_eigen_vals, s_eigen_vects);

    // compute tau^2

    for (i = 0; i < p; i++)
        log_lambda_s += log(vget(s_eigen_vals, i));
    log_lambda_s = log_lambda_s/p;
    for (i = 0; i < p; i++)
        tau += pow(log(vget(lambda_star, i)) - log_lambda_s, 2)/((double) p + 4) - 2.0 / (double) n;

    // shrink \lambda* towards the structured eigenvalues

    for (i = 0; i < p; i++)
        vset(lambda_star, i,
             exp((2.0 / (double) n ) / ((2.0 / (double) n) + tau) * log_lambda_s +
                 tau/((2.0 / (double) n) + tau) * log(vget(lambda_star, i))));

    // put the eigenvalues in a diagonal matrix

    t = gsl_matrix_diagonal(d).vector;
    gsl_vector_memcpy(&t, lambda_star);

    // reconstruct the covariance matrix

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, d, eigen_vects, 0, result1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, eigen_vects, result1, 0, result);

    // clean up

    gsl_vector_free(lambda_star);
    gsl_vector_free(eigen_vals);
    gsl_vector_free(s_eigen_vals);
    gsl_matrix_free(d);
    gsl_matrix_free(eigen_vects);
    gsl_matrix_free(s_eigen_vects);
    gsl_matrix_free(result1);
}



void maximization(llna_model* model, llna_ss* ss)
{
    int i, j;
    double sum;

    // mean maximization

    for (i = 0; i < model->k-1; i++)
        vset(model->mu, i, vget(ss->mu_ss, i) / ss->ndata);

    // covariance maximization

    for (i = 0; i < model->k-1; i++)
    {
        for (j = 0; j < model->k-1; j++)
        {
            mset(model->cov, i, j,
                 (1.0 / ss->ndata) *
                 (mget(ss->cov_ss, i, j) +
                  ss->ndata * vget(model->mu, i) * vget(model->mu, j) -
                  vget(ss->mu_ss, i) * vget(model->mu, j) -
                  vget(ss->mu_ss, j) * vget(model->mu, i)));
        }
    }
    if (PARAMS.cov_estimate == SHRINK)
    {
        cov_shrinkage(model->cov, (int) ss->ndata, model->cov);
    }
    matrix_inverse(model->cov, model->inv_cov);
    model->log_det_inv_cov = log_det(model->inv_cov);

    // topic maximization

    for (i = 0; i < model->k; i++)
    {
        sum = 0;
        for (j = 0; j < model->log_omega->size2; j++)
            sum += mget(ss->omega_ss, i, j);

        if (sum == 0) sum = safe_log(sum) * model->log_omega->size2;
        else sum = safe_log(sum);

        for (j = 0; j < model->log_omega->size2; j++)
            mset(model->log_omega, i, j, safe_log(mget(ss->omega_ss, i, j)) - sum);
    }
}


/*
 * run em
 *
 */

llna_model* em_initial_model(int k, corpus* corpus, char* start)
{
    int d, r, t, range_t = 1 + corpus->max_t - corpus->min_t;
    llna_model* model;
    printf("starting from %s\n", start);
    if (strcmp(start, "rand")==0)
        model = random_init(k, corpus->nterms, range_t);
    else if (strcmp(start, "seed")==0)
        model = corpus_init(k, corpus);
    else
        model = read_llna_model(start);
    if (model == NULL)
        return NULL;
    gsl_vector* events = gsl_vector_calloc(range_t);
    gsl_vector* xb = gsl_vector_calloc(range_t);
    corpus->zbar = gsl_matrix_calloc(corpus->ndocs, k);
    corpus->zbar_scaled = gsl_matrix_calloc(corpus->ndocs, k);
    for (d = (corpus->ndocs) - 1; d >= 0; d--)
    {
        if (corpus->docs[d].t_exit > 0)
        {
            vinc(events, corpus->docs[d].t_exit, corpus->docs[d].label);
           // printf("Start %d \t end %d\n", corpus->docs[d].t_enter, corpus->docs[d].t_exit);
            for (r = corpus->docs[d].t_enter; r <= corpus->docs[d].t_exit; r++)
                vinc(xb, r, 1.0);

        }
    }

    vset(model->cbasehazard, 0, vget(events, 0) / vget(xb, 0));
    vset(model->basehazard, 0, vget(events, 0) / vget(xb, 0));
    for (t = 1; t < range_t; t++)
    {
        vset(model->basehazard, t, vget(events, t) / vget(xb, t));
        vset(model->cbasehazard, t, vget(model->cbasehazard, t - 1) + vget(events, t) / vget(xb, t));
    }

    gsl_vector_free(events);
    gsl_vector_free(xb);
    return(model);
}

void cumulative_basehazard(corpus* corpus, llna_model* model)
{    
    double xb2, exb2, exb;
    int d, r;
    
    gsl_vector* xb = gsl_vector_calloc(model->range_t);
    gsl_vector* zbeta = gsl_vector_calloc(corpus->ndocs);
    gsl_vector_set_zero(model->basehazard);
    gsl_vector_set_zero(xb);
    gsl_vector_set_zero(zbeta);
    gsl_blas_dgemv(CblasNoTrans, 1, corpus->zbar, model->topic_beta, 0, zbeta);
       
    exb = 0.0;
    for (d = (corpus->ndocs) - 1; d >= 0; d--)
    {
        xb2 = 0.0;
        exb2 = 0.0;
        xb2 += vget(zbeta, d);
        if (corpus->docs[d].label > 0)
            exb2 += vget(zbeta, d);

        //std::cout << "exb " << exb << " exb2 " << exb2;
        for (r = corpus->docs[d].t_enter; r <= corpus->docs[d].t_exit; r++)
            vset(xb, r, log_sum(vget(xb, r), xb2));
        if (corpus->docs[d].label > 0)
            exb = log_sum(exb, exb2);

        //std::cout << " = " << exb << std::endl << " xb " << xb << " xb2 " << xb2;

        if (d == 0 || vget(corpus->cmark, d) > 0)
        {
            for (r = 0; r < vget(corpus->cmark, d); r++)
                vinc(model->basehazard, corpus->docs[d].t_exit,
                    1.0
                    /
                    (
                        (1.0 + exp(vget(xb, corpus->docs[d].t_exit)) - exp(exb)) + 
                        (
                            ((vget(corpus->cmark, d) - r) 
                            / vget(corpus->cmark, d)) 
                            * exp(exb)
                        )
                    )
                ); //efron's method as in survfit4.c in R survival function
            exb = 0.0;
        }

        //std::cout << " = " << xb << std::endl;
     /*   if (d == 0 || vget(corpus->cmark, d) > 0)
        {
            for (r = 0; r < vget(corpus->cmark, d); r++)
                vinc(model->basehazard, corpus->docs[d].t_exit,
                1.0
                /
                (exp(vget(xb, corpus->docs[d].t_exit) -
                        ((r / vget(corpus->cmark, d)) * exp(exb)))
                )   ); //efron's method as in survfit4.c in R survival function
            exb = 0.0;
        }*/
    }
    vset(model->cbasehazard, 0, vget(model->basehazard, 0));
    for (r = 1; r < model->range_t; r++)
    {
        if (isnan(vget(model->basehazard,r) || vget(model->basehazard, r) < 1e-100))
        {
            //			std::cout << "Base haz set to 1e-100 because xb == " << xb[ss->times[d] - 1] << " and exb == " << exb << " so basehaz[time_index_exit] ==" << basehaz[time_index_exit] << std::endl;
            vset(model->basehazard, r, 1e-100); //log(basehaz) required so a minimum measureable hazard is required to avoid NaN errors.
        }
        vset(model->cbasehazard, r, vget(model->cbasehazard, r - 1) + vget(model->basehazard, r));
        if (isnan(vget(model->cbasehazard, r) || vget(model->cbasehazard, r) < 1e-100))
        {
            //			std::cout << "Base haz set to 1e-100 because xb == " << xb[ss->times[d] - 1] << " and exb == " << exb << " so basehaz[time_index_exit] ==" << basehaz[time_index_exit] << std::endl;
            vset(model->cbasehazard, r, 1e-100); //log(basehaz) required so a minimum measureable hazard is required to avoid NaN errors.
        }
    }
    gsl_vector_free(xb);;
    gsl_vector_free(zbeta);
};


double cstat(corpus* corpus, llna_model* model)
{
    int nd = corpus->ndocs;
    double num = 0.0, den = 0.0;
    gsl_vector* zbeta = gsl_vector_calloc(nd);
    gsl_vector_set_zero(zbeta);
    gsl_blas_dgemv(CblasNoTrans, 1, corpus->zbar, model->topic_beta, 0, zbeta);

#pragma omp parallel reduction(+:num,den) default(none) shared(zbeta, nd, corpus)
    {
        int dl, ddl;
        int size = omp_get_num_threads(); // get total number of processes
        int rank = omp_get_thread_num(); // get rank of current

        for (dl = (rank * nd / size); dl < (rank + 1) * nd / size; dl++)
        {
            if (corpus->docs[dl].label > 0)
            {
                for (ddl = dl + (int)  vget(corpus->cmark, dl); ddl < nd; ddl++)
                {
                    if (corpus->docs[dl].t_exit >= corpus->docs[ddl].t_enter)
                    {
                        den += 1.0;
                        if (vget(zbeta, dl) > vget(zbeta, ddl))
                            num += 1.0;
                        else if (vget(zbeta, dl) == vget(zbeta, ddl))
                            num += 0.5;
                    }


                }
            }
        }
    }
    gsl_vector_free(zbeta);
    return((double) num / (double) den);
}
/*
void permute_groups(corpus* corpus)
{
    //Allocate each person to a random subset for distributed cox regression
    int ngroups = omp_get_num_procs();
    gsl_rng* r = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(r, (unsigned long) time(NULL));
    gsl_vector* random = gsl_vector_calloc(corpus->ndocs);
    gsl_vector* permuted = gsl_vector_calloc(corpus->ndocs);
    for (int d = 0; d < corpus->ndocs; d++)
    {
        vset(random, d, gsl_rng_uniform(r));
        vset(permuted, d, d);
    }
    gsl_sort_vector2(random, permuted);
    int group_length = ceil((double)corpus->ndocs / (double) ngroups);
    gsl_matrix_set_all(corpus->group, corpus->ndocs);
    gsl_vector* temp = gsl_vector_calloc(group_length);
    int cumulative = 0;
    for (int g = 0;  g < ngroups; g++)
    {
        for (int d = 0; d < group_length; d++)
        {
            if (cumulative >= corpus->ndocs) continue;
            vset(temp, d, vget(permuted, cumulative));
            cumulative++;
        }
        gsl_sort_vector(temp);
        gsl_matrix_set_col(corpus->group, g, temp);
        for (int d = 0; d < group_length; d++)
            if (mget(corpus->group, d, g) >= corpus->ndocs) mset(corpus->group, d, g, -1);
    }
    gsl_rng_free(r);
    gsl_vector_free(temp);
    gsl_vector_free(random);
    gsl_vector_free(permuted);
}
*/

void em(char* dataset, int k, char* start, char* dir)
{
    FILE* lhood_fptr;
    char string[100];
    double convergence = 1, lhood = 0, lhood_old = 0;
    corpus* corpus;
    llna_model *model;
    llna_ss* ss;
    time_t t1,t2;
    double avg_niter, converged_pct;
    gsl_matrix *corpus_lambda, *corpus_nu, *corpus_phi_sum;

    // read the data and make the directory

    corpus = read_data(dataset);
    if (corpus == NULL)
    {
        printf("Unable to read data\n");
        return;
    }
    //mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);
    int chk = 1;
    if (mkd(dir) != 0)
    {
        if (errno == EEXIST)
        {
            int p = 0;
            char up[10];
            char* dirupdated;
            while (chk!=0 && errno == EEXIST)
            {
                sprintf(up, "%d", p);
                dirupdated = (char*)malloc(1 + strlen(dir) + strlen(up));
                strcpy(dirupdated, dir);
                strcat(dirupdated, up);
                printf("Directory already exists. Changing to %s\n", dirupdated);
                chk = mkd(dirupdated);
                p += 1;
             }
            dir = dirupdated;
        }
        if (errno == ENOENT)
        {
            printf("Directory path invalid. Please try again\n");
            return;
        }
    }
    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");
    write_params(lhood_fptr);
    // run em

    model = em_initial_model(k, corpus, start);
    if (model == NULL)
    {
        return;
    }
	printf("model initialised\n");
    ss = new_llna_ss(model);
	printf("New ss\t");

    //Working memory allocation done once at start - speed over memory efficiency
    corpus_lambda = gsl_matrix_alloc(corpus->ndocs, model->k);
    corpus_nu = gsl_matrix_alloc(corpus->ndocs, model->k);
    corpus_phi_sum = gsl_matrix_alloc(corpus->ndocs, model->k);
    


    //cox_reg_hes_intitialise(model->k - 1, model->range_t, model, sum_zbar, gdiag, hdiag, mean_z,
    //    scale_z, beta, newbeta, cumulxb, /*cumul2risk, cumulg2diag,
    //    cumulh2diag,*/ cumulrisk_start, cumulgdiag_start, cumulhdiag_start, cumulrisk_end,
    //    cumulgdiag_end, cumulhdiag_end, running_gdiag, running_hdiag, htemp, atemp,
    //    cumulxb_private,/* cumul2risk_private, cumulh2diag_private, cumulg2diag_private,*/ cumulrisk_start_private,
    //    cumulgdiag_start_private, cumulhdiag_start_private, cumulrisk_end_private, cumulgdiag_end_private, cumulhdiag_end_private);

	printf("gsl allocated\t");
    time(&t1);
    //init_temp_vectors((model->k)-1); // !!! hacky
    model->iteration = 0;
    sprintf(string, "%s/%03d", dir, model->iteration);
    write_llna_model(model, string);
    double C = 0.0;
    double newC = 0.0;
    double Change = 0.0;
    int reset_var = 1;
    int cox_iter = 0;


    do
    {
        if (convergence <= model->em_convergence && model->iteration > 0)
        {
            if (model->em_convergence > PARAMS.em_convergence)  model->em_convergence /= 10;
            if (model->var_convergence > PARAMS.var_convergence)  model->var_convergence /= 10;
            if (model->surv_convergence > PARAMS.surv_convergence)  model->surv_convergence /= 10;
            if (model->cg_convergence > PARAMS.cg_convergence)  model->cg_convergence /= 10;
        }

        printf("***** EM ITERATION %d *****\n", model->iteration);
        printf("***** Target convergence = %f\n", model->em_convergence);
        time(&t2);

        expectation(corpus, model, ss,  &avg_niter, &lhood,
                    corpus_lambda, corpus_nu, corpus_phi_sum,
                    reset_var, &converged_pct);


        printf("Expectation likelihood %5.5e \t ", lhood);
        printf("%3.0f percent documents converged, averaged %3.1f iterations.\n", converged_pct*100, avg_niter);
        convergence = (lhood_old - lhood) / lhood_old;
        //int base_index = 0;
        double f = 0.0; 
     //   gsl_vector_set_zero(model->topic_beta);


     //   cox_iter = cox_reg_hes(model, corpus, &f);

       cox_iter = cox_reg_dist(model, corpus, &f);
     //   cox_iter = cox_reg(model, corpus, &f);
     /*   while (cox_iter <= 0)
        {
            PARAMS.surv_penalty /= 10;
           cox_iter = cox_reg_dist(model, corpus, &f);
       //     cox_iter = cox_reg(model, corpus, &f);
        }*/
        printf("Cox liklihood %5.5e,  in %d iterations \t C statistic = %f\n", f, cox_iter, cstat(corpus, model));

   /*     if (cox_iter > PARAMS.surv_max_iter && PARAMS.surv_penalty>1e-6)
            PARAMS.surv_penalty /= 10;
        else if (cox_iter <= 5 && PARAMS.surv_penalty<1e6)
            PARAMS.surv_penalty *= 10;*/

        cumulative_basehazard(corpus, model);
        vprint(model->topic_beta);
        newC = cstat(corpus, model);

        if (convergence < 0 && PARAMS.runin!=model->iteration )
        {
            reset_var = 0; //retry using global lambda and mu for starting point for variational inference parameters if didn't converge from random start
        //    if (PARAMS.surv_penalty>1e-6) PARAMS.surv_penalty /= 10; //reduce magnitude of beta coefficients for next calculation to shrink extreme allocations
            if (PARAMS.var_max_iter > 0) PARAMS.var_max_iter += 10; // provide longer for convergence
            else model->var_convergence /= 10;  
        }
        else
        {
            printf("Maximisation....\n");
            reset_var = 1;
            maximization(model, ss);
            lhood_old = lhood;
            model->iteration++;
        }


        time(&t1);
        fprintf(lhood_fptr, "%d %5.5e %5.5e %5.5f %1.5f %1.5f\n",
            model->iteration, lhood, convergence, avg_niter, converged_pct, newC);
        Change = newC - C;
        C = newC;
        fflush(lhood_fptr);
        reset_llna_ss(ss);
    }
    while ((model->iteration <= PARAMS.runin + 1) || ( (model->iteration < PARAMS.em_max_iter) &&
           (fabs(convergence) > PARAMS.em_convergence) ));


    printf("Converged: \n Final likelihood %5.5e \t final C statistic = %f\n", lhood, cstat(corpus, model));

    sprintf(string, "%s/final", dir);
    write_llna_model(model, string);
    sprintf(string, "%s/final-lambda.dat", dir);
    printf_matrix(string, corpus_lambda);
    sprintf(string, "%s/final-nu.dat", dir);
    printf_matrix(string, corpus_nu);
    sprintf(string, "%s/final-zbar.dat", dir);
    printf_matrix(string, corpus->zbar);

    fclose(lhood_fptr);
    gsl_matrix_free(corpus_lambda);
    gsl_matrix_free(corpus_nu);
    gsl_matrix_free(corpus_phi_sum);
 
    return;
}


/*
 * load a model, and do approximate inference for each document in a corpus
 *
 */

void inference(char* dataset, char* model_root, char* out)
{
    char fname[100];
    // read the data and model
    corpus * corpus = read_data(dataset);
    llna_model * model = read_llna_model(model_root);
    if (model == NULL)
        return;
    int chk = 1;
    if (mkd(out) != 0)
    {
        if (errno == EEXIST)
        {
            int p = 0;
            char up[10];
            char* outupdated;
            while (chk != 0 && errno == EEXIST)
            {
                sprintf(up, "%d", p);
                outupdated = (char*)malloc(1 + strlen(out) + strlen(up));
                strcpy(outupdated, out);
                strcat(outupdated, up);
                printf("Directory already exists. Changing to %s\n", outupdated);
                chk = mkd(outupdated);
                p += 1;
            }
            out = outupdated;
        }
        if (errno == ENOENT)
        {
            printf("Directory path invalid. Please try again\n");
            return;
        }
    }

    gsl_vector * lhood = gsl_vector_calloc(corpus->ndocs);
    gsl_matrix * corpus_nu = gsl_matrix_calloc(corpus->ndocs, model->k);
    gsl_matrix * corpus_lambda = gsl_matrix_calloc(corpus->ndocs, model->k);
    corpus->zbar = gsl_matrix_calloc(corpus->ndocs, model->k);
    gsl_matrix * phi_sums = gsl_matrix_calloc(corpus->ndocs, model->k);
    corpus->zbar_scaled = gsl_matrix_calloc(corpus->ndocs, model->k);


    // approximate inference
   // init_temp_vectors(model->k-1); // !!! hacky
 //   sprintf(fname, "%s-word-assgn.dat", out);
 //   FILE* word_assignment_file = fopen(fname, "w");
#pragma omp parallel default(none) shared(corpus, model, corpus_lambda,  corpus_nu, phi_sums, lhood, PARAMS) /* for (i = 0; i < corpus->ndocs; i++) */
    {
        int i, j;
        int size = omp_get_num_threads(); // get total number of processes
        int rank = omp_get_thread_num(); // get rank of current
        for (i = (rank * corpus->ndocs / size); i < (rank + 1) * corpus->ndocs / size; i++)
        {
            doc doc = corpus->docs[i];
            llna_var_param* var = new_llna_var_param(doc.nterms, model->k);
            init_var_unif(var, &doc, model);

            vset(lhood, i, var_inference(var, &doc, model));
            gsl_matrix_set_row(corpus_lambda, i, var->lambda);
            gsl_matrix_set_row(corpus_nu, i, var->nu);
            gsl_vector curr_row = gsl_matrix_row(phi_sums, i).vector;
            col_sum(var->phi, &curr_row);
            // Allocated topics for survival supervision
            for (int n = 0; n < doc.nterms; n++)
                for (j = 0; j < model->k; j++)
                    minc(corpus->zbar, i, j, mget(var->phi, n, j) * (double)doc.count[n] / (double)doc.total);
        //    write_word_assignment(word_assignment_file, &doc, var->phi);

        //    printf("document %05d, niter = %05d\n", i, var->niter);
            free_llna_var_param(var);
        }
    }

    cumulative_basehazard(corpus, model);
    printf("\n C statistic = %1.3f\n", cstat(corpus, model));

    // output likelihood and some variational parameters
    sprintf(fname, "%s/inf-ctm-lhood.dat", out);
    printf_vector(fname, lhood);
    sprintf(fname, "%s/inf-lambda.dat", out);
    printf_matrix(fname, corpus_lambda);
    sprintf(fname, "%s/inf-nu.dat", out);
    printf_matrix(fname, corpus_nu);
    sprintf(fname, "%s/inf-sum.dat", out);
    printf_matrix(fname, phi_sums);
    sprintf(fname, "%s/inf-zbar.dat", out);
    printf_matrix(fname, corpus->zbar);

}


/*
 * split documents into two random parts
 *
 */

void within_doc_split(char* dataset, char* src_data, char* dest_data, double prop)
{
    int i;
    corpus * corp, * dest_corp;

    corp = read_data(dataset);
    dest_corp = malloc(sizeof(corpus));
    if (dest_corp != NULL)
    {
        printf("splitting %d docs\n", corp->ndocs);
        dest_corp->docs = malloc(sizeof(doc) * corp->ndocs);
        if (dest_corp->docs != NULL)
        {
            dest_corp->nterms = corp->nterms;
            dest_corp->ndocs = corp->ndocs;
            for (i = 0; i < corp->ndocs; i++)
                split(&(corp->docs[i]), &(dest_corp->docs[i]), prop);
            write_corpus(dest_corp, dest_data);
            write_corpus(corp, src_data);
        }
    }
}


/*
 * for each partially observed document: (a) perform inference on the
 * observations (b) take expected theta and compute likelihood
 *
 */

int pod_experiment(char* observed_data, char* heldout_data,
                   char* model_root, char* out)
{
    corpus *obs, *heldout;
    llna_model *model;
    llna_var_param *var;
    int i;
    gsl_vector *log_lhood, *e_theta;
    doc obs_doc, heldout_doc;
    char string[100];
    double total_lhood = 0, total_words = 0, l;
    FILE* e_theta_file = fopen("/Users/blei/llna050_e_theta.txt", "w");

    // load model and data
    obs = read_data(observed_data);
    heldout = read_data(heldout_data);
    assert(obs->ndocs == heldout->ndocs);
    model = read_llna_model(model_root);

    // run experiment
   // init_temp_vectors(model->k-1); // !!! hacky
    log_lhood = gsl_vector_alloc((size_t)obs->ndocs + 1);
    e_theta = gsl_vector_alloc(model->k);
    for (i = 0; i < obs->ndocs; i++)
    {
        // get observed and heldout documents
        obs_doc = obs->docs[i];
        heldout_doc = heldout->docs[i];
        // compute variational distribution
        var = new_llna_var_param(obs_doc.nterms, model->k);
        init_var_unif(var, &obs_doc, model);
        var_inference(var, &obs_doc, model);
        expected_theta(var, &obs_doc, model, e_theta);

        vfprint(e_theta, e_theta_file);

        // approximate inference of held out data
        l = log_mult_prob(&heldout_doc, e_theta, model->log_omega);
        vset(log_lhood, i, l);
        total_words += heldout_doc.total;
        total_lhood += l;
        printf("hid doc %d    log_lhood %5.5f\n", i, vget(log_lhood, i));
        // save results?
        free_llna_var_param(var);
    }
    vset(log_lhood, obs->ndocs, exp(-total_lhood/total_words));
    printf("perplexity : %5.10f", exp(-total_lhood/total_words));
    sprintf(string, "%s-pod-llna.dat", out);
    printf_vector(string, log_lhood);
    return(0);
}


/*
 * little function to count the words in each document and spit it out
 *
 */

void count(char* corpus_name, char* output_name)
{
    corpus *c;
    int i;
    FILE *f;
    int j;
    f = fopen(output_name, "w");
    c = read_data(corpus_name);
    for (i = 0; i < c->ndocs; i++)
    {
        j = c->docs[i].total;
        fprintf(f, "%5d\n", j);
    }
}

/*
 * main function
 *
 */

int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        if (strcmp(argv[1], "est")==0)
        {
            read_params(argv[6]);
            print_params();
            em(argv[2], atoi(argv[3]), argv[4], argv[5]);
            return(1);
        }
        if (strcmp(argv[1], "inf")==0)
        {
            read_params(argv[5]);
            print_params();
            inference(argv[2], argv[3], argv[4]);
            return(1);
        }
    }
    printf("usage : ctm est <dataset> <# topics> <rand/seed/model> <dir> <settings>\n");
    printf("        ctm inf <dataset> <model-prefix> <results-prefix> <settings>\n");
    return(0);
}
