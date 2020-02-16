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

#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <assert.h>
#include <stdio.h>
#include "gsl-wrappers.h"
#include "corpus.h"
#include "ctm.h"
#include "params.h"
#include "inference.h"


#ifdef main
#undef main
#endif



extern llna_params PARAMS;

double f_lambda(const gsl_vector * p, void * params);
void df_lambda(const gsl_vector * p, void * params, gsl_vector * df);
void fdf_lambda(const gsl_vector * p, void * params, double * f, gsl_vector * df);

/*
 * likelihood bound
 *
 */

double expect_mult_norm(llna_var_param * var)
{
    int i;
    double sum_exp = 0;
    int niter = (int) var->lambda->size;

    for (i = 0; i < niter; i++)
        sum_exp += exp(vget(var->lambda, i) + (0.5) * vget(var->nu,i));

    return((1.0/var->zeta) * sum_exp - 1.0 + safe_log(var->zeta));
}
/*
void lhood_bnd_old(llna_var_param* var, doc* doc, llna_model* mod)
{
    int i = 0, j = 0, k = mod->k;
    gsl_vector_set_zero(var->topic_scores);

    // E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu)

    double lhood = (0.5) * mod->log_det_inv_cov + (0.5) * (mod->k - 1);
    for (i = 0; i < k - 1; i++)
    {
        double v = -(0.5) * vget(var->nu, i) * mget(mod->inv_cov, i, i);
        for (j = 0; j < mod->k - 1; j++)
        {
            v -= (0.5) *
                (vget(var->lambda, i) - vget(mod->mu, i)) *
                mget(mod->inv_cov, i, j) *
                (vget(var->lambda, j) - vget(mod->mu, j));
        }
        v += (0.5) * log(vget(var->nu, i));
        lhood += v;
    }

    // E[log p(z_n | \eta)] + E[log p(w_n | \omega)] + H(q(z_n | \phi_n))

    lhood -= expect_mult_norm(var) * doc->total;
    for (i = 0; i < doc->nterms; i++)
    {
        // !!! we can speed this up by turning it into a dot product
        // !!! profiler says this is where some time is spent
        for (j = 0; j < mod->k; j++)
        {
            double phi_ij = mget(var->phi, i, j);
            double log_phi_ij = mget(var->log_phi, i, j);
            if (phi_ij > 0)
            {
                vinc(var->topic_scores, j, phi_ij * doc->count[i]);
                lhood +=
                    doc->count[i] * phi_ij *
                    (vget(var->lambda, j) +
                        mget(mod->log_omega, j, doc->word[i]) -
                        log_phi_ij);
            }
        }
    }
    var->lhood = lhood;
    assert(!isnan(var->lhood));
}
*/

void lhood_bnd(llna_var_param* var, doc* doc, llna_model* mod)
{
    int n = 0, i = 0, k = mod->k;
//    gsl_vector_set_zero(var->topic_scores);
    // E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu) NB log 2pi cancels out in likelihood
    double lhood = (0.5) * mod->log_det_inv_cov + (0.5)  * ((double) k - 1.0);
    double v = 0.0;
    gsl_vector_const_view inv_cov_diag = gsl_matrix_const_diagonal(mod->inv_cov);
    gsl_vector_const_view nu = gsl_vector_const_subvector(var->nu, 0, (size_t)k - 1);
    gsl_blas_ddot(&nu.vector, &inv_cov_diag.vector, &v);
    lhood -= 0.5 * v;

    // compute lambda - mu (= temp1)
    gsl_vector_const_view lambda = gsl_vector_const_subvector(var->lambda, 0, (size_t)k - 1);
    gsl_vector_const_view mu = gsl_vector_const_subvector(mod->mu, 0, (size_t)k - 1);
    gsl_blas_dcopy(&lambda.vector, var->tempvector[1]);
    gsl_blas_daxpy(-1.0, &mu.vector, var->tempvector[1]);
    // compute (lambda - mu)^T Sigma^-1 (lambda - mu)
    gsl_blas_dsymv(CblasUpper, 1, mod->inv_cov, var->tempvector[1], 0, var->tempvector[2]);
    // gsl_blas_dgemv(CblasNoTrans, 1, mod->inv_cov, temp[1], 0, temp[2]);
    v = 0.0;
    gsl_blas_ddot(var->tempvector[1], var->tempvector[2], &v);
    lhood -= 0.5 * v;

    for (i = 0; i < k - 1; i++)
        lhood += (0.5) * safe_log(vget(var->nu, i));

    // E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu)

    // E[log p(z_n | \eta)] + E[log p(w_n | \omega)] + H(q(z_n | \phi_n))
    
    //Loop is needed because phi is a representation of a sparse matrix for only the observed words
    for (n = 0; n < doc->nterms; n++)
    {
        gsl_vector_const_view nphi = gsl_matrix_const_row(var->phi, n);
        gsl_vector_view nlogphi = gsl_matrix_row(var->log_phi, n);
        gsl_vector_const_view logomega = gsl_matrix_const_column(mod->log_omega, doc->word[n]);
        
 //       gsl_blas_daxpy(doc->count[n], &nphi.vector, var->topic_scores);
        gsl_blas_daxpy(-1.0, var->lambda, &nlogphi.vector);
        gsl_blas_daxpy(-1.0, &logomega.vector, &nlogphi.vector);
        gsl_vector_scale(&nlogphi.vector, -1.0);
        v = 0.0;
        gsl_blas_ddot(&nphi.vector, &nlogphi.vector, &v);
        lhood += (double)doc->count[n] * v;

    }
    var->lhood = lhood;
    assert(!isnan(var->lhood));
}

void lhood_bnd_surv(llna_var_param* var, doc* doc, llna_model* mod)
{
 //   gsl_vector_set_zero(var->topic_scores);

    // E[log p(z_n | \eta)] + E[log p(w_n | \omega)] + H(q(z_n | \phi_n))
    double cbhz_prod = 1.0;
    double lhood = 0.0;

    for (int n = 0; n < doc->nterms; n++)
    {
        gsl_vector_const_view nphi = gsl_matrix_const_row(var->phi, n);
        gsl_vector_const_view cbhz_params = gsl_matrix_const_row(var->cbhz_params_matrix, n);
        gsl_vector_const_view scaledbeta = gsl_matrix_const_row(var->scaledbetamatrix, n);
        double temp = 0.0;
        gsl_blas_ddot(&nphi.vector, &cbhz_params.vector, &temp);
     //   temp += mod->intercept;
        //for (i = 0; i < k; i++)
         //   temp += vget(&nphi.vector, i) * exp(vget(scaledbeta,i));
        cbhz_prod *= temp;
        if (doc->label > 0)
        {
            gsl_blas_ddot(&nphi.vector, &scaledbeta.vector, &lhood);
         //   lhood += mod->intercept;
        }
        /*for (i = 0; i < k; i++)
        {
            double phi_ij = mget(var->phi, n, i);
            double log_phi_ij = mget(var->log_phi, n, i);
            if (phi_ij > 0)
            {
                vinc(var->topic_scores, i, phi_ij * (double) doc->count[n]);
                lhood += ((double) doc->label * phi_ij * vget(mod->topic_beta, i) * (double) doc->count[n] / (double) doc->total)
                    + ((double) doc->count[n] * phi_ij * (vget(var->lambda, i) + mget(mod->log_omega, i, doc->word[n]) - log_phi_ij));
                for (i = 0; i < k; i++)
                    temp += vget(&nphi.vector, i) * exp(vget(scaledbeta, i));
                cbhz_prod *= temp;
            }
            
        }*/
    }

    lhood -= cbhz_prod * vget(mod->cbasehazard, doc->t_exit) * exp(mod->intercept);
    if (doc->label > 0 && doc->t_exit < mod->range_t - 1)
        lhood += safe_log(vget(mod->basehazard, doc->t_exit)) + mod->intercept;

    lhood_bnd(var, doc, mod);
    var->lhood += lhood;
    assert(!isnan(var->lhood));
}




/**
 * optimize zeta
 *
 */

int opt_zeta(llna_var_param * var, doc * doc, llna_model * mod)
{
    int i;

    var->zeta = 1.0;
    for (i = 0; i < mod->k-1; i++)
        var->zeta += exp(vget(var->lambda, i) + (0.5) * vget(var->nu, i));
    assert(var->zeta != 0 || !isnan(var->zeta));
    return(0);
}


/**
 * optimize phi
 *
 */

void opt_phi(llna_var_param * var, doc * doc, llna_model * mod)
{
    int i, n, K = mod->k;
    double log_sum_n = 0;

    // compute phi proportions in log space
 

    for (n = 0; n < doc->nterms; n++)
    {  
        gsl_vector_view nlogphi = gsl_matrix_row(var->log_phi, n);
        gsl_vector_view nphi = gsl_matrix_row(var->phi, n);
        gsl_vector_const_view nlogomega = gsl_matrix_const_column(mod->log_omega, doc->word[n]);
        gsl_blas_dcopy(&nlogomega.vector , &nlogphi.vector); //use log phi for temporary vector as and phi for phi calculatations
        gsl_blas_daxpy(1.0, var->lambda, &nlogphi.vector);
        log_sum_n = vget(&nlogphi.vector, 0);
        for (i = 1; i < K; i++)
        {
            log_sum_n = log_sum(log_sum_n, vget(&nlogphi.vector, i));
           // double expphi= exp(vget(&nlogphi.vector, i));
           // log_sum_n += expphi;
        }
        //gsl_vector_(&nphi.vector, 1.0 /log_sum_n);
        //log_sum_n = safe_log(log_sum_n);
        //        
        gsl_vector_add_constant(&nlogphi.vector, -log_sum_n);
        for (i = 0; i < K; i++)
        {
            vset(&nphi.vector, i, exp(vget(&nlogphi.vector, i)));
            assert(!isnan(vget(&nphi.vector,i)));
        }
    }
}

void opt_phi_surv(llna_var_param* var, doc* doc, llna_model* mod)
{
    int i, n, K = mod->k;
    double log_sum_n = 0, temp = 0;
    double label = (double) doc->label;
    double cbhz_prod = 1.0; // vget(mod->cbasehazard, doc->t_exit); Doesn't effect phi convergence on a document level
    for (n = 0; n < doc->nterms; n++)
    {
        gsl_vector_const_view nphi = gsl_matrix_const_row(var->phi, n);
        gsl_vector_const_view cbhz_params = gsl_matrix_const_row(var->cbhz_params_matrix, n);
        gsl_blas_ddot(&nphi.vector, &cbhz_params.vector, &temp);
    //    temp += mod->intercept;
        assert(!isnan(temp) && !isinf(temp));
        cbhz_prod *= temp;
    }

    //double dif = 0.0;
    // compute phi proportions in log space
   
        for (n = 0; n < doc->nterms; n++)
        {
            gsl_vector_const_view cbhz_params = gsl_matrix_const_row(var->cbhz_params_matrix, n);
            gsl_vector_const_view scaledbeta = gsl_matrix_const_row(var->scaledbetamatrix, n);
            gsl_vector_view nphi = gsl_matrix_row(var->phi, n);
            gsl_vector_view nlogphi = gsl_matrix_row(var->log_phi, n);
            gsl_blas_ddot(&nphi.vector, &cbhz_params.vector, &temp);
            cbhz_prod /= temp; //remove the contribution of word n
            assert(!isnan(cbhz_prod) && !isinf(cbhz_prod));

            gsl_vector_const_view nlogomega = gsl_matrix_const_column(mod->log_omega, doc->word[n]);
            gsl_blas_dcopy(&nlogomega.vector, &nlogphi.vector);
            gsl_blas_daxpy(1.0, var->lambda, &nlogphi.vector);
            gsl_blas_daxpy(label, &scaledbeta.vector, &nlogphi.vector);
            gsl_blas_daxpy(-cbhz_prod, &cbhz_params.vector, &nlogphi.vector);

            log_sum_n = vget(&nlogphi.vector, 0);
            for (i = 1; i < K; i++)
            {
                log_sum_n = log_sum(log_sum_n, vget(&nlogphi.vector, i));
                assert(!isnan(log_sum_n) && !isinf(log_sum_n));
            }
            gsl_vector_add_constant(&nlogphi.vector, -log_sum_n);
            for (i = 0; i < K; i++)
            {
                //vset(var->log_phi, n, i, mget(var->log_phi, n, i) - log_sum_n);
                vset(&nphi.vector, i, exp(vget(&nlogphi.vector, i)));
                assert(!isnan(vget(&nlogphi.vector, i)));
            }
            gsl_blas_ddot(&nphi.vector, &cbhz_params.vector, &temp);
         //   temp += mod->intercept;

            assert(!isnan(temp) && !isinf(temp));
            cbhz_prod *= temp;
            assert(!isnan(cbhz_prod) && !isinf(cbhz_prod));
        }
    
}


/**
 * optimize lambda
 *
 */

void fdf_lambda(const gsl_vector * p, void * params, double * f, gsl_vector * df)
{
   // *f = f_lambda(p, params);
   // df_lambda(p, params, df);

    double term1, term2, term3;
    int i;
    llna_var_param* var = ((bundle*)params)->var;
    doc* doc = ((bundle*)params)->doc;
    llna_model* mod = ((bundle*)params)->mod;
   // gsl_vector* sum_phi = ((bundle*)params)->sum_phi;
    gsl_vector_view sum_phi = gsl_vector_subvector(var->sum_phi, 0, mod->k - 1);

    gsl_vector_set_zero(var->tempvector[0]);

    // compute lambda^T \sum phi = term 1
    //gsl_blas_ddot(p, ((bundle*)params)->sum_phi, &term1);
    gsl_blas_ddot(p, &sum_phi.vector, &term1);
    //  lambda (= temp1)
    gsl_blas_dcopy(p, var->tempvector[1]);
    // (\lambda -\mu) = temp1
    gsl_blas_daxpy(-1.0, mod->mu, var->tempvector[1]);

    // compute \Sigma^{-1} (\lambda - \mu) = temp0
    gsl_blas_dsymv(CblasUpper, 1, mod->inv_cov, var->tempvector[1], 0, var->tempvector[0]);
    // compute - (N / \zeta) * exp(\lambda + \nu^2 / 2) = temp3
    // last term in f_lambda
    term3 = 0;
    gsl_vector_const_view nu = gsl_vector_const_subvector(var->nu, 0, (size_t)(mod->k) - 1);
    gsl_blas_dcopy(p, var->tempvector[2]);
    gsl_blas_daxpy(0.5, &nu.vector, var->tempvector[2]);
    for (i = 0; i < (mod->k) - 1; i++)
    {
        vset(var->tempvector[2], i, exp(vget(var->tempvector[2], i)));
        term3 += vget(var->tempvector[2],i);
    }
    gsl_blas_daxpy(-((double)doc->total / var->zeta), var->tempvector[2], var->tempvector[3]);
    term3 = -((1.0 / var->zeta) * term3 - 1.0 + safe_log(var->zeta)) * (double) doc->total;

    gsl_vector_set_all(df, 0.0);
    gsl_vector_add(df, var->tempvector[0]);
    gsl_vector_sub(df, &sum_phi.vector);
    gsl_vector_sub(df, var->tempvector[3]);

    // compute (lambda - mu)^T Sigma^-1 (lambda - mu) = term 2
    gsl_blas_ddot(var->tempvector[1], var->tempvector[0], &term2);
    term2 = -0.5 * term2;

    *f = (-(term1 + term2 + term3));
    // negate for minimization
    return;
}


// set return value (note negating derivative of bound)



double f_lambda(const gsl_vector * p, void * params)
{
    double term1, term2, term3;
    int i;
    llna_var_param * var = ((bundle *) params)->var;
    doc * doc = ((bundle *) params)->doc;
    llna_model * mod = ((bundle *) params)->mod;
    gsl_vector_view sum_phi = gsl_vector_subvector(var->sum_phi, 0, mod->k - 1);
    // compute lambda^T \sum phi
    //gsl_blas_ddot(p,((bundle *) params)->sum_phi, &term1);
    gsl_blas_ddot(p, &sum_phi.vector, &term1);
    assert(!isnan(term1));

    // compute lambda - mu (= temp1)
    gsl_blas_dcopy(p, var->tempvector[1]);
    gsl_blas_daxpy (-1.0, mod->mu, var->tempvector[1]);
    // compute (lambda - mu)^T Sigma^-1 (lambda - mu)
    gsl_blas_dsymv(CblasUpper, 1, mod->inv_cov, var->tempvector[1], 0, var->tempvector[2]);
    // gsl_blas_dgemv(CblasNoTrans, 1, mod->inv_cov, temp[1], 0, temp[2]);
    gsl_blas_ddot(var->tempvector[2], var->tempvector[1], &term2);
    term2 = - 0.5 * term2;
    assert(!isnan(term2));
    // last term
    term3 = 0;
    gsl_vector_const_view nu = gsl_vector_const_subvector(var->nu, 0, (size_t)(mod->k) - 1);
    gsl_blas_dcopy(p, var->tempvector[3]);
    gsl_blas_daxpy(0.5, &nu.vector, var->tempvector[3]);
    for (i = 0; i < (mod->k) - 1; i++)
         term3 += exp(vget(var->tempvector[3], i));
    assert(!isnan(term3));
    term3 = -((1.0 / var->zeta) * term3 - 1.0 + safe_log(var->zeta)) * (double)doc->total;
    assert(!isnan(term3));
    // negate for minimization
    return(-(term1+term2+term3));
}


void df_lambda(const gsl_vector * p, void * params, gsl_vector * df)
{
    // cast bundle {variational parameters, model, document}

    llna_var_param * var = ((bundle *) params)->var;
    doc * doc = ((bundle *) params)->doc;
    llna_model * mod = ((bundle *) params)->mod;
    //gsl_vector * sum_phi = ((bundle *) params)->sum_phi;
    gsl_vector_view sum_phi = gsl_vector_subvector(var->sum_phi, 0, mod->k - 1);

    // compute \Sigma^{-1} (\mu - \lambda)

    gsl_vector_set_zero(var->tempvector[0]);
    gsl_blas_dcopy(mod->mu, var->tempvector[1]);
    gsl_vector_sub(var->tempvector[1], p);
    gsl_blas_dsymv(CblasLower, 1, mod->inv_cov, var->tempvector[1], 0, var->tempvector[0]);

    // compute - (N / \zeta) * exp(\lambda + \nu^2 / 2)
    gsl_vector_const_view nu = gsl_vector_const_subvector(var->nu, 0, (size_t)(mod->k) - 1);
    gsl_vector_memcpy(var->tempvector[2], p);
    gsl_blas_daxpy(0.5, &nu.vector, var->tempvector[2]);
    for (int i = 0; i < (mod->k) - 1; i++)
        vset(var->tempvector[2], i, exp(vget(var->tempvector[2], i)));

    gsl_blas_daxpy(-((double)doc->total / var->zeta), var->tempvector[2], var->tempvector[3]);

    // set return value (note negating derivative of bound)

    gsl_vector_set_all(df, 0.0);
    gsl_vector_sub(df, var->tempvector[0]);
    gsl_vector_sub(df, &sum_phi.vector);
    gsl_vector_sub(df, var->tempvector[3]);
}


int opt_lambda(llna_var_param * var, doc * doc, llna_model * mod)
{
    gsl_multimin_function_fdf lambda_obj;
    const gsl_multimin_fdfminimizer_type * T;
    gsl_multimin_fdfminimizer * s;
    bundle b;
    int iter = 0, i;
    int status;

    b.var = var;
    b.doc = doc;
    b.mod = mod;

    // precompute \sum_n \phi_n and put it in the bundle
    int k = mod->k;
    
    gsl_vector_set_zero(var->sum_phi);
    for (i = 0; i < doc->nterms; i++)
    {
        gsl_vector_const_view iphi = gsl_matrix_const_row(var->phi, i);
        gsl_blas_daxpy((double)doc->count[i], &iphi.vector, var->sum_phi);
    }

    lambda_obj.f = &f_lambda;
    lambda_obj.df = &df_lambda;
    lambda_obj.fdf = &fdf_lambda;
    lambda_obj.n = (size_t)k - 1;
    lambda_obj.params = (void *)&b;

    // starting value
    // T = gsl_multimin_fdfminimizer_vector_bfgs;
    T = gsl_multimin_fdfminimizer_conjugate_fr;
    // T = gsl_multimin_fdfminimizer_steepest_descent;
    s = gsl_multimin_fdfminimizer_alloc (T, (size_t)(mod->k) - 1);
    
    gsl_vector_view lambda = gsl_vector_subvector(var->lambda, 0, k - 1);
    gsl_blas_dcopy(&lambda.vector, var->tempvector[4]);
    gsl_multimin_fdfminimizer_set (s, &lambda_obj, var->tempvector[4], 0.1, 0.01);
    do
    {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate (s);
        // double converged = fabs((f_old - s->f) / f_old);
        // printf("f(lambda) = %5.17e ; conv = %5.17e\n", s->f, converged);
        if (status) break;
        status = gsl_multimin_test_gradient (s->gradient, PARAMS.cg_convergence);
    }
    while ((status == GSL_CONTINUE) &&
           ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    // while ((converged > PARAMS.cg_convergence) &&
    // ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
  //  if (iter == PARAMS.cg_max_iter)
   //     printf("warning: cg didn't converge (lambda)\n");
    for (i = 0; i < mod->k - 1; i++)
    {
        vset(var->lambda, i, vget(s->x, i));
        assert(!isnan(vget(var->lambda,i)));
    }
    vset(var->lambda, mod->k - 1, 0);

    gsl_multimin_fdfminimizer_free(s);
    return(0);
}

/**
 * optimize nu
 *
 */

double f_nu_i(double nu_i, int i, llna_var_param * var,
              llna_model * mod, doc * d)
{
    double v;

    v = - (nu_i * mget(mod->inv_cov, i, i) * 0.5)
        - (((double) d->total/var->zeta) * exp(vget(var->lambda, i) + nu_i/2))
        + (0.5 * safe_log(nu_i));

    return(v);
}


double df_nu_i(double nu_i, int i, llna_var_param * var,
               llna_model * mod, doc * d)
{
    double v;

    v = - (mget(mod->inv_cov, i, i) * 0.5)
        - (0.5 * ((double) d->total/var->zeta) * exp(vget(var->lambda, i) + nu_i/2))
        + (0.5 * (1.0 / nu_i));

    return(v);
}


double d2f_nu_i(double nu_i, int i, llna_var_param * var, llna_model * mod, doc * d)
{
    double v;

    v = - (0.25 * ((double) d->total/var->zeta) * exp(vget(var->lambda, i) + nu_i/2))
        - (0.5 * (1.0 / (nu_i * nu_i)));

    return(v);
}


void opt_nu(llna_var_param * var, doc * d, llna_model * mod)
{
    int i;

    // !!! here i changed to k-1
    for (i = 0; i < mod->k-1; i++)
        opt_nu_i(i, var, mod, d);
}


double fixed_point_iter_i(int i, llna_var_param * var, llna_model * mod, doc * d)
{
    double v;
    double lambda = vget(var->lambda, i);
    double nu = vget(var->nu, i);
    double c = ((double) d->total / var->zeta);

    v = mget(mod->inv_cov,i,i) + c * exp(lambda + nu/2);

    return(v);
}


void opt_nu_i(int i, llna_var_param * var, llna_model * mod, doc * d)
{
    double init_nu = 10;
    double nu_i = 0, log_nu_i = 0, df = 0, d2f = 0;
    int iter = 0;

    log_nu_i = safe_log(init_nu);
    do
    {
        iter++;
        nu_i = exp(log_nu_i);
        // assert(!isnan(nu_i));
        if (isnan(nu_i))
        {
            init_nu = init_nu*2;
            printf("warning : nu is nan; new init = %5.5f\n", init_nu);
            log_nu_i = safe_log(init_nu);
            nu_i = init_nu;
        }
        //double f = f_nu_i(nu_i, i, var, mod, d);
       //  printf("%5.5f  %5.5f \n", nu_i, f);
        df = df_nu_i(nu_i, i, var, mod, d);
        d2f = d2f_nu_i(nu_i, i, var, mod, d);
        log_nu_i = log_nu_i - (df*nu_i)/(d2f*nu_i*nu_i+df*nu_i);
    }
    while (fabs(df) > NEWTON_THRESH);
    vset(var->nu, i, exp(log_nu_i));
    assert(!isnan(vget(var->nu,i)));
}

/**
 * initial variational parameters
 *
 */

void init_var_unif(llna_var_param * var, doc * doc, llna_model * mod)
{
    int i;
    gsl_matrix_view phi = gsl_matrix_submatrix(var->phi, 0, 0, doc->nterms, mod->k);
    gsl_matrix_view log_phi = gsl_matrix_submatrix(var->log_phi, 0, 0, doc->nterms, mod->k);
    gsl_matrix_set_all(&phi.matrix, 1.0/mod->k);
    gsl_matrix_set_all(&log_phi.matrix, log(1.0 / (double) mod->k));

    var->zeta = 10;
    for (i = 0; i < mod->k-1; i++)
    {
        vset(var->nu, i, 10.0);
        vset(var->lambda, i, 0);
    }
    vset(var->nu, i, 0);
    vset(var->lambda, i, 0);
    var->niter = 0;
    var->lhood = 0;
    for (i = 0; i< 5; i++)
        gsl_vector_set_zero(var->tempvector[i]);
    for (int n = 0; n < doc->nterms; n++)
    {
        gsl_vector_view scaledbeta = gsl_matrix_row(var->scaledbetamatrix, n);
        gsl_vector_view cbhz_params = gsl_matrix_row(var->cbhz_params_matrix, n);
        gsl_blas_dcopy(mod->topic_beta, &scaledbeta.vector);
        gsl_vector_scale(&scaledbeta.vector, (double)doc->count[n] / (double)doc->total);
        for (int i = 0; i < mod->k; i++)
        {
            vset(&cbhz_params.vector, i, exp(vget(&scaledbeta.vector, i)));
#pragma omp critical
            assert(!isnan(vget(&cbhz_params.vector, i) && !isinf(vget(&cbhz_params.vector, i))));
        }
    }
}


void init_var(llna_var_param * var, doc * doc, llna_model * mod, gsl_vector *lambda, gsl_vector *nu)
{
    gsl_vector_memcpy(var->lambda, lambda);
    gsl_vector_memcpy(var->nu, nu);
    opt_zeta(var, doc, mod);
    opt_phi(var, doc, mod);

    var->niter = 0;
    for (int n = 0; n < doc->nterms; n++)
    {
        gsl_vector_view scaledbeta = gsl_matrix_row(var->scaledbetamatrix, n);
        gsl_vector_view cbhz_params = gsl_matrix_row(var->cbhz_params_matrix, n);
        gsl_blas_dcopy(mod->topic_beta, &scaledbeta.vector);
        gsl_vector_scale(&scaledbeta.vector, (double)doc->count[n] / (double)doc->total);
        for (int i = 0; i < mod->k; i++)
        {
            vset(&cbhz_params.vector, i, exp(vget(&scaledbeta.vector, i)));
#pragma omp critical
            assert(!isnan(vget(&cbhz_params.vector, i) && !isinf(vget(&cbhz_params.vector, i))));

        }
    }
}




/**
 *
 * variational inference
 *
 */

llna_var_param * new_llna_var_param(int nterms, int k)
{
    llna_var_param * ret = malloc(sizeof(llna_var_param));
    if (ret != NULL)
    {
        int ntemp = 5;
        ret->lambda = gsl_vector_calloc(k);
        ret->nu = gsl_vector_calloc(k);
        ret->phi = gsl_matrix_calloc(nterms, k);
        ret->log_phi = gsl_matrix_calloc(nterms, k);
        ret->sum_phi = gsl_vector_calloc((size_t)k);
        ret->zeta = 0;
     //   ret->topic_scores = gsl_vector_calloc(k);
        ret->tempvector = malloc(sizeof(gsl_vector*) * ntemp);
        if (ret->tempvector != NULL && ntemp > 0)
        {
            for (int i = 0; i < ntemp; i++)
                ret->tempvector[i] = gsl_vector_calloc((size_t)k - 1);
        }
        ret->cbhz_params_matrix = gsl_matrix_calloc(nterms, k); //for temporary working 
        ret->scaledbetamatrix = gsl_matrix_calloc(nterms, k); //for temporary working 

        return(ret);
    }
    else
    {
        printf("Out of memory\n");
        return NULL;
    }
}


void free_llna_var_param(llna_var_param * v)
{
    int ntemp = 5;
    gsl_vector_free(v->lambda);
    gsl_vector_free(v->nu);
    gsl_matrix_free(v->phi);
    gsl_matrix_free(v->log_phi);
    gsl_vector_free(v->sum_phi);
 //   gsl_vector_free(v->topic_scores);
    for (int i = 0; i < ntemp; i++)
        free(v->tempvector[i]);
    gsl_matrix_free(v->scaledbetamatrix);
    gsl_matrix_free(v->cbhz_params_matrix);
    free(v);
}


double var_inference(llna_var_param * var,
                     doc * doc,
                     llna_model * mod)
{
    double lhood_old = 0;
    double convergence;

    do
    {
        opt_zeta(var, doc, mod);
        opt_lambda(var, doc, mod);
        opt_zeta(var, doc, mod);
        opt_nu(var, doc, mod);
        opt_zeta(var, doc, mod);
        if (mod->iteration >= PARAMS.runin)
        {
            opt_phi_surv(var, doc, mod);
            lhood_old = var->lhood;
            lhood_bnd_surv(var, doc, mod);
        }
        else
        {
            opt_phi(var, doc, mod);
            lhood_old = var->lhood;
            lhood_bnd(var, doc, mod);
        }

        convergence = fabs((lhood_old - var->lhood) / lhood_old);
        // printf("lhood = %8.6f (%7.6f)\n", var->lhood, convergence);
        var->niter++;

      /*  if ((lhood_old > var->lhood) && (var->niter != PARAMS.runin+1 && var->niter > 1 ))
            printf("WARNING: iter %05d %5.5f > %5.5f\n",
                   var->niter, lhood_old, var->lhood); */
    }
    while ((convergence > mod->var_convergence) &&
           ((PARAMS.var_max_iter < 0) || (var->niter < PARAMS.var_max_iter)));

    if (convergence > mod->var_convergence) var->converged = 0;
    else var->converged = 1;

    return(var->lhood);
}


void update_expected_ss(llna_var_param* var, doc* d, llna_ss* ss)
{
    int i, j, w, c;
    double lilj;

    // covariance and mean suff stats
    for (i = 0; i < ss->cov_ss->size1; i++)
    {
#pragma omp atomic update
        ss->mu_ss->data[i] += var->lambda->data[i];
       // vinc(ss->mu_ss, i, vget(var->lambda, i));
        for (j = 0; j < ss->cov_ss->size2; j++)
        {
            lilj = vget(var->lambda, i) * vget(var->lambda, j);
            if (i == j)
            {
#pragma omp atomic update 
                ss->cov_ss->data[(i * ss->cov_ss->tda) + j] += var->nu->data[i] + lilj;
             //   mset(ss->cov_ss, i, j,
               //     mget(ss->cov_ss, i, j) + vget(var->nu, i) + lilj);
            }
            else
            {
#pragma omp atomic update 
                ss->cov_ss->data[(i * ss->cov_ss->tda) + j] += lilj;
//                mset(ss->cov_ss, i, j, mget(ss->cov_ss, i, j) + lilj);
            }
        }
    }
    // topics suff stats
    for (i = 0; i < d->nterms; i++)
    {
        for (j = 0; j < ss->omega_ss->size1; j++)
        {
            w = d->word[i];
            c = d->count[i];
#pragma omp atomic update
            ss->omega_ss->data[(j * ss->omega_ss->tda) + w] += c * var->phi->data[(i * var->phi->tda) + j];
 //           mset(ss->omega_ss, j, w,
 //               mget(ss->omega_ss, j, w) + c * mget(var->phi, i, j));
        }
    }
    // number of data
#pragma omp atomic update
    ss->ndata++;
}

/*
 * importance sampling the likelihood based on the variational posterior
 *
 */

double sample_term(llna_var_param* var, doc* d, llna_model* mod, double* eta)
{
    int i, j, n;
	double t1, t2, sum;
	double *theta;
	theta = (double *) calloc((mod->k),sizeof(double));
	if (theta == NULL) {
		printf("malloc of size %d failed!\n", 50);   // could also call perror here
		exit(1);   // or return an error to caller
	}

    double word_term;

    t1 = (0.5) * mod->log_det_inv_cov;
    t1 += - (0.5) * (mod->k) * 1.837877;
    for (i = 0; i < mod->k; i++)
        for (j = 0; j < mod->k ; j++)
            t1 -= (0.5) *
                (eta[i] - vget(mod->mu, i)) *
                mget(mod->inv_cov, i, j) *
                (eta[j] - vget(mod->mu, j));

    // compute theta
    sum = 0;
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = exp(eta[i]);
        sum += theta[i];
    }
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = theta[i] / sum;
    }

    // compute word probabilities
    for (n = 0; n < d->nterms; n++)
    {
        word_term = 0;
        for (i = 0; i < mod->k; i++)
            word_term += theta[i]*exp(mget(mod->log_omega,i,d->word[n]));
        t1 += d->count[n] * safe_log(word_term);
    }
	free(theta);
    // log(q(\eta | lambda, nu))
    t2 = 0;
    for (i = 0; i < mod->k; i++)
        t2 += safe_log(gsl_ran_gaussian_pdf(eta[i] - vget(var->lambda,i), sqrt(vget(var->nu,i))));
    return(t1-t2);
}


double sample_lhood(llna_var_param* var, doc* d, llna_model* mod)
{
    int nsamples, i, n;
	double *eta = (double *) calloc(mod->k,sizeof(double));
    if (eta != NULL)
    {
        double log_prob, sum = 0, v;
        gsl_rng* r = gsl_rng_alloc(gsl_rng_taus);

        gsl_rng_set(r, (long)1115574245);
        nsamples = 10000;

        // for each sample
        for (n = 0; n < nsamples; n++)
        {
            // sample eta from q(\eta)
            for (i = 0; i < mod->k; i++)
            {
                v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu, i)));
                eta[i] = v + vget(var->lambda, i);
            }
            // compute p(w | \eta) - q(\eta)
            log_prob = sample_term(var, d, mod, eta);
            // update log sum
            if (n == 0) sum = log_prob;
            else sum = log_sum(sum, log_prob);
            // printf("%5.5f\n", (sum - log(n+1)));
        }
        free(eta);
        sum = sum - safe_log((double)nsamples);
        return(sum);
    }
    else
        return DBL_MAX;
}


/*
 * expected theta under a variational distribution
 *
 * (v is assumed allocated to the right length.)
 *
 */


void expected_theta(llna_var_param *var, doc* d, llna_model *mod, gsl_vector* val)
{
    int nsamples, i, n;
    double *eta = (double *) calloc(mod->k,sizeof(double));
	double *theta = (double *) calloc(mod->k, sizeof(double));
	double *e_theta = (double *)calloc(mod->k, sizeof(double));
    if (eta != NULL && theta != NULL && e_theta != NULL)
    {
        double sum, w, v;
        gsl_rng* r = gsl_rng_alloc(gsl_rng_taus);

        gsl_rng_set(r, (long)1115574245);
        nsamples = 100;

        // initialize e_theta
        for (i = 0; i < mod->k; i++) e_theta[i] = -1;
        // for each sample
        for (n = 0; n < nsamples; n++)
        {
            // sample eta from q(\eta)
            for (i = 0; i < mod->k; i++)
            {
                v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu, i)));
                eta[i] = v + vget(var->lambda, i);
            }
            // compute p(w | \eta) - q(\eta)
            w = sample_term(var, d, mod, eta);
            // compute theta
            sum = 0;
            for (i = 0; i < mod->k; i++)
            {
                theta[i] = exp(eta[i]);
                sum += theta[i];
            }
            for (i = 0; i < mod->k; i++)
                theta[i] = theta[i] / sum;
            // update e_theta
            for (i = 0; i < mod->k; i++)
                e_theta[i] = log_sum(e_theta[i], w + safe_log(theta[i]));
        }
        // normalize e_theta and set return vector
        sum = -1;
        for (i = 0; i < mod->k; i++)
        {
            e_theta[i] = e_theta[i] - safe_log(nsamples);
            sum = log_sum(sum, e_theta[i]);
        }
        for (i = 0; i < mod->k; i++)
            vset(val, i, exp(e_theta[i] - sum));
        free(eta);
        free(theta);
        free(e_theta);
    }
    else
        return;
}

/*
 * log probability of the document under proportions theta and topics
 * omega
 *
 */

double log_mult_prob(doc* d, gsl_vector* theta, gsl_matrix* log_omega)
{
    int i, k;
    double ret = 0;
    double term_prob;

    for (i = 0; i < d->nterms; i++)
    {
        term_prob = 0;
        for (k = 0; k < log_omega->size1; k++)
            term_prob += vget(theta, k) * exp(mget(log_omega, k, d->word[i]));
        ret = ret + safe_log(term_prob) * d->count[i];
    }
    return(ret);
}


/*
 * writes the word assignments line for a document to a file
 *
 */

void write_word_assignment(FILE* f, doc* d, gsl_matrix* phi)
{
    int n;

    fprintf(f, "%03d", d->nterms);
    for (n = 0; n < d->nterms; n++)
    {
        gsl_vector phi_row = gsl_matrix_row(phi, n).vector;
        fprintf(f, " %04d:%02d", d->word[n], argmax(&phi_row));
    }
    fprintf(f, "\n");
    fflush(f);
}
